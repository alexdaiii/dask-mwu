"""
Calculate p-values for the Mann-Whitney U test using Dask arrays.

This code uses functions from the SciPy library.
SciPy is licensed under the BSD license.
For more information, visit: https://www.scipy.org/
"""

from typing import NamedTuple


from statsmodels.stats.multitest import multipletests


import numpy as np


from scipy import stats


import dask.array as da


__all__ = ["mann_whitney_u"]


def _get_mwu_z(
    u_stat: np.ndarray[np.int64],
    n1: np.ndarray[np.int64],
    n2: np.ndarray[np.int64],
    tie_term: np.ndarray[np.int64],
):
    """
    Copied from https://github.com/scipy/scipy/blob/0f1fd4a7268b813fa2b844ca6038e4dfdf90084a/scipy/stats/_mannwhitneyu.py#L153

    Args:
        u_stat: The U statistic. An array of shape (n_features, n_conditions)
        n1: The number of samples in the first group. An array of shape (n_conditions,)
        n2: The number of samples in the second group. An array of shape (n_conditions,)
        tie_term: A precomputed array of shape (n_features,) that contains the tie term for each feature.

    Returns:
        z: The Z statistic. An array of shape (n_features, n_conditions)
    """
    # Follows mannwhitneyu [2]
    mu = n1 * n2 / 2
    n = n1 + n2

    # Tie correction according to [2], "Normal approximation and tie correction"
    # "A more computationally-efficient form..."
    # the tie term is repeated across the 2nd dimension to allow for broadcasting
    sigma = np.sqrt(n1 * n2 / 12 * ((n + 1) - tie_term[:, None] / (n * (n - 1))))

    numerator = u_stat - mu

    # Continuity correction.
    # Because SF is always used to calculate the p-value, we can always
    # _subtract_ 0.5 for the continuity correction. This always increases the
    # p-value to account for the rest of the probability mass _at_ q = U.
    numerator -= 0.5

    # no problem evaluating the norm SF at an infinity
    with np.errstate(divide="ignore", invalid="ignore"):
        z = numerator / sigma
    return z


class MannWhitneyU(NamedTuple):
    """
    A class that contains the U statistic and p-values for the Mann-Whitney U test.
    """

    U: np.ndarray[np.float64]
    p_vals: np.ndarray[np.float64]
    p_adj: np.ndarray[np.float64]


def _mwu_from_rank_sums(
    rank_sum: np.ndarray[np.float64],
    tie_term: np.ndarray[np.int64],
    masks: da.Array,
) -> tuple[
    np.ndarray[np.float64],
    np.ndarray[np.float64],
]:
    """
    Compute the U statistic and p-values from the ranksums and masks. This
    assumes that ranksum is a matrix of (n_features, n_groups) where each
    position is the sum of the ranks for group G1 for that feature. The masks
    should be a matrix of (n_observations, n_groups).

    In the Mann-Whitney U test, two-tailed, we compare groups G1 and G2.
    The ranksums should be the sum of the ranks for group G1.
    The masks should be a boolean matrix that selects observations that
    belong to G1.

    This function assumes there are ties and uses the formula for the standard
    deviation that accounts for ties.

    Args:
        rank_sum: A numpy array of shape (n_features, n_groups) that contains
            the sum of the ranks for group G1 for each feature.
        tie_term: A numpy array of shape (n_features, ) that contains the tie
            term for each feature.
        masks: A dask array of shape (n_observations, n_groups) where each row
            is a boolean mask that selects the observations that belong to
            that group.

    Returns:
        U: A numpy array of shape (n_features, ) that contains the U statistic
            for each feature.
        p_vals: A numpy array of shape (n_features, ) that contains the p-value
            (uncorrected) for each feature. You need to apply FDR or FWER
            correction to these p-values because n_features will usually be
            very large.
    """
    n1 = masks.sum(axis=0).compute()
    n2 = masks.shape[0] - n1

    r1 = rank_sum - (n1 * (n1 + 1)) / 2
    r2 = (n1 * n2) - r1
    u_stat = np.maximum(r1, r2)

    z = _get_mwu_z(u_stat, n1, n2, tie_term)
    p = 2 * stats.norm.sf(z)

    # Ensure that test statistic is not greater than 1
    # This could happen for exact test when U = m*n/2
    p = np.clip(p, 0, 1)

    return r1, p


def _bh_vec(chunk: np.ndarray[np.float64 | np.float32]):
    chunk = chunk.flatten()

    _, bh_corrected, _, _ = multipletests(chunk, method="fdr_bh")

    return bh_corrected[:, None]


def _get_padj(pvals: np.ndarray[np.float64 | np.float32]):
    p_adj = da.from_array(pvals, chunks=(-1, 1))

    p_adj = p_adj.map_blocks(_bh_vec, dtype=p_adj.dtype).compute()

    return p_adj


def mann_whitney_u(
    rank_sum: np.ndarray[np.float64],
    tie_term: np.ndarray[np.int64],
    masks: da.Array,
) -> MannWhitneyU:
    """
    Compute the U statistic and p-values for the Mann-Whitney U test.

    Args:
        rank_sum: A dask array of shape (n_features, n_groups) where each position
            is the sum of the ranks for that feature.
        tie_term: A dask array of shape (n_features,) where each position is the
            tie term (t^3 - t) for that feature.
        masks: A dask array of shape (n_observations, n_groups) where each row
            is a boolean mask that selects the observations that belong to
            that group.

    Returns:
        A tuple of -
        U: A numpy array of shape (n_features, n_groups) that contains the U statistic
        for each feature.
        p_vals: A numpy array of shape (n_features, n_groups) that contains the p-value
        (uncorrected) for each feature. You need to apply FDR or FWER
        correction to these p-values because n_features will usually be
        very large.
        p_adj: A numpy array of shape (n_features, n_groups) that contains the adjusted
        p-values for each feature.

    """
    u_stat, p_vals = _mwu_from_rank_sums(rank_sum, tie_term, masks)
    p_adj = _get_padj(p_vals)

    return MannWhitneyU(U=u_stat, p_vals=p_vals, p_adj=p_adj)
