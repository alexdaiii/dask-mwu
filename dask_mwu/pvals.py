"""
Calculate p-values for the Mann-Whitney U test using Dask arrays.

This code uses functions from the SciPy library.
SciPy is licensed under the BSD license.
For more information, visit: https://www.scipy.org/
"""

try:
    # this must be first since its actually a dep of dask
    import numpy as np
except ImportError:
    raise ImportError(
        "numpy is not installed. Please install numpy to use this package."
    )


try:
    import dask.array as da
except ImportError:
    raise ImportError("dask is not installed. Please install dask to use this package.")



def _get_mwu_z(u_stat: np.ndarray[np.int64],
               n1: np.ndarray[np.int64],
               n2: np.ndarray[np.int64],
               ties: da.Array):
    """
    Copied from https://github.com/scipy/scipy/blob/0f1fd4a7268b813fa2b844ca6038e4dfdf90084a/scipy/stats/_mannwhitneyu.py#L153

    Args:
        u_stat: The U statistic. An array of shape (n_features, n_conditions)
        n1: The number of samples in the first group. An array of shape (n_conditions,)
        n2: The number of samples in the second group. An array of shape (n_conditions,)
        ties: The number of ties in the data. An array of shape (n_features, n_conditions)

    Returns:
        z: The Z statistic. An array of shape (n_features, n_conditions)
    """
    # Follows mannwhitneyu [2]
    mu = n1 * n2 / 2
    n = n1 + n2

    # Tie correction according to [2], "Normal approximation and tie correction"
    # "A more computationally-efficient form..."
    tie_term = (t**3 - t).sum(axis=-1)
    s = np.sqrt(n1*n2/12 * ((n + 1) - tie_term/(n*(n-1))))

    numerator = U - mu - 0.5

    # Continuity correction.
    # Because SF is always used to calculate the p-value, we can always
    # _subtract_ 0.5 for the continuity correction. This always increases the
    # p-value to account for the rest of the probability mass _at_ q = U.
    numerator -= 0.5

    # no problem evaluating the norm SF at an infinity
    with np.errstate(divide='ignore', invalid='ignore'):
        z = numerator / s
    return z


def mwu_from_rank_sums(
        ranks: da.Array,
        ties: da.Array,
        masks: da.Array,
):
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
        rank_sums: A dask array of shape (n_features, n_groups) where each
            position is the sum of the ranks for group G1 for that feature.
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

    u1 = ranks - (n1 * (n1 + 1)) / 2
    u_stat = np.minimum(u1, n1 * n2 - u1)





def compute_tie_term(
        ranks: da.Array,
):
    ...

