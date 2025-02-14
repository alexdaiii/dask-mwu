

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




def mwu_from_rank_sums(
        rank_sums: da.Array,
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

    u1 = rank_sums - (n1 * (n1 + 1)) / 2
    u_stat = np.minimum(u1, n1 * n2 - u1)





def compute_tie_term(
        ranks: da.Array,
):
    ...

