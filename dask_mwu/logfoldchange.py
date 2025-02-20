import dask.array as da
import numpy as np


def compute_logfoldchange(
    data: da.Array,
    mask: da.Array,
):
    """
    Computes the logfoldchange of the mean expression for every group.

    Args:
        data: Data that represents a measure of counts or abundances. Should
            be a (n_obs, n_features) array
        mask: Data that is a one-hot encoded matrix of observations belonging
            in a group. A (n_obs, n_groups) array

    Returns:

    """
    total_sum = data.sum(axis=0).compute()[:, None]

    n1 = np.sum(mask, axis=0).compute()
    n2 = data.shape[0] - n1

    # mu_1 = (data[:, :, None] * mask[:, None, :]).sum(axis=0).compute()
    mu_1 = da.tensordot(data, mask, axes=((0,), (0,))).compute()
    mu_2 = total_sum - mu_1

    mu_1 = mu_1 / n1
    mu_2 = mu_2 / n2

    # prevent log(0)
    small_offset = 1e-9

    lfc = np.log2(np.expm1(mu_1) + small_offset) - np.log2(np.expm1(mu_2) + small_offset)

    return lfc
