import dask.array as da
import numpy as np

from dask_mwu._utils import validate_ranks_and_masks


def _compute_group_means(data: da.Array, mask: da.Array):
    total_sum = data.sum(axis=0).compute()[:, None]

    n1 = np.sum(mask, axis=0).compute()
    n2 = data.shape[0] - n1

    # mu_1 = (data[:, :, None] * mask[:, None, :]).sum(axis=0).compute()
    mu_1 = da.tensordot(data, mask, axes=((0,), (0,))).compute()
    mu_2 = total_sum - mu_1

    mu_1 = mu_1 / n1
    mu_2 = mu_2 / n2

    return mu_1, mu_2


def compute_logfoldchange(
    data: da.Array, mask: da.Array, base: float | None = None
) -> np.ndarray[np.float64]:
    """
    Computes the logfoldchange of the mean expression for every group.

    Args:
        data: Data that represents a measure of counts or abundances. Should
            be a (n_obs, n_features) array
        mask: Data that is a one-hot encoded matrix of observations belonging
            in a group. A (n_obs, n_groups) array

    Returns: a matrix of shape (n_features, n_groups) that contains the
        logfoldchange of the mean expression for every group.

    """
    validate_ranks_and_masks(data, mask)

    mu_1, mu_2 = _compute_group_means(data, mask)

    # prevent log(0)
    small_offset = 1e-9

    expm1_func = np.expm1 if base is None else lambda x: np.expm1(x * np.log(base))

    lfc = np.log2(expm1_func(mu_1) + small_offset) - np.log2(
        expm1_func(mu_2) + small_offset
    )

    return lfc
