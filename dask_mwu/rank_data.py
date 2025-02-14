"""
Contains utility functions for the dask-mwu package.

1. Computes the ranks of the data using scipy.stats.rankdata.
2. Creates the masks to perform element wise masking of in-group and
   out-group data.
3. Broadcasts the masks to create multiple matrices for each condition.
4. Saving, and rechunking zarr arrays
"""

import logging

from dask_mwu.errors import (
    InvalidDimensionError,
    EmptyArrayError,
    InvalidChunkSizeError,
)

# check if deps installed
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

try:
    from scipy.stats import rankdata
except ImportError:
    raise ImportError(
        "scipy is not installed. Please install scipy to use this package."
    )


_LOG_FORMAT_DEBUG = (
    "%(asctime)s:: %(levelname)s:: %(message)s:: %(pathname)s:%(funcName)s:%(lineno)d"
)


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def get_masks(
    choices: np.ndarray,
) -> tuple[da.Array, np.ndarray]:
    """
    Computes one-hot encoded masks for each group in the choices array.
    The choices array is a 1D or 2D array where each element represents
    the group that the observation belongs to. This will return a 2D array
    of masks where each column is a boolean mask that selects the observations
    that belong to that group.

    Args:
        choices: A vector of (n_groups, ) or (n_groups, 1)
        categorical like data that represents
        what group each observation belongs to.

    Returns:
        masks: A dask array of shape (n_observations, n_groups) where each row
            is a boolean mask that selects the observations that belong to
            that group.
        groups: A numpy array of unique group identifiers.
    """
    # Reshape choices if it's 1D

    if choices.shape[0] == 0:
        raise EmptyArrayError("choices must have at least one element.")

    print(choices.ndim, choices.shape)

    if choices.ndim > 2:
        raise InvalidDimensionError(
            "choices must be a 1D or 2D array with a single column."
        )

    # Flatten the array if it has more than one dimension
    if choices.ndim > 1:
        choices = choices.flatten()

    # groups = da.unique(choices).compute()
    #
    # logger.debug(f"There are {len(groups)} unique groups.")
    #
    # masks = da.from_array(
    #     da.concatenate(
    #         [da.where(choices == group, True, False) for group in groups], axis=1
    #     ).compute(),
    #     chunks=(-1, 1),
    # )

    # Step 1: Find unique elements and their indices
    unique_elements, indices = np.unique(choices, return_inverse=True)
    # Step 2: Create one-hot encoding using indices and identity matrix
    one_hot = np.eye(len(unique_elements))[indices]

    logger.debug(f"Mask shape: {one_hot.shape}. {one_hot.__str__()}")

    return da.from_array(one_hot, chunks=(-1, 1)), unique_elements


def compute_rank(data: da.Array, *, n_features_per_chunk: int) -> da.Array:
    """
    Applies the scipy rankdata function to the data to rank the data + figure out ties.
    Will map the rankdata function to the data array per chunk. Your 1st dim (row) CANNOT
    be chunked. If they are, you will get an error since each row (observation)
    needs to be ranked.

    Memory usage: `2 * 8 * ncpus * n_observations * input_features_per_chunk * 8 bytes (int64)`
    This is because scipy rank data will allocate ~8 additional arrays of equal
    size to the input to rankdata. Dask usually loads 2 * ncpus chunks at a time.

    It is HIGHLY recommended to save this data to disk after computing the ranks
    because this is one of the more expensive operations.

    Args:
        data: A dask array of shape (n_observations, n_features) where each row
        is a float or integer value. The chunks must be such that each row is
        in a single chunk.
        n_features_per_chunk: The number of features per chunk. This function will
        also rechunk the data array to be (n_observations, n_features_per_chunk).

    Returns:
        ranks: A dask array of shape (n_observations, n_features) where each row
        is the rank of the corresponding row in the data array. The data will be
        (lazily) rechunked to (n_observations, n_features_per_chunk).
    """
    # perform the checks
    if data.ndim != 2:
        raise InvalidDimensionError("data must be a 2D array.")

    if len(data.chunks[0]) != 1:
        # in dask for a 2D array if the 1 dim chunks are not (N,) then the 1 dim are chunked
        raise InvalidChunkSizeError(
            "The rows of the data array must be in a single chunk."
        )

    meta_output = np.array([], dtype=np.int64)

    ranks = data.map_blocks(rankdata, axis=0, meta=meta_output, dtype=np.int64).rechunk(
        (-1, n_features_per_chunk)
    )

    return ranks


def compute_ranks_per_group(
    ranks: da.Array,
    masks: da.Array,
):
    """
    Taking in a (n_obs, n_features) array of ranks and a (n_obs, n_groups) array of masks,
    it will take the ranks and convert it into a 3D array of (n_obs, n_features, 1).
    It converts the masks into a 3D array of (n_obs, 1, n_groups).

    It will broadcast and element-wise multiply the two groups to get an (n_obs, n_features, n_groups)
    array. What that means is that it repeats the ranks by the number of groups.
    Then it takes each column of the mask and replicates the mask to match the
    shape of the ranks. Then it element-wise multiplies the two arrays.

    By default, masks from the get_masks function are chunked so that each column
    is in a single chunk. The ranks from the compute_rank function are chunked
    so that only the 2nd dimension is chunked by n_features_per_chunk. This
    allows for the computation to load all observations for a group for n_features_per_chunk
    at a time. The expected memory usage is 2 * ncpus * n_obs * n_features_per_chunk * 8 bytes
    since each group is inside a different chunk.

    Args:
        ranks: A dask array of shape (n_obs, n_features) where each row
        is the rank of the corresponding row in the data array
        masks: A dask array of shape (n_observations, n_groups) where each row
        is a boolean mask that selects the observations that belong to that group.

    Returns:
        ranks_per_group: This is a 3D array of shape (n_obs, n_features, n_groups) where
        each slice along the last dimension is the ranks of the data for that group.
    """
    # check the shapes to make sure they are compatible
    if ranks.shape[0] != masks.shape[0]:
        raise InvalidDimensionError(
            "The number of observations in the ranks and masks arrays must be the same."
        )

    # This is just to make implementation easier
    if ranks.ndim != 2:
        raise InvalidDimensionError("ranks must be a 2D array.")

    if masks.ndim != 2:
        raise InvalidDimensionError("masks must be a 2D array.")

    if not np.any(masks, axis=1).all():
        raise ValueError("Each observation must belong to at least one group.")

    if np.any(masks.sum(axis=1) > 1):
        raise ValueError("Each observation can only belong to one group.")

    return ranks[:, :, None] * masks[:, None, :]


def compute_in_group_ranksum(
    ranks: da.Array,
    masks: da.Array,
):
    """
    When comparing 2 groups G1 and G2 in the MWU test, we need to compute the
    sum of ranks for the in-group (G1) data. For every feature, this function
    will sum the ranks of the observations that belong to group G1.

    The sum of all ranks is equal to (n_obs * (n_obs + 1)) / 2. To get the
    sum of the ranks for the other group (G2), we can subtract the sum of
    the ranks of G1 (results from this function) from the total sum of ranks.

    This function WILL compute the ranksums (not lazy evaluate) and store the
    ranksums in memory.

    Args:
        ranks: A dask array of shape (n_obs, n_features) where each position
            is the rank of the data within the column.
        masks: A dask array of shape (n_obs, n_groups) where each row is a boolean
            mask that selects the observations that belong to group G1.

    Returns: A dask array of shape (n_features, n_groups) where each column
        is the sum of the ranks of the observations that belong to group G1.

    """
    ranksums = compute_ranks_per_group(ranks, masks).sum(axis=0).compute()

    return ranksums


__all__ = [
    "get_masks",
    "compute_rank",
    "compute_ranks_per_group",
    "compute_in_group_ranksum",
]
