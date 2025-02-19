"""
Contains utility functions for the dask-mwu package.

1. Computes the ranks of the data using scipy.stats.rankdata.
2. Creates the masks to perform element wise masking of in-group and
   out-group data.
3. Broadcasts the masks to create multiple matrices for each condition.

This code uses functions from the SciPy library.
SciPy is licensed under the BSD license.
For more information, visit: https://www.scipy.org/
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
    import dask
except ImportError:
    raise ImportError("dask is not installed. Please install dask to use this package.")

try:
    from scipy._lib._util import _contains_nan
    from scipy.stats._stats_py import _order_ranks, _rankdata
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

    # Step 1: Find unique elements and their indices
    unique_elements, indices = np.unique(choices, return_inverse=True)
    # Step 2: Create one-hot encoding using indices and identity matrix
    one_hot = np.eye(len(unique_elements))[indices]

    logger.debug(f"Mask shape: {one_hot.shape}. {one_hot.__str__()}")

    return da.from_array(one_hot, chunks=(-1, 1)), unique_elements


def rankdata(a, method='average', *, axis=None, nan_policy='propagate') -> np.ndarray:
    """
    From: https://github.com/scipy/scipy/blob/0f1fd4a7268b813fa2b844ca6038e4dfdf90084a/scipy/stats/_stats_py.py#L10108

    Assign ranks to data, dealing with ties appropriately.

    By default (``axis=None``), the data array is first flattened, and a flat
    array of ranks is returned. Separately reshape the rank array to the
    shape of the data array if desired (see Examples).

    Ranks begin at 1.  The `method` argument controls how ranks are assigned
    to equal values.  See [1]_ for further discussion of ranking methods.

    Parameters
    ----------
    a : array_like
        The array of values to be ranked.
    method : {'average', 'min', 'max', 'dense', 'ordinal'}, optional
        The method used to assign ranks to tied elements.
        The following methods are available (default is 'average'):

          * 'average': The average of the ranks that would have been assigned to
            all the tied values is assigned to each value.
          * 'min': The minimum of the ranks that would have been assigned to all
            the tied values is assigned to each value.  (This is also
            referred to as "competition" ranking.)
          * 'max': The maximum of the ranks that would have been assigned to all
            the tied values is assigned to each value.
          * 'dense': Like 'min', but the rank of the next highest element is
            assigned the rank immediately after those assigned to the tied
            elements.
          * 'ordinal': All values are given a distinct rank, corresponding to
            the order that the values occur in `a`.
    axis : {None, int}, optional
        Axis along which to perform the ranking. If ``None``, the data array
        is first flattened.
    nan_policy : {'propagate', 'omit', 'raise'}, optional
        Defines how to handle when input contains nan.
        The following options are available (default is 'propagate'):

          * 'propagate': propagates nans through the rank calculation
          * 'omit': performs the calculations ignoring nan values
          * 'raise': raises an error

        .. note::

            When `nan_policy` is 'propagate', the output is an array of *all*
            nans because ranks relative to nans in the input are undefined.
            When `nan_policy` is 'omit', nans in `a` are ignored when ranking
            the other values, and the corresponding locations of the output
            are nan.

        .. versionadded:: 1.10

    Returns
    -------
    ranks : ndarray
         An array of size equal to the size of `a`, containing rank
         scores.

    References
    ----------
    .. [1] "Ranking", https://en.wikipedia.org/wiki/Ranking

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import rankdata
    >>> rankdata([0, 2, 3, 2])
    array([ 1. ,  2.5,  4. ,  2.5])
    >>> rankdata([0, 2, 3, 2], method='min')
    array([ 1,  2,  4,  2])
    >>> rankdata([0, 2, 3, 2], method='max')
    array([ 1,  3,  4,  3])
    >>> rankdata([0, 2, 3, 2], method='dense')
    array([ 1,  2,  3,  2])
    >>> rankdata([0, 2, 3, 2], method='ordinal')
    array([ 1,  2,  4,  3])
    >>> rankdata([[0, 2], [3, 2]]).reshape(2,2)
    array([[1. , 2.5],
          [4. , 2.5]])
    >>> rankdata([[0, 2, 2], [3, 2, 5]], axis=1)
    array([[1. , 2.5, 2.5],
           [2. , 1. , 3. ]])
    >>> rankdata([0, 2, 3, np.nan, -2, np.nan], nan_policy="propagate")
    array([nan, nan, nan, nan, nan, nan])
    >>> rankdata([0, 2, 3, np.nan, -2, np.nan], nan_policy="omit")
    array([ 2.,  3.,  4., nan,  1., nan])

    """
    methods = ('average', 'min', 'max', 'dense', 'ordinal')
    if method not in methods:
        raise ValueError(f'unknown method "{method}"')

    x = np.asarray(a)

    if axis is None:
        x = x.ravel()
        axis = -1

    if x.size == 0:
        dtype = float if method == 'average' else np.dtype("long")
        return np.empty(x.shape, dtype=dtype)

    contains_nan, nan_policy = _contains_nan(x, nan_policy)

    x = np.swapaxes(x, axis, -1)
    ranks, ties = _rankdata(x, method, return_ties=True)

    if contains_nan:
        i_nan = (np.isnan(x) if nan_policy == 'omit'
                 else np.isnan(x).any(axis=-1))
        ranks = ranks.astype(float, copy=False)
        ranks[i_nan] = np.nan

    ranks = np.swapaxes(ranks, axis, -1)
    ties = np.swapaxes(ties, axis, -1)

    return np.stack((ranks, ties), axis=-1)


def compute_rank(data: da.Array, *, n_features_per_chunk: int) -> da.Array:
    """
    Applies the scipy rankdata function to the data to rank the data + figure out ties.
    Will map the rankdata function to the data array per chunk. Your 1st dim (row) CANNOT
    be chunked. If they are, you will get an error since each row (observation)
    needs to be ranked.

    Memory usage when performing compute() on any of the 2 returning arrays:

    >>> 2 * 10 * ncpus * n_observations * n_features_per_chunk * 8 bytes (int64)

    This is because scipy _rankdata will allocate ~10 additional arrays of equal
    size to the input to _rankdata. Dask usually loads 2 * ncpus chunks at a time.

    It is HIGHLY recommended to save this data to disk after computing the ranks
    because this is one of the more expensive operations.

    Args:
        data: A dask array of shape (n_observations, n_features) where each row
        is a float or integer value. The chunks must be such that each row is
        in a single chunk.
        n_features_per_chunk: The number of features per chunk. This function will
        also rechunk the data array to be (n_observations, n_features_per_chunk).

    Returns:
        A dask array of shape (n_obs, n_features, 2).
        The first array in dimension 2 is the ranks of the data (array[:, :, 0])
        array. The second array is the ties of the data array (array[:, :, 1]).
        The arrays will be chunked by (n_observations, n_features_per_chunk, 1).
        This means that if you save it to zarr, the ranks dimension and ties
        dimension can be loaded independently.
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

    rank_ties = data.map_blocks(rankdata,
                            axis=0,
                            meta=meta_output,
                            dtype=np.int64,
                            chunks=(
                                data.chunks[0],
                                data.chunks[1],
                                (2,)
                            ),
                            new_axis=[2]
                            ).rechunk((data.chunks[0], n_features_per_chunk, 1))

    return rank_ties


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

    This function will compute the ranksums (not lazy evaluate) and store the
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
