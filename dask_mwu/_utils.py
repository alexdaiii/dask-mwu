import numpy as np
from dask import array as da


__all__ = [
    "InvalidDimensionError",
    "InvalidChunkSizeError",
    "EmptyArrayError",
    "validate_ranks_and_masks"
]

class InvalidDimensionError(ValueError):
    pass


class InvalidChunkSizeError(ValueError):
    pass


class EmptyArrayError(ValueError):
    pass


def validate_ranks_and_masks(arr: da.Array, oh: da.Array) -> None:
    """
    Validates the array and one hot encoded array of observations in a group
    are valid

    Args:
        arr: Array of observations for every feature (n, m)
        oh: One hot encoded matrix (n, p)
    """
    # check the shapes to make sure they are compatible
    if arr.shape[0] != oh.shape[0]:
        raise InvalidDimensionError(
            "The number of observations in the ranks and masks arrays must be the same."
        )

    # This is just to make implementation easier
    if arr.ndim != 2:
        raise InvalidDimensionError("ranks must be a 2D array.")

    if oh.ndim != 2:
        raise InvalidDimensionError("masks must be a 2D array.")

    if not np.any(oh, axis=1).all():
        raise ValueError("Each observation must belong to at least one group.")

    if np.any(oh.sum(axis=1) > 1):
        raise ValueError("Each observation can only belong to one group.")
