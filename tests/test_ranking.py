from typing import Type

import numpy as np
import pytest
import dask.array as da

from scipy.stats import rankdata

from dask_mwu.errors import InvalidDimensionError, InvalidChunkSizeError
from dask_mwu.rank_data import compute_rank, compute_ranks_per_group, \
    compute_in_group_ranksum

rng = np.random.default_rng(42)


@pytest.mark.parametrize("chunks", [(-1, 2), (-1, 3), (-1, 6)])
@pytest.mark.parametrize("output_chunks", [2, 3, 6])
@pytest.mark.parametrize(
    "_name, data",
    [
        ["all_positive_and_0", rng.integers(0, 100, size=(25, 12))],
        ["all_negative", rng.integers(-100, 0, size=(25, 12))],
        ["mixed", rng.integers(-100, 100, size=(25, 12))],
        [
            "with ties",
            np.array(
                [
                    [1, 2, 3, 4, 5, 6],
                    [6, 5, 4, 3, 2, 1],
                    [2, 2, 3, 2, 3, 3],
                    [4, 4, 4, 4, 4, 4],
                    [0, 1, 3, 5, 7, 9],
                ]
            ),
        ],
    ],
)
def test_compute_rank(
    chunks: tuple[int, int], _name: str, data: np.ndarray, output_chunks: int
) -> None:
    arr = da.from_array(data, chunks=chunks)

    # rank the data and apply data so each row is ranked, columns are independent
    expected = rankdata(data, axis=0)

    actual = compute_rank(arr, n_features_per_chunk=output_chunks)
    actual_cmp = actual.compute()

    assert expected.shape == actual_cmp.shape
    assert np.allclose(expected, actual_cmp)
    assert actual.chunksize == (data.shape[0], output_chunks)


@pytest.mark.parametrize(
    "_name, data, error",
    [
        ["1d_array", da.array([1, 2, 3, 4, 5]), InvalidDimensionError],
        [
            "3d_array",
            da.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
            InvalidDimensionError,
        ],
        [
            "row_chunked_1",
            da.array(
                [
                    [1, 2, 3, 4, 5],
                    [6, 7, 8, 9, 10],
                    [11, 12, 13, 14, 15],
                ],
            ).rechunk(chunks=(2, 5)),
            InvalidChunkSizeError,
        ],
    ],
)
def test_invalid_compute_rank(_name, data: da.Array, error: Type[Exception]) -> None:
    with pytest.raises(error):
        compute_rank(data, n_features_per_chunk=2)


@pytest.mark.parametrize(
    "_name, ranks, masks",
    [
        (
            "5 obs, 3 features, 2 groups",
            da.from_array([[1, 5, 3], [2, 4, 1], [3, 3, 2], [4, 2, 5], [5, 1, 4]]),
            da.from_array(
                [
                    [True, False],
                    [False, True],
                    [False, True],
                    [True, False],
                    [True, False],
                ]
            ),
        ),
        (
            "1 obs, 1 feature, 1 group",
            da.from_array([[10000]]),
            da.from_array([[True]]),
        ),
        (
            "with ties",
            da.from_array([[1.5, 2.5], [3.5, 4.5]]),
            da.from_array([[True, False], [False, True]]),
        ),
    ],
)
def test_compute_ranks_per_group(_name, ranks, masks):
    actual = compute_ranks_per_group(ranks, masks)

    for mask_col in range(masks.shape[1]):
        expected_group = ranks * masks[:, mask_col].reshape(-1, 1)
        actual_col = actual[:, :, mask_col].compute()

        assert expected_group.shape == actual_col.shape

        assert np.allclose(expected_group, actual_col)


@pytest.mark.parametrize(
    "_name, ranks, masks, error",
    [
        (
            "Somehow all true",
            da.from_array(
                [
                    [1, 5, 3],
                    [2, 4, 1],
                    [3, 3, 2],
                ]
            ),
            da.from_array(
                [
                    [True, True],
                    [True, True],
                    [True, True],
                ]
            ),
            ValueError,
        ),
        (
            "Somehow all false",
            da.from_array(
                [
                    [1, 5, 3],
                    [2, 4, 1],
                    [3, 3, 2],
                ]
            ),
            da.from_array(
                [
                    [False, False],
                    [False, False],
                    [False, False],
                ]
            ),
            ValueError,
        ),
        (
            "Mask is larger than ranks",
            da.from_array(
                [
                    [1, 5, 3],
                    [2, 4, 1],
                    [3, 3, 2],
                ]
            ),
            da.from_array(
                [
                    [True, False],
                    [False, True],
                    [False, True],
                    [True, False],
                ]
            ),
            InvalidDimensionError,
        ),
        (
            "Mask is smaller than ranks",
            da.from_array(
                [
                    [1, 5, 3],
                    [2, 4, 1],
                    [3, 3, 2],
                ]
            ),
            da.from_array(
                [
                    [True, False],
                    [False, True],
                ]
            ),
            InvalidDimensionError,
        ),
        (
            "Ranks is 1D",
            da.from_array([1, 2]),
            da.from_array(
                [
                    [True, False],
                    [False, True],
                ]
            ),
            InvalidDimensionError,
        ),
        (
            "Mask is 1D",
            da.from_array(
                [
                    [1, 5, 3],
                    [2, 4, 1],
                    [3, 3, 2],
                ]
            ),
            da.from_array([True, False, True]),
            InvalidDimensionError,
        ),
    ],
)
def test_invalid_compute_ranks_per_group(
    _name: str, ranks: da.Array, masks: da.Array, error: Type[Exception]
):
    with pytest.raises(error):
        compute_ranks_per_group(ranks, masks)


@pytest.mark.parametrize(
    "_name, ranks, masks",
    [
        (
            "5 obs, 3 features, 2 groups",
            rankdata(np.array([[0, 1, 1],
                               [1, 4, 1],
                               [1, 3, 1],
                               [1, 2, 1],
                               [2, 6, 1]]), axis=0),
            da.from_array(
                [
                    [True, False],
                    [False, True],
                    [False, True],
                    [True, False],
                    [True, False],
                ]
            ),
        ),
        (
            "1 obs, 1 feature, 1 group",
            rankdata(np.array([[10000]]), axis=0),
            da.from_array([[True]]),
        ),
    ],
)
def test_ranksum(_name: str, ranks: da.Array, masks: da.Array):

    actual = compute_in_group_ranksum(da.from_array(ranks), masks)

    # should be the sum of ranks for the group, per feature
    assert actual.shape == (ranks.shape[1], masks.shape[1])

    for mask_col in range(masks.shape[1]):
        mask = masks[:, mask_col]
        expected_group = (ranks * mask.reshape(-1, 1)).sum(axis=0).compute()

        actual_col = actual[:, mask_col]

        assert np.allclose(expected_group, actual_col)
