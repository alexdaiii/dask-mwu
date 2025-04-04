from typing import Type

import dask.array as da

import numpy as np
import pytest
from scipy.stats import rankdata
from scipy.stats._stats_py import _rankdata
from sklearn.preprocessing import OneHotEncoder

from dask_mwu._utils import InvalidChunkSizeError, InvalidDimensionError
from dask_mwu.rank_data import (
    compute_in_group_ranksum,
    rank_data,
)


class TestRankData:
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
            [
                "with nans",
                np.array(
                    [
                        [1, 2, 3, np.nan, 11, 12],
                        [4, 5, np.nan, 6, 13, 14],
                        [7, np.nan, 9, 10, np.nan, np.nan],
                    ]
                ),
            ],
        ],
    )
    def test_compute_rank(
        self, chunks: tuple[int, int], _name: str, data: np.ndarray, output_chunks: int
    ) -> None:
        arr = da.from_array(data, chunks=chunks)

        # rank the data and apply data so each row is ranked, columns are independent
        expected = rankdata(data, axis=0)

        # From scipy.stats.rankdata
        x = np.swapaxes(data, 0, -1)
        _, expected_ties = _rankdata(x, method="average", return_ties=True)
        expected_ties = np.swapaxes(expected_ties, 0, -1)

        # compute the ranks
        actual = rank_data(arr, n_features_per_chunk=output_chunks)
        actual_cmp = actual.compute()

        actual_ranks = actual_cmp[:, :, 0]
        actual_ties = actual_cmp[:, :, 1]

        assert expected.shape == actual_ranks.shape
        assert expected_ties.shape == actual_ties.shape
        assert np.allclose(expected, actual_ranks, equal_nan=True)
        assert np.allclose(expected_ties, actual_ties, equal_nan=True)
        assert actual.chunksize == (data.shape[0], output_chunks, 1)

        # with dask.config.set({"visualization.engine": "cytoscape"}):
        #     actual.visualize(f"../output/test_compute_rank_{_name}.png")

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
    def test_invalid_compute_rank(
        self, _name, data: da.Array, error: Type[Exception]
    ) -> None:
        with pytest.raises(error):
            rank_data(data, n_features_per_chunk=2)


class TestRankSum:
    rng = np.random.default_rng(42)
    enc = OneHotEncoder(sparse_output=False)

    @pytest.mark.parametrize("chunks", [1, 2, 3, 4, 6])
    @pytest.mark.parametrize(
        "_name, ranks, masks",
        [
            (
                "5 obs, 12 features, 4 groups",
                rankdata(
                    rng.integers(0, 100, size=(25, 12)),
                    axis=0,
                ),
                da.from_array(enc.fit_transform(rng.integers(5, size=(25, 1)))),
            ),
            (
                "1 obs, 1 feature, 1 group",
                rankdata(np.array([[10000]]), axis=0),
                da.from_array([[True]]),
            ),
        ],
    )
    def test_ranksum(
        self, _name: str, ranks: da.Array, masks: da.Array, chunks: int
    ) -> None:
        actual = compute_in_group_ranksum(
            da.from_array(ranks, chunks=(-1, chunks)), masks
        )

        # with dask.config.set({"visualization.engine": "cytoscape"}):
        #     actual.visualize(f"../output/test_ranksum_{_name}_chunks_{chunks}.png")

        # should be the sum of ranks for the group, per feature
        assert actual.shape == (ranks.shape[1], masks.shape[1])

        # should be a np float64
        assert actual.dtype == np.float64

        for mask_col in range(masks.shape[1]):
            mask = masks[:, mask_col]
            expected_group = (ranks * mask.reshape(-1, 1)).sum(axis=0).compute()

            actual_col = actual[:, mask_col]

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
        self, _name: str, ranks: da.Array, masks: da.Array, error: Type[Exception]
    ):
        with pytest.raises(error):
            compute_in_group_ranksum(ranks, masks)
