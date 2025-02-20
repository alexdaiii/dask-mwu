import numpy as np
import pytest
from scipy.stats import mannwhitneyu
from scipy.stats._stats_py import _rankdata
from sklearn.preprocessing import OneHotEncoder

import dask.array as da
from statsmodels.stats.multitest import multipletests

from dask_mwu.pvals import mann_whitney_u
from dask_mwu.rank_data import compute_tie_term

rng = np.random.default_rng(42)


@pytest.mark.parametrize("chunks", [1, 2, 3, 6])
@pytest.mark.parametrize(
    "_name, data, classes",
    [
        (
            "all_positive_and_0",
            rng.integers(0, 100, size=(25, 12)),
            rng.integers(5, size=25),
        ),
        (
            "all_negative",
            rng.integers(-100, 0, size=(25, 12)),
            rng.integers(5, size=25),
        ),
        (
            "mixed",
            rng.integers(-100, 100, size=(25, 12)),
            rng.integers(5, size=25),
        ),
        (
            "with ties",
            np.array(
                [
                    [1, 2, 3, 4, 5, 6],
                    [1, 2, 3, 4, 5, 6],
                    [1, 2, 3, 4, 5, 6],
                    [1, 2, 3, 4, 5, 6],
                    [1, 2, 3, 4, 5, 6],
                ]
            ),
            rng.integers(6, size=5),
        ),
    ],
)
def test_p_vals(_name: str, data: np.ndarray, classes: np.ndarray, chunks: int) -> None:
    x = np.swapaxes(data, 0, -1)
    ranks, ties = _rankdata(x, method="average", return_ties=True)

    ties = da.from_array(np.swapaxes(ties, 0, -1), chunks=(-1, chunks))
    ranks = np.swapaxes(ranks, 0, -1)

    tie_term = compute_tie_term(ties)

    # one hot encode the masks
    enc = OneHotEncoder(sparse_output=False)
    masks = enc.fit_transform(classes.reshape(-1, 1))

    # broadcast multiply the ranks and masks
    ranks_sum = (ranks[:, :, None] * masks[:, None, :]).sum(axis=0)

    # actual
    actual_mwu = mann_whitney_u(
        ranks_sum, tie_term, da.from_array(masks, chunks=(-1, 1))
    )

    u_expected, p_expected, p_adj = [], [], []

    # expected - the slow unvectorized version
    for experiment in masks.T:
        group1 = data[experiment == 1]
        group2 = data[experiment == 0]

        u_stat, p_vals = mannwhitneyu(group1, group2)

        u_expected.append(u_stat)
        p_expected.append(p_vals)
        p_adj.append(multipletests(p_vals, method="fdr_bh")[1])

    u_expected = np.array(u_expected).T
    p_expected = np.array(p_expected).T
    p_adj_expected = np.array(p_adj).T

    np.testing.assert_allclose(actual_mwu.U, u_expected)
    np.testing.assert_allclose(actual_mwu.p_vals, p_expected)
    np.testing.assert_allclose(actual_mwu.p_adj, p_adj_expected)
