import numpy as np
import pandas as pd
import pytest
import dask.array as da
import anndata as ad

from dask_mwu.logfoldchange import compute_logfoldchange, _compute_group_means
from dask_mwu.rank_data import get_masks

rng = np.random.default_rng(42)


class TestLogFoldChange:
    chunks = [1, 2, 3, 4, 6]
    examples = [
        ("positive", rng.integers(100, size=(25, 12)), rng.integers(5, size=(25, 1))),
        ("all zero", np.zeros((25, 12)), rng.integers(5, size=(25, 1))),
        ("2 groups", rng.integers(50, size=(25, 12)), rng.integers(2, size=(25, 1))),
    ]

    def setup_anndata(
        self, data: np.ndarray, groups: np.ndarray, base: float | None = None
    ):
        obs = pd.DataFrame(
            {
                "class": groups.flatten(),
            }
        )

        adata = ad.AnnData(
            X=data,
            obs=obs,
        )
        adata.obs["class"] = adata.obs["class"].astype("category")
        # sc.pp.log1p(adata, base=base)

        return adata

    def get_data_masks(self, data: np.ndarray, groups: np.ndarray):
        data = da.from_array(data, chunks=(-1, 1))
        mask, categories = get_masks(groups)
        data = np.log1p(data)

        return data, mask, categories

    @pytest.mark.parametrize("chunks", chunks)
    @pytest.mark.parametrize("_name, data, groups", examples)
    def test_compute_means_for_logfoldchange(
        self,
        chunks: int,
        _name: str,
        data: np.ndarray,
        groups: np.ndarray,
        get_ranked_data,
    ):
        data, mask, categories = self.get_data_masks(data, groups)
        # expected (from scanpy - which the lfc code is copied from)
        adata = self.setup_anndata(data, groups)

        rg = get_ranked_data(
            adata,
        )

        mu1, mu2 = _compute_group_means(data, mask)

        for idx, _, _ in rg.wilcoxon(tie_correct=False):
            assert np.allclose(rg.means[idx], mu1[:, idx].T)

            assert np.allclose(rg.means_rest[idx], mu2[:, idx].T)

        assert np.allclose(mu1, rg.means.T)
        assert np.allclose(mu2, rg.means_rest.T)

    @pytest.mark.parametrize("base", [None, 2, 10])
    @pytest.mark.parametrize("chunks", chunks)
    @pytest.mark.parametrize("_name, data, groups", examples)
    def test_compute_logfoldchange(
        self,
        chunks: int,
        _name: str,
        data: np.ndarray,
        groups: np.ndarray,
        base: float | None,
        get_ranked_data,
    ):
        data, mask, categories = self.get_data_masks(data, groups)
        adata = self.setup_anndata(data, groups, base=base)

        rg = get_ranked_data(
            adata,
            base
        )

        actual = compute_logfoldchange(
            data, mask, base=adata.uns.get("log1p", {}).get("base")
        )

        # make sure its the correct shape
        assert actual.shape == (data.shape[1], len(categories))

        for i, cat in enumerate(categories):
            actual_lfc = actual[:, i]

            expected = rg.stats[(f"{i}", "logfoldchanges")].values.T

            assert expected.shape == actual_lfc.shape

            assert np.allclose(expected, actual_lfc)


