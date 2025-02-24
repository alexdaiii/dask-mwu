import pytest
import anndata as ad
import scanpy as sc
from scanpy.tools._rank_genes_groups import _RankGenes


@pytest.fixture
def setup_anndata():
    def _setup_anndata(adata: ad.AnnData, base: float):
        sc.pp.log1p(adata, base=base)

        return adata

    return _setup_anndata


@pytest.fixture
def get_ranked_data(setup_anndata):
    def _get_ranked_data(adata: ad.AnnData, base: float | None = None):
        adata = setup_anndata(adata, base)

        rg = _RankGenes(
            adata=adata,
            groups="all",
            groupby="class",
            use_raw=adata.raw is not None,
            mask_var=None,
            reference="rest",
            layer=None,
            comp_pts=False,
        )

        rg.compute_statistics(
            method="wilcoxon",
            tie_correct=False,
            rankby_abs=False,
            corr_method="benjamini-hochberg",
            # If this is not NONE then it will try to sort which is not what we want
            n_genes_user=None,
        )

        return rg

    return _get_ranked_data
