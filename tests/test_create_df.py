import numpy as np
import pandas as pd
import pytest
import anndata as ad

from dask_mwu.create_df import create_df

rng = np.random.default_rng(42)


@pytest.mark.parametrize("top_n", [None, 5, 10, 12])
@pytest.mark.parametrize(
    "sort",
    [
        "asc",
        "desc",
    ],
)
@pytest.mark.parametrize(
    "_name, data",
    [
        (
            "positive",
            ad.AnnData(
                X=rng.integers(100, size=(25, 12)),
                obs=pd.DataFrame({"class": rng.integers(5, size=(25,))}),
                var=pd.DataFrame(
                    {"gene_names": [f"gene_{i}" for i in range(12)]},
                    index=[f"gene_{i}" for i in range(12)],
                ),
            ),
        ),
        (
            "zeros",
            ad.AnnData(
                X=np.zeros((25, 12)),
                obs=pd.DataFrame({"class": rng.integers(5, size=(25,))}),
                var=pd.DataFrame(
                    {"gene_names": [f"gene_{i}" for i in range(12)]},
                    index=[f"gene_{i}" for i in range(12)],
                ),
            ),
        ),
    ],
)
def test_create_df(
    _name: str, data: ad.AnnData, get_ranked_data, sort: str, top_n: int | None
):
    rg = get_ranked_data(data)

    u_stat = []
    pvals = []
    padj = []
    lfc = []
    categories = []

    for cat in data.obs["class"].cat.categories:
        lfc.append(rg.stats[(f"{cat}", "logfoldchanges")].values)
        u_stat.append(rg.stats[(f"{cat}", "scores")].values)
        pvals.append(rg.stats[(f"{cat}", "pvals")].values)
        padj.append(rg.stats[(f"{cat}", "pvals_adj")].values)
        categories.append(cat)

    u_stat = np.array(u_stat).T
    pvals = np.array(pvals).T
    padj = np.array(padj).T
    lfc = np.array(lfc).T
    categories = np.array(categories)

    for cat, df in create_df(
        lfc=lfc,
        p_vals=pvals,
        p_adj=padj,
        u_stat=u_stat,
        gene_names=data.var["gene_names"].values,
        categories=categories,
        sort_by=sort,
        top_n=top_n,
    ):
        df.set_index("gene", inplace=True)
        expected_df = rg.stats[f"{cat}"]
        expected_df["abs_logfoldchange"] = np.abs(expected_df["logfoldchanges"])
        expected_df = expected_df.sort_values(
            "abs_logfoldchange", ascending=sort == "asc"
        )

        assert np.allclose(
            df, expected_df.head(top_n if top_n is not None else len(expected_df))
        )


def test_invalid_sort():
    n_features = 10
    n_groups = 2

    with pytest.raises(ValueError):
        for _ in create_df(
            lfc=np.zeros((n_features, n_groups)),
            p_vals=np.zeros((n_features, n_groups)),
            p_adj=np.zeros((n_features, n_groups)),
            u_stat=np.zeros((n_features, n_groups)),
            gene_names=np.array([f"gene_{i}" for i in range(n_features)]),
            categories=np.array([f"group_{i}" for i in range(n_groups)]),
            sort_by="invalid",
        ):
            ...


@pytest.mark.parametrize(
    "feature",
    [
        "lfc",
        "p_vals",
        "p_adj",
        "u_stat",
    ],
)
@pytest.mark.parametrize(
    "valid_shape, invalid_shape",
    [
        ((10, 5), (5, 10)),
        ((10, 5), (10, 6)),
        ((10, 5), (11, 5)),
    ],
)
def test_invalid_shape(
    feature: str, valid_shape: tuple[int, int], invalid_shape: tuple[int, int]
):
    n_features, n_groups = valid_shape

    args = {
        "lfc": np.zeros((n_features, n_groups)),
        "p_vals": np.zeros((n_features, n_groups)),
        "p_adj": np.zeros((n_features, n_groups)),
        "u_stat": np.zeros((n_features, n_groups)),
        "gene_names": np.array([f"gene_{i}" for i in range(n_features)]),
        "categories": np.array([f"group_{i}" for i in range(n_groups)]),
    }

    args[feature] = np.zeros(invalid_shape)

    with pytest.raises(ValueError):
        for _ in create_df(**args):
            ...


@pytest.mark.parametrize(
    "gene_or_category, n_features, n_groups, invalid_shape",
    [
        ("gene_names", 10, 12, (11,)),
        ("gene_names", 10, 12, (11, 1)),
        ("gene_names", 10, 12, (9, 1)),
        ("gene_names", 10, 12, (9,)),
        ("gene_names", 10, 12, (10, 1, 1)),
        ("categories", 12, 4, (5,)),
        ("categories", 12, 4, (5, 1)),
        ("categories", 12, 4, (3,)),
        ("categories", 12, 4, (3, 1)),
        ("categories", 12, 4, (4, 1, 1)),
    ],
)
def test_invalid_gene_categories(gene_or_category, n_features, n_groups, invalid_shape):
    args = {
        "lfc": np.zeros((n_features, n_groups)),
        "p_vals": np.zeros((n_features, n_groups)),
        "p_adj": np.zeros((n_features, n_groups)),
        "u_stat": np.zeros((n_features, n_groups)),
        "gene_names": np.zeros((n_features, 1)),
        "categories": np.zeros((n_groups, 1)),
    }

    args[gene_or_category] = np.zeros(invalid_shape)

    with pytest.raises(ValueError):
        for _ in create_df(**args):
            ...


@pytest.mark.parametrize(
    "n_features, top_n",
    [
        (100, 101),
        (100, -1),
    ],
)
def test_invalid_n(n_features: int, top_n: int):
    n_groups = 10

    with pytest.raises(ValueError):
        for _ in create_df(
            lfc=np.zeros((n_features, n_groups)),
            p_vals=np.zeros((n_features, n_groups)),
            p_adj=np.zeros((n_features, n_groups)),
            u_stat=np.zeros((n_features, n_groups)),
            gene_names=np.array([f"gene_{i}" for i in range(n_features)]),
            categories=np.array([f"group_{i}" for i in range(n_groups)]),
            top_n=top_n,
        ):
            ...
