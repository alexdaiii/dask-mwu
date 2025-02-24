from typing import Literal, Iterator, TypeVar

import numpy as np
import pandas as pd

__all__ = [
    "create_df",
]


def _check_shapes(
    gene_names: np.ndarray[np.str_],
    u_stat: np.ndarray[np.float64],
    p_vals: np.ndarray[np.float64],
    p_adj: np.ndarray[np.float64],
    lfc: np.ndarray[np.float64],
    categories: np.ndarray,
    top_n: int,
):
    if gene_names.ndim > 2 or (gene_names.ndim == 2 and gene_names.shape[1] != 1):
        raise ValueError("gene_names must be of shape (n_features,) or (n_features, 1)")

    if categories.ndim > 2 or (categories.ndim == 2 and categories.shape[1] != 1):
        raise ValueError("categories must be of shape (n_groups,) or (n_groups, 1)")

    n_genes_genes = gene_names.shape[0]
    n_genes_u_stat = u_stat.shape[0]
    n_genes_p_vals = p_vals.shape[0]
    n_genes_p_adj = p_adj.shape[0]
    n_genes_lfc = lfc.shape[0]

    if not (
        n_genes_genes
        == n_genes_u_stat
        == n_genes_p_vals
        == n_genes_p_adj
        == n_genes_lfc
    ):
        raise ValueError(
            "The number of genes in gene_names, u_stat, p_vals, p_adj, and lfc must be the same"
        )

    n_groups_cats = categories.shape[0]
    n_groups_u_stat = u_stat.shape[1]
    n_groups_p_vals = p_vals.shape[1]
    n_groups_p_adj = p_adj.shape[1]
    n_groups_lfc = lfc.shape[1]

    if not (
        n_groups_cats
        == n_groups_u_stat
        == n_groups_p_vals
        == n_groups_p_adj
        == n_groups_lfc
    ):
        raise ValueError(
            "The number of groups in categories, u_stat, p_vals, p_adj, and lfc must be the same"
        )

    if top_n > n_genes_genes:
        raise ValueError("top_n must be less than the number of genes")

    if top_n < 0:
        raise ValueError("top_n must be greater than 0")


T = TypeVar("T", bound=np.generic)


def create_df(
    *,
    gene_names: np.ndarray[np.str_],
    u_stat: np.ndarray[np.float64],
    p_vals: np.ndarray[np.float64],
    p_adj: np.ndarray[np.float64],
    lfc: np.ndarray[np.float64],
    categories: np.ndarray[T],
    top_n: int | None = None,
    sort_by: Literal["asc", "desc"] = "desc",
) -> Iterator[tuple[T, pd.DataFrame]]:
    """
    Given the results of the Mann-Whitney U test and the logfoldchange, create
    a pandas DataFrame that contains the results.

    Args:
        gene_names: The names of the genes that were tested. This would be in
            the adata.var_names attribute. Must be of the shape (n_features,)
            or (n_features, 1)
        u_stat: The U statistic for each gene for each group. The matrix is
            of the shape (n_features, n_groups)
        p_vals: The p-values for each gene for each group. The matrix is of
            the shape (n_features, n_groups)
        p_adj: The adjusted p-values for each gene for each group. The matrix
            is of the shape (n_features, n_groups)
        lfc: The logfoldchange for each gene for each group. The matrix is of
            the shape (n_features, n_groups)
        categories: The categories that were tested. The order of the categories
            should be the same as the order of the columns in the u_stat, p_vals,
            and logfoldchange matrices. meaning the ith categority is the ith
            column in the matrices.
        top_n: How many genes to return. If None, return all genes.
        sort_by: How to sort the genes by absolute logfoldchange. If "asc", sort
            by ascending order, if "desc", sort by descending order.

    Returns: A generator that yields a pandas DataFrame for each category.
    """

    # validation
    if sort_by != "asc" and sort_by != "desc":
        raise ValueError("sort_by must be either 'asc' or 'desc'")

    if top_n is None:
        top_n = gene_names.shape[0]

    _check_shapes(gene_names, u_stat, p_vals, p_adj, lfc, categories, top_n)

    for i, cat in enumerate(categories):
        df = pd.DataFrame(
            {
                "gene": gene_names,
                "U": u_stat[:, i],
                "p_value": p_vals[:, i],
                "p_adjusted": p_adj[:, i],
                "logfoldchange": lfc[:, i],
                "abs_logfoldchange": np.abs(lfc[:, i]),
            }
        )

        if sort_by == "asc":
            df = df.sort_values("abs_logfoldchange", ascending=True)
        else:
            df = df.sort_values("abs_logfoldchange", ascending=False)

        yield cat, df.head(top_n)
