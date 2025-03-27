"""
Runs the Wilcoxon rank-sum test on each gene to find marker genes for each group.

Make sure to add the project root to the PYTHONPATH before running this script.

Example:
    export PYTHONPATH=$PYTHONPATH:$(pwd)

To run without holding the terminal, use the following command from the project directory:

nohup python -u scratch/rank_gene_groups.py > .nohup/find_marker_genes.rank_gene_groups.log &

To monitor the progress, use the following command:

tail -f .nohup/find_marker_genes.rank_gene_groups.log
"""

import os

from dask_mwu import (
    get_masks,
    rank_data,
    compute_in_group_ranksum,
    compute_tie_term,
    mann_whitney_u,
    compute_logfoldchange,
    create_df,
)

# set env variables to turn off multithreading for numpy and openblas
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


from pathlib import Path
import re

import numpy as np
import pandas as pd
import dask.array as da
from pydantic import BaseModel, DirectoryPath, NewPath, FilePath, FileUrl


class Params(BaseModel):
    counts_file: FilePath | FileUrl | DirectoryPath
    """
    The location of the 1logp normalized counts matrix. Should be in zarr format 
    which allows partial loading in chunks.
    """

    obs_file: FilePath
    """
    A parquet file containing the observations. The rows should correspond to
    individual cells and the columns should contain metadata about the cells.
    """
    group_col: str
    """
    The column in the obs file to group the cells by. This will be used to
    find the marker genes in each group compared to all other cells.
    """

    vars_file: FilePath
    """
    The location of the vars file. This should be a parquet file with the rows
    corresponding to genes and the columns containing metadata about the genes.
    Usually this will only contain the gene names and maybe if the gene is a
    highly vartiable gene.
    """
    genes_col: str
    """
    THe column in the vars_file that contains the gene names
    """

    intermediate_dir: DirectoryPath | NewPath
    """
    A location to store intermediate zarr files.
    """

    # intermediate files
    ranks_zarr: str = "ranks.zarr"
    """
    Name of the zarr file to store the ranks and ties. It is a (cells, genes, 2)
    array where [:, :, 0] is the ranks and [:, :, 1] is the ties.
    """
    recompute_ranks: bool = False
    """
    Recompute the ranks even if the ranks file already exists. 
    """

    output_dir: DirectoryPath | NewPath
    n_feats_per_chunk: int
    """
    Number of features to calculate the fold change for in each chunk. This
    controls how much memory is used when calculating the fold change.
    """
    rechunk_feats_per_chunk: int
    """
    Number of features to rechunk the zarr array to. Since scipy _rankdata is
    the most memory intensive part of the calculation, this can be used to
    increase the number of features that are processed at once.
    """


def remove_non_alphanumeric(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "_", s)


def read_data(*, counts_file: Path, obs_file: Path, vars_file: Path, group_col: str):
    print(f"Reading data from {counts_file}")

    arr = da.from_zarr(counts_file)

    print(f"Reading obs from {obs_file}")

    obs = pd.read_parquet(obs_file)

    if len(obs) != arr.shape[0]:
        raise ValueError(
            f"Number of observations in {obs_file} does not match number of cells in {counts_file}"
        )

    print(f"There are {len(np.unique(obs[group_col]))} groups in the data")

    print(f"Reading vars from {vars_file}")

    vars_df = pd.read_parquet(vars_file)

    if len(vars_df) != arr.shape[1]:
        raise ValueError(
            f"Number of genes in {vars_file} does not match number of genes in {counts_file}"
        )

    print(f"Successfully read data with {arr.shape[0]} cells and {arr.shape[1]} genes")

    return arr, obs, vars_df


def get_params():
    obs_path = (
        Path.home()
        / "data"
        / "scratch-data"
        / "find_marker_genes"
        / "combined_log1p_norm"
        / "obs.parquet"
    )
    vars_path = (
        Path.home() / "data" / "abc_atlas_combined" / "intermediates" / "vars.parquet"
    )
    counts = (
        Path.home()
        / "data"
        / "scratch-data"
        / "find_marker_genes"
        / "combined_log1p_norm"
        / "combined_log1p_norm_counts.zarr"
    )
    genes_col = "gene_symbol"
    # group_col = "subclass"
    group_col = "class"

    output_dir_base = Path.home() / "data" / "abc_atlas_combined" / "rank_gene_groups"
    intermediate_dir = output_dir_base / "computation_cache"
    output_dir = output_dir_base / "results_cluster_level"

    # ~154 GB of memory
    n_feats_per_chunk = 6
    rechunk_feats_per_chunk = 60

    return Params(
        counts_file=counts,
        obs_file=obs_path,
        group_col=group_col,
        vars_file=vars_path,
        genes_col=genes_col,
        intermediate_dir=intermediate_dir,
        output_dir=output_dir,
        n_feats_per_chunk=n_feats_per_chunk,
        rechunk_feats_per_chunk=rechunk_feats_per_chunk,
    )


def main():
    params = get_params()

    arr, obs, vars_df = read_data(
        counts_file=params.counts_file,
        obs_file=params.obs_file,
        vars_file=params.vars_file,
        group_col=params.group_col,
    )


    if not params.output_dir.exists():
        params.output_dir.mkdir(parents=True)
        print(f"Created output directory at {params.output_dir}")
    if not params.intermediate_dir.exists():
        params.intermediate_dir.mkdir(parents=True)
        print(f"Created intermediate directory at {params.intermediate_dir}")

    print("Finding marker genes")

    rank_gene_groups_vec(
        arr,
        choices=obs[params.group_col].values,
        genes=vars_df[params.genes_col].values,
        output_dir=params.output_dir,
        intermediate_dir=params.intermediate_dir,
        ranks_zarr=params.ranks_zarr,
        n_feats_per_chunk=params.n_feats_per_chunk,
        recompute_ranks=params.recompute_ranks,
        rechunk_feats_per_chunk=params.rechunk_feats_per_chunk,
    )

    print("Successfully found marker genes")


def get_ranks(
    data: da.Array,
    *,
    n_features_per_chunk: int,
    recompute_ranks: bool,
    intermediate_dir: Path,
    ranks_zarr: str,
    rechunk_feats_per_chunk: int,
) -> tuple[da.Array, da.Array]:
    # check if ranks already exists
    ranks_file = intermediate_dir / ranks_zarr
    if ranks_zarr.endswith(".zarr"):
        pre_ranks_file = intermediate_dir / ranks_zarr.replace(".zarr", "_pre.zarr")
    else:
        pre_ranks_file = intermediate_dir / f"{ranks_zarr}_pre.zarr"

    if ranks_file.exists() and not recompute_ranks:
        print(f"Reading ranks from {ranks_file}")
        return get_ranks_and_ties(ranks_file)

    print("Computing ranks")
    rank_ties = rank_data(data, n_features_per_chunk=n_features_per_chunk)

    # add the _pre to the filename to avoid overwriting the file
    print(f"Saving ranks to {ranks_file}")
    rank_ties.to_zarr(pre_ranks_file, overwrite=True)

    # rechunk the zarr array
    print(f"Rechunking ranks to {rechunk_feats_per_chunk}")
    rank_ties = da.from_zarr(pre_ranks_file).rechunk((-1, rechunk_feats_per_chunk, 1))
    rank_ties.to_zarr(ranks_file, overwrite=True)

    print("Reading ranks")
    return get_ranks_and_ties(ranks_file)


def get_ranks_and_ties(ranks_file: Path) -> tuple[da.Array, da.Array]:
    ranks = da.from_zarr(ranks_file)[:, :, 0]
    ties = da.from_zarr(ranks_file)[:, :, 1]
    return ranks, ties


def rank_gene_groups_vec(
    data: da.Array,
    *,
    choices: np.ndarray[np.generic],
    genes: np.ndarray[np.str_],
    output_dir: Path,
    intermediate_dir: Path,
    ranks_zarr: str,
    n_feats_per_chunk: int,
    recompute_ranks: bool,
    rechunk_feats_per_chunk: int,
):
    in_groups, groups = get_masks(choices)

    ranks, ties = get_ranks(
        data,
        n_features_per_chunk=n_feats_per_chunk,
        recompute_ranks=recompute_ranks,
        intermediate_dir=intermediate_dir,
        ranks_zarr=ranks_zarr,
        rechunk_feats_per_chunk=rechunk_feats_per_chunk,
    )

    print("Computing Rank Sums and Ties")
    rank_sums = compute_in_group_ranksum(ranks, in_groups)
    tie_term = compute_tie_term(ties)

    print("Computing Mann-Whitney U")
    mwu = mann_whitney_u(rank_sums, tie_term, in_groups)

    print("Computing logfoldchange")
    lfc = compute_logfoldchange(data, in_groups)

    for cat, df in create_df(
        gene_names=genes,
        u_stat=mwu.U,
        p_vals=mwu.p_vals,
        p_adj=mwu.p_adj,
        lfc=lfc,
        categories=groups,
    ):
        print(f"Saving {cat} results to csv")

        csv_filename = output_dir / f"{remove_non_alphanumeric(cat)}_stats.csv"
        print(f"Writing to csv at {csv_filename}")

        df.to_csv(csv_filename)

    print("Successfully saved results")


if __name__ == "__main__":
    main()
