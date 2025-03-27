# Dask MWU

Computes the Mann Whitney U test and logfoldchanges for M groups of data
using Dask. Equivalent to scanpy.stats.rank_genes_groups with method='wilcoxon'
but much faster and able to handle larger than memory datasets.

Example usage is in the `scratch/rank_gene_groups.py` script.

## Install Requirements

```bash
poetry install
```

## Run Tests

```bash
poetry run pytest
```