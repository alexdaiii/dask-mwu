import numpy as np
import pandas as pd
import pytest
import anndata as ad

rng = np.random.default_rng(42)


@pytest.mark.parametrize(
    "_name, data",
    [
        (
            "positive",
            ad.AnnData(
                X=rng.integers(100, size=(25, 12)),
                obs=pd.DataFrame({"class": rng.integers(5, size=(25, 1))}),
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
                obs=pd.DataFrame({"class": rng.integers(5, size=(25, 1))}),
                var=pd.DataFrame(
                    {"gene_names": [f"gene_{i}" for i in range(12)]},
                    index=[f"gene_{i}" for i in range(12)],
                ),
            ),
        ),
    ],
)
def test_create_df(_name: str, data: ad.AnnData): ...
