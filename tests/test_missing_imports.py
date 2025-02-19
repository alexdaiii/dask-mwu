import importlib
import sys
from unittest.mock import patch

import pytest


@pytest.mark.parametrize(
    "module_name, package_name, target_module",
    [
        ("dask.array", "dask", "rank_data"),
        ("scipy._lib._util", "scipy", "rank_data"),
        ("scipy.stats._stats_py", "scipy", "rank_data"),
        ("numpy", "numpy", "rank_data"),
    ],
)
def test_missing_dependencies(module_name, package_name, target_module):
    with patch.dict(sys.modules, {module_name: None}):
        with pytest.raises(
            ImportError,
            match=f"{package_name} is not installed. Please install {package_name} to use this package.",
        ):
            importlib.reload(importlib.import_module(f"dask_mwu.{target_module}"))
