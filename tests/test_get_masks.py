from typing import Type

import pytest
import numpy as np

from dask_mwu.errors import EmptyArrayError, InvalidDimensionError
from dask_mwu.rank_data import get_masks


@pytest.mark.parametrize(
    "reshape",
    [
        True,
        False,
    ],
)
@pytest.mark.parametrize(
    "_name, choices, expected_mask, expected_groups",
    [
        [
            "2 choices",
            np.array([0, 1, 0, 1, 0]),
            np.array(
                [
                    [True, False],
                    [False, True],
                    [True, False],
                    [False, True],
                    [True, False],
                ]
            ),
            {0, 1},
        ],
        [
            "N choices",
            np.array([0, 1, 2, 1, 0]),
            np.array(
                [
                    [True, False, False],
                    [False, True, False],
                    [False, False, True],
                    [False, True, False],
                    [True, False, False],
                ]
            ),
            {0, 1, 2},
        ],
        [
            "not number",
            np.array(["a", "b", "d", "b", "a", "c"]),
            np.array(
                [
                    [
                        True,
                        False,
                        False,
                        False,
                    ],
                    [
                        False,
                        True,
                        False,
                        False,
                    ],
                    [
                        False,
                        False,
                        False,
                        True,
                    ],
                    [
                        False,
                        True,
                        False,
                        False,
                    ],
                    [
                        True,
                        False,
                        False,
                        False,
                    ],
                    [
                        False,
                        False,
                        True,
                        False,
                    ],
                ]
            ),
            {"a", "b", "c", "d"},
        ],
    ],
)
def test_get_masks(
    reshape: bool,
    _name: str,
    choices: np.ndarray,
    expected_mask: np.ndarray[bool],
    expected_groups: set,
):
    if reshape:
        choices = choices.reshape(-1, 1)

    mask, groups = get_masks(choices)

    assert mask.shape == expected_mask.shape
    assert np.all(mask.compute() == expected_mask)
    assert set(groups) == expected_groups
    assert mask.chunksize == (mask.shape[0], 1)


@pytest.mark.parametrize(
    "_name, choices, error",
    [
        ["Empty", np.array([]), EmptyArrayError],
        ["3D choices", np.array([[[0], [1], [0], [1], [0]]]), InvalidDimensionError],
    ],
)
def test_get_masks_invalid(
    _name: str,
    choices: np.ndarray,
    error: Type[Exception],
):
    with pytest.raises(error):
        get_masks(choices)
