# ----------------------------------------------------------------------------
# Copyright 2020 Benoit Fuentes <bf@benoit-fuentes.fr>
#
# This file is part of Wonterfact.
#
# Wonterfact is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# Wonterfact is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Wonterfact. If not, see <https://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------

"""Tests for all examples in the wonterfact/examples directory"""

# Python standard library

# Third-party imports
import pytest
import numpy as np

try:
    import cupy as cp  # pylint: disable=import-error
except:
    cp = None

# wonterfact and relative imports
import wonterfact.cupy_utils as c_utils

pytestmark = pytest.mark.gpu  # marks all file methods as "gpu"


@pytest.mark.parametrize("dim_cumsum", [10, 100, 1000])
def test_cupy_cumsum_2d(dim_cumsum):
    arr_in = cp.ones((2, dim_cumsum), dtype=float)
    arr_out = cp.empty_like(arr_in)
    c_utils.cupy_cumsum_2d(arr_in, arr_out)
    assert cp.allclose(arr_out, cp.cumsum(arr_in, axis=-1))


def test_find_cumsum_max_threads():
    max_threads = c_utils.find_cumsum_max_threads(10, 100)
    assert isinstance(max_threads, int)


def test_min_clip():
    arr = cp.array([3.0, 4.0])
    assert (c_utils.min_clip(arr, 3.5) == cp.array([3.5, 4.0])).all()


def test_max_clip():
    arr = cp.array([3.0, 4.0])
    assert (c_utils.max_clip(arr, 3.5) == cp.array([3.0, 3.5])).all()


def test_normalize_l1_l2_tensor_numba_core():
    dim0, dim1 = 10, 10
    tensor_init = 100 * cp.random.rand(dim0, dim1)
    tensor_out = tensor_init.copy()
    # pylint: disable=unsubscriptable-object
    c_utils.normalize_l1_l2_tensor_numba_core[dim0, (dim1 + 1) // 2](
        tensor_out, 20, 0.01
    )
    # pylint: enable=unsubscriptable-object
    from wonterfact import LeafGammaNorm

    tensor_out_np = cp.asnumpy(tensor_init)
    LeafGammaNorm._normalize_l1_l2_tensor(tensor_out_np, (1,), 20, 0.01)

    assert np.allclose(tensor_out_np, cp.asnumpy(tensor_out))


def test_set_bezier_point():
    arr1 = cp.random.rand(10)
    arr2 = cp.random.rand(10)
    arr3 = cp.random.rand(10)
    arr_out = cp.empty_like(arr1)
    param = 0.5

    c_utils._set_bezier_point(
        arr1, arr2, arr3, param, arr_out,
    )
    arr_out2 = (
        (1 - param) ** 2 * arr1 + 2 * (1 - param) * param * arr2 + (param ** 2) * arr3
    )
    assert cp.allclose(arr_out, arr_out2)


def test_multiply_and_sum():
    arr1 = cp.ones(4, dtype=float)
    arr2 = cp.ones(4, dtype=float)
    arr_out = c_utils.multiply_and_sum(arr1, arr2)
    assert arr_out == 4.0


def test_hyp0f1ln():
    # already tested in test_utils
    pass


def test_bessel_ratio():
    # already tested in test_utils
    pass


def test_inclusive_scan_2d():
    # tested through cupy_cumsum_2d
    pass


def test_sum_inclusive_scan_2d():
    # tested through cupy_cumsum_2d
    pass
