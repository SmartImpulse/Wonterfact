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

"""Tests for all methods in utils module"""

# Python standard library

# Third-party imports
import numpy as np
import numpy.random as npr
import pytest

try:
    import cupy as cp  # pylint: disable=import-error
except:
    cp = None

# wonterfact imports
from wonterfact import utils
from wonterfact import glob


def prepare_for_numpy_einsum(args):
    letter_to_int = {}
    for num_letter, letter in enumerate(set(args[1] + args[3] + args[5])):
        letter_to_int[letter] = num_letter
    sub0 = [letter_to_int[letter] for letter in args[1]]
    sub1 = [letter_to_int[letter] for letter in args[3]]
    subout = [letter_to_int[letter] for letter in args[5]]
    return [args[0], sub0, args[2], sub1, subout]


@pytest.fixture(
    scope="module", params=["cpu", pytest.param("gpu", marks=pytest.mark.gpu)]
)
def backend_and_operations_as_list(request):
    """
    Returns a list of (op0, sub0, op1, sub1, out, subout)
    """
    xp = cp if request.param == "gpu" else np
    arr0 = xp.random.rand(4, 4, 4)
    arr1 = xp.random.rand(4, 4, 4)
    # Please update 'expected_func' dictionary in
    # 'test_parse_einsum_two_operands_input' if you update the following list.
    input_list = [
        ["ijk", "ijk", "ijk"],
        ["ijk", "ijk", "ij"],
        ["ijk", "ijk", "i"],
        ["ijk", "ijk", ""],
        ["ijk", "ijk", "kji"],
        ["ijk", "jik", "jk"],
        ["ijk", "klm", "ijlm"],
        ["ijk", "klm", "im"],
        ["ijk", "mkj", "im"],
        ["ijk", "imj", "im"],
        ["ijk", "ikl", "ijl"],
        ["ijk", "ikl", "jl"],
        ["ijk", "kli", "ji"],
    ]
    operation_list = [
        (
            arr0,
            elem[0],
            arr1,
            elem[1],
            xp.zeros((4,) * len(elem[2]), dtype=float),
            elem[2],
        )
        for elem in input_list
    ]
    return xp, operation_list


@pytest.fixture(scope="module")
def operations_as_list(backend_and_operations_as_list):
    return backend_and_operations_as_list[1]


@pytest.fixture(scope="module")
def operations_as_string(operations_as_list):
    """
    Returns a list of ('sub0,sub1->subout', op0, op1, out)
    """
    operation_list = [
        ("->".join([",".join(elem[1:4:2]), elem[5]]), elem[0], elem[2], elem[4])
        for elem in operations_as_list
    ]
    return operation_list


@pytest.fixture(scope="module")
def xp(backend_and_operations_as_list):
    return backend_and_operations_as_list[0]


def test_has_a_shared_sublist():
    assert utils._has_a_shared_sublist([], [1, 2, 3])
    assert utils._has_a_shared_sublist([1, 2, 3, 4], [2, 5, 3])
    assert not utils._has_a_shared_sublist([1, 2, 3, 4], [0, 3, 2])
    assert not utils._has_a_shared_sublist(
        [1, 2, 3, 4], [1, 2, 5, 6, 7, 8, 5, 76, 4, 3]
    )


def test_get_transpose_and_slice():
    sub1, sub2, sub_out = "ij", "jkl", "kilj"
    transpose1, slice1 = utils.get_transpose_and_slice(sub1, sub_out)
    transpose2, slice2 = utils.get_transpose_and_slice(sub2, sub_out)
    assert transpose1 == (0, 1)
    assert transpose2 == (1, 2, 0)
    assert slice1 == (None, slice(None), None, slice(None))
    assert slice2 == (slice(None), None, slice(None), slice(None))


def test_supscript_summation(xp):
    sub1, sub2, sub_out = "ij", "jkl", "kilj"
    op1 = xp.arange(3 * 5).reshape((3, 5))
    op2 = xp.arange(5 * 2 * 4).reshape((5, 2, 4))
    res = utils.supscript_summation(op1, sub1, op2, sub2, sub_out)
    assert xp.allclose(res, op1[:, None] + op2.transpose((1, 2, 0))[:, None])


def test_parse_einsum_args():
    args1 = ("fd,dt,d->ft", 1, 2, 3)
    args2 = (1, "fd", 2, "dt", 3, "d", "ft")
    out1 = utils._parse_einsum_args(*args1)
    out2 = utils._parse_einsum_args(*args2)
    assert tuple(out1[:-1:2]) == args2[:-1:2]
    assert tuple(out2[:-1:2]) == args2[:-1:2]


def test_einsum(operations_as_string, xp):
    for elem in operations_as_string:
        assert xp.allclose(utils.einsum(*elem[:3]), xp.einsum(*elem[:3]))
    for elem in operations_as_string:
        elem[3][...] = 0.0
        utils.einsum(*elem[:3], out=elem[3])
        assert xp.allclose(elem[3], xp.einsum(*elem[:3]))

    arr1 = xp.random.rand(4, 4, 4)
    arr2 = xp.random.rand(4, 4, 4)
    arr3 = xp.random.rand(4, 4)
    assert xp.allclose(
        utils.einsum("ijk,ikl,il->ijl", arr1, arr2, arr3),
        xp.einsum("ijk,ikl,il->ijl", arr1, arr2, arr3),
    )


def test_einsum_two_operands(operations_as_list, xp):
    backend = xp.__name__
    for elem in operations_as_list:
        if backend == "cupy":
            pass
        assert xp.allclose(
            utils._einsum_two_operands(*elem[:4], elem[5], backend),
            xp.einsum(*prepare_for_numpy_einsum(elem)),
        )
    for elem in operations_as_list:
        elem[4][...] = 0.0
        utils._einsum_two_operands(*elem[:4], elem[5], backend, out=elem[4]),
        assert xp.allclose(elem[4], xp.einsum(*prepare_for_numpy_einsum(elem)))


def test_element_wise_mult(xp):
    arr1 = xp.random.rand(4, 4, 4)
    arr2 = xp.random.rand(4, 4)
    out = xp.ones((4, 4, 4))
    sub1 = sub2 = sub_out = None
    backend = xp.__name__
    transpose1 = (2, 0, 1)
    transpose2 = (1, 0)
    slice1 = Ellipsis
    slice2 = (slice(None), None)
    utils._element_wise_mult(
        arr1,
        sub1,
        arr2,
        sub2,
        out,
        sub_out,
        backend,
        transpose1,
        transpose2,
        slice1,
        slice2,
    )
    assert xp.allclose(out, xp.einsum("ijk,jk->kij", arr1, arr2))


def test_regular_einsum(operations_as_list, xp):
    backend = xp.__name__
    for elem in operations_as_list:
        elem[4][...] = 0.0
        utils._regular_einsum(*elem, backend)
        assert xp.allclose(elem[4], xp.einsum(*prepare_for_numpy_einsum(elem)))


def test_sequential_tensor_dot(xp):
    op1 = xp.random.rand(4, 4, 4)
    sub1 = "ijk"
    op2 = xp.random.rand(4, 4, 4)
    sub2 = "kil"
    out = xp.random.rand(4, 4, 4)
    sub_out = "lij"
    backend = xp.__name__
    list_of_slice_out = [(slice(None), dim) for dim in range(4)]
    list_of_slice_1 = [(dim,) for dim in range(4)]
    list_of_slice_2 = [(slice(None), dim) for dim in range(4)]
    new_sub1 = ["j", "k"]
    new_sub2 = ["k", "l"]
    new_sub_out = ["l", "j"]
    out_shape = (4, 4, 4)
    utils._sequential_tensor_dot(
        op1,
        sub1,
        op2,
        sub2,
        out,
        sub_out,
        backend,
        list_of_slice_out,
        list_of_slice_1,
        list_of_slice_2,
        new_sub1,
        new_sub2,
        new_sub_out,
        out_shape,
    )
    assert xp.allclose(
        out, xp.einsum("{},{}->{}".format(sub1, sub2, sub_out), op1, op2)
    )


def test_parse_einsum_two_operands_input(operations_as_list, xp):
    # this refers operations_as_list
    expected_func = {
        # cupy
        cp: [
            utils._element_wise_mult,
            utils._element_wise_mult_and_sum,
            utils._element_wise_mult_and_sum,
            utils._element_wise_mult_and_sum,
            utils._element_wise_mult,
            utils._element_wise_mult_and_sum,
            utils._sequential_tensor_dot,
            utils._regular_einsum,
            utils._sequential_tensor_dot,
            utils._regular_einsum,
            utils._sequential_tensor_dot,
            utils._sequential_tensor_dot,
            utils._regular_einsum,
        ],
        # numpy
        np: [
            utils._element_wise_mult,
            utils._regular_einsum,
            utils._regular_einsum,
            utils._regular_einsum,
            utils._element_wise_mult,
            utils._regular_einsum,
            utils._sequential_tensor_dot,
            utils._regular_einsum,
            utils._sequential_tensor_dot,
            utils._regular_einsum,
            utils._sequential_tensor_dot,
            utils._sequential_tensor_dot,
            utils._regular_einsum,
        ],
    }
    backend = xp.__name__
    for elem, func in zip(operations_as_list, expected_func[xp]):
        assert (
            utils._parse_einsum_two_operands_input(
                elem[0].shape, elem[1], elem[2].shape, elem[3], elem[5], backend
            )[0]
            == func
        )


def test_element_wise_mult_and_sum(xp):
    if xp == cp:
        op1 = xp.random.rand(4, 4, 4)
        sub1 = "ijk"
        op2 = xp.random.rand(4, 4, 4)
        sub2 = "jik"
        out = xp.random.rand(4, 4)
        sub_out = "jk"
        backend = xp.__name__
        switch_operand = False
        transpose1 = (1, 2, 0)
        transpose2 = (0, 2, 1)
        slice1 = Ellipsis
        sum_axis = 2
        utils._element_wise_mult_and_sum(
            op1,
            sub1,
            op2,
            sub2,
            out,
            sub_out,
            backend,
            switch_operand,
            transpose1,
            transpose2,
            slice1,
            sum_axis,
        )
        assert xp.allclose(
            out, xp.einsum("{},{}->{}".format(sub1, sub2, sub_out), op1, op2)
        )
    else:
        assert True


def test_einsum_as_dot(xp):
    op1 = xp.random.rand(4, 4, 4)
    sub1 = "ijk"
    op2 = xp.random.rand(4, 4, 4)
    sub2 = "kil"
    sub_out = "lj"
    backend = xp.__name__
    assert xp.allclose(
        utils._einsum_as_dot(op1, sub1, op2, sub2, sub_out, backend),
        xp.einsum("{},{}->{}".format(sub1, sub2, sub_out), op1, op2),
    )


def test_einconv(operations_as_list, xp):
    # first, we test with no convolution
    for elem in operations_as_list:
        elem[4][...] = 0.0
        utils.einconv(*elem[:4], elem[5], out=elem[4])
        assert xp.allclose(elem[4], xp.einsum(*prepare_for_numpy_einsum(elem)))

    # get convolution method for numpy or scipy
    if xp == np:
        from scipy.ndimage import convolve
    elif xp == cp:
        from cupyx.scipy.ndimage import convolve  # pylint: disable=import-error

    # 1D convolution
    arr1 = xp.random.rand(10, 3, 10)
    arr2 = xp.random.rand(2, 3)
    sub1 = "ijk"
    sub2 = "kj"
    subout = "jki"
    conv_idx_list = [
        "k",
    ]
    args = (arr1, sub1, arr2, sub2, subout)
    out = utils.einconv(*args, conv_idx_list=conv_idx_list)
    out2 = xp.empty((3, 9, 10))
    for ii in range(10):
        for jj in range(3):
            out2[jj, :, ii] = convolve(arr1[ii, jj, :], arr2[:, jj])[:-1]
    assert xp.allclose(out, out2)

    # independent with respect to operands order ?
    assert xp.allclose(
        out, utils.einconv(*args[2:4], *args[:2], args[-1], conv_idx_list=conv_idx_list)
    )

    # in correlation mode
    out = utils.einconv(*args, conv_idx_list=conv_idx_list, compute_correlation=True)
    for ii in range(10):
        for jj in range(3):
            out2[jj, :, ii] = convolve(arr1[ii, jj, :], arr2[::-1, jj])[:-1]
    assert xp.allclose(out, out2)

    # 2D convolution
    conv_idx_list = ["k", "j"]
    out = utils.einconv(*args, conv_idx_list=conv_idx_list)
    out2 = xp.empty((1, 9, 10))
    for ii in range(10):
        out2[:, :, ii] = convolve(arr1[ii, :, :], arr2.T)[1:-1, :-1]
    assert xp.allclose(out, out2)

    # independent with respect to operands order ?
    assert xp.allclose(
        out, utils.einconv(*args[2:4], *args[:2], args[-1], conv_idx_list=conv_idx_list)
    )
    # in correlation mode
    out = utils.einconv(*args, conv_idx_list=conv_idx_list, compute_correlation=True)
    for ii in range(10):
        for jj in range(3):
            out2[:, :, ii] = convolve(arr1[ii, :, :], arr2.T[::-1, ::-1])[1:-1, :-1]
    assert xp.allclose(out, out2)

    # should raise an error if one array is not larger than the other in every
    # dimension where convolution is performed
    with pytest.raises(ValueError, match=r".*must be at least as large*"):
        utils.einconv(
            xp.ones((2, 1)),
            "ij",
            xp.ones((1, 2)),
            "ij",
            "ij",
            conv_idx_list=["i", "j"],
        )
    with pytest.raises(ValueError, match=r".*must be at least as large*"):
        utils.einconv(
            xp.ones((1, 2)),
            "ij",
            xp.ones((2, 1)),
            "ij",
            "ij",
            conv_idx_list=["i", "j"],
        )


def test_make_unique_hashable():
    forbiden_set = set([0, 1, 2, 3, "a", "b", "c"])
    assert utils._make_unique_hashable(forbiden_set) not in forbiden_set


def test_find_equality_root(xp):
    backend = xp.__name__
    max_iter = 10

    # inequality constraint should not be binding
    arr = 100 * xp.ones((2, 2, 2))
    const_coef = xp.array([1, -1])
    arr[:, :, :1] = 200
    norm_arr = arr.sum(2, keepdims=True)
    sigma = utils._find_equality_root(
        arr, norm_arr, const_coef, max_iter, type="inequality", backend=backend
    )
    assert xp.allclose(sigma, 0)
    assert xp.allclose((arr / norm_arr).sum(2), 1.0)

    # inequality constraint should be binding
    arr[:, :, :1] = 50
    norm_arr = arr.sum(2, keepdims=True)
    sigma = utils._find_equality_root(
        arr, norm_arr, const_coef, max_iter, type="inequality", backend=backend
    )
    assert xp.allclose(((arr / (norm_arr - sigma * const_coef)) * const_coef).sum(2), 0)
    assert xp.allclose((arr / (norm_arr - sigma * const_coef)).sum(2), 1.0)

    # force equality constraint
    arr[:, :, :1] = 200
    norm_arr = arr.sum(2, keepdims=True)
    sigma = utils._find_equality_root(
        arr, norm_arr, const_coef, max_iter, type="equality", backend=backend
    )
    assert np.allclose(((arr / (norm_arr - sigma * const_coef)) * const_coef).sum(2), 0)
    assert np.allclose((arr / (norm_arr - sigma * const_coef)).sum(2), 1.0)

    # inequality constraint should not be binding
    const_coef = xp.ones((2, 2))
    const_coef[:, 1] = -1
    arr[:, :, 0] = 200
    norm_arr = arr.sum((1, 2), keepdims=True)
    sigma = utils._find_equality_root(
        arr, norm_arr, const_coef, max_iter, type="inequality", backend=backend
    )
    assert xp.allclose(sigma, 0)

    # inequality constraint should be binding
    arr[:, :, 0] = 50
    norm_arr = arr.sum((1, 2), keepdims=True)
    sigma = utils._find_equality_root(
        arr, norm_arr, const_coef, max_iter, type="inequality", backend=backend
    )
    assert xp.allclose(
        ((arr / (norm_arr - sigma * const_coef)) * const_coef).sum((1, 2)), 0
    )
    assert xp.allclose((arr / (norm_arr - sigma * const_coef)).sum((1, 2)), 1.0)

    # when no normalization
    norm_arr = xp.array(1 + 0.001)
    const_coef = xp.ones((2, 2, 2))
    const_coef[:, :, 1] = -1
    arr[:, :, 0] = 200
    sigma = utils._find_equality_root(
        arr, norm_arr, const_coef, max_iter, type="inequality", backend=backend
    )
    assert xp.allclose(sigma, 0)

    arr[:, :, 0] = 50
    sigma = utils._find_equality_root(
        arr, norm_arr, const_coef, max_iter, type="inequality", backend=backend
    )
    assert xp.allclose(((arr / (norm_arr - sigma * const_coef)) * const_coef).sum(), 0)


def test_xlogy(xp):
    arr1 = xp.ones(1, dtype=float)
    arr2 = xp.ones(1, dtype=float)
    assert xp.allclose(utils.xlogy(arr1, arr2), 0.0)

    arr1[...] = 0.0
    arr2[...] = 0.0
    assert xp.allclose(utils.xlogy(arr1, arr2), 0.0)

    arr1[...] = 2.0
    arr2[...] = 2.0
    assert xp.allclose(utils.xlogy(arr1, arr2), arr1 * xp.log(arr2))

    out = xp.ones_like(arr1)
    utils.xlogy(arr1, arr2, out=out)
    assert xp.allclose(utils.xlogy(arr1, arr2), out)


def test_cumsum_last_axis(xp):
    arr = xp.random.rand(4, 4, 10)
    out = xp.ones_like(arr)
    utils.cumsum_last_axis(arr, out)
    assert xp.allclose(out, xp.cumsum(arr, axis=-1))


def test_next_pow_of_two():
    assert utils.next_pow_of_two(0) == 1
    assert utils.next_pow_of_two(0.5) == 1
    assert utils.next_pow_of_two(1) == 1
    assert utils.next_pow_of_two(1.1) == 2
    assert utils.next_pow_of_two(2) == 2
    assert utils.next_pow_of_two(8.5) == 16


def test_exp_digamma(xp):
    if xp == np:
        from scipy.special import digamma
    elif xp == cp:
        from cupyx.scipy.special import digamma  # pylint: disable=import-error

    arr = xp.array(1.0)
    assert xp.allclose(utils.exp_digamma(arr), xp.exp(digamma(arr)))
    arr[...] = 0.0
    assert xp.allclose(utils.exp_digamma(arr), xp.exp(digamma(arr)))
    arr[...] = 10.0
    assert xp.allclose(utils.exp_digamma(arr), xp.exp(digamma(arr)))


def test_hyp0f1ln(xp):
    from scipy.special import hyp0f1

    if xp == cp:
        hyp0f1_np = hyp0f1

        def hyp0f1(arr1, arr2):  # pylint: disable=function-redefined
            arr1 = xp.asnumpy(arr1)
            arr2 = xp.asnumpy(arr2)
            return xp.array(hyp0f1_np(arr1, arr2))

    arr1 = xp.array([2.0, 1e-8, 1e5])
    arr2 = xp.array([3.0, 2e-8, 2e5])
    assert xp.allclose(utils.hyp0f1ln(arr1, arr2), xp.log(hyp0f1(arr1, arr2)))
    arr2[...] = 0.0
    assert xp.allclose(utils.hyp0f1ln(arr1, arr2), 0)


def test_bessel_ratio(xp):
    from scipy.special import iv

    if xp == cp:
        iv_np = iv

        def iv(arr1, arr2):  # pylint: disable=function-redefined
            arr1 = xp.asnumpy(arr1)
            arr2 = xp.asnumpy(arr2)
            return xp.array(iv_np(arr1, arr2))

    arr1 = xp.array([2.0, 1e-8, 1e2])
    arr2 = xp.array([3.0, 2e-8, 2e2])
    assert xp.allclose(
        utils.bessel_ratio(arr1, arr2), arr2 * iv(arr1 + 1, arr2) / iv(arr1, arr2)
    )
    arr2[...] = 0.0
    assert xp.allclose(utils.bessel_ratio(arr1, arr2), 0)


def test_forced_iter():
    elem = [3, 4]
    output = tuple(ii for ii in utils.forced_iter(elem))
    assert output == tuple(elem)
    elem = 3
    output = tuple(ii for ii in utils.forced_iter(elem))
    assert output == (elem,)
    elem = ()
    output = tuple(ii for ii in utils.forced_iter(elem))
    assert output == ()


def test_explicit_slice():
    ndim = 3
    list_of_input = [
        Ellipsis,
        (Ellipsis,),
        (slice(None), Ellipsis),
        (slice(None), Ellipsis, slice(None)),
        (Ellipsis, slice(None)),
        (slice(None),),
        slice(None),
    ]
    for sl in list_of_input:
        assert utils.explicit_slice(sl, ndim) == (slice(None),) * ndim

    list_of_input = [
        1,
        (1,),
        (1, Ellipsis),
        (1, Ellipsis, slice(None)),
        (1, slice(None)),
        (1, slice(None), Ellipsis),
    ]
    for sl in list_of_input:
        assert utils.explicit_slice(sl, ndim) == (1,) + (slice(None),) * (ndim - 1)

    list_of_input = [(slice(1), 2), (slice(1), 2, Ellipsis)]
    for sl in list_of_input:
        assert utils.explicit_slice(sl, ndim) == (slice(1), 2, slice(None))

    assert utils.explicit_slice(Ellipsis, 0) == Ellipsis
    assert utils.explicit_slice(slice(None), 1) == (slice(None),)

    mask = np.array([[True, False], [False, True]])
    list_of_input = [
        mask,
        (mask, slice(None)),
        (mask, Ellipsis),
    ]
    for sl in list_of_input:
        explicit_sl = utils.explicit_slice(sl, ndim)
        assert len(explicit_sl) == ndim - 1
        assert np.alltrue(explicit_sl[0] == mask)
        assert explicit_sl[1:] == (slice(None),) * (ndim - 2)


def test_real_to_2D_nonnegative():
    arr = np.array([[1, -1, 2], [-3, 3.2, 0]])
    arr_nonneg = utils.real_to_2D_nonnegative(arr)
    assert arr_nonneg.shape == arr.shape + (2,)
    assert np.allclose(arr_nonneg[..., 0], arr.clip(min=0))
    assert np.allclose(arr_nonneg[..., 1], (-arr).clip(min=0))


def test_complex_to_2D_real():
    arr = np.array([[1, -1, 2], [-3, 3.2, 0]]) + 1j * np.array(
        [[3, 1, -2], [3, 0, -1.1]]
    )
    arr_real = utils.complex_to_2D_real(arr)
    assert arr_real.shape == arr.shape + (2,)
    assert np.allclose(arr_real[..., 0], arr.real)
    assert np.allclose(arr_real[..., 1], arr.imag)


def test_complex_to_4D_nonnegative():
    arr = np.array([[1, -1, 2], [-3, 3.2, 0]]) + 1j * np.array(
        [[3, 1, -2], [3, 0, -1.1]]
    )
    arr_real = utils.complex_to_4D_nonnegative(arr)
    assert arr_real.shape == arr.shape + (2, 2)
    assert np.allclose(arr_real[..., 0, 0], arr.real.clip(min=0))
    assert np.allclose(arr_real[..., 0, 1], (-arr.real).clip(min=0))
    assert np.allclose(arr_real[..., 1, 0], arr.imag.clip(min=0))
    assert np.allclose(arr_real[..., 1, 1], (-arr.imag).clip(min=0))


def test_clip_inplace(xp):
    arr_init = xp.array([3.0, 4.0])

    arr_out = arr_init.copy()
    utils.clip_inplace(arr_out, a_min=3.5)
    assert (arr_out == xp.array([3.5, 4.0])).all()

    arr_out = arr_init.copy()
    utils.clip_inplace(arr_out, a_max=3.5)
    assert (arr_out == xp.array([3.0, 3.5])).all()

    arr_out = arr_init.copy()
    utils.clip_inplace(arr_out, a_min=3.5, a_max=3.8)
    assert (arr_out == xp.array([3.5, 3.8])).all()


def test_inverse_gamma(xp):
    if xp == np:
        from scipy.special import digamma
    elif xp == cp:
        from cupyx.scipy.special import digamma  # pylint: disable=import-error
    input_arr_pos = xp.array([1.0, 2.0, 3.443])
    assert xp.allclose(utils.inverse_digamma(digamma(input_arr_pos)), input_arr_pos)
    assert xp.allclose(digamma(utils.inverse_digamma(input_arr_pos)), input_arr_pos)
    input_arr_neg = xp.array([-1.0, -2.0, -3.443])
    assert xp.allclose(digamma(utils.inverse_digamma(input_arr_neg)), input_arr_neg)

    output_arr = xp.zeros_like(input_arr_pos)
    utils.inverse_digamma(input_arr_pos, out=output_arr)
    assert xp.allclose(utils.inverse_digamma(input_arr_pos), output_arr)
