# ----------------------------------------------------------------------------
# Copyright 2020 Smart Impulse SAS, Benoit Fuentes <bf@benoit-fuentes.fr>
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

"""Module for useful methods used in wonterfact"""

# Python System imports
import sys
import math
from itertools import product
import inspect
from functools import wraps, lru_cache

# Third-party imports
import functools as cached_p
from numba import jit, vectorize
import numpy as np
from numpy.lib.stride_tricks import as_strided
import opt_einsum
import scipy.special as sps


class BackendSpecific:
    @staticmethod
    def raise_backend_error(backend):
        raise ValueError("Unknown backend '{}'".format(backend))

    @staticmethod
    @lru_cache(maxsize=16)
    def back(backend):
        if backend == "numpy":
            return np
        elif backend == "cupy":
            import cupy  # pylint: disable=import-error

            return cupy
        else:
            BackendSpecific.raise_backend_error(backend)

    @staticmethod
    @lru_cache(maxsize=16)
    def as_strided(backend):
        if backend == "numpy":
            from numpy.lib.stride_tricks import as_strided as as_strided_numpy

            return as_strided_numpy
        elif backend == "cupy":
            from cupy.lib.stride_tricks import (
                as_strided as as_strided_cupy,
            )  # pylint: disable=import-error

            return as_strided_cupy
        else:
            BackendSpecific.raise_backend_error(backend)

    @lru_cache(maxsize=16)
    def cumsum_last_axis(self, backend):
        if backend == "numpy":

            def cumsum_last_axis_numpy(arr, out):
                arr.cumsum(axis=-1, out=out)

            return cumsum_last_axis_numpy

        elif backend == "cupy":

            def cumsum_last_axis_cupy(arr, out):
                size_cumsum = arr.shape[-1]
                arr = arr.reshape((-1, size_cumsum))
                max_threads = self.get_cupy_utils(backend).find_cumsum_max_threads(
                    *arr.shape
                )
                self.get_cupy_utils(backend).cupy_cumsum_2d(
                    arr, out.reshape(arr.shape), max_threads=max_threads
                )

            return cumsum_last_axis_cupy
        else:
            BackendSpecific.raise_backend_error(backend)

    @staticmethod
    @lru_cache(maxsize=16)
    def get_cupy_utils(backend):

        if backend == "numpy":
            raise NotImplementedError

        elif backend == "cupy":
            from . import cupy_utils

            return cupy_utils
        else:
            BackendSpecific.raise_backend_error(backend)

    @staticmethod
    @lru_cache(maxsize=4)
    def digamma(backend):
        if backend == "numpy":
            return sps.digamma
        elif backend == "cupy":
            from cupyx.scipy.special import (
                digamma as digamma_cupy,
            )  # pylint: disable=import-error

            return digamma_cupy

    @staticmethod
    @lru_cache(maxsize=4)
    def polygamma(backend):
        if backend == "numpy":
            return sps.polygamma
        elif backend == "cupy":
            from cupyx.scipy.special import (
                polygamma as polygamma_cupy,
            )  # pylint: disable=import-error

            return polygamma_cupy


xp_utils = BackendSpecific()


def cupy_alternative(infer_backend_from):
    """
    A decorator that allows to call an alternative cupy function in case
    input arrays are cupy arrays instead of numpy arrays. The cupy alternative
    method must have the same name, and the same signature as the decorated
    method and must be place in cupy_utils module.

    Parameters
    ----------
    infer_backend_from: str
        Name of an input array in decorated function signature from which
        backend (numpy or cupy) is inferred.
    """

    def _cupy_alternative(numpy_func):
        func_args = list(inspect.signature(numpy_func).parameters.keys())

        @wraps(numpy_func)
        def method_call(*args, backend=None, **kwargs):
            array_level = func_args.index(infer_backend_from)
            array = (
                args[array_level]
                if array_level < len(args)
                else kwargs.get(infer_backend_from)
            )
            backend = backend or infer_backend(array)
            func_to_execute = (
                numpy_func
                if backend == "numpy"
                else getattr(xp_utils.get_cupy_utils("cupy"), numpy_func.__name__)
            )
            return func_to_execute(*args, **kwargs)

        return method_call

    return _cupy_alternative


def _has_a_shared_sublist(ls1, ls2):
    """
    Given two lists ls1 and ls2, if the two sub-lists defined as "all elements
    in ls1 that belong to ls2 and all elements in ls2 that belong to ls1" are
    equals it returns True, otherwise it returns False.

    Example
    -------
    >>> _has_a_shared_sublist([], [1,2,3])
    True
    >>> _has_a_shared_sublist([1,2,3,4], [2,5,3])
    True
    >>> _has_a_shared_sublist([1,2,3,4], [0,3,2])
    False
    >>> _has_a_shared_sublist([1,2,3,4], [1,2,5,6,7,8,5,76,4,3])
    False
    """

    def get_all_in(one, another):
        for element in one:
            if element in another:
                yield element

    for x1, x2 in zip(get_all_in(ls1, ls2), get_all_in(ls2, ls1)):
        if x1 != x2:
            return False

    return True


def infer_backend(x):
    backend = x.__class__.__module__.split(".")[0]
    if backend == "builtins":
        return "numpy"
    return backend


@lru_cache(maxsize=2048)
def get_transpose_and_slice(sub, sub_out):
    """
    Gives transposition and slicing to apply to an array whose supscripts name
    are sub so it can be sum to another array whose supscripts name are sub_out.
    """
    transpose = tuple(sub.index(idx) for idx in sub_out if idx in sub)
    slice_to_apply = tuple(None if idx not in sub else slice(None) for idx in sub_out)
    slice_to_apply = Ellipsis if slice_to_apply == () else slice_to_apply
    return transpose, slice_to_apply


def supscript_summation(op1, sub1, op2, sub2, sub_out):
    """
    Equivalent to einsum but for summation of two tensors instead of multiplication.

    Parameters
    ----------
    op1: array_like
        First operand
    sub1: tuple of hashable or string
        Sequence of IDs for each axis of operand0
    op2: array_like
        Second operand
    sub2: tuple of hashable or string
        Sequence of IDs for each axis of operand1
    sub_out: tuple of hashable or string
    """
    transpose1, slice1 = get_transpose_and_slice(sub1, sub_out)
    transpose2, slice2 = get_transpose_and_slice(sub2, sub_out)
    return op1.transpose(transpose1)[slice1] + op2.transpose(transpose2)[slice2]


def _parse_einsum_args(*args):
    if isinstance(args[0], str):
        try:
            einsum_str = args[0].translate({ord(" "): None})
        except TypeError:
            einsum_str = args[0].translate(" ", None)
        input_str, output_str = einsum_str.split("->")
        input_str_list = input_str.split(",")
        number_tensor = len(input_str_list)
        args2 = [
            elem
            for tensor_subscripts in zip(args[1 : 1 + number_tensor], input_str_list)
            for elem in tensor_subscripts
        ]
        args2.append(output_str)
        args2 += list(args[1 + number_tensor :])
    else:
        args2 = list(args)
    # change all hashable idx to int
    size_args = len(args2)
    idx_set = set.union(
        *(set(args2[ii]) for ii in range(1, size_args, 2)), set(args2[-1])
    )
    idx_to_int = {idx: num for num, idx in enumerate(idx_set)}
    for ii in list(range(1, size_args, 2)) + [
        -1,
    ]:
        args2[ii] = tuple(idx_to_int[key] for key in args2[ii])
    return args2


def einsum(*args, **kwargs):
    """
    An optimized version of einsum. Can works with both numpy and cupy arrays.
    """
    args2 = _parse_einsum_args(*args)
    backend = kwargs.get("backend", None)
    backend = backend or infer_backend(args2[0])
    kwargs["backend"] = backend
    out = kwargs.pop("out", None)
    if len(args2) == 5:
        output = _einsum_two_operands(*args2, out=out, **kwargs)
    else:
        if backend == "cupy" and out is not None:  # cupy does not support "out"
            out[...] = opt_einsum.contract(*args2, **kwargs)
            return
        output = opt_einsum.contract(*args2, out=out, **kwargs)
    if out is None:
        return output


def _einsum_two_operands(op1, sub1, op2, sub2, sub_out, backend, out=None):
    """
    Call for dot or multiply instead of einsum
    """
    # TODO: eventually squeeze input operands' axis that can be pre-summed
    all_return = _parse_einsum_two_operands_input(
        op1.shape, tuple(sub1), op2.shape, tuple(sub2), tuple(sub_out), backend
    )
    func = all_return[0]
    args = all_return[1:]
    return func(op1, sub1, op2, sub2, out, sub_out, backend, *args)


def _element_wise_mult(
    op1, sub1, op2, sub2, out, sub_out, backend, transpose1, transpose2, slice1, slice2
):
    if transpose1:
        op1 = op1.transpose(transpose1)
    if transpose2:
        op2 = op2.transpose(transpose2)
    return xp_utils.back(backend).multiply(op1[slice1], op2[slice2], out=out)


def _regular_einsum(op1, sub1, op2, sub2, out, sub_out, backend):
    if out is None:
        return opt_einsum.contract(op1, sub1, op2, sub2, sub_out, backend=backend)
    else:
        if backend == "cupy" or out.ndim == 0:
            out[...] = opt_einsum.contract(
                op1, sub1, op2, sub2, sub_out, backend=backend
            )
        else:
            opt_einsum.contract(op1, sub1, op2, sub2, sub_out, backend=backend, out=out)
        return out


def _sequential_tensor_dot(
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
):
    # declaration of output tensor
    if out is None:
        out = xp_utils.back(backend).empty(out_shape, dtype=op1.dtype)

    for slice_out, slice_1, slice_2 in zip(
        list_of_slice_out, list_of_slice_1, list_of_slice_2
    ):
        out[slice_out] = _einsum_as_dot(
            op1[slice_1], new_sub1, op2[slice_2], new_sub2, new_sub_out, backend
        )
    return out


@lru_cache(maxsize=1024)
def _parse_einsum_two_operands_input(shape_1, sub1, shape_2, sub2, sub_out, backend):
    """
    Choose method and compute necessarily values for the chosen method
    """

    # values that will be useful
    sub1 = list(sub1)
    sub2 = list(sub2)
    sub_out = list(sub_out)
    set1 = set(sub1)
    set2 = set(sub2)
    set_out = set(sub_out)

    no_duplication_idx = (
        len(set1) == len(sub1)
        and len(set2) == len(sub2)
        and len(set_out) == len(sub_out)
    )

    # Is it a simple multiplication ?
    if set1.issubset(sub_out) and set2.issubset(sub_out) and no_duplication_idx:
        transpose2 = None
        if not _has_a_shared_sublist(sub2, sub_out):
            transpose2 = tuple(sub2.index(idx) for idx in sub_out if idx in sub2)
        transpose1 = None
        if not _has_a_shared_sublist(sub1, sub_out):
            transpose1 = tuple(sub1.index(idx) for idx in sub_out if idx in sub1)

        slice2 = tuple(slice(None) if idx in sub2 else None for idx in sub_out)
        slice1 = tuple(slice(None) if idx in sub1 else None for idx in sub_out)
        return _element_wise_mult, transpose1, transpose2, slice1, slice2

    # Is it a multiplication and then a reduction (with no extra dimension)?
    # (only interesting if backend is cupy due to poor perf of cupy.einsum)
    # let's  suppose set1 is smaller than set2
    if (
        (set1.issubset(set2) or set2.issubset(set1))
        and no_duplication_idx
        and backend == "cupy"
    ):
        switch_operand = False
        if len(set1) > len(set2):
            shape_1, sub1, shape_2, sub2 = shape_2, sub2, shape_1, sub1
            switch_operand = True
        transpose2 = tuple(sub2.index(idx) for idx in sub_out)
        transpose2 += tuple(ii for ii in range(len(sub2)) if ii not in transpose2)

        new_sub2 = [sub2[idx] for idx in transpose2]
        transpose1 = tuple(sub1.index(idx) for idx in new_sub2 if idx in sub1)
        slice1 = tuple(slice(None) if idx in sub1 else None for idx in new_sub2)
        sum_axis = tuple(range(len(sub_out), len(new_sub2)))

        return (
            _element_wise_mult_and_sum,
            switch_operand,
            transpose1,
            transpose2,
            slice1,
            sum_axis,
        )

    # Do we have to use einsum?
    double_id = set1.intersection(set2)
    sum_id = set1.union(set2) - set_out
    has_single_marginalization = (set1.difference(set2).difference(set_out)) or (
        set2.difference(set1).difference(set_out)
    )
    if (
        not sum_id.intersection(double_id)
        or set1.issubset(set2)
        or set2.issubset(set1)
        or not no_duplication_idx
        or has_single_marginalization
    ):
        return (_regular_einsum,)

    # Else we use tensordot
    inner_id = set_out.intersection(double_id)
    inner_id_list = list(inner_id)

    # construct id2shape dict:
    id2shape_dict = {
        idx: shape[index_id.index(idx)]
        for shape, index_id in zip([shape_1, shape_2], [sub1, sub2])
        for idx in index_id
    }

    # construct id2axis for all tensors
    id2axis_dict_1 = {idx: sub1.index(idx) for idx in set1}
    id2axis_dict_2 = {idx: sub2.index(idx) for idx in set2}
    id2axis_dict_out = {idx: sub_out.index(idx) for idx in set_out}

    # Instantiation of output tensor
    out_shape = tuple(id2shape_dict[idx] for idx in sub_out)

    # subscripts with no inner indexes
    for idx in inner_id:
        sub1.remove(idx)
        sub2.remove(idx)
        sub_out.remove(idx)

    # then we apply einsum (from opt_einsum) for each values of inner_id
    list_of_slice_out = []
    list_of_slice_1 = []
    list_of_slice_2 = []

    shape_inner_id = tuple(id2shape_dict[idx] for idx in inner_id_list)
    for inner_id_values in product(*(range(ii) for ii in shape_inner_id)):
        slice_1 = [
            slice(None),
        ] * len(shape_1)
        slice_2 = [
            slice(None),
        ] * len(shape_2)
        slice_out = [
            slice(None),
        ] * len(out_shape)
        for idx, num_idx in zip(inner_id_list, inner_id_values):
            slice_1[id2axis_dict_1[idx]] = num_idx
            slice_2[id2axis_dict_2[idx]] = num_idx
            slice_out[id2axis_dict_out[idx]] = num_idx
        list_of_slice_out.append(tuple(slice_out))
        list_of_slice_1.append(tuple(slice_1))
        list_of_slice_2.append(tuple(slice_2))

    return (
        _sequential_tensor_dot,
        list_of_slice_out,
        list_of_slice_1,
        list_of_slice_2,
        sub1,
        sub2,
        sub_out,
        out_shape,
    )


def _element_wise_mult_and_sum(
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
):
    if switch_operand:
        op1, sub1, op2, sub2 = op2, sub2, op1, sub1
    op1 = op1.transpose(transpose1)
    op2 = op2.transpose(transpose2)
    return xp_utils.get_cupy_utils("cupy").multiply_and_sum(
        op1[slice1], op2, out=out, axis=sum_axis
    )


def _einsum_as_dot(op1, sub1, op2, sub2, sub_out, backend):
    """
    Only works if calculus can be done only with reshapes and dot
    """
    sub1 = list(sub1)
    sub2 = list(sub2)

    # find transposes
    double_idx = [ii for ii in sub1 if ii in sub2]
    double_axes1 = [sub1.index(idx) for idx in double_idx]
    double_axes2 = [sub2.index(idx) for idx in double_idx]

    transpose_out = [sub_out.index(idx) for idx in sub1 + sub2 if idx not in double_idx]

    tensordot_out_sub = [idx for idx in sub1 + sub2 if idx not in double_idx]
    transpose_out = [tensordot_out_sub.index(idx) for idx in sub_out]

    return (
        xp_utils.back(backend)
        .tensordot(op1, op2, (double_axes1, double_axes2))
        .transpose(transpose_out)
    )


def einconv(
    operand0,
    sublist0,
    operand1,
    sublist1,
    sublistout,
    *args,
    conv_idx_list=None,
    backend=None,
    compute_correlation=False,
    **kwargs
):
    """
    A upgraded version of einsum for two operands operations that can also
    compute convolutions on one or more axes. Works with numpy and cupy arrays.
    Only 'valid' mode is available for convolution (see docstring of
    numpy.convolve for the definition of 'valid' mode)

    Parameters
    ----------
    operand0: array_like
        First operand
    sublist0: sequence of hashable
        Sequence of IDs for each axis of operand0
    operand1: array_like
        Second operand
    sublist1: sequence of hashable
        Sequence of IDs for each axis of operand1
    sublistout: sequence of hashable
        Sequence of IDs for each axis of desired output array
    args: sequence of arguments
        All arguments that can be passed to numpy.einsum (see corresponding
        docstring).
    conv_idx_list: sequence hashable or None, optional, default None
        If None, works as numpy.einsum. If not None, specify the axes IDs of
        operand0 and operand1 on which to perform convolution in 'valid' mode
        (they must have the same ID). Convolution can be performed along several
        dimension. Either operand0 or operand1 must be at least as large as the
        other in every dimension where convolution is performed.
    backend: 'numpy', 'cupy' or None, optional, default None
        Specifies input arrays' backend. If None, the backend is automatically
        inferred.
    compute_correlation: bool, optional, default False
        If True, correlation is computed instead of convolution (it applies to
        all axes on which to perform convolution).
    kwargs: dict
        Other keyword arguments that can be passed to numpy.einsum.
    """
    backend = backend or infer_backend(operand0)
    kwargs["backend"] = backend
    conv_idx_list = conv_idx_list or []
    if conv_idx_list == []:
        return einsum(
            operand0, sublist0, operand1, sublist1, sublistout, *args, **kwargs
        )

    slice_direction = 1 if compute_correlation else -1
    sub_list = [list(sublist0), list(sublist1)]
    op_list = [operand0, operand1]
    all_idx_set = set(sublist0).union(set(sublist1)).union(set(sublistout))
    for num, conv_idx in enumerate(conv_idx_list):
        # operand with the greatest convolution axis size is put in first position
        num_dim_list = [sub.index(conv_idx) for sub in sub_list]
        size_list = [op.shape[num_dim] for op, num_dim in zip(op_list, num_dim_list)]
        if size_list[0] < size_list[1]:
            if num > 0:
                raise ValueError(
                    "Either operand0 or operand1 must be at least as large as the other in every dimension where convolution is performed"
                )
            for elem in [num_dim_list, size_list, op_list, sub_list]:
                elem.reverse()
        size_out = size_list[0] - size_list[1] + 1
        size_small = op_list[1].shape[num_dim_list[1]]

        shape = (
            op_list[0].shape[: num_dim_list[0]]
            + (size_out, size_small)
            + op_list[0].shape[num_dim_list[0] + 1 :]
        )

        strides = (
            op_list[0].strides[: num_dim_list[0]]
            + (op_list[0].strides[num_dim_list[0]],) * 2
            + op_list[0].strides[num_dim_list[0] + 1 :]
        )
        new_idx = _make_unique_hashable(all_idx_set)
        all_idx_set.add(new_idx)
        sub_list[0].insert(num_dim_list[0] + 1, new_idx)
        sub_list[1][num_dim_list[1]] = new_idx
        tuple_of_slices = (slice(None),) * (num_dim_list[0] + 1) + (
            slice(None, None, slice_direction),
        )
        op_list[0] = xp_utils.as_strided(backend)(
            op_list[0], shape=shape, strides=strides
        )[tuple_of_slices]

    return einsum(
        *[elem for tuple_of_elem in zip(op_list, sub_list) for elem in tuple_of_elem],
        sublistout,
        *args,
        **kwargs
    )


def _make_unique_hashable(forbidden_set):
    ii = 0
    while ii in forbidden_set:
        ii += 1
    return ii


def _find_equality_root(
    input_array,
    input_denominator,
    const_coef,
    max_iter,
    type="equality",
    atol=1e-15,
    backend=None,
):
    """
    const_coef must have at least one positive value
    """
    backend = backend or infer_backend(input_array)
    if input_denominator.size == input_array.size:
        input_denominator = input_denominator.reshape(input_array.shape)
        denominator_full_size = True
    elif input_denominator.ndim == input_array.ndim:
        denominator_full_size = True
    elif input_denominator.size == const_coef.size:
        input_denominator = input_denominator.reshape(const_coef.shape)
        denominator_full_size = False
    else:
        input_denominator = (
            xp_utils.back(backend).zeros(input_array.shape, dtype=input_array.dtype)
            + input_denominator
        )
        denominator_full_size = True

    extra_dim = False
    if input_array.ndim == const_coef.ndim:
        input_array = input_array[None, :]
        if denominator_full_size:
            input_denominator = input_denominator[None, :]
        extra_dim = True

    degree = const_coef.size
    sum_axis = tuple(range(input_array.ndim - const_coef.ndim, input_array.ndim))
    flag_shape = tuple(
        input_array.shape[ii] for ii in range(0, input_array.ndim - const_coef.ndim)
    )
    sum_axis_flag = tuple(range(1, 1 + const_coef.ndim))
    shape_sigma = (
        input_array.shape[slice(input_array.ndim - const_coef.ndim)]
        + (1,) * const_coef.ndim
    )
    proj = (input_array * const_coef).sum(axis=sum_axis, keepdims=True)
    sigma = xp_utils.back(backend).zeros(shape_sigma, dtype=input_array.dtype)

    if type == "inequality":
        flag = proj.reshape(flag_shape) < 0
    elif type == "equality":
        flag = xp_utils.back(backend).ones(flag_shape, dtype=bool)
    else:
        raise ValueError("Argument 'type' must be 'equality' or 'inequality'")

    for __ in range(max_iter):
        if not flag.any():
            break
        sigma_flag = sigma[flag]
        if denominator_full_size:
            den_array_flag = input_denominator[flag]
        else:
            den_array_flag = input_denominator
        input_array_flag = input_array[flag]
        temp1 = const_coef / (den_array_flag - sigma_flag * const_coef)
        S1 = temp1.sum(axis=sum_axis_flag, keepdims=True)
        temp2 = temp1 * input_array_flag
        S2 = temp2.sum(axis=sum_axis_flag, keepdims=True)
        temp3 = temp2 * temp1
        S3 = temp3.sum(axis=sum_axis_flag, keepdims=True)
        S4 = (temp1 ** 2).sum(axis=sum_axis_flag, keepdims=True)
        S5 = (temp3 * temp1).sum(axis=sum_axis_flag, keepdims=True)

        H1 = -S1 * S2 + S3
        H2 = -S4 * (S2 ** 2) + 2 * S2 * S5 - S3 ** 2
        H3 = ((1 - degree) * (degree * H2 + H1 ** 2)) ** 0.5

        new_sigma = sigma_flag - degree * S2 / (
            H1 + xp_utils.back(backend).sign(H1) * H3
        )
        sigma[flag] = new_sigma
        flag[flag] = flag[flag] & xp_utils.back(backend).atleast_1d(
            ~xp_utils.back(backend).isclose(S2, 0, atol=atol).squeeze()
        )
    if extra_dim:
        sigma = sigma[0]
    return sigma


@cupy_alternative(infer_backend_from="x_arr")
def xlogy(x_arr, y_arr, out=None, backend=None):
    """
    Same as scipy.special.xlogy but works either with numpy of cupy arrays.
    Used backend can be manually specified via keyword argument 'backend' which
    can be 'numpy', 'cupy' or None. If None, the backend is automatically
    inferred.
    """
    return sps.xlogy(x_arr, y_arr, out=out)  # pylint: disable=no-member


def cumsum_last_axis(arr, out, backend=None):
    """
    Compute the cumulative sum on the last axis of input array.

    Parameters
    ----------
    arr: numpy or cupy ndarray
        Input array
    out: numpy or cupy ndarray
        Array on which to write result
    backend: 'numpy', 'cupy' or None
        Specifies the backend of input and output arrays. If None, it is
        automatically inferred.
    """
    backend = backend or infer_backend(arr)
    xp_utils.cumsum_last_axis(backend)(arr, out)


def next_pow_of_two(val):
    """
    Returns the lowest power of two that is greater or equal to input value.
    """
    if val < 1:
        return 1
    return 2 ** (int(np.ceil(val)) - 1).bit_length()


def scalar_compatible(*arg_names):
    """
    A decorator that deals with scalar or 0-dim array input for methods that
    only take array input with array.ndim > 0. Only available for methods whose
    array inputs must have the same shape. If an array is "None", then it is
    passed as is.

    Parameters
    ----------
    *args_names: sequence of str Sequence of input array names of the function
        to be decorated.
    """

    def _scalar_compatible(func):
        func_args = list(inspect.signature(func).parameters.keys())

        @wraps(func)
        def scalar_compatible_func(*args, **kwargs):
            args = list(args)
            backend = None
            for arg_name in arg_names:
                arg_level = func_args.index(arg_name)
                if arg_level < len(args):
                    arg_value = args[arg_level]
                    in_args = True
                else:
                    arg_value = kwargs.get(arg_name)
                    in_args = False
                if backend is None and arg_value is not None:
                    backend = infer_backend(arg_value)
                    xp = xp_utils.back(backend)
                    isscalar = xp.isscalar(arg_value)
                    ndim = 0 if isscalar else arg_value.ndim
                if arg_value is not None:
                    arg_value = xp.atleast_1d(arg_value)
                if in_args:
                    args[arg_level] = arg_value
                else:
                    kwargs[arg_name] = arg_value
            output_arr = func(*args, **kwargs)
            if output_arr is None:
                return
            if isscalar:
                return output_arr[0]
            if ndim == 0:
                return output_arr.squeeze()

            return output_arr

        return scalar_compatible_func

    return _scalar_compatible


@scalar_compatible("input_arr")
@cupy_alternative(infer_backend_from="input_arr")
@jit
def exp_digamma(input_arr):  # TODO: allow greater order, with tolerance stop
    """
    Exponential of Digamma function.
    """
    coef_list = [
        0.041666666666666664,  # 1 / 24,
        -0.006423611111111111,  # - 37 / 5760,
        0.003552482914462081,  # 10313 / 2903040,
        -0.0039535574489730305,  # - 5509121 / 1393459200,
    ]
    order = 4
    shape = input_arr.shape
    input_arr = input_arr.ravel()
    size_arr = input_arr.size
    output_arr = np.zeros(size_arr)

    for index in range(size_arr):
        input_plus_n = input_arr[index]
        if input_plus_n == 0.0:
            output_arr[index] = 0.0
        else:
            tmp2 = 0

            while input_plus_n < 10:
                tmp2 -= 1 / input_plus_n
                input_plus_n += 1

            input_plus_n -= 0.5
            output_arr[index] = input_plus_n
            temp = input_plus_n
            input_plus_n = input_plus_n ** 2

            for ind_coef in range(order):
                temp /= input_plus_n
                output_arr[index] += coef_list[ind_coef] * temp

            if tmp2 != 0:
                output_arr[index] *= np.exp(tmp2)

    output_arr = output_arr.reshape(shape)
    return output_arr


@scalar_compatible("v_arr", "z_arr")
@cupy_alternative(infer_backend_from="v_arr")
@jit
def hyp0f1ln(v_arr, z_arr, tol=1e-16):
    """
    Log of confluent hypergeometric limit function 0F1.

    Parameters
    ----------
    v_arr, z_arr: array_like
        Input values.
    tol: float, optional, default 1e-4
        Precision
    """
    # TODO: fixe overflow issue when z_arr>>1, a lead would to check there:
    # https://github.com/stan-dev/math/blob/92075708b1d1796eb82e3b284cd11e544433518e/stan/math/prim/fun/log_modified_bessel_first_kind.hpp

    # TODO: v_arr and z_arr must have the same shape for now
    shape = v_arr.shape
    v_arr = v_arr.ravel()
    z_arr = z_arr.ravel()
    input_size = z_arr.size
    output_val = np.zeros(input_size)
    output_temp_val = np.ones(input_size)
    for ii in range(input_size):
        v_val = v_arr[ii]
        temp_r = 1
        temp_a = 1
        keep_going = True
        while keep_going:
            if output_temp_val[ii] > 1e300:
                output_val[ii] = output_val[ii] + math.log(output_temp_val[ii])
                temp_a /= output_temp_val[ii]
                output_temp_val[ii] = 1
            temp_a *= z_arr[ii] / (temp_r * v_val)
            output_temp_val[ii] += temp_a
            temp_r += 1
            v_val += 1
            keep_going = temp_a > tol * output_temp_val[ii]
    output_val += np.log(output_temp_val)
    output_val = output_val.reshape(shape)
    return output_val


@scalar_compatible("v_arr", "z_arr")
@cupy_alternative(infer_backend_from="v_arr")
@jit
def bessel_ratio(v_arr, z_arr, tol=1e-16):
    # TODO: v_arr and z_arr must have the same shape for now
    """
    Calculates the function z_arr * iv(v_arr + 1,z_arr)/iv(v_arr,z_arr) where iv
    is the modified bessel function of the first kind. The approximation
    algorithm [Gautschi 1978] is used

    Parameters
    ----------
    v_arr, z_arr : array_like. Must be nonnegative (an error could be raised
          otherwise) Input data.
    tol : tolerance (number of accurate significant figures)
    """
    shape = v_arr.shape
    v_arr = v_arr.ravel()
    z_arr = z_arr.ravel()
    input_size = v_arr.size
    z_arr2 = z_arr / 2
    output_val = np.ones(input_size)
    for ii in range(input_size):
        temp_pr = 1
        temp_v0 = v_arr[ii] + z_arr2[ii] + 1
        temp_v = v_arr[ii] + z_arr[ii] + 1.5
        temp_u = (v_arr[ii] + 1 + z_arr[ii]) * temp_v
        temp_w = z_arr2[ii] * (v_arr[ii] + 1.5)
        temp_p = temp_w / (temp_v0 * temp_v - temp_w)
        tol2 = tol * (1 + temp_p)
        temp_pr *= temp_p
        output_val[ii] += temp_pr
        keep_going = True
        while keep_going:
            temp_u += temp_v
            temp_v += 0.5
            temp_w += z_arr2[ii]
            temp_t = temp_w * (1 + temp_p)
            temp_p = temp_t / (temp_u - temp_t)
            temp_pr *= temp_p
            output_val[ii] += temp_pr
            keep_going = temp_pr > tol2
        output_val[ii] *= z_arr[ii] ** 2 / (z_arr[ii] + 2 * v_arr[ii] + 2)
    output_val = output_val.reshape(shape)
    return output_val


def forced_iter(input):
    """
    Returns `iter(input)` if input is iterable, otherwise returns
    `iter((input, ))`
    """
    try:
        it = iter(input)
    except TypeError:
        it = iter((input,))
    return it


def _is_bool_masking(elem):
    """
    Returns boolean masking as numpy array if elem is a boolean mask. Returns None
    otherwise
    """
    try:
        iter(elem)
        elem_as_np = np.array(elem)
        if elem_as_np.dtype == bool:
            return elem_as_np
    except TypeError:
        pass
    return None


def explicit_slice(input_slice, ndim):
    """
    Returns explicit slicing as tuple of a ndim-ndarray given an input slicing.
    input slicing must be either a tuple or a single element (slice for the first
    dimension).
    If `ndim` equals 0, then input_slice is returned as is.

    Examples
    --------
    >>> explicit_slice(Ellipsis, 2)
    (slice(None, None, None), slice(None, None, None))
    >>> explicit_slice([Ellipsis, 1], 2)
    (slice(None, None, None), 1)
    >>> explicit_slice([slice(1), 1], 3)
    (slice(None, 1, None), 1, slice(None, None, None))
    >>> explicit_slice(Ellipsis, 0)
    Ellipsis
    """

    if ndim == 0:
        return input_slice
    if type(input_slice) != tuple:
        input_slice = (input_slice,)

    # check if there is a boolean masking, in which case ndim might be lower
    for elem in input_slice:
        elem_as_np = _is_bool_masking(elem)
        if elem_as_np is not None:
            ndim -= elem_as_np.ndim - 1

    explicit_slice = [
        slice(None),
    ] * ndim
    found_ellipsis = False
    for num_dim, sl in enumerate(input_slice):
        if sl is Ellipsis:
            found_ellipsis = True
            break
        explicit_slice[num_dim] = sl
    if found_ellipsis:
        for num_dim, sl in enumerate(input_slice[::-1]):
            if sl is Ellipsis:
                break
            explicit_slice[-1 - num_dim] = sl

    return tuple(explicit_slice)


def real_to_2D_nonnegative(input_array):
    """
    Transforms a real-valued numpy array for a nonnegative one.
    Extra dimension (pos/neg part) corresponds to the last axis
    Input:
      - a [dim_1 x dim_2 x .... dim_N] real-valued numpy array
    Output:
      - a [dim_1 x dim_2 x .... dim_N x 2] numpy array with only nonnegative values
    """
    output_array = np.expand_dims(input_array, -1)
    output_array = np.abs(
        np.concatenate(
            (np.maximum(output_array, 0), np.maximum(-output_array, 0)), axis=-1
        )
    )
    return output_array


def complex_to_2D_real(input_array, returns_view=True):
    """
    Transforms a complex-valued numpy array for a real-valued one.
    Extra dimension (real part / imaginary part) corresponds to the last axis
    By default, it returns a new view of the input array
    Input:
      - [dim_1 x dim_2 x .... dim_N] complex-valued numpy array
      - returns_view (opt., default True): if False (resp. True), it returns a copy (resp. a new view) of the input numpy array
    Output:
      - [dim_1 x dim_2 x .... dim_N x 2] real-valued numpy array
    """
    itemsize = int(input_array.dtype.itemsize / 2)
    output_array = as_strided(
        input_array,
        shape=input_array.shape + (2,),
        strides=input_array.strides + (itemsize,),
    ).real
    if not returns_view:
        output_array = output_array.copy()
    return output_array


def complex_to_4D_nonnegative(input_array):
    """
    Transforms a complex-valued numpy array for a nonnegative-valued one.
    Extra dimensions (real/imaginary part and pos/neg part) corresponds to the last two axis
    It returns a new copy of the input array
    Input:
      - [dim_1 x dim_2 x .... dim_N] complex-valued numpy array
    Output:
      - [dim_1 x dim_2 x .... dim_N x 2 x 2] numpy array with only nonnegative values
    """

    return real_to_2D_nonnegative(complex_to_2D_real(input_array))


def create_filiation(parent, child, **kwargs):
    """
    Calls parent.new_child(child, **kwargs)
    """
    parent.new_child(child, **kwargs)


def clip_inplace(arr, a_min=None, a_max=None, backend=None):
    backend = backend or infer_backend(arr)
    if backend == "numpy":
        arr.clip(min=a_min, max=a_max, out=arr)
    elif backend == "cupy":
        if a_min is not None:
            xp_utils.get_cupy_utils("cupy").min_clip(arr, a_min, arr)
        if a_max is not None:
            xp_utils.get_cupy_utils("cupy").max_clip(arr, a_max, arr)


@scalar_compatible("input_arr", "out")
def inverse_digamma(input_arr, num_iter=5, backend=None, out=None):
    """
    Computes the inverse digamma function using Newton's method
    See Appendix c of Minka, T. P. (2003). Estimating a Dirichlet distribution.
    Annals of Physics, 2000(8), 1-13. http://doi.org/10.1007/s00256-007-0299-1 for details.
    """
    backend = backend or infer_backend(input_arr)

    xp = xp_utils.back(backend)
    # initialization
    if out is None:
        # to be sure int are converted into floats
        output_arr = xp.zeros_like(input_arr) + 0.0
    else:
        output_arr = out
    mask = input_arr >= -2.22
    output_arr[mask] = xp.exp(input_arr[mask]) + 0.5
    output_arr[~mask] = -(1 / (input_arr[~mask] + -xp_utils.digamma(backend)(1)))
    # do Newton update here
    order = xp.array(1)
    for __ in range(num_iter):
        output_arr -= (
            xp_utils.digamma(backend)(output_arr) - input_arr
        ) / xp_utils.polygamma(backend)(order, output_arr)

    if out is None:
        return output_arr


def normalize(tensor, axis):
    """
    Returns a normalized version of the input tensor (l1 normalization) along
    given axis

    Parameters
    ----------
    tensor: array_like,
        Must contain nonnegative coefficients
    axis: tuple
        Axis along which tensor should be normalized

    Returns
    -------
    array_like
        output tensor, equals to `tensor / tensor.sum(axis=axis, keepdims=True)`

    """
    return tensor / tensor.sum(axis=axis, keepdims=True)
