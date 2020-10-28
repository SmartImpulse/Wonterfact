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

"""Module for all operator classes"""

# Python System imports
from functools import cached_property

# Third-party imports
import numpy as np
from methodtools import lru_cache

# Relative imports
from . import utils, core_nodes
from .glob_var_manager import glob


class _Operator(core_nodes._ChildNode, core_nodes._DynNodeData):
    """
    Mother class of all operators
    """

    @cached_property
    def tensor_has_energy(self):
        return any(parent.tensor_has_energy for parent in self.list_of_parents)

    @cached_property
    def norm_axis(self):
        if self.tensor_has_energy:
            return None
        norm_axis = []
        for num_idx, idx in enumerate(self.index_id):
            is_idx_in_norm_axis = next(
                (
                    bool(parent)
                    for parent in self.list_of_parents
                    if idx in parent.get_index_id_for_children(self)
                    and parent.get_index_id_for_children(self).index(idx)
                    in parent.get_norm_axis_for_children(self)
                ),
                False,
            )
            if is_idx_in_norm_axis:
                norm_axis.append(num_idx)
        return tuple(norm_axis)


class Proxy(_Operator):
    """
    A proxy class to any node.

    Can be useful when one wants a parent and a child being linked several
    times (which is not possible straightforward). A proxy can have only one
    parent.
    """

    # TODO: compatible with Mutliplexer as child
    # TODO: if no brother, could optimize tensor_update which is copied by its parent

    max_parents = 1

    def _give_update(self, parent, out=None):
        if out is None:
            return self.tensor_update
        out[...] = self.tensor_update

    def _initialization(self):
        self.tensor = self.first_parent.get_tensor_for_children(self)
        if not glob.xp.may_share_memory(self.tensor, self.first_parent.tensor):
            raise ValueError(
                """A Proxy's tensor should be a view to its parent's:
                I guess you cannot use this 'slice_for_child'"""
            )
        self.tensor_update = glob.xp.zeros(self.tensor.shape, dtype=glob.float)

    @lru_cache(maxsize=1)
    def _get_raw_mean_tensor_for_VBEM(self, current_iter):
        raw_tensor = self.first_parent._get_mean_tensor_for_VBEM(self, current_iter)
        return raw_tensor

    def _check_filiation_ok(self, child=None, parent=None, **kwargs):
        if child is not None:
            if kwargs.get("slice_for_child", Ellipsis) != Ellipsis:
                raise ValueError(
                    "`slice_for_child` argument cannot be specified when parent is a {} object".format(
                        type(self)
                    )
                )
            if kwargs.get("strides_for_child", None) != None:
                raise ValueError(
                    "`strides_for_child` argument cannot be specified when parent is a {} object".format(
                        type(self)
                    )
                )
            if isinstance(child, Multiplexer):
                raise ValueError(
                    "Proxy object cannot be a parent of Multiplexer object."
                )
        super()._check_filiation_ok(child=child, parent=parent, **kwargs)


class Multiplier(_Operator):
    """
    Class of multiplier operator.

    Must have two and only two parents. Inner tensor is the multiplication of
    its parent's tensor. It can also be used to convoluate its parent's tensor
    when the attribute 'conv_idx_ids' is provided (cf __init__ doctstring). The
    way the two parent's tensor are multiplied or convolved is automatically
    computed according the index IDs of all involved tensors. Only 'valid' mode
    for convolution is available (cf docstring of numpy.convolve for the
    definition of this mode).
    """

    max_parents = 2

    def __init__(self, conv_idx_ids=None, **kwargs):
        """
        Returns a Multiplier object which multiplies or convolves the parent's
        tensors.

        Parameters
        ----------
        conv_idx_ids: sequence hashable, optional, default None
            When provided and different from None, the operator acts as a
            convolver. Must be of the form [idx1, idx2, ...]
            where each idx is the name of parents' and self's axes to be
            convolved (convolution can be performed on several axes). If
            parent's names of axes to be convolved are different, you should
            use option `index_id_for_child` in the filiation creation process
            (i.e. when calling `new_child` or `new_parent` method) to change
            them.
        """
        self.conv_idx_ids = conv_idx_ids or []
        super().__init__(**kwargs)

    @cached_property
    def conv_idx_list_for_einconv(self):
        return [(idx,) * 3 for idx in self.conv_idx_ids]

    def _update_tensor(self, **kwargs):
        input_einconv = [
            elem
            for node in self.list_of_parents
            for elem in [
                node.get_tensor_for_children(self),
                node.get_index_id_for_children(self),
            ]
        ] + [self.index_id]

        utils.einconv(
            *input_einconv, out=self.tensor[...], conv_idx_list=self.conv_idx_ids
        )

    @lru_cache(maxsize=1)
    def _get_raw_mean_tensor_for_VBEM(self, current_iter):
        input_einconv = [
            elem
            for node in self.list_of_parents
            for elem in [
                node._get_mean_tensor_for_VBEM(self, current_iter),
                node.get_index_id_for_children(self),
            ]
        ] + [self.index_id]

        raw_tensor = np.zeros_like(self.tensor)
        utils.einconv(
            *input_einconv, conv_idx_list=self.conv_idx_ids, out=raw_tensor[...]
        )
        return raw_tensor

    def _bump(self, **kwargs):
        self._update_tensor()

    def _give_update(self, parent, out=None):
        input_einconv = [
            elem
            for node in self.list_of_parents
            if node != parent
            for elem in [
                node.get_tensor_for_children(self),
                node.get_index_id_for_children(self),
            ]
        ] + [
            self.full_tensor_update,
            self.index_id,
            parent.get_index_id_for_children(self),
        ]
        update_tensor = utils.einconv(
            *input_einconv,
            compute_correlation=True,
            conv_idx_list=self.conv_idx_ids,
            out=out
        )
        if out is None:
            return update_tensor

    def _initialization(self):
        # instantiation of tensor
        shape_dict = {}
        # first find all indexes that are involved in convolution
        for parent in self.list_of_parents:
            parent_tensor_shape = parent.get_tensor_for_children(self).shape
            parent_idx = parent.get_index_id_for_children(self)
            for idx, size in zip(parent_idx, parent_tensor_shape):
                if idx not in self.conv_idx_ids:
                    shape_dict.update({idx: size})
        # then we compute the resulting dimension for convolution indexes
        shape_dict_conv_diff = {}
        shape_dict_conv_sum = {}
        shape_dict_conv_min = {}
        for idx in self.conv_idx_ids:
            dim_iter = iter(
                parent.idx_to_shape_for_child(idx, self)
                for parent in self.list_of_parents
                if idx in parent.get_index_id_for_children(self)
            )
            dim1, dim2 = next(dim_iter), next(dim_iter)
            dim1, dim2 = (dim2, dim1) if dim1 >= dim2 else (dim1, dim2)
            shape_dict_conv_diff.update({idx: dim2 - dim1 + 1})
            shape_dict_conv_sum.update({idx: dim2 + dim1 - 1})
            shape_dict_conv_min.update({idx: dim1})

        shape_dict.update(shape_dict_conv_diff)  # for inner tensor
        self.tensor = glob.xp.empty(
            tuple(shape_dict[idx] for idx in self.index_id), dtype=glob.float
        )
        if self.update_period != 0:
            shape_dict.update(shape_dict_conv_sum)  # for inner full_tensor_update
            full_shape = tuple(shape_dict[idx] for idx in self.index_id)
            self.full_tensor_update = glob.xp.ones(full_shape, dtype=glob.float)
            tensor_update_slice = tuple(
                slice(None)
                if idx not in self.conv_idx_ids
                else slice(shape_dict_conv_min[idx] - 1, -shape_dict_conv_min[idx] + 1)
                for idx in self.index_id
            )
            self.tensor_update = self.full_tensor_update[tensor_update_slice]

        self._update_tensor()

    def _check_model_validity(self):
        super()._check_model_validity()
        parent_with_energy_list = [
            parent for parent in self.list_of_parents if parent.tensor_has_energy
        ]
        if len(parent_with_energy_list) > 1:
            raise ValueError(
                "Problem with {}'s parents. At most one parent can have a "
                "tensor having energy i.e. no subject to normalization "
                "constraint.".format(self)
            )
        if not self.tensor_has_energy and self.conv_idx_ids:
            raise ValueError(
                "Problem with {}. In convolution mode, inner tensor of a "
                "Multiplier must have energy. Please put a LeafGamma (or any "
                "Leaf carrying energy) upstream.".format(self)
            )
        set_of_idx = set.union(
            *(
                set(parent.get_index_id_for_children(self))
                for parent in self.list_of_parents
            )
        )
        set_of_idx.update(self.index_id)
        dict_idx = {idx: {} for idx in set_of_idx}
        for parent in self.list_of_parents:
            parent_norm_axis = parent.get_norm_axis_for_children(self)
            parent_index_id = parent.get_index_id_for_children(self)
            for idx in parent_index_id:
                dict_idx[idx].update(
                    {
                        parent: {
                            "has_energy": parent.tensor_has_energy,
                            "is_normalized": (
                                True
                                if parent.tensor_has_energy
                                else parent_index_id.index(idx) in parent_norm_axis
                            ),  # it is simpler to consider normalized when tensor_has_energy
                            "is_marginalized": not idx in self.index_id,
                            "is_convolved": idx in self.conv_idx_ids,
                        }
                    }
                )
        for idx, parents_dict in dict_idx.items():
            # if idx is convolved, it must appear in two parents and it must
            # be normalized in both parents.
            if idx in self.conv_idx_ids:
                if len(parents_dict) != 2:
                    raise ValueError(
                        "Cannot convolve '{}' in {}. This index should belong "
                        "to two parents. You might make use of `index_id_for_child`"
                        " argument during filiation creation.".format(idx, self)
                    )
                if not all(info["is_normalized"] for info in parents_dict.values()):
                    raise ValueError(
                        "Model is wrong at {} level. Axis to be convolved "
                        "should be normalized or belong to a tensor that has energy."
                    )
            # idx can only be normalized once unless it is convolved
            normalized_parent_no_conv_list = [
                parent
                for parent, info in parents_dict.items()
                if info["is_normalized"] and not info["is_convolved"]
            ]
            normalized_parent_list = [
                parent for parent, info in parents_dict.items() if info["is_normalized"]
            ]
            if len(normalized_parent_no_conv_list) > 1:
                raise ValueError(
                    "Model is wrong at {} level. An index_id cannot be "
                    "normalized twice".format(self)
                )
            # if there is a parent with energy, idx must be normalized once
            if parent_with_energy_list and not normalized_parent_list:
                raise ValueError(
                    "Model is wrong at {} level. An index_id should be "
                    "normalized once before multiplication with a tensor that "
                    "has energy.".format(self)
                )
            # if idx is not normalized, it cannot be marginalized
            if (
                not normalized_parent_no_conv_list
                and next(iter(parents_dict.values()))["is_marginalized"]
            ):
                raise ValueError(
                    "Model is wrong at {} level. An index_id should be "
                    "normalized before marginalization".format(self)
                )


class Multiplexer(_Operator):
    """
    Class of Multiplexer operator.

    It aims at concatenate inner tensor of the parent nodes. Concatenation can
    be performed on a new axis (stacking) wich is automatically detected
    according to parents and self index IDs. It can also be performed on an
    existing axis when attributes `multiplexer_idx` is provided (see `__ini__`
    docstring).
    """

    # TODO: if no brother, could optimize tensor_update which is copied by its parent

    def __init__(self, multiplexer_idx=None, **kwargs):
        """
        Returns a Multiplexer object which concatenate or stack the parent's
        tensors.

        Parameters
        ----------
        multiplexer_idx: hashable or None, optional, default None
            If None, multiplexer will stack all parents' tensor along a new axis
            which is automatically detected by comparing self's `index_id`
            argument with its parents' ones. If not None, should refer to the
            axis ID along which parents' tensors are concatenated.
        """
        self.multiplexer_idx = multiplexer_idx
        super().__init__(**kwargs)
        self.parent_slicing_dict = {}

    def _check_filiation_ok(self, child=None, parent=None, **kwargs):
        if parent is not None:
            if kwargs.get("slice_for_child", None) != None:
                raise ValueError(
                    "`slice_for_child` argument cannot be specified when child is a {} object".format(
                        type(self)
                    )
                )
            if kwargs.get("strides_for_child", None) != None:
                raise ValueError(
                    "`strides_for_child` argument cannot be specified when child is a {} object".format(
                        type(self)
                    )
                )
            if isinstance(parent, Proxy):
                raise ValueError(
                    "Proxy object cannot be a parent of Multiplexer object."
                )
        super()._check_filiation_ok(child=child, parent=parent, **kwargs)

    def _give_update(self, parent, out=None):
        update = self.tensor_update[self.parent_slicing_dict[parent]]
        if out is None:
            return update
        out[...] = update

    def _initialization(self):
        if (
            len(
                set(
                    parent.get_index_id_for_children(self)
                    for parent in self.list_of_parents
                )
            )
            > 1
        ):
            raise ValueError(
                "All parents of a multiplexer object must have the same index_id"
            )

        multiplexer_idx_set = set(self.index_id) - set(
            self.list_of_parents[0].get_index_id_for_children(self)
        )
        # if concatenation is performed along a new axis
        if len(multiplexer_idx_set) == 1:
            multiplexer_idx = multiplexer_idx_set.pop()
            multiplexer_idx_number = self.index_id.index(multiplexer_idx)

            # tensor definition (concatenation of parents' tensors)
            self.tensor = glob.xp.concatenate(
                tuple(
                    parent.get_tensor_for_children(self)[
                        (slice(None),) * multiplexer_idx_number + (None,)
                    ]
                    for parent in self.list_of_parents
                ),
                axis=multiplexer_idx_number,
            )
            for num_parent, parent in enumerate(self.list_of_parents):
                self.parent_slicing_dict.update(
                    {parent: (slice(None),) * multiplexer_idx_number + (num_parent,)}
                )

        # if concatenation is performed along an existing axis
        elif not multiplexer_idx_set:
            if self.multiplexer_idx is None:
                raise ValueError("Defining multiplexer_idx is mandatory in that case")
            multiplexer_idx_number = self.index_id.index(self.multiplexer_idx)

            # tensor definition (concatenation of parents' tensors)
            self.tensor = glob.xp.concatenate(
                tuple(
                    parent.get_tensor_for_children(self)
                    for parent in self.list_of_parents
                ),
                axis=multiplexer_idx_number,
            )
            index_init = 0
            for num_parent, parent in enumerate(self.list_of_parents):
                index_end = (
                    index_init
                    + parent.get_tensor_for_children(self).shape[multiplexer_idx_number]
                )
                self.parent_slicing_dict.update(
                    {
                        parent: (slice(None),) * multiplexer_idx_number
                        + (slice(index_init, index_end),)
                    }
                )
                index_init = index_end
        else:
            raise ValueError(
                "index_id problem between multiplexer object and its parents"
            )

        if self.tensor_update is None and self.update_period != 0:
            self.tensor_update = glob.xp.ones_like(self.tensor)

        self._redefine_parent_tensor()

    def _redefine_parent_tensor(self):
        for parent in self.list_of_parents:
            parent.tensor = self.tensor[self.parent_slicing_dict[parent]].reshape(
                parent.tensor.shape
            )
            # must be recursive in case consecutive multiplexers
            if isinstance(parent, Multiplexer):
                parent._redefine_parent_tensor()

    @cached_property
    def multiplexer_idx_number(self):
        if self.multiplexer_idx:
            return self.index_id.index(self.multiplexer_idx)
        return 0

    def _check_model_validity(self):
        super()._check_model_validity()
        if not all(parent.tensor_has_energy for parent in self.list_of_parents) and any(
            parent.tensor_has_energy for parent in self.list_of_parents
        ):
            raise ValueError(
                "Model is wrong at {} level. Either all the parents' tensor or "
                "none of them should have energy.".format(self)
            )
        if self.multiplexer_idx is not None and not self.tensor_has_energy:
            raise ValueError(
                "Problem at {} level. When `multiplexer_idx` is provided, i.e. "
                "when a Multiplexer concatenates its parent's tensor along an "
                "existing axis, all parents' tensor should have energy."
            )

    @lru_cache(maxsize=1)
    def _get_raw_mean_tensor_for_VBEM(self, current_iter):
        raw_tensor = glob.xp.zeros_like(self.tensor)
        for parent in self.list_of_parents:
            parent_tensor = parent._get_mean_tensor_for_VBEM(self, current_iter)
            self_shape = raw_tensor[self.parent_slicing_dict[parent]].shape
            raw_tensor[self.parent_slicing_dict[parent]] = parent_tensor.reshape(
                self_shape
            )
        return raw_tensor


class Integrator(_Operator):
    """
    Class for Integrator operation.

    Its goal is to compute integration (i.e. cumulative sum) of its single
    parent's tensor along the last axis. Actually, since theory force operators
    to respect some kind of energy conservation, the real operation here is a
    weighted cumulative sum where weights are defined as [1/T, 1/(T-1), ... 1/2,
    1] and where T is the size of the last axis of parent's tensor. This
    weighted cumulative sum can also be performed backwards (see docstring of
    `__init__` method).
    """

    max_parents = 1

    def __init__(self, **kwargs):
        """
        Returns an Integrator object which computes a weighted cumulative sum
        along last axis of a parent's tensor.

        Parameters
        ----------
        backward_integration: bool, optional, default False
            If True, integration is performed backwards.
        """
        self.backward_integration = kwargs.pop("backward_integration", False)
        super().__init__(**kwargs)

    def _update_tensor(self, **kwargs):
        direction = -1 if self.backward_integration else 1
        parent_tensor = (
            self.first_parent.get_tensor_for_children(self) * self._normalization_coef
        )
        utils.cumsum_last_axis(
            parent_tensor[..., ::direction], self.tensor[..., ::direction]
        )

    @lru_cache(maxsize=1)
    def _get_raw_mean_tensor_for_VBEM(self, current_iter):
        parent_tensor = (
            self.first_parent._get_mean_tensor_for_VBEM(self, current_iter)
            * self._normalization_coef
        )
        direction = -1 if self.backward_integration else 1
        raw_tensor = glob.xp.zeros_like(self.tensor)
        utils.cumsum_last_axis(
            parent_tensor[..., ::direction], raw_tensor[..., ::direction]
        )
        return raw_tensor

    def _give_update(self, parent, out=None):
        direction = 1 if self.backward_integration else -1
        utils.cumsum_last_axis(
            self.tensor_update[..., ::direction], self.update_to_give[..., ::direction]
        )
        self.update_to_give *= self._normalization_coef
        if out is None:
            return self.update_to_give
        out[...] = self.update_to_give

    def _initialization(self):
        self.tensor = glob.xp.empty_like(
            self.first_parent.get_tensor_for_children(self)
        )
        self._update_tensor()
        if self.update_period != 0:
            self.full_tensor_update = glob.xp.zeros_like(self.tensor)
            self.tensor_update = self.full_tensor_update
            self.update_to_give = glob.xp.empty_like(self.tensor)
            self.integration_dim = self.tensor.shape[-1]

    @cached_property
    def _normalization_coef(self):
        parent_tensor = self.first_parent.get_tensor_for_children(self)
        if self.backward_integration:
            norm_coef = 1.0 / glob.xp.arange(
                1, parent_tensor.shape[-1] + 1, dtype=glob.float
            )
        else:
            norm_coef = 1.0 / glob.xp.arange(
                parent_tensor.shape[-1], 0, -1, dtype=glob.float
            )
        return norm_coef

    def _check_model_validity(self):
        super()._check_model_validity()
        if not self.tensor_has_energy and self.tensor.ndim - 1 not in self.norm_axis:
            raise ValueError(
                "Model is wrong at {} level. Parent's tensor must have energy "
                "or its last axis must be normalized."
            )


class Adder(_Operator):
    """
    Class for Adder operator.

    It aims at summing tensors of all parents into a single tensor. Allows
    self's tensor to be manipulated with reshapes and slices before actual
    summation (see `wonterfact.DynNodeData.new_parent` method's docstring).
    """

    def __init__(self, **kwargs):
        self.pre_slice_dict = {}
        self.shape_dict = {}
        self.post_slice_dict = {}
        super().__init__(**kwargs)

    def _update_tensor(self, **kwargs):
        self.tensor[...] = 0
        for parent in self.list_of_parents:
            tensor = self.apply_slice_and_shape(self.tensor, parent)
            if not glob.xp.may_share_memory(tensor, self.tensor):
                raise ValueError("something is wrong in the reslicing and reshaping")
            tensor += parent.get_tensor_for_children(self)

    @lru_cache(maxsize=1)
    def _get_raw_mean_tensor_for_VBEM(self, current_iter):
        raw_tensor = glob.xp.zeros_like(self.tensor)
        for parent in self.list_of_parents:
            tensor = self.apply_slice_and_shape(raw_tensor, parent)
            if not glob.xp.may_share_memory(tensor, raw_tensor):
                raise ValueError("something is wrong in the reslicing and reshaping")
            tensor += parent._get_mean_tensor_for_VBEM(self, current_iter)
        return raw_tensor

    def _give_update(self, parent, out=None):
        update = self.apply_slice_and_shape(self.tensor_update, parent)
        if out is None:
            return update
        out[...] = update

    def _initialization(self):
        if self.tensor is None:
            if self.parent_full_shape is None:
                raise ValueError(
                    "Please manually instantiate a tensor for this Adder (cannot infer the proper shape)"
                )
            self.tensor = glob.xp.zeros_like(
                self.parent_full_shape.get_tensor_for_children(self)
            )
        self._update_tensor()
        if self.update_period != 0:
            self.tensor_update = glob.xp.zeros_like(self.tensor)

    def _parse_kwargs_for_filiation(self, parent, **kwargs):
        pre_slice_for_adder = kwargs.pop("pre_slice_for_adder", Ellipsis)
        shape_for_adder = kwargs.pop("shape_for_adder", None)
        post_slice_for_adder = kwargs.pop("post_slice_for_adder", Ellipsis)
        try:
            pre_slice_for_adder = tuple(pre_slice_for_adder)
        except TypeError:
            pass
        try:
            post_slice_for_adder = tuple(post_slice_for_adder)
        except TypeError:
            pass
        self.pre_slice_dict[parent] = pre_slice_for_adder
        self.shape_dict[parent] = shape_for_adder
        self.post_slice_dict[parent] = post_slice_for_adder
        return kwargs

    def apply_slice_and_shape(self, tensor, parent):
        tensor1 = tensor[self.pre_slice_dict[parent]]
        if self.shape_dict[parent]:
            tensor1 = tensor1.reshape(self.shape_dict[parent])
        if self.post_slice_dict[parent]:
            tensor1 = tensor1[self.post_slice_dict[parent]]
        return tensor1

    @cached_property
    def parent_full_shape(self):
        parent_full_shape = next(
            (
                parent
                for parent in self.list_of_parents
                if self.pre_slice_dict[parent] == Ellipsis
                and self.post_slice_dict[parent] == Ellipsis
                and self.shape_dict[parent] is None
            ),
            None,
        )
        return parent_full_shape

    def _check_model_validity(self):
        super()._check_model_validity()
        if not all(parent.tensor_has_energy for parent in self.list_of_parents):
            raise ValueError(
                "Model is wrong at {} level. Parents' tensors of an Adder "
                "should all have energy."
            )


# class RealMultiplier(DynNodeData):
#     def __init__(self, **kwargs):
#         self.conv_idx_ids = kwargs.pop('conv_idx_ids', [])
#         self.sign_id = kwargs.pop('sign_id', None)
#         self.parent_sign_id_dict = {}
#         super(RealMultiplier, self).__init__(**kwargs)

#     def parent_sign_axis(self, parent):
#         return parent.index_id.index(self.parent_sign_id_dict[parent])

#     def _new_parent(self, parent, **kwargs):
#         parent_sign_id = kwargs.pop('parent_sign_id', None)
#         if parent_sign_id is None:
#             ValueError("Please provide 'parent_sign_id' during call of create_filiation")
#         self.parent_sign_id_dict[parent] = parent_sign_id

#     def get_parent_slice(self, parent, sign_num):
#         return [slice(None), ] * self.parent_sign_axis(parent) + [sign_num, ]

#     def _update_tensor(self, **kwargs):
#         pass
