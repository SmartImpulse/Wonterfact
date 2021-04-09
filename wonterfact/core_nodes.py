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

"""Module for all base classes used in wonterfact"""

# Python System imports
from functools import cached_property
from methodtools import lru_cache  # allows cache decorator per instance

# Relative imports
from . import utils
from .glob_var_manager import glob

# Third-party imports
import numpy as np
from custom_inherit import DocInheritMeta
from baseconv import base62


class _Node(
    metaclass=DocInheritMeta(style="numpy_with_merge", include_special_methods=True)
):
    """
    Base class of all nodes in a wonterfact tree,  i.e. the graphical
    representation of a tensor factorization model.
    """

    def __init__(self, name=None, **kwargs):
        """
        Parameters
        ----------
        name: str or None, optional, default None
            A name for the node. Very useful for debugging but not necessary.
            Each node's name in a tree should be unique. If None, a unique name
            is generated.
        """
        self.name = name
        if not self.name:
            base62.sign = "_"
            self.name = "n_" + base62.encode(id(self))
        self.already_been_counted = False

    def census(self, nodebook=None):  # Needs to be overridden in ChildNode class.
        """
        Returns a single element set with itself in it.

        Returns
        -------
        set
        """
        if nodebook is None:
            nodebook = set()
        nodebook.add(self)
        return nodebook

    # Needs to be overridden in DynNodeData
    def should_update(self, iteration_number=None):
        """
        Always returns True

        Returns
        -------
        bool
        """
        return True

    def _set_inference_mode(self, mode="EM"):
        if mode not in ("EM", "VBEM"):
            raise ValueError("Wrong inference mode value: must be 'EM of 'VBEM' ")
        self._inference_mode = mode

    def _check_filiation_ok(self, child=None, parent=None, **kwargs):
        """
        Raise an error if the node cannot accept to bound with either a new
        parent or a new child, given self class, input node's class and input
        kwargs.
        """
        pass

    def _check_model_validity(self):
        """
        Raise an error if one suspects that the model (i.e. the tree structure)
        is locally wrong
        """
        pass

    def __repr__(self):
        str_out = "{}(name='{}')".format(type(self).__name__, self.name)
        return str_out


class _ParentNode(_Node):
    """
    Base class for all parent nodes, i.e. all nodes that have children
    """

    max_children = None

    def __init__(self, **kwargs):
        self._list_of_children = []  # list of all child nodes
        super().__init__(**kwargs)

    @property
    def list_of_children(self):
        """
        Gives the list of all child nodes
        """
        return self._list_of_children

    @cached_property
    def has_a_single_child(self):
        return len(self.list_of_children) == 1

    @property
    def first_child(self):
        return self.list_of_children[0]

    def _update_child_list(self, child):
        self._list_of_children.append(child)

    def remove_child(self, child):
        """
        Deletes filiation between node and input child if exists, i.e. removes
        child from node's children and removes node from child's parents.
        """
        if child in self.list_of_children:
            self._list_of_children.remove(child)
        if self in child.list_of_parents:
            child._list_of_parents.remove(self)

        # TODO : clear all cached_property

    def _check_filiation_ok(self, child=None, parent=None, **kwargs):
        if child is not None and child in self.list_of_children:
            raise ValueError(
                """
                A parent and a child cannot be linked several times.
                Please make use of wonterfact.Proxy class if needed.
                """
            )
        if (
            child is not None
            and self.max_children is not None
            and len(self.list_of_children) == self.max_children
        ):
            raise ValueError(
                """
                {} nodes cannot have more than {} child(ren)
                """.format(
                    type(self), self.max_children
                )
            )
        super()._check_filiation_ok(child=child, parent=parent, **kwargs)

    def new_child(self, child, **kwargs):
        """
        Creates a new filiation between node and a new child.

        Parameters
        ----------
        child: wonterfact.Node object
            A child to connect with.
        """
        # first, one checks if filiation can be done
        self._check_filiation_ok(child=child, **kwargs)
        child._check_filiation_ok(parent=self, **kwargs)
        # then we give kwargs to child to it can parse useful arguments
        kwargs = child._parse_kwargs_for_filiation(self, **kwargs)
        # finally, create filiation
        self._create_filiation_to_child(child)

    def _create_filiation_to_child(self, child):
        """
        Creates a filiation between node and input child, i.e. adds child to
        node's children and adds node to child's parents.
        """
        try:
            self._list_of_children.append(child)
            child._list_of_parents.append(self)
        except Exception as exception:
            self.remove_child(child)
            raise type(exception)(str(exception))

    def new_children(self, *children):
        """
        Creates regular filiations between node and all children passed through.
        If you need to pass a specific option for the filiation creation, please
        make use of `new_child` method with each child you want to add.
        """
        for child in children:
            self.new_child(child)


class _ChildNode(_Node):
    """
    Base for all child nodes, i.e. all nodes that have parents
    """

    max_parents = None

    def __init__(self, **kwargs):
        self._list_of_parents = []  # list of all parent nodes
        super().__init__(**kwargs)

    @property
    def list_of_parents(self):
        """
        Gives the list of all parent nodes.

        Returns
        -------
        list
        """
        return self._list_of_parents

    @property
    def first_parent(self):
        return self.list_of_parents[0]

    def census(self, nodebook=None):
        """
        Returns the set of all ancestor nodes (parents, parents of parents,
        etc.). Usefull to make a census of all nodes in a tree.
        """
        nodebook = super().census(nodebook)
        for parent in self.list_of_parents:
            if parent not in nodebook:
                nodebook = parent.census(nodebook)

        return nodebook

    @cached_property
    def level(self):
        """
        Returns the node level in the tree (0 for a hyperparameter bud, 1 for a
        leaf, and 1 + distance from farest parent leaf otherwise).
        """
        return max(parent.level for parent in self.list_of_parents) + 1

    def new_parent(self, parent, **kwargs):
        """
        Creates a filiation between node and input parent.
        Calls parent's `new_child` method. Please see the corresponding
        docstring for full description of input parameters.
        """
        parent.new_child(self, **kwargs)

    def new_parents(self, *parents):
        """
        Creates regular filiations between node and all parents passed through.
        If you need to pass a specific option for the filiation creation, please
        make use of `new_parent` method with each parent you want to add.
        """
        for parent in parents:
            self.new_parent(parent)

    def _check_filiation_ok(self, child=None, parent=None, **kwargs):
        if parent is not None and parent in self.list_of_parents:
            raise ValueError(
                """
                A parent and a child cannot be linked several times.
                Please make use of wonterfact.Proxy class if needed.
                """
            )
        if (
            parent is not None
            and self.max_parents is not None
            and len(self.list_of_parents) == self.max_parents
        ):
            raise ValueError(
                """
                {} nodes cannot have more than {} parent(s)
                """.format(
                    type(self), self.max_parents
                )
            )
        super()._check_filiation_ok(child=child, parent=parent, **kwargs)

    def _parse_kwargs_for_filiation(self, parent, **kwargs):
        return kwargs


class _NodeData(_ParentNode):
    """
    Base class of all nodes that carry data. Inherits from ParentNode class
    since a NodeData necessarily have at least one child.
    """

    def __init__(self, tensor=None, index_id=None, **kwargs):
        """
        Parameters
        ----------
        tensor: array_like or None, optional, default None
            Values of the data. Must be a ``numpy.ndarray``, ndarray like
            or castable to any ``ndarray``. Can be a scalar (for 0-dim array)
            The given object will be copied during class __init__ method.
            It will be cast to cupy.ndarray if cupy backend is used.
        index_id: tuple of hashables or str or None, optional, default None
            Name(s) of the data dimension. Must be a tuple of hashable or str with
            length equal to the number of data dimension.
        """
        self.tensor = tensor
        self.index_id = index_id
        if self.tensor is not None:
            # one make a copy of self.tensor to be sure it is unique and not
            # shared with other tensors
            self.tensor = glob.xp.array(self.tensor, dtype=glob.float)
        super().__init__(**kwargs)

    def cast_array(self, tensor, force_numpy=False):
        """
        If force_numpy, changes input tensor to a numpy ndarray if not already.
        Otherwise returns input tensor as is.
        """
        if force_numpy and utils.infer_backend(tensor) == glob.CUPY:
            return glob.xp.asnumpy(tensor)
        return tensor

    def get_tensor(self, force_numpy=False):
        """
        Return inner tensor. If force_numpy is true, the tensor is casted to
        numpy ndarray if needed (if cupy backend is used)
        """
        return self.cast_array(self.tensor, force_numpy=force_numpy)

    def tensor_has_energy(self):
        """
        Tells if inner tensor has energy. If not, it means that inner tensor
        complies to some normalization constraint.
        """
        raise NotImplementedError

    def __repr__(self):
        str_out = super().__repr__()
        if isinstance(self.index_id, str):
            print_index_id = "'{}'".format(self.index_id)
        else:
            print_index_id = "{}".format(self.index_id)
        return str_out[:-1] + ", index_id={})".format(print_index_id)


class _DynNodeData0(_NodeData):
    """
    Base class for all nodes that carry dynamic data, i.e. data that can evolve
    during estimation algorithms.
    """

    def __init__(self, update_period=1, update_offset=0, update_succ=1, **kwargs):
        """
        Parameters
        ----------
        update_period: int, optional, default 1
            If 0, the node and its ancestors (if any) never update, otherwise see
            update_succ doc.
        update_succ: int, optional, default 1
            Every ``update_period`` iterations, the node and its parents update
            ``update_succ`` successive times and then freeze.
        update_offset: int, optional, default 0
            Number of iterations before first update for the node and its
            ancestors (if any).
        """
        self.update_offset = update_offset
        self.update_succ = update_succ
        self.update_period = update_period
        super().__init__(**kwargs)
        self.tensor_update = None  # Needs to be define in _initialization method
        self.slicing_for_children_dict = {}
        self.reshape_for_children_dict = {}
        self.index_id_for_children_dict = {}
        self.strides_for_children_dict = {}

    def get_tensor(self, force_numpy=False, raw_tensor=None):
        """
        Returns inner tensor of the node.

        Parameters
        ----------
        force_numpy : bool, optional, default False
            if True, the returned tensor is casted to numpy ndarray no matter
            the used backend.
        raw_tensor: array_like or None, default None
            If provided, returns raw_tensor instead of inner tensor.
        """
        tensor = self.tensor if raw_tensor is None else raw_tensor
        return self.cast_array(tensor, force_numpy=force_numpy)

    def get_tensor_for_children(self, child, force_numpy=False, raw_tensor=None):
        """
        Returns inner tensor or a modified view or modified copy of the inner
        tensor. The result might differ depending on who is asking: in some
        cases, the node might lie to its children (cf
        wonterfact.create_filiation docstring). Children of the node should
        always use this method instead of the ``get_tensor`` method.

        Parameters
        ----------
        child : a Node object
            The node who is asking for the tensor of its parent.
        force_numpy : bool, optional, default False
            If True, the returned tensor is casted to numpy ndarray no matter
            the used backend.
        raw_tensor: array_like or None, default None
            If provided, returns a modified view of raw_tensor instead of inner
            tensor.
        """
        if child not in self.list_of_children:
            return self.get_tensor(force_numpy=force_numpy, raw_tensor=raw_tensor)
        tensor = self.tensor if raw_tensor is None else raw_tensor
        if self.strides_for_children_dict[child]:
            if not all(
                tensor.strides[ii] >= tensor.strides[ii + 1]
                for ii in range(len(tensor.strides) - 1)
            ):
                raise ValueError(
                    "Strides of {}'s inner tensor should always be in descending"
                    "order".format(self)
                )
            return glob.as_strided(
                tensor,
                shape=self.reshape_for_children_dict[child],
                strides=self.strides_for_children_dict[child],
            )
        if tensor.ndim == 0:
            return self.get_tensor(force_numpy=force_numpy, raw_tensor=raw_tensor)
        tensor_to_give = tensor[self.slicing_for_children_dict[child]]
        if self.reshape_for_children_dict[child]:
            tensor_to_give = tensor_to_give.reshape(
                self.reshape_for_children_dict[child]
            )
        return self.cast_array(tensor_to_give, force_numpy=force_numpy)

    @lru_cache(maxsize=64)
    def no_tensor_transform_for_child(self, child):
        gives_raw_tensor = (
            self.slicing_for_children_dict[child] is Ellipsis
            and self.reshape_for_children_dict[child] is None
            and self.strides_for_children_dict[child] is None
        )

        return gives_raw_tensor

    @lru_cache(maxsize=16)
    def should_update(self, iteration_number=None):
        """
        Tells if the node should update or not given input iteration number (see
        ``update_period``, ``update_succ`` and ``update_offset``in nodes' init
        docstring). Regardless of its attributes, it cannot update if one of
        its child cannot either.

        Parameters
        ----------
        iteration_number : str or None Number of current iteration. If None,
            always return True
        """
        if iteration_number is None:
            return True
        should_update = self.update_period != 0 and (
            (iteration_number - self.update_offset) % self.update_period
            < self.update_succ
        )
        all_children_should_update = all(
            child.should_update(iteration_number)
            for child in self.list_of_children
            if hasattr(child, "should_update")
        )
        return should_update and all_children_should_update

    def new_child(self, child, **kwargs):
        """
        Parameters
        ----------
        slice_for_child: sequence, optional, default Ellipsis
            Slice to apply on parent's tensor when child calls
            `get_tensor_for_children`.
        shape_for_child: sequence of int or None, optional, default None
            Shape used to reshape the sliced parent's tensor when child calls
            `get_tensor_for_children`. If None, the sliced tensor is not
            reshape.
        index_id_for_child: sequence of hashable, default None
            Index IDs that parent gives to the child when child calls
            `get_tensor_for_children`. If None, parent gives its
            `parent.index_id`
        strides_for_child: sequence of int or None, default None
            If `slice_for_child` is not Ellipsis, has to be None.
            New strides to apply to parent's tensor (after a possible reshaping)
            when child calls `get_tensor_for_children`.
        pre_slice_for_adder: sequence, optional, default Ellipsis
            Can be set only if `child` is an wonterfact.Adder object. Slice to
            apply first to child's tensor before summation of the parent's
            tensor.
        shape_for_adder: sequence of int or None, optional, default None
            Can be set only if `child` is an wonterfact.Adder object. Shape used
            to reshape the child's tensor after the first slicing (see
            `pre_slice_for_adder`) before summation of the parent's tensor.
        post_slice_for_adder: sequence, optional, default Ellipsis
            Can be set only if `child` is an wonterfact.Adder object. Slice to
            apply last to child's tensor after a first slicing (see
            `pre_slice_for_adder`) and reshaping (see `shape_for_adder`) before
            summation of the parent's tensor. Cautious, let `tensor` be the
            child's inner tensor, the operation
            `tensor[pre_slice_for_adder].reshape(shape_for_adder)[post_slice_for_adder]`
            should return a view to (a piece of) `tensor` otherwise it will
            raise an error.
        """
        slice_for_child = kwargs.get("slice_for_child", Ellipsis)
        shape_for_child = kwargs.get("shape_for_child", None)
        index_id_for_child = kwargs.get("index_id_for_child", None)
        strides_for_child = kwargs.get("strides_for_child", None)

        if slice_for_child != Ellipsis and strides_for_child != None:
            raise ValueError(
                "One cannot have both `slice_for_child` and `strides_for_child` different from their default value."
            )
        super().new_child(child, **kwargs)
        index_id_for_child = index_id_for_child or self.index_id
        try:
            slice_for_child = tuple(slice_for_child)
        except TypeError:
            pass
        self.slicing_for_children_dict[child] = slice_for_child
        self.reshape_for_children_dict[child] = shape_for_child
        self.index_id_for_children_dict[child] = index_id_for_child
        self.strides_for_children_dict[child] = strides_for_child

    def idx_to_shape_for_child(self, idx, child):
        """
        Gives to a given child the size corresponding to the dimension name idx.
        The result might differ depending on the child.
        """
        shape = self.get_tensor_for_children(child).shape[
            self.get_index_id_for_children(child).index(idx)
        ]
        return shape

    def _compute_tensor_update_aux(
        self,
        child,
        tensor_to_fill,
        value_to_force=None,
        cumsum=True,
        method_to_call="_give_update",
    ):
        tensor_to_fill = (
            self.tensor_update if tensor_to_fill is None else tensor_to_fill
        )
        child_slicing = self.slicing_for_children_dict[child]
        if value_to_force is None:
            fill_tensor = child.__getattribute__(method_to_call)(self)
        else:
            fill_tensor = value_to_force
        if self.strides_for_children_dict[child]:
            strided_tensor_update = glob.as_strided(
                tensor_to_fill,
                shape=self.reshape_for_children_dict[child],
                strides=self.strides_for_children_dict[child],
            )
            if cumsum:
                strided_tensor_update[...] += fill_tensor
            else:
                strided_tensor_update[...] = fill_tensor
        else:
            if value_to_force is None:
                fill_tensor = fill_tensor.reshape(tensor_to_fill[child_slicing].shape)
            if cumsum:
                tensor_to_fill[child_slicing] += fill_tensor
            else:
                tensor_to_fill[child_slicing] = fill_tensor

    def _compute_tensor_update_aux2(self, tensor_to_fill, method_to_call):
        if self.has_a_single_child and self.no_tensor_transform_for_child(
            self.first_child
        ):
            self.first_child.__getattribute__(method_to_call)(self, out=tensor_to_fill)
        else:
            if self.are_all_tensor_coefs_linked_to_at_least_one_child:
                tensor_to_fill[...] = 0.0
                for child in self.list_of_children:
                    self._compute_tensor_update_aux(
                        child,
                        tensor_to_fill=tensor_to_fill,
                        cumsum=True,
                        method_to_call=method_to_call,
                    )
            else:
                # only possible for nodes that have energy and therefore all tensors
                # coefs are seen by at most one child
                tensor_to_fill[...] = 1.0
                for child in self.list_of_children:
                    self._compute_tensor_update_aux(
                        child,
                        tensor_to_fill=tensor_to_fill,
                        cumsum=False,
                        method_to_call=method_to_call,
                    )

    def compute_tensor_update(self):
        """
        Computes the multiplicative update if self is an Operator or a parameter
        Leave. Computes the sum of expected sufficient statistics of its
        children if self is a hyperparameter bud. In eather case, the resulting
        update is the sum of all updates given by the children.
        """

        self._compute_tensor_update_aux2(
            tensor_to_fill=self.tensor_update, method_to_call="_give_update"
        )

    def get_index_id_for_children(self, child):
        """
        Returns index_id. Answer might differ depending on who is asking (cf Demultiplexer object)
        """
        return self.index_id_for_children_dict[child]

    @cached_property
    def are_all_tensor_coefs_linked_to_at_least_one_child(self):
        test_tensor = glob.xp.zeros_like(self.tensor)
        for child in self.list_of_children:
            self._compute_tensor_update_aux(
                child, tensor_to_fill=test_tensor, value_to_force=1.0, cumsum=False
            )
        return test_tensor.all()

    @cached_property
    def are_all_tensor_coefs_linked_to_at_most_one_child(self):
        test_tensor = glob.xp.zeros_like(self.tensor)
        for child in self.list_of_children:
            self._compute_tensor_update_aux(
                child, tensor_to_fill=test_tensor, value_to_force=1.0, cumsum=True
            )
        return (test_tensor < 2.0).all()

    def _backup_tensor(self):  # TODO eventually don't backup proxys and multiplexers
        self._tensor_backup = self.tensor.copy()

    def _restore_tensor(self):
        self.tensor[...] = self._tensor_backup
        del self._tensor_backup


class _DynNodeData(_DynNodeData0):
    """
    Base class for all nodes that carry values belonging in the domain of
    paramaters (basically paramater leaves and operators)
    """

    def get_norm_axis_for_children(self, child):
        if self.tensor_has_energy or self.tensor.ndim == 0 or self.norm_axis == None:
            return None
        if self.strides_for_children_dict[child] is not None:
            raise NotImplementedError
        # first we deal with the slicing for child
        norm_axis_list = list(self.norm_axis)
        explicit_slice = utils.explicit_slice(
            self.slicing_for_children_dict[child], self.tensor.ndim
        )
        num_axis = 0
        for sl in explicit_slice:
            if isinstance(sl, int):
                norm_axis_list = [
                    val - 1 if val > num_axis else val for val in norm_axis_list
                ]
            else:
                num_axis += 1
        # then the reshape for child
        shape_sliced_tensor = self.tensor[self.slicing_for_children_dict[child]].shape
        shape_for_child = self.get_tensor_for_children(child).shape
        if shape_sliced_tensor == shape_for_child:
            return tuple(norm_axis_list)
        if 1 in shape_sliced_tensor:
            raise ValueError(
                "{} cannot yet automatically infer `norm_axis_for_child` for its "
                "child {} due to the presence of 1-size dimension of the sliced tensor. "
                "You can either explicitly specify `norm_axis_for_child` during the filiation "
                "creation, redefine `slice_for_child` during the filiation creation"
                "so that 1-size dimensions are squeezed or run `Root.estimate_param` "
                "with `check_model_validity=False` to bypass the validity check of the model."
            )
        cumprod_shape_1 = np.cumprod(shape_sliced_tensor)
        if 1 in shape_for_child:
            raise ValueError(
                "{} cannot automatically infer `norm_axis_for_child` for its "
                "child {} due to the presence of 1-size dimensions in `shape_for_child`"
                "Please explicitly specify `norm_axis_for_child` during the filiation "
                "creation, or run `Root.estimate_param` with `check_model_validity=False`"
                "to bypass the validity check of the model."
            )
        cumprod_shape_2 = np.cumprod(shape_for_child)
        cumprod_intersect = sorted(
            list(set(cumprod_shape_1).intersection(cumprod_shape_2))
        )
        list_of_clust1, list_of_clust2 = [], []
        min_dim = 0
        for max_dim in cumprod_intersect:
            list_of_clust1.append(
                [
                    dim
                    for (dim, val) in enumerate(cumprod_shape_1)
                    if (val > min_dim and val <= max_dim)
                ]
            )
            list_of_clust2.append(
                [
                    dim
                    for (dim, val) in enumerate(cumprod_shape_2)
                    if (val > min_dim and val <= max_dim)
                ]
            )
            min_dim = max_dim
        norm_axis = []
        for clust1, clust2 in zip(list_of_clust1, list_of_clust2):
            if all(elem in norm_axis_list for elem in clust1):
                norm_axis += clust2
            elif any(elem in norm_axis_list for elem in clust1):
                raise ValueError(
                    "Invalid shape_for_child between {} and {}. "
                    "Normalized axis should not be spitted during the reshaping"
                    " process".format(self, child)
                )
        return tuple(norm_axis)

    @cached_property
    def norm_axis(self):
        """
        If inner tensor has no energy, specifies axis subject to a normalization
        constraint. In such case, `self.tensor.sum(axis=self.norm_axis))` should
        be filled with `1.0`.
        """
        raise NotImplementedError

    def _check_model_validity(self):
        super()._check_model_validity()

        # If tensor has energy, each of its value can be seen at most by one
        # child otherwise it would mean that energy is duplicated. On the other
        # hand, if a value is seen by no child, it is not a problem because
        # tensor_update can have default value = 1 as if we had masked
        # observations
        if self.tensor_has_energy:
            if not self.are_all_tensor_coefs_linked_to_at_most_one_child:
                raise ValueError(
                    "Model invalid at {} level. Each coefficient of inner tensor "
                    "should be seen by at most one child, unless inner tensor has "
                    "no energy".format(self)
                )
            return

        #  Otherwise, all values of inner tensor should be seen by at least one child.
        if not self.are_all_tensor_coefs_linked_to_at_least_one_child:
            raise ValueError(
                "Model incomplete at {} level. Each coefficient of inner tensor "
                "should be seen by at least one child, unless inner tensor has "
                "energy. Please find an equivalent model by putting, for "
                "instance, a LeafGamma or any other leaf that has energy "
                "upstream".format(self)
            )

        # Also, slice_for_child should not slice incomplete pieces of normalized
        # distributions (actually it could, but it is dangerous and we should
        # find a workaround to avoid that.)
        for child in self.list_of_children:
            explicit_slice = utils.explicit_slice(
                self.slicing_for_children_dict[child], self.tensor.ndim
            )
            for axis in self.norm_axis:
                dim_axis = self.tensor.shape[axis]
                test_arr = np.arange(dim_axis)
                if (
                    test_arr[explicit_slice[axis]].size != test_arr.size
                    or (test_arr[explicit_slice[axis]] != test_arr).any()
                ):
                    raise ValueError(
                        "Model might be wrong at {} level. If inner tensor is "
                        "subject to normalization constraint (i.e. it has no energy)"
                        ", its children should not see an incomplete piece of "
                        "normalized distribution. Please find an equivalent model by"
                        " putting, for instance, a LeafGamma or any other leaf that "
                        "has energy upstream".format(self)
                    )

    def _total_energy_leak(self):
        if not self.tensor_has_energy:
            return 0
        tensor_copy = self.tensor.copy()
        for child in self.list_of_children:
            self._compute_tensor_update_aux(
                child, tensor_to_fill=tensor_copy, value_to_force=0.0, cumsum=False
            )
        return tensor_copy.sum().item()
