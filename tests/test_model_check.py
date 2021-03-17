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

"""Tests for the model validation feature of wonterfact"""

# Python standard library

# Third-party imports
import numpy as np
import numpy.random as npr
import pytest

# wonterfact imports
import wonterfact as wtf
import wonterfact.core_nodes as wtfc
import wonterfact.operators as wtfo


def test_get_norm_for_children():
    parent = wtfc._DynNodeData()
    parent.tensor_has_energy = False
    child = wtfo._Operator()
    sl = slice(None)
    # list of (parent_shape, parent_norm_axis, slice_for_child, shape_for_child, norm_axis_for_child)
    input_output_list = [
        ((2,) * 7, (1, 3, 4), (0, sl, 1, sl, sl, 1, sl), (2, 2, 2, 2), (0, 1, 2)),
        ((2,) * 7, (1, 3, 4), (0, sl, 1, sl, sl, 1, slice(1)), (2, 2, 2, 1), (0, 1, 2)),
        ((2,) * 7, (1, 3, 4), (0, sl, 1, sl, sl, 1, 0), (2, 2, 2), (0, 1, 2)),
        ((2,) * 7, (1, 3, 4), (0, sl, 1, sl, sl, 1, sl), (2, 4, 2), (0, 1)),
        ((2,) * 7, (1, 3, 4), (0, sl, 1, sl, sl, sl, sl), (2, 4, 4), (0, 1)),
        ((2,) * 7, (4,), (0, sl, 1, sl, sl, 1, sl), (4, 2, 2), (1,)),
        ((2, 3, 4, 2), (1, 2), (0, Ellipsis), (2, 6, 2), (0, 1)),
        ((2, 3, 4, 1), (1, 2), (0, Ellipsis, 0), (2, 6), (0, 1)),
        ((2,), (), Ellipsis, None, ()),
        ((2, 2, 2), (1,), (slice(1),), (1, 2, 2), (1,)),
    ]

    def setup_nodes(parent_shape, parent_norm_axis, slice_for_child, shape_for_child):
        parent.remove_child(child)
        parent.tensor = npr.rand(*parent_shape)
        parent.norm_axis = parent_norm_axis
        parent.new_child(
            child, slice_for_child=slice_for_child, shape_for_child=shape_for_child
        )

    for elem in input_output_list:
        setup_nodes(*elem[:-1])
        assert parent.get_norm_axis_for_children(child) == elem[-1]

    # expected errors
    # list of (parent_shape, parent_norm_axis, slice_for_child, shape_for_child, expected error)
    input_output_list = [
        ((2, 2, 2), (2,), Ellipsis, (2, 4), "Invalid shape_for_child .*"),
        ((2, 2, 2), (1,), Ellipsis, (4, 2), "Invalid shape_for_child .*"),
        (
            (2, 2, 2),
            (1,),
            (slice(1),),
            (2, 2),
            ".* 1-size dimension of the sliced tensor.*",
        ),
        (
            (2, 2, 2),
            (2,),
            Ellipsis,
            (1, 4, 2),
            ".* 1-size dimensions in `shape_for_child`.*",
        ),
    ]
    for elem in input_output_list:
        setup_nodes(*elem[:-1])
        with pytest.raises(ValueError, match=elem[-1]):
            parent.get_norm_axis_for_children(child)


def test_check_model_validity_dynnodedata():
    parent = wtfc._DynNodeData(index_id="d", tensor=npr.rand(2))
    parent.tensor_has_energy = False
    child1 = wtfo._Operator(index_id="d", tensor=npr.rand(1))
    parent.new_child(child1, slice_for_child=slice(0, 1))
    with pytest.raises(
        ValueError,
        match=".*Each coefficient of inner tensor should be seen by at least",
    ):
        parent._check_model_validity()

    parent = wtfc._DynNodeData(index_id="td", tensor=npr.rand(2, 2))
    parent.tensor_has_energy = False
    parent.norm_axis = (1,)
    child1 = wtfo._Operator(index_id="td", tensor=npr.rand(2, 1))
    parent.new_child(child1, slice_for_child=(..., slice(0, 1)))
    child2 = wtfo._Operator(index_id="td", tensor=npr.rand(2, 1))
    parent.new_child(child2, slice_for_child=(..., slice(1, 2)))
    with pytest.raises(
        ValueError, match=".*its children should not see an incomplete piece of"
    ):
        parent._check_model_validity()

    parent = wtfc._DynNodeData(index_id="td", tensor=npr.rand(2, 2))
    parent.tensor_has_energy = False
    parent.norm_axis = (1,)
    child1 = wtfo._Operator(index_id="t", tensor=npr.rand(1))
    parent.new_child(child1, slice_for_child=(..., 0))
    child2 = wtfo._Operator(index_id="td", tensor=npr.rand(1))
    parent.new_child(child2, slice_for_child=(..., [1,]))
    with pytest.raises(
        ValueError, match=".*its children should not see an incomplete piece of"
    ):
        parent._check_model_validity()

    parent = wtfc._DynNodeData(index_id="d", tensor=npr.rand(3))
    parent.tensor_has_energy = True
    child1 = wtfo._Operator(index_id="d", tensor=npr.rand(2))
    parent.new_child(child1, slice_for_child=slice(0, 2))
    child2 = wtfo._Operator(index_id="d", tensor=npr.rand(1))
    parent.new_child(child2, slice_for_child=slice(0, 1))
    with pytest.raises(
        ValueError,
        match=".*Each coefficient of inner tensor should be seen by at most",
    ):
        parent._check_model_validity()


def test_norm_axis_and_check_model_validity_multiplier():
    tensor1 = npr.rand(2, 2)
    tensor2 = npr.rand(2, 2)

    # list of (has_energy1, has_energy2, index_id1, index_id2, index_id_child, norm_axis1, norm_axis2, norm_axis_child)
    input_output_list = [
        (True, False, "td", "df", "tf", None, (1,), None),
        (False, False, "td", "df", "tf", (0, 1), (1,), (0, 1)),
        (False, False, "td", "df", "tf", (1,), (1,), (1,)),
        (False, False, "td", "df", "tf", (0,), (1, 1), (0, 1)),
    ]

    def setup(
        has_energy1,
        has_energy2,
        index_id1,
        index_id2,
        index_id_child,
        norm_axis1,
        norm_axis2,
    ):
        parent1 = wtfc._DynNodeData(tensor=tensor1, index_id=index_id1)
        parent2 = wtfc._DynNodeData(tensor=tensor2, index_id=index_id2)
        child = wtfo.Multiplier(
            index_id=index_id_child, tensor=npr.rand(*(2,) * len(index_id_child))
        )
        obs = wtf.PosObserver(tensor=child.tensor)
        parent1.tensor_has_energy = has_energy1
        parent2.tensor_has_energy = has_energy2
        parent1.norm_axis = norm_axis1
        parent2.norm_axis = norm_axis2
        child.new_parents(parent1, parent2)
        child.new_child(obs)
        return child

    for elem in input_output_list:
        child = setup(*elem[:-1])
        assert child.norm_axis == elem[-1]

    # list of (has_energy1, has_energy2, index_id1, index_id2, index_id_child, norm_axis1, norm_axis2, expected_error)
    input_output_list = [
        (True, True, "td", "df", "tf", None, None, ".*At most one parent can have.*"),
        (
            False,
            False,
            "td",
            "df",
            "tf",
            (1,),
            (0,),
            ".*An index_id cannot be normalized twice.*",
        ),
        (
            True,
            False,
            "t",
            "df",
            "tdf",
            None,
            (0,),
            ".*before multiplication with a tensor that has energy.*",
        ),
        (
            False,
            False,
            "td",
            "df",
            "tf",
            (0,),
            (1,),
            ".*should be normalized before marginalization.*",
        ),
    ]
    for elem in input_output_list:
        child = setup(*elem[:-1])
        with pytest.raises(ValueError, match=elem[-1]):
            child._check_model_validity()

    # The followings should work
    child = setup(True, False, "td", "df", "tf", None, (1,))
    assert child._check_model_validity() is None
    child = setup(False, False, "td", "df", "tf", (0, 1), (1,))
    assert child._check_model_validity() is None


def test_check_model_validity_convolver():
    tensor1 = npr.rand(4, 4, 2)
    tensor2 = npr.rand(2, 2, 2)
    tensor_child = npr.rand(3, 3, 2)

    def setup(
        has_energy1,
        has_energy2,
        index_id1,
        index_id2,
        index_id_child,
        norm_axis1,
        norm_axis2,
        conv_idx_ids,
    ):
        parent1 = wtfc._DynNodeData(tensor=tensor1, index_id=index_id1)
        parent1.tensor_has_energy = has_energy1
        parent1.norm_axis = norm_axis1
        parent2 = wtfc._DynNodeData(tensor=tensor2, index_id=index_id2)
        parent2.tensor_has_energy = has_energy2
        parent2.norm_axis = norm_axis2
        child = wtfo.Multiplier(
            index_id=index_id_child, tensor=tensor_child, conv_idx_ids=conv_idx_ids
        )
        child.new_parents(parent1, parent2)
        obs = wtf.PosObserver(tensor=child.tensor)
        obs.new_parent(child)
        return child

    # list of (has_energy1, has_energy2, index_id1, index_id2, index_id_child,...
    # ... norm_axis1, norm_axis2, conv_idx_ids, expected_error)
    input_output_list = [
        (
            False,
            False,
            "ftd",
            "ftd",
            "ftd",
            (0, 1, 2),
            (0, 1),
            ("f", "t"),
            ".*inner tensor of a Multiplier must have energy.*",
        ),
        (
            True,
            False,
            "fid",
            "ftd",
            "ftd",
            None,
            (0, 1),
            ("f", "t"),
            ".*This index should belong to two parents.*",
        ),
        (
            True,
            False,
            "ftd",
            "ftd",
            "ftd",
            None,
            (0,),
            ("f", "t"),
            ".*Axis to be convolved should be normalized.*",
        ),
    ]
    for elem in input_output_list:
        child = setup(*elem[:-1])
        with pytest.raises(ValueError, match=elem[-1]):
            child._check_model_validity()

    # should not raise any error
    child = setup(True, False, "ftd", "ftd", "ftd", None, (0, 1), ("f", "t"))
    assert child._check_model_validity() is None


def test_check_model_validity_multiplexer():
    def setup(has_energy1, has_energy2, norm_axis1, norm_axis2, concatenate):
        parent1 = wtfc._DynNodeData(index_id="f", tensor=npr.rand(2))
        parent1.tensor_has_energy = has_energy1
        parent1.norm_axis = norm_axis1
        parent2 = wtfc._DynNodeData(index_id="f", tensor=npr.rand(2))
        parent2.tensor_has_energy = has_energy2
        parent2.norm_axis = norm_axis2
        index_id_child = "f" if concatenate else "fd"
        multiplexer_idx = "f" if concatenate else None
        tensor_child = npr.rand(4) if concatenate else npr.rand(2, 2)
        multiplexer = wtf.Multiplexer(
            index_id=index_id_child,
            tensor=tensor_child,
            multiplexer_idx=multiplexer_idx,
        )
        multiplexer.new_parents(parent1, parent2)
        obs = wtf.PosObserver(index_id=index_id_child, tensor=tensor_child)
        obs.new_parent(multiplexer)
        return multiplexer

    with pytest.raises(ValueError, match=".*Either all the parents' tensor.*"):
        child = setup(False, True, (0,), None, True)
        child._check_model_validity()
    with pytest.raises(ValueError, match=".*When `multiplexer_idx` is provided.*"):
        child = setup(False, False, (0,), (0,), True)
        child._check_model_validity()

    child = setup(False, False, (0,), (0,), False)
    assert child._check_model_validity() is None
    child = setup(True, True, (0,), (0,), False)
    assert child._check_model_validity() is None
    child = setup(True, True, None, None, True)
    assert child._check_model_validity() is None


def test_check_model_validity_integrator():
    parent = wtfc._DynNodeData(index_id="dt", tensor=npr.rand(2, 4))
    parent.tensor_has_energy = False
    parent.norm_axis = (0,)
    child = wtf.Integrator(index_id="dt", tensor=npr.rand(2, 4))
    obs = wtf.PosObserver(index_id="dt", tensor=npr.rand(2, 4))
    parent.new_child(child)
    child.new_child(obs)
    with pytest.raises(ValueError, match=".*or its last axis must be normalized.*"):
        child._check_model_validity()


def test_check_model_validity_adder():
    parent1 = wtfc._DynNodeData(index_id="ft", tensor=npr.rand(2, 2))
    parent1.tensor_has_energy = False
    parent1.norm_axis = (0, 1)
    parent2 = wtfc._DynNodeData(index_id="ft", tensor=npr.rand(2, 2))
    parent2.tensor_has_energy = True
    child = wtf.Adder(index_id="ft", tensor=npr.rand(2, 2))
    child.new_parents(parent1, parent2)
    obs = wtf.PosObserver(index_id="ft", tensor=npr.rand(2, 2))
    child.new_child(obs)
    with pytest.raises(ValueError, match=".*should all have energy.*"):
        child._check_model_validity()

