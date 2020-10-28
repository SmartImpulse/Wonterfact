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

"""Tests for basic methods of nodes"""

# Python standard library
from pathlib import Path
import tempfile

# Third-party imports
import numpy as np
import numpy.random as npr
import pytest

# wonterfact imports
import wonterfact as wtf

# relative imports
from . import utils as t_utils

# let us create a wonterfact tree once for all tests
def make_test_tree():
    """
    A toy wonterfact tree is designed in order to test the maximum
    possibilities.
    """

    dim_k, dim_s, dim_t, dim_f, dim_c = 3, 2, 10, 4, 2

    leaf_energy = wtf.LeafGamma(
        name="leaf_energy",
        index_id="",
        tensor=np.array(1),
        prior_shape=1.1,
        prior_rate=0.01,
    )
    leaf_k = wtf.LeafDirichlet(
        name="leaf_k",
        index_id="k",
        norm_axis=(0,),
        tensor=np.ones(dim_k),
        prior_shape=1 + 0.1 * npr.rand(dim_k),
    )
    mul_k = wtf.Multiplier(name="mul_k", index_id=("k",))
    mul_k.new_parents(leaf_energy, leaf_k)

    leaf_kts_0 = wtf.LeafDirichlet(
        name="leaf_kts_0",
        index_id="kts",
        norm_axis=(1, 2),
        tensor=np.ones((dim_k, dim_t, dim_s)),
        prior_shape=1 + 0.00001 * npr.rand(dim_k, dim_t, dim_s),
    )
    mul_kts = wtf.Multiplier(name="mul_kts", index_id="kts")
    mul_kts.new_parents(leaf_kts_0, mul_k)

    leaf_kts_1 = wtf.LeafGamma(
        name="leaf_kts_1",
        index_id="kts",
        tensor=np.ones((dim_k, dim_t, dim_s)),
        prior_shape=1 + 0.001 * npr.rand(dim_k, dim_t, dim_s),
        prior_rate=0.001,
    )
    mult_ktsc = wtf.Multiplexer(name="mult_ktsc", index_id="ktsc")
    mult_ktsc.new_parents(leaf_kts_1, mul_kts)

    # no update for this leaf
    # it acts like an Adder (no normalization)
    leaf_c = wtf.LeafDirichlet(
        name="leaf_c",
        norm_axis=(),
        index_id="c",
        tensor=np.ones(dim_c),
        update_period=0,
    )
    mul_kts_1 = wtf.Multiplier(name="mul_kts_1", index_id="kts")
    mul_kts_1.new_parents(mult_ktsc, leaf_c)

    # two updates every 3 iterations
    leaf_kf = wtf.LeafDirichlet(
        name="leaf_kf",
        index_id="kf",
        norm_axis=(1,),
        tensor=np.ones((dim_k, dim_f)),
        prior_shape=1 + 0.01 * npr.rand(dim_k, dim_f),
        update_period=3,
        update_succ=2,
        update_offset=0,
    )
    mul_tfs = wtf.Multiplier(name="mul_tfs", index_id="tfs")
    mul_tfs.new_parents(mul_kts_1, leaf_kf)

    obs_tf = wtf.RealObserver(
        name="obs_tf", index_id="tf", tensor=100 * npr.randn(dim_t, dim_f)
    )
    mul_tfs.new_child(obs_tf)

    root = wtf.Root(
        name="root",
        cost_computation_iter=1,
        stop_estim_threshold=0,
        update_type="regular",
        inference_mode="EM",
        verbose_iter=10,
    )
    obs_tf.new_child(root)

    leaf_k_1 = wtf.LeafGamma(
        name="leaf_k_1",
        index_id="k",
        tensor=np.ones(dim_k),
        prior_shape=1 + 0.001 * npr.rand(dim_k),
        prior_rate=0.001,
    )
    # one update every 3 iterations
    leaf_kt = wtf.LeafDirichlet(
        name="leaf_kt",
        index_id="kt",
        norm_axis=(1,),
        tensor=np.ones((dim_k, dim_t)),
        prior_shape=1,
        update_period=3,
        update_succ=1,
        update_offset=2,
    )
    mul_kt = wtf.Multiplier(name="mul_kt", index_id="kt")
    mul_kt.new_parents(leaf_k_1, leaf_kt)
    mul_tf = wtf.Multiplier(name="mul_tf", index_id="tf")
    mul_tf.new_parents(leaf_kf, mul_kt)

    obs_tf_2 = wtf.PosObserver(
        name="obs_tf_2", index_id="tf", tensor=100 * npr.rand(dim_t, dim_f)
    )
    mul_tf.new_child(obs_tf_2)
    obs_tf_2.new_child(root)

    root.dim_k, root.dim_s, root.dim_t, root.dim_f, root.dim_c = (
        dim_k,
        dim_s,
        dim_t,
        dim_f,
        dim_c,
    )

    return root


@pytest.fixture(
    scope="module", params=["cpu", pytest.param("gpu", marks=pytest.mark.gpu)]
)
def tree(request):
    backend = request.param
    wtf.glob.set_backend_processor(backend, force=True)
    return make_test_tree()


def test_filiation(tree):
    with pytest.raises(ValueError, match=r".* cannot be linked several times.*"):
        tree.leaf_kf.new_child(tree.mul_tf)
    with pytest.raises(ValueError, match=r".* nodes cannot have more than .*"):
        tree.leaf_energy.new_child(tree.mul_tf)
    assert set(tree.leaf_kf.list_of_children) == set([tree.mul_tfs, tree.mul_tf])
    assert set(tree.mul_tf.list_of_parents) == set([tree.leaf_kf, tree.mul_kt])
    assert tree.leaf_energy.first_child == tree.mul_k
    assert tree.mul_k.first_parent == tree.leaf_energy
    assert tree.leaf_k_1.has_a_single_child
    assert not tree.leaf_kf.has_a_single_child


def test_level(tree):
    assert tree.leaf_energy.level == 0
    assert tree.leaf_energy.level == 0
    assert tree.mult_ktsc.level == 3
    assert tree.obs_tf_2.level == 3
    assert tree.obs_tf.level == 6
    assert tree.root.level == 7


def test_census(tree):
    all_nodes = set(
        [
            tree.leaf_energy,
            tree.leaf_k,
            tree.leaf_k_1,
            tree.leaf_kf,
            tree.leaf_kt,
            tree.leaf_kts_0,
            tree.leaf_kts_1,
            tree.leaf_c,
            tree.mul_k,
            tree.mul_kts,
            tree.mul_kts_1,
            tree.mul_tf,
            tree.mul_tfs,
            tree.mul_kt,
            tree.mult_ktsc,
            tree.obs_tf,
            tree.obs_tf_2,
            tree.root,
        ]
    )
    assert all_nodes == tree.root.census()


def test_get_tensor(tree):
    tree.leaf_c._set_inference_mode(mode="EM")
    tree.leaf_c._initialization()
    tensor1 = tree.leaf_c.get_tensor(force_numpy=True)
    assert np.allclose(tensor1, 1.0)
    tensor2 = tree.leaf_c.get_tensor_for_children(tree.mul_kts_1, force_numpy=True)
    assert np.allclose(tensor2, 1.0)


def test_message_passing(tree):
    def set_foo(iteration_number=None, mode="top-down"):
        tree.root.tree_traversal(
            "__setattr__",
            mode=mode,
            method_input=(("foo", iteration_number), {}),
            iteration_number=iteration_number,
        )

    all_nodes = tree.root.census()
    all_nodes_that_always_update = all_nodes.difference(
        set([tree.leaf_c, tree.leaf_kf, tree.leaf_kt])
    )
    set_foo(None)
    assert all(node.foo == None for node in all_nodes)
    # pylint: disable=no-member
    set_foo(0)
    assert all(node.foo == 0 for node in all_nodes_that_always_update)
    assert tree.leaf_c.foo is None
    assert tree.leaf_kf.foo == 0
    assert tree.leaf_kt.foo is None
    set_foo(1)
    assert all(node.foo == 1 for node in all_nodes_that_always_update)
    assert tree.leaf_c.foo is None
    assert tree.leaf_kf.foo == 1
    assert tree.leaf_kt.foo is None

    set_foo(2)
    assert all(node.foo == 2 for node in all_nodes_that_always_update)
    assert tree.leaf_c.foo is None
    assert tree.leaf_kf.foo == 1
    assert tree.leaf_kt.foo == 2

    set_foo(3)
    assert all(node.foo == 3 for node in all_nodes_that_always_update)
    assert tree.leaf_c.foo is None
    assert tree.leaf_kf.foo == 3
    assert tree.leaf_kt.foo == 2
    # pylint: enable=no-member


def test_initialization(tree):
    tree.root.tree_traversal(
        "_set_inference_mode",
        mode="top-down",
        method_input=((), dict(mode=tree.root.inference_mode)),
    )
    tree.root.tree_traversal(
        "_initialization", mode="top-down",
    )
    norm_tensor_k = tree.leaf_kf.get_tensor(force_numpy=True).sum(1)
    assert np.allclose(norm_tensor_k, 1.0)
    norm_tensor_k = tree.leaf_kts_0.get_tensor(force_numpy=True).sum((1, 2))
    assert np.allclose(norm_tensor_k, 1.0)
    assert np.isclose(tree.leaf_k.get_tensor(force_numpy=True).sum(), 1.0)

    assert tree.mul_tfs.get_tensor().shape == (tree.dim_t, tree.dim_f, tree.dim_s)
    assert tree.mul_tf.get_tensor().shape == (tree.dim_t, tree.dim_f)
    assert tree.mul_kt.get_tensor().shape == (tree.dim_k, tree.dim_t)
    assert tree.mult_ktsc.get_tensor().shape == (
        tree.dim_k,
        tree.dim_t,
        tree.dim_s,
        tree.dim_c,
    )

    all_nodes = tree.root.census()
    assert all(
        node.tensor_update.shape == node.tensor.shape
        for node in all_nodes
        if getattr(node, "tensor_update", None) is not None
    )
    assert tree.leaf_c.tensor_update == None


def test_compute_tensor_update(tree):
    tree.root.tree_traversal(
        "_set_inference_mode",
        mode="top-down",
        method_input=((), dict(mode=tree.root.inference_mode)),
    )
    tree.root.tree_traversal(
        "_initialization", mode="top-down",
    )
    tree.root.tree_traversal(
        "compute_tensor_update", mode="bottom-up", iteration_number=0
    )

    assert wtf.glob.xp.allclose(
        tree.mul_tf.tensor_update, tree.obs_tf_2.tensor / tree.mul_tf.tensor
    )
    assert wtf.glob.xp.allclose(
        tree.leaf_k_1.tensor_update,
        wtf.glob.xp.einsum("kt,kt->k", tree.mul_kt.tensor_update, tree.leaf_kt.tensor),
    )
    assert wtf.glob.xp.allclose(
        tree.leaf_energy.tensor_update,
        (tree.mul_k.tensor_update * tree.leaf_k.tensor).sum(),
    )
    assert wtf.glob.xp.allclose(
        tree.leaf_kf.tensor_update,
        (
            wtf.glob.xp.einsum(
                "tfs,kts->kf", tree.mul_tfs.tensor_update, tree.mul_kts_1.tensor
            )
            + wtf.glob.xp.einsum(
                "tf,kt->kf", tree.mul_tf.tensor_update, tree.mul_kt.tensor
            )
        ),
    )


def test_param_estimation(tree):
    tree.root.estimate_param(n_iter=100)
    cost_func = np.array(tree.root.cost_record)
    assert all(cost_func[:-1] >= cost_func[1:])


def test_graphviz(tree):
    legend_dict = {
        "k": {"description": "atom", "letter": "k"},
        "f": {"description": "frequency"},
        "t": {"description": "time"},
        "s": {"description": "sign"},
        "c": {"description": "complex part"},
    }
    assert t_utils._assert_graphviz_ok(tree, legend_dict)

