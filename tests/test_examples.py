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
from itertools import product

# Third-party imports
import pytest
import numpy as np
import numpy.random as npr

# wonterfact and relative imports
from wonterfact import glob, LeafGammaNorm
from wonterfact import utils as wtfu
from wonterfact.examples import snmf
from wonterfact.examples import nmf
from wonterfact.examples import conv_nmf

# relative imports
from . import utils as t_utils

# args for make_snmf
data = 100 * npr.randn(10, 5)
atoms_nonneg_init = npr.rand(3, 5, 2)
activations_init = np.ones((10, 3))


list_of_tree_makers_tuple = [
    (nmf.make_nmf, (), {}),
    (nmf.make_smooth_activation_nmf, (), {}),
    (nmf.make_smooth_activation_nmf2, (), {}),
    (nmf.make_sparse_nmf, (), {}),
    (nmf.make_sparse_nmf2, (), {}),
    (nmf.make_sparse_nmf3, (), {}),
    (snmf.make_snmf, (data, atoms_nonneg_init, activations_init), {"fix_atoms": True}),
    (snmf.make_snmf, (data, atoms_nonneg_init, activations_init), {"fix_atoms": False}),
    (snmf.make_cluster_snmf, (), {}),
    (snmf.make_cluster_snmf2, (), {}),
    (conv_nmf.make_deconv_tree, (), {}),
]
# let us add unique_id to the tuples
list_of_tree_makers_tuple = [
    elem + (num,) for num, elem in enumerate(list_of_tree_makers_tuple)
]


@pytest.fixture(scope="module")
def cost_record_results():
    return {}


@pytest.mark.parametrize("tree_maker_tuple", list_of_tree_makers_tuple)
@pytest.mark.parametrize("inference_mode", ["EM", "VBEM"])
@pytest.mark.parametrize("update_type", ["regular", "parabolic"])
@pytest.mark.parametrize("limit_skellam", [True, False])
@pytest.mark.parametrize("backend", ["cpu", pytest.param("gpu", marks=pytest.mark.gpu)])
def test_example(
    tree_maker_tuple,
    inference_mode,
    update_type,
    limit_skellam,
    backend,
    cost_record_results,
):
    glob.set_backend_processor(backend, force=True)
    np.random.seed(0)
    tree_maker, args, kwargs, unique_id = tree_maker_tuple
    tree = tree_maker(*args, **kwargs)
    t_utils._setup_tree_for_decreasing_cost(
        tree,
        inference_mode=inference_mode,
        update_type=update_type,
        limit_skellam=limit_skellam,
    )
    if inference_mode == "VBEM":
        if any(type(node) == LeafGammaNorm for node in tree.census()):
            with pytest.raises(NotImplementedError):
                tree.estimate_param(n_iter=100)
            return
        for ii in range(10):
            tree.estimate_param(n_iter=10)
            tree.estimate_hyperparam(n_iter=ii)
    else:
        tree.estimate_param(n_iter=100)
    assert t_utils._assert_cost_decrease(tree)
    assert t_utils._assert_graphviz_ok(tree)
    base_key = (
        unique_id,
        inference_mode,
        update_type,
        limit_skellam,
    )

    cost_record_results[base_key + (backend,)] = np.array(tree.cost_record)
    if (
        base_key + ("cpu",) in cost_record_results
        and base_key + ("gpu",) in cost_record_results
    ):
        assert np.allclose(
            cost_record_results[base_key + ("cpu",)],
            cost_record_results[base_key + ("gpu",)],
        )
