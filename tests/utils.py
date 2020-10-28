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

"""Useful methods for testing a wonterfact model"""

# Python standard library
from pathlib import Path
import tempfile

# Third-party imports
import numpy as np

# wonterfact imports


def _assert_graphviz_ok(tree, legend_dict=None):
    with tempfile.TemporaryDirectory() as tmpdir:
        filemane = Path(tmpdir) / "test.pdf"
        tree.draw_tree(
            filename=filemane, legend_dict=legend_dict, prior_nodes=True, view=False
        )
        return filemane.exists()


def _assert_cost_decrease(tree):
    return (np.diff(tree.cost_record) <= 0).all()


def _setup_tree_for_decreasing_cost(
    tree, inference_mode="EM", update_type="parabolic", limit_skellam=True
):
    tree.cost_computation_iter = 1
    tree.acceleration_start_iter = 10
    for parent in tree.list_of_parents:
        parent.drawings_step = parent.drawings_max
        parent.drawings = parent.drawings_max
        parent.limit_skellam_update = limit_skellam
    tree.inference_mode = inference_mode
    tree.update_type = update_type
    tree.stop_estim_threshold = 1e-10
