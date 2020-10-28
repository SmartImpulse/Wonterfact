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

"""Examples of semi-nonnegative models for real-valued observed tensor"""


# Python standard library

# Third-party imports
import numpy as np
import numpy.random as npr

# wonterfact imports
import wonterfact as wtf
import wonterfact.utils as wtfu


def make_snmf_tree(fix_atoms=False):
    dim_k, dim_f, dim_t = 5, 20, 100

    atoms_kf = npr.choice([-1, 1], size=[dim_k, dim_f]) * npr.dirichlet(
        np.ones(dim_f) * 0.9, size=dim_k
    )

    activations_tk = npr.gamma(shape=0.6, scale=200, size=(dim_t, dim_k))

    observations_tf = np.einsum("tk,kf->tf", activations_tk, atoms_kf)
    observations_tf += npr.randn(dim_t, dim_f) * 1e-4

    leaf_kfs = wtf.LeafDirichlet(
        name="atoms",
        index_id="kfs",
        norm_axis=(1, 2),
        tensor=wtf.utils.real_to_2D_nonnegative(atoms_kf),
        update_period=0 if fix_atoms else 1,
        prior_shape=1 if fix_atoms else 1 + 1e-5 * npr.rand(dim_k, dim_f, 2),
    )
    leaf_tk = wtf.LeafGamma(
        name="activations",
        index_id="tk",
        tensor=np.ones_like(activations_tk),
        prior_rate=1e-4,
    )
    mul_tfs = wtf.Multiplier(name="multiplier", index_id="tfs")
    mul_tfs.new_parents(leaf_kfs, leaf_tk)
    obs_tf = wtf.RealObserver(name="observer", index_id="tf", tensor=observations_tf)
    mul_tfs.new_child(obs_tf)
    root = wtf.Root()
    obs_tf.new_child(root)

    return root


def make_convex_clustering():
    """
    Convex S-NMF for automatic clustering, inspired by Ding2010_IEEE
    """
    dim_d, dim_f = 400, 10
    tensor_df = np.zeros((dim_d, dim_f))
    for ii in range(4):
        shape = npr.dirichlet(np.ones(dim_f)) ** 2 * 100
        tensor_df[ii * dim_d // 4 : (ii + 1) * dim_d // 4, :] = npr.dirichlet(
            shape, size=dim_d // 4
        ) * npr.choice([-1, 1], size=(dim_f))

    tensor_df = tensor_df[npr.permutation(dim_d)]
    tensor_df *= 200
    # tensor_df[tensor_df<1e-10] = 0

    dim_q = 10

    tensor_dfs = wtfu.real_to_2D_nonnegative(tensor_df)
    tensor_dfs /= tensor_dfs.sum(axis=(1, 2), keepdims=True)
    leaf_dfs = wtf.LeafDirichlet(
        name="samples",
        index_id="dfs",
        norm_axis=(1, 2),
        tensor=tensor_dfs,
        update_period=0,
    )

    tensor_qd = np.ones((dim_q, dim_d))
    leaf_qd = wtf.LeafDirichlet(
        name="samples_by_class",
        index_id="qd",
        norm_axis=(1,),
        tensor=tensor_qd,
        prior_shape=2,
    )
    mul_qfs = wtf.Multiplier(name="barycenters", index_id="qfs")
    mul_qfs.new_parents(leaf_qd, leaf_dfs)

    tensor_q = 1 + 1e-4 * (npr.rand(dim_q))
    leaf_q = wtf.LeafGamma(
        name="class_energy",
        index_id="q",
        norm_axis=(0,),
        tensor=tensor_q,
        # total_max_energy=2 * np.abs(tensor_df).sum(),
        prior_shape=1.1 * tensor_q,
        prior_rate=1e-4,
    )
    mul_qd = wtf.Multiplier(name="sample_energy", index_id="qd")
    mul_qd.new_parents(leaf_q, leaf_qd)

    mul_dfs = wtf.Multiplier(name="reconstruction", index_id="dfs")
    mul_dfs.new_parents(mul_qfs, mul_qd)

    # observations
    obs_df = wtf.RealObserver(
        name="observations",
        index_id="df",
        tensor=tensor_df,
        # drawings_max=drawing_bin_max * tensor_df.size,
        # drawings_step = drawings_bin_step * tensor_df.size,
    )
    mul_dfs.new_child(obs_df)
    root = wtf.Root(
        name="root",
        verbose_iter=200,
        cost_computation_iter=10,
        # update_type='regular'
    )
    obs_df.new_child(root)
    return root
