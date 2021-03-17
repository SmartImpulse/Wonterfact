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


def make_snmf(fix_atoms=False):
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


def make_cluster_snmf():
    """
    Convex S-NMF for automatic clustering, inspired by Ding2010_IEEE
    Data to cluster are considered scale invariant, meaning we cluster directions
    rather than points in some n-dimensional space.
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


def make_cluster_snmf2(nb_cluster=4, prior_rate=0.001):
    """
    Convex S-NMF for automatic clustering, inspired by Ding2010_IEEE
    Data to cluster are points in some n-dimensional space.
    """
    dim_d, dim_f = nb_cluster * 100, 2
    tensor_df = np.zeros((dim_d, dim_f))
    for ii in range(nb_cluster):
        mean = npr.uniform(low=-10, high=10, size=dim_f)
        cov = npr.uniform(low=0.0, high=1.0, size=(dim_f, dim_f))
        cov = np.dot(cov, cov.T) + np.diag(npr.uniform(low=0.0, high=0.5, size=dim_f))
        tensor_df[
            ii * dim_d // nb_cluster : (ii + 1) * dim_d // nb_cluster, :
        ] = npr.multivariate_normal(mean, cov, size=dim_d // nb_cluster)

    tensor_df = tensor_df[npr.permutation(dim_d)].copy()

    dim_q = nb_cluster

    tensor_dfs = wtfu.real_to_2D_nonnegative(tensor_df)
    tensor_dfs /= tensor_dfs.sum()
    leaf_dfs = wtf.LeafDirichlet(
        name="samples",
        index_id="dfs",
        norm_axis=(0, 1, 2),
        tensor=tensor_dfs,
        update_period=0,
    )

    tensor_dq = np.ones((dim_d, dim_q))
    leaf_dq = wtf.LeafDirichlet(
        name="class_by_sample",
        index_id="dq",
        norm_axis=(1,),
        tensor=tensor_dq,
        prior_shape=1 + 1e-5 * npr.rand(dim_d, dim_q),
    )
    mul_pfs = wtf.Multiplier(name="barycenters", index_id="pfs")
    leaf_dq.new_child(mul_pfs, index_id_for_child="dp")
    leaf_dfs.new_child(mul_pfs)

    tensor_pqm = np.zeros((dim_q, dim_q, 2))
    tensor_pqm[:, :, 0] = np.eye(dim_q)
    tensor_pqm[:, :, 1] = 1 - np.eye(dim_q)
    leaf_pqm = wtf.LeafDirichlet(
        name="energy dispatcher",
        index_id="pqm",
        tensor=tensor_pqm,
        norm_axis=(2,),
        update_period=0,
    )

    mul_qfsm = wtf.Multiplier(index_id="qfsm")
    mul_qfsm.new_parents(mul_pfs, leaf_pqm)

    leaf_d = wtf.LeafGamma(
        name="sample_energy",
        index_id="d",
        norm_axis=(0,),
        tensor=np.ones(dim_d),
        prior_shape=1,
        prior_rate=prior_rate,
    )

    mul_dq = wtf.Multiplier(name="sample_class_energy", index_id="dq")
    mul_dq.new_parents(leaf_d, leaf_dq)

    mul_dfsm = wtf.Multiplier(name="reconstruction", index_id="dfsm")
    mul_dfsm.new_parents(mul_qfsm, mul_dq)

    # observations
    obs_df = wtf.RealObserver(name="observations", index_id="df", tensor=tensor_df,)
    mul_dfsm.new_child(obs_df, index_id_for_child="dfs", slice_for_child=(Ellipsis, 0))
    root = wtf.Root(
        name="root",
        verbose_iter=50,
        cost_computation_iter=10,
        # update_type='regular'
    )
    obs_df.new_child(root)

    # # null obs
    # obs_df2 = wtf.PosObserver(
    #     name="null_observations", index_id="dfs", tensor=np.zeros((dim_d, dim_f, 2))
    # )
    # mul_dfsm.new_child(obs_df2, index_id_for_child="dfs", slice_for_child=(Ellipsis, 1))

    # obs_df2.new_child(root)

    return root
