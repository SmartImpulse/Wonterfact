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


def make_snmf(
    data,
    atoms_nonneg_init,
    activations_init,
    fix_atoms=False,
    atoms_shape_prior=1,
    activations_shape_prior=1,
    activations_rate_prior=0.001,
    inference_mode="EM",
    integer_data=False,
):
    """
    This return a wonterfact tree corresponding to the Skellam-SNMF algorithm.

    Parameters
    ----------
    data: array_like of shape [J x I]
        Real-valued input array to factorize
    atoms_nonneg_init: array_like of shape [K x I x 2]
        Initialization for nonnegative atoms tensor
    activations_init: array_like of shape [J x K]
        Initialization for activations matrix
    fix_atoms: bool, default False
        Whether atoms should be updated or left to initial value
    atoms_shape_prior: array_like or float
        Shape hyperparameters for atoms (`atoms_shape_prior + atoms_nonneg_init`)
        should raise no Error
    activations_shape_prior: array_like or float
        Shape hyperparameters for activations (`activations_shape_prior + activations_init`)
        should raise no Error
    activations_rate_prior: array_like or float
        Shape hyperparameters for atoms (`activations_rate_prior + activations_init`)
        should raise no Error.
    inference_mode: 'EM' or 'VBEM', default 'EM'
        Algorithm that should be used to infer parameters
    integer_data: bool, default False
        Whether data are integers or real numbers. If True, Skellam-SNMF is
        performed with :math: `M=1`, otherwise with :math: `M=\\infty` (see [1])

    Returns
    -------
    wonterfact.Root
        root of the wonterfact tree

    Notes
    ------
    Allows to solve the following problem
    .. math::
        X_{ji} \\approx \\sum_{k} \\lambda_{jk} * W_{ki} \\textrm{with}\\
        W_{ki} = \\theta_{ki, s=0} - \\theta_{ki, s=1}
    Beware that axis are reversed compared to the model in [1]. This due to the
    terms of use of wonterfact.

    References
    ----------
    ..[1] B.Fuentes et. al., Probabilistic semi-nonnegative matrix factorization:
    a Skellam-based framework, 2021

    """
    ### creation of atoms leaf
    # to be sure atoms are well normalized
    atoms_nonneg_init /= atoms_nonneg_init.sum(axis=(1, 2), keepdims=True)
    leaf_kis = wtf.LeafDirichlet(
        name="atoms",  # name of the node
        index_id="kis",  # name of the axis
        norm_axis=(1, 2),  # normalization axis
        tensor=atoms_nonneg_init,  # instantiation of leaf's tensor
        init_type="custom",  # to be sure value of `tensor` is kept as initialization
        update_period=0 if fix_atoms else 1,  # whether atoms should be updated or not
        prior_shape=atoms_shape_prior,  # shape hyperparameters
    )
    ### creation of activations leaf
    leaf_jk = wtf.LeafGamma(
        name="activations",  # name of the node
        index_id="jk",  # name of the axis
        tensor=activations_init,  # instantiation of leaf's tensor
        init_type="custom",  # to be sure value of `tensor` is kept as initialization
        prior_rate=activations_rate_prior,  # rate hyperparameters
        prior_shape=activations_shape_prior,  # shape hyperparameters
    )
    ### resulting skellam parameters for observed data data
    mul_jis = wtf.Multiplier(name="multiplier", index_id="jis")
    mul_jis.new_parents(leaf_kis, leaf_jk)  # creation of filiations

    ### observed real-valued data
    obs_ji = wtf.RealObserver(
        name="observer",
        index_id="ji",
        tensor=data,
        limit_skellam_update=not integer_data,  # whether data are considered as integers
    )
    mul_jis.new_child(obs_ji)  # filiation between data and model parameters

    ### creation or the root
    root = wtf.Root(
        inference_mode=inference_mode,
        stop_estim_threshold=1e-7,
        cost_computation_iter=50,
    )
    obs_ji.new_child(root)
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

    tensor_dfs = wtfu.normalize(wtfu.real_to_2D_nonnegative(tensor_df), (1, 2))
    leaf_dfs = wtf.LeafDirichlet(
        name="samples",
        index_id="dfs",
        norm_axis=(1, 2),
        tensor=tensor_dfs,
        update_period=0,
    )

    tensor_qd = np.ones((dim_q, dim_d)) / dim_d
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

    tensor_dfs = wtfu.normalize(wtfu.real_to_2D_nonnegative(tensor_df), None)
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
        tensor=wtfu.normalize(npr.rand(dim_d, dim_q), (1,)),
        prior_shape=1,
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
    obs_df = wtf.RealObserver(
        name="observations",
        index_id="df",
        tensor=tensor_df,
    )
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
