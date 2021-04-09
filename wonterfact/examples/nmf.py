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

"""Examples of nonnegative models for nonnegative observed tensor"""


# Python standard library

# Third-party imports
import numpy as np
import numpy.random as npr

# wonterfact imports
import wonterfact as wtf
import wonterfact.utils as wtfu


def make_nmf(fix_atoms=False):
    dim_k, dim_f, dim_t = 5, 20, 100

    atoms_kf = npr.dirichlet(np.ones(dim_f) * 0.9, size=dim_k)

    activations_tk = npr.gamma(shape=0.6, scale=200, size=(dim_t, dim_k))

    observations_tf = np.einsum("tk,kf->tf", activations_tk, atoms_kf)
    observations_tf += npr.randn(dim_t, dim_f) * 1e-4

    leaf_kf = wtf.LeafDirichlet(
        name="atoms",
        index_id="kf",
        norm_axis=(1,),
        tensor=atoms_kf,
        update_period=0 if fix_atoms else 1,
        prior_shape=1 if fix_atoms else 1 + 1e-5 * npr.rand(dim_k, dim_f),
    )
    leaf_tk = wtf.LeafGamma(
        name="activations",
        index_id="tk",
        tensor=np.ones_like(activations_tk),
        prior_rate=1e-5,
        prior_shape=1,
    )
    mul_tf = wtf.Multiplier(name="multiplier", index_id="tf")
    mul_tf.new_parents(leaf_kf, leaf_tk)
    obs_tf = wtf.PosObserver(name="observer", index_id="tf", tensor=observations_tf)
    mul_tf.new_child(obs_tf)
    root = wtf.Root(name="nmf")
    obs_tf.new_child(root)

    return root


def _aux_smooth_activation_nmf():
    dim_k, dim_f, dim_t = 5, 20, 100
    dim_w = 9

    atoms_kf = npr.dirichlet(np.ones(dim_f) * 0.9, size=dim_k)

    activations_tk = npr.gamma(shape=0.6, scale=200, size=(dim_t + dim_w - 1, dim_k))
    spread_tkw = npr.dirichlet(3 * np.ones(dim_w), size=(dim_t + dim_w - 1, dim_k))
    act_tkw = np.einsum("tk,tkw->tkw", activations_tk, spread_tkw)
    activations_tk = np.zeros((dim_t, dim_k))
    for ww in range(dim_w):
        activations_tk += act_tkw[ww : dim_t + ww, :, ww]

    observations_tf = np.einsum("tk,kf->tf", activations_tk, atoms_kf)
    observations_tf += npr.randn(dim_t, dim_f) * 1e-4

    leaf_kf = wtf.LeafDirichlet(
        name="atoms",
        index_id="kf",
        norm_axis=(1,),
        tensor=atoms_kf,
        update_period=1,
        prior_shape=1 + 1e-5 * npr.rand(dim_k, dim_f),
    )

    leaf_kt = wtf.LeafGamma(
        name="impulse",
        index_id="kt",
        tensor=np.ones((dim_k, dim_t + dim_w - 1)),
        prior_shape=1,
        prior_rate=1e-5,
    )
    leaf_ktw = wtf.LeafDirichlet(
        name="spreading",
        index_id="ktw",
        norm_axis=(2,),
        tensor=np.ones((dim_k, dim_t + dim_w - 1, dim_w)),
        prior_shape=10,
    )
    mul_kwt = wtf.Multiplier(name="mul_kwt", index_id="kwt")
    mul_kwt.new_parents(leaf_kt, leaf_ktw)

    mul_tf = wtf.Multiplier(name="reconstruction", index_id="tf")
    mul_tf.new_parent(leaf_kf)
    obs_tf = wtf.PosObserver(name="observer", index_id="tf", tensor=observations_tf)
    mul_tf.new_child(obs_tf)

    return leaf_kt, leaf_ktw, mul_kwt, mul_tf, obs_tf


def make_smooth_activation_nmf():
    """
    NMF with overlapping activation (each time coefficient spreads out on
    both sides). We uses this model in order to test strides_for_child
    """
    _, leaf_ktw, mul_kwt, mul_tf, obs_tf = _aux_smooth_activation_nmf()

    dim_k, _, dim_w = leaf_ktw.tensor.shape
    dim_t = obs_tf.tensor.shape[0]

    leaf_w = wtf.LeafDirichlet(
        name="adder", index_id="w", norm_axis=(), tensor=np.ones(dim_w), update_period=0
    )
    mul_tk = wtf.Multiplier(name="activations", index_id="tk")
    shape = (dim_k, dim_w, dim_t + dim_w - 1)
    strides_for_child = ([8,] + list(np.cumprod(shape[:0:-1]) * 8))[::-1]
    strides_for_child[-2] += 8
    mul_kwt.new_child(
        mul_tk,
        shape_for_child=(dim_k, dim_w, dim_t),
        strides_for_child=strides_for_child,
    )
    leaf_w.new_child(mul_tk)

    mul_tf.new_parent(mul_tk)
    root = wtf.Root(name="smooth_activation_nmf")
    obs_tf.new_child(root)

    return root


def make_smooth_activation_nmf2():
    """
    Same model as in the `make_smooth_activation_nmf` method, but with Proxys
    and Adder instead of strides
    """
    _, leaf_ktw, mul_kwt, mul_tf, obs_tf = _aux_smooth_activation_nmf()

    dim_w = leaf_ktw.tensor.shape[2]
    dim_t = obs_tf.tensor.shape[0]

    add_kt = wtf.Adder(name="activations", index_id="kt")
    for ww in range(dim_w):
        proxy = wtf.Proxy(index_id="kt")
        proxy.new_parent(
            mul_kwt,
            slice_for_child=(slice(None), ww, slice(ww, ww + dim_t)),
            index_id_for_child="kt",
        )
        proxy.new_child(add_kt)
    mul_tf.new_parent(add_kt)
    root = wtf.Root(name="smooth_activation_nmf2")
    obs_tf.new_child(root)

    return root


def make_sparse_nmf(prior_rate=0.001, obs=None, atoms=None):
    """
    NMF with minimization of \sum_{k != k'} P(k|t)P(k'|t)E(t) where P(k|t) are
    the activations and E(t) total energy at time t
    """
    dim_f, dim_t, dim_k = 2, 100, 2

    # gt_kf = npr.dirichlet(np.ones(dim_f), size=dim_k)
    gt_kf = np.array([[4.0, 1.0], [4.0, 3.0]])
    gt_kf /= gt_kf.sum(1, keepdims=True)
    gt_tk = npr.gamma(shape=0.3, scale=100, size=(dim_t, dim_k))
    gt_tf = np.dot(gt_tk, gt_kf)
    gt_tf += npr.rand(dim_t, dim_f)
    if obs is not None:
        gt_tf = obs

    leaf_t = wtf.LeafGamma(
        name="time_energy",
        index_id="t",
        tensor=np.ones(dim_t),
        prior_rate=prior_rate,
        prior_shape=1,
    )

    leaf_tk = wtf.LeafDirichlet(
        name="activations",
        index_id="tk",
        norm_axis=(1,),
        tensor=np.ones((dim_t, dim_k)),
        prior_shape=1,
    )

    mul_tk = wtf.Multiplier(index_id="tk")
    mul_tk.new_parents(leaf_t, leaf_tk)

    mul_tkl = wtf.Multiplier(name="activations_square", index_id="tkl")
    leaf_tk.new_child(mul_tkl, index_id_for_child="tl")
    mul_tk.new_child(mul_tkl)
    if atoms is None:
        atoms = np.ones((dim_k, dim_f))
        update_period = 1
    else:
        update_period = 0
    leaf_kf = wtf.LeafDirichlet(
        name="atoms",
        index_id="kf",
        norm_axis=(1,),
        tensor=atoms,
        prior_shape=1 + 1e-4 * npr.rand(dim_k, dim_f),
        update_period=update_period,
    )
    mul_tf = wtf.Multiplier(name="reconstruction", index_id="tf")
    leaf_kf.new_child(mul_tf)

    test_arr = npr.rand(2, dim_k, dim_k)
    strides = (test_arr.strides[0],) + np.diag(test_arr[0]).strides
    mul_tkl.new_child(
        mul_tf,
        shape_for_child=(dim_t, dim_k),
        strides_for_child=strides,
        index_id_for_child="tk",
    )

    obs_tf = wtf.PosObserver(name="observations", index_id="tf", tensor=gt_tf)
    mul_tf.new_child(obs_tf)
    root = wtf.Root(name="root", verbose_iter=50, cost_computation_iter=10)
    obs_tf.new_child(root)
    return root


def make_sparse_nmf2(prior_rate=0.001, obs=None):
    """
    NMF with l1/l2 sparse constraint on atoms
    """
    dim_f, dim_t, dim_k = 2, 100, 2

    # gt_kf = npr.dirichlet(np.ones(dim_f), size=dim_k)
    gt_kf = np.array([[4.0, 1.0], [4.0, 3.0]])
    gt_kf /= gt_kf.sum(1, keepdims=True)
    gt_tk = npr.gamma(shape=0.3, scale=100, size=(dim_t, dim_k))
    gt_tf = np.dot(gt_tk, gt_kf)
    gt_tf += npr.rand(dim_t, dim_f)
    if obs is not None:
        gt_tf = obs

    leaf_kf = wtf.LeafGammaNorm(
        name="atoms",
        index_id="kf",
        tensor=np.ones((dim_k, dim_f)),
        l2_norm_axis=(1,),
        prior_rate=prior_rate,
        prior_shape=1 + 1e-4 * npr.rand(dim_k, dim_k),
    )

    leaf_kt = wtf.LeafDirichlet(
        name="activations",
        index_id="kt",
        norm_axis=(1,),
        tensor=np.ones((dim_k, dim_t)),
        prior_shape=1,
    )

    mul_tf = wtf.Multiplier(name="reconstruction", index_id="tf")
    mul_tf.new_parents(leaf_kt, leaf_kf)

    obs_tf = wtf.PosObserver(name="observations", index_id="tf", tensor=gt_tf)
    mul_tf.new_child(obs_tf)
    root = wtf.Root(name="root", verbose_iter=50, cost_computation_iter=10)
    obs_tf.new_child(root)
    return root


def make_sparse_nmf3(prior_rate=0.001, obs=None):
    """
    NMF with approximation of l2 norm for atoms
    """
    dim_f, dim_t, dim_k, dim_a = 2, 100, 2, 2

    gt_kf = npr.dirichlet(np.ones(dim_f), size=dim_k)
    # gt_kf = np.array([[4.0, 1.0], [4.0, 3.0]])
    gt_kf /= gt_kf.sum(1, keepdims=True)
    gt_tk = npr.gamma(shape=0.3, scale=100, size=(dim_t, dim_k))
    gt_tf = np.dot(gt_tk, gt_kf)
    gt_tf += npr.rand(dim_t, dim_f)
    if obs is not None:
        gt_tf = obs

    leaf_k = wtf.LeafGamma(
        name="atoms_energy",
        index_id="k",
        tensor=np.ones((dim_k)),
        prior_shape=1,
        prior_rate=prior_rate,
    )

    leaf_kf = wtf.LeafDirichlet(
        name="atoms_init",
        index_id="kf",
        norm_axis=(1,),
        tensor=np.ones((dim_k, dim_f)),
        prior_shape=1 + 1e-4 * npr.rand(dim_k, dim_f),
    )
    mul_kf = wtf.Multiplier(index_id="kf")
    mul_kf.new_parents(leaf_kf, leaf_k)
    mul_kfg = wtf.Multiplier(index_id="kfg")
    mul_kf.new_child(mul_kfg)
    leaf_kf.new_child(mul_kfg, index_id_for_child="kg")

    leaf_c = wtf.LeafDirichlet(
        index_id="c", norm_axis=(0,), tensor=np.array([0.5, 0.5]), update_period=0
    )
    mul_ckf = wtf.Multiplier(index_id="ckf")
    leaf_c.new_child(mul_ckf)
    test_arr = npr.rand(2, dim_f, dim_f)
    strides = (test_arr.strides[0],) + np.diag(test_arr[0]).strides
    mul_kfg.new_child(
        mul_ckf,
        index_id_for_child="kf",
        shape_for_child=(dim_k, dim_f),
        strides_for_child=strides,
    )

    leaf_g = wtf.LeafDirichlet(
        index_id="g", norm_axis=(), tensor=np.ones(dim_f - 1), update_period=0
    )
    mul_kf2 = wtf.Multiplier(index_id="kf")
    leaf_g.new_child(mul_kf2)
    mask = np.logical_not(np.eye(dim_f, dtype=bool))
    mul_kfg.new_child(
        mul_kf2,
        slice_for_child=[slice(None), mask],
        shape_for_child=(dim_k, dim_f, dim_f - 1),
    )

    add_kf = wtf.Adder(name="atoms", index_id="kf")
    mul_kf2.new_child(add_kf)
    mul_ckf.new_child(add_kf, index_id_for_child="kf", slice_for_child=(0, Ellipsis))

    # leaf_ka = wtf.LeafDirichlet(
    #     name="angle",
    #     index_id="ka",
    #     norm_axis=(1,),
    #     tensor=np.ones((dim_k, dim_a)),
    #     l2_norm_axis=(1,),
    #     prior_shape=1 + 1e-4 * npr.rand(dim_k, dim_a),
    # )

    # mul_ka = wtf.Multiplier(index_id="ka")
    # mul_ka.new_parents(leaf_k, leaf_ka)

    # mul_kab = wtf.Multiplier(index_id="kab")
    # mul_ka.new_child(mul_kab)
    # leaf_ka.new_child(mul_kab, index_id_for_child="kb")

    # leaf_abm = wtf.LeafDirichlet(
    #     index_id="abm",
    #     norm_axis=(2,),
    #     tensor=np.array([[[1, 0, 0], [0, 1, 0]], [[0, 1, 0], [0, 0, 1]]]),
    #     update_period=0,
    # )

    # mul_km = wtf.Multiplier(index_id="km")
    # mul_km.new_parents(leaf_abm, mul_kab)

    # leaf_mnf = wtf.LeafDirichlet(
    #     name="basis",
    #     index_id="mnf",
    #     norm_axis=(1, 2),
    #     tensor=np.array(
    #         [[[0, 0.5], [0.5, 0]], [[0.5, 0.5], [0, 0]], [[0.5, 0], [0, 0.5]]]
    #     ),
    #     update_period=0,
    # )

    # mul_nkf = wtf.Multiplier(index_id="nkf", name="atoms")
    # mul_nkf.new_parents(leaf_mnf, mul_km)

    leaf_kt = wtf.LeafDirichlet(
        name="activations",
        index_id="kt",
        norm_axis=(1,),
        tensor=np.ones((dim_k, dim_t)),
        prior_shape=1,
    )

    mul_tf = wtf.Multiplier(name="reconstruction", index_id="tf")
    leaf_kt.new_child(mul_tf)
    # mul_nkf.new_child(mul_tf, index_id_for_child="kf", slice_for_child=[0, Ellipsis])
    add_kf.new_child(mul_tf)

    obs_tf = wtf.PosObserver(name="observations", index_id="tf", tensor=gt_tf)
    mul_tf.new_child(obs_tf)
    root = wtf.Root(name="root", verbose_iter=50, cost_computation_iter=10)
    obs_tf.new_child(root)
    return root
