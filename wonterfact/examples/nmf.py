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


def make_nmf_tree(fix_atoms=False):
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
