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

"""Examples of wonterfact models using involving convolution"""

# Python standard library

# Third-party imports
import numpy as np
import numpy.random as npr
from scipy.signal import convolve2d

# wonterfact imports
import wonterfact as wtf


def make_deconv_tree():
    dim_x0, dim_y0, dim_x1, dim_y1 = 20, 20, 3, 3
    dim_x = dim_x0 - dim_x1 + 1
    dim_y = dim_y0 - dim_y1 + 1
    dim_k = 2

    kernel_kyx = np.zeros((dim_k, dim_y1, dim_x1))
    kernel_kyx[0, [1, 0, 1, 2, 1], [0, 1, 1, 1, 2]] = 1
    kernel_kyx[1, [0, 0, 1, 1, 2, 2, 2], [0, 2, 0, 2, 0, 1, 2]] = 1
    kernel_kyx /= kernel_kyx.sum((1, 2), keepdims=True)

    impulse_kyx = npr.gamma(shape=0.08, scale=200, size=(dim_k, dim_y0, dim_x0))
    impulse_kyx[impulse_kyx < 200] = 0
    impulse_kyx[impulse_kyx >= 200] = 200
    image_yx = np.zeros((dim_y, dim_x))
    for kk in range(dim_k):
        image_yx += convolve2d(impulse_kyx[kk], kernel_kyx[kk], mode="valid")

    leaf_kernel = wtf.LeafDirichlet(
        name="kernel",
        index_id=("k", "j", "i"),
        norm_axis=(1, 2),
        tensor=np.ones((dim_k, dim_y1, dim_x1)),
        prior_shape=100 + 1e-4 * npr.rand(dim_k, dim_y1, dim_x1),
    )
    leaf_impulse = wtf.LeafGamma(
        name="impulse",
        index_id=("k", "y", "x"),
        tensor=np.ones((dim_k, dim_y0, dim_x0)),
        prior_shape=1,
        prior_rate=0.0001,
    )
    mul_image = wtf.Multiplier(
        name="reconstruction", index_id="yx", conv_idx_ids=("y", "x")
    )
    leaf_kernel.new_child(mul_image, index_id_for_child=("k", "y", "x"))
    leaf_impulse.new_child(mul_image, index_id_for_child=("k", "y", "x"))
    obs = wtf.PosObserver(
        name="image",
        index_id="yx",
        tensor=image_yx,
        drawings_max=200 * dim_x * dim_y,
        drawings_step=10,
    )
    mul_image.new_child(obs)
    root = wtf.Root(name="root")
    obs.new_child(root)
    return root
