# ----------------------------------------------------------------------------
# Copyright 2020 Smart Impulse SAS, Benoit Fuentes <bf@benoit-fuentes.fr>
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

"""Module for all observer classes"""


# Python Future imports
from __future__ import division, unicode_literals, print_function, absolute_import

# Python System imports

# Third-party imports
import numpy.random as npr
from functools import cached_property


# Relative imports
from . import utils, core_nodes
from .glob_var_manager import glob


class _Observer(
    core_nodes._NodeData, core_nodes._ChildNode
):  # TODO: optimize mask_data
    """
    Base class for all Observer nodes, i.e. nodes that carry observed data.
    """

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        mask_data: array_like of booleans or None, default None
            Boolean mask to apply to observed 2 to specify which
            coefficients are masked and which are not. If masked, a coefficient
            plays no role in the optimization process.
        drawings_max: float or None, optional, default None
            During optimization algorithm, inner tensor is normalized with a
            coefficient which can increase along iterations up to a limit.
            value. If not None, this limit value is computed as `drawings_max /
            abs(self.tensor).sum()`. If None, the limit value is `1`.
        drawings_update_iter: int, optional, default 1
            The normalization coefficient (see `drawings_max` section) is
            updated every `drawings_update_iter` iterations.
        drawings_step: float or None, optional, default None
            Normalization coefficient is initialized as `drawings_step /
            abs(self.tensor).sum()` and when it has to be updated (see
            `drawings_update_iter`), the same amount is added to the current
            normalization coefficient until it reaches its limit (see
            `drawings_max` section). If None, `drawings_step` is set to
            `drawings_max` value so that the normalization coefficient remains
            fixed during the algorithm.

        Notes
        -----
        The dynamic normalization feature (see `drawings_max`,
        `drawings_update_iter` and `drawings_step` sections) aims at giving more
        weight to the priors in the early stage of the algorithm. If you do not
        want to use this feature, just leave default values for those arguments.
        """
        self.mask_data = kwargs.pop("mask_data", None)
        self.drawings_max = kwargs.pop("drawings_max", None)
        self.drawings_step = kwargs.pop("drawings_step", None)
        self.drawings_update_iter = kwargs.pop("drawings_update_iter", 1)
        super().__init__(**kwargs)

        self.drawings_max = self.drawings_max or self.sum_tensor
        self.drawings_step = self.drawings_step or self.sum_tensor
        self.drawings = self.drawings_step
        self.drawings_update_counter = 0

    @cached_property
    def is_tensor_null(self):
        return (self.tensor == 0).astype(glob.float)

    @cached_property
    def is_tensor_pos(self):
        return self.tensor > 0

    @cached_property
    def sum_tensor(self):
        return glob.xp.abs(self.tensor).sum()

    def get_current_mult_factor(self):
        return self.drawings / self.sum_tensor if self.sum_tensor else 1

    def update_drawings(self):
        self.drawings_update_counter += 1
        if self.drawings_update_counter % self.drawings_update_iter == 0:
            self.drawings = min(self.drawings_max, self.drawings + self.drawings_step)

    def apply_mask_to_tensor_update(self, tensor_update):
        if self.mask_data is not None:
            tensor_update[self.mask_data] = 1

    def get_current_reconstruction(self):
        """
        Returns the current approximation of the observed tensor.
        """
        raise NotImplementedError

    def tensor_has_energy(self):
        return True


class PosObserver(_Observer):
    """
    Class for nonnegative observations.
    """

    def _initialization(self):
        pass

    def _give_update(self, parent, out=None):
        parent_tensor = parent.get_tensor_for_children(self)

        if out is None:
            tensor_update = glob.xp.empty_like(self.tensor)
        else:
            tensor_update = out

        denominator = parent_tensor + self.is_tensor_null
        if self._inference_mode == "VBEM":
            # IN VBEM mode, underflow might happen, leading to null parent_tensor
            denominator += parent_tensor == 0

        tensor_update[...] = self.get_current_mult_factor() * self.tensor / denominator

        self.update_drawings()
        self.apply_mask_to_tensor_update(tensor_update)

        return tensor_update

    def get_current_reconstruction(self, parent, force_numpy=False):
        tensor_to_give = (
            parent.get_tensor_for_children(self) / self.get_current_mult_factor()
        )
        if force_numpy and utils.infer_backend(tensor_to_give) == glob.CUPY:
            return glob.xp.asnumpy(tensor_to_give)
        return tensor_to_give

    def get_kl_divergence(self):
        """
        Returns the kullback-Leibler divergence between observer tensor and the
        current reconstruction.
        """
        kl_div = 0
        for parent in self.list_of_parents:
            reconstruction = self.get_current_reconstruction(parent)
            my_tensor = self.tensor
            kl_div -= utils.xlogy(my_tensor, reconstruction).sum()
            kl_div += reconstruction.sum()
            kl_div -= my_tensor.sum() - utils.xlogy(my_tensor, my_tensor).sum()
        return float(kl_div)

    def _get_data_fitting(self):
        """
        Returns minus log-likelihood of Poisson distribution
        """
        lh = glob.xp.zeros_like(self.tensor)
        for parent in self.list_of_parents:
            my_tensor = self.get_current_mult_factor() * self.tensor
            parent_tensor = parent.get_tensor_for_children(self)
            if self._inference_mode == "EM":
                lh -= parent_tensor
            elif self._inference_mode == "VBEM":
                lh -= parent._get_mean_tensor_for_VBEM(self)
            lh += utils.xlogy(my_tensor, parent_tensor)
            lh -= glob.sps.gammaln(my_tensor + 1)
        if self.mask_data is not None:
            lh[self.mask_data] = 0
        return -float(lh.sum())


class RealObserver(_Observer):
    """
    Class for real observations
    """

    def __init__(self, **kwargs):
        self.limit_skellam_update = kwargs.pop("limit_skellam_update", True)
        super().__init__(**kwargs)

    @cached_property
    def abs_tensor(self):
        return glob.xp.abs(self.tensor)

    @cached_property
    def abs_tensor_plus_1(self):
        return self.abs_tensor + 1

    @cached_property
    def abs_tensor_power2(self):
        return self.abs_tensor ** 2

    def _initialization(self):

        self.tensor_update_dict = {}
        for parent in self.list_of_parents:
            self.tensor_update_dict[parent] = glob.xp.ones_like(
                parent.get_tensor_for_children(self)
            )

    def get_current_reconstruction(self, parent, force_numpy=False):
        parent_tensor = parent.get_tensor_for_children(self)
        real_parent_tensor = parent_tensor[..., 0] - parent_tensor[..., 1]
        tensor_to_give = real_parent_tensor / self.get_current_mult_factor()
        if force_numpy and utils.infer_backend(tensor_to_give) == glob.CUPY:
            return glob.xp.asnumpy(tensor_to_give)
        return tensor_to_give

    def _get_data_fitting(self):
        """
        Returns minus log-likelihood of either Skellam distribution or extended
        real KL divergence.
        """
        lh = glob.xp.zeros_like(self.tensor)
        for parent in self.list_of_parents:
            mult_fact = self.get_current_mult_factor()
            parent_tensor = parent.get_tensor_for_children(self)
            if self._inference_mode == "EM":
                lh -= parent_tensor.sum(-1)
            elif self._inference_mode == "VBEM":
                lh -= parent._get_mean_tensor_for_VBEM(self).sum(-1)
            abs_tensor = mult_fact * self.abs_tensor
            inside_log = (
                parent_tensor[..., 0] * self.is_tensor_pos
                + parent_tensor[..., 1] * glob.xp.logical_not(self.is_tensor_pos)
                + self.is_tensor_null
            )
            lh += utils.xlogy(abs_tensor, inside_log)
            if not self.limit_skellam_update:
                abs_tensor += 1
                lh += utils.hyp0f1ln(
                    abs_tensor, parent_tensor[..., 0] * parent_tensor[..., 1]
                )
                lh -= glob.sps.gammaln(abs_tensor)
            else:
                temp_calculus = self.temp_calculus(parent)
                lh += temp_calculus
                lh -= utils.xlogy(
                    abs_tensor, (abs_tensor + temp_calculus + self.is_tensor_null) / 2
                )
        if self.mask_data is not None:
            lh[self.mask_data] = 0
        return -float(lh.sum())

    def temp_calculus(self, parent):
        parent_tensor = parent.get_tensor_for_children(self)
        model_param_prod = parent_tensor[..., 0] * parent_tensor[..., 1]
        temp = (
            4 * model_param_prod
            + self.get_current_mult_factor() ** 2 * self.abs_tensor_power2
        ) ** 0.5
        return temp

    def temp_calculus_skellam(self, parent):
        parent_tensor = parent.get_tensor_for_children(self)
        model_param_prod = parent_tensor[..., 0] * parent_tensor[..., 1]
        temp = utils.bessel_ratio(
            self.get_current_mult_factor() * self.abs_tensor + 1,
            2 * (model_param_prod ** 0.5),
            1e-16,
        )
        return temp

    def _give_update(self, parent, out=None):

        # model
        parent_tensor = parent.get_tensor_for_children(self)
        mult_fact = self.get_current_mult_factor()

        if out is None:
            tensor_update = glob.xp.empty_like(parent_tensor)
        else:
            tensor_update = out

        # compute tensor_update
        mask = parent_tensor > 0
        mask[..., 0] &= self.is_tensor_pos
        mask[..., 1] &= glob.xp.logical_not(self.is_tensor_pos)

        if not self.limit_skellam_update:
            temp_calculus = self.temp_calculus_skellam(parent)
            tensor_update[...] = (2 * parent_tensor[..., ::-1]) / (
                2 * (1 + mult_fact * self.abs_tensor[..., None])
                + temp_calculus[..., None]
            )
        else:
            temp_calculus = self.temp_calculus(parent)
            is_temp_calculus_null = temp_calculus == 0
            tensor_update[...] = (2 * parent_tensor[..., ::-1]) / (
                mult_fact * self.abs_tensor[..., None]
                + temp_calculus[..., None]
                + is_temp_calculus_null[
                    ..., None
                ]  # to avoid x/0, in which case value of tensor_update is not important
            )
        tensor_update += (mult_fact * self.abs_tensor[..., None] * mask) / (
            parent_tensor + ~mask
        )

        # heuristic (for now) in order to improve convergence rate
        self.tensor_update_dict[parent] = tensor_update

        self.update_drawings()
        self.apply_mask_to_tensor_update(self.tensor_update_dict[parent])

        return self.tensor_update_dict[parent]


class BlindObs(core_nodes._ChildNode, core_nodes._ParentNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _give_update(self, parent, out=None):
        if out is None:
            return glob.xp.ones_like(parent.get_tensor_for_children(self))
        out[...] = 1.0

        return out

    @staticmethod
    def _get_data_fitting():
        return 0
