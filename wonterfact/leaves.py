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

"""Module for all leave classes"""

# Python System imports
from functools import cached_property

# Third-party imports
import numpy as np
from methodtools import lru_cache

# Relative imports
from . import utils, core_nodes
from .glob_var_manager import glob


class _Leaf(core_nodes._DynNodeData):
    """
    Mother class for the leaves of a graphical model, i.e. tensors to estimate in a factorization model.
    """

    def __init__(
        self,
        prior_shape=1.0,
        constraint_coeffs=None,
        constraint_type="inequality",
        constraint_max_iter=10,
        prior_accelerator=None,
        learn_prior_alpha=False,
        **kwargs
    ):
        """
        Parameters
        ----------
        prior_shape: array_like or float, optional, default 1
            Shape hyperparameter of the prior distribution.
        constraint_coeffs: array_like or None, optional, default None
            If None, no constraint is applied. Otherwise, coefficients for
            linear equality (resp. inequality) constraints. Inner tensor will
            always comply to the constraint `(self.tensor *
            self.constraint_coeffs).sum(axis_to_sum) == 0` (resp `>=0`)
            where `axis_to_sum` are the `ndim` last axes of `self.tensor` and
            `ndim=self.constraint_coeffs.ndim`.
        constraint_type: 'equality' or 'inequality', optional, default 'inequality'
            Defines the type of linear constraint to apply if
            `constraint_coeffs` is provided.
        constraint_max_iter: int, optional, default 10
            Maximum number of iterations for the inner algorithm used to comply
            to the linear constraints. The greater it is, the more precise it is
            but also the slower.
        prior_accelerator: float or None, optional, default None
            Allows to give more importance to the priors without having to
            change their value. Useful for accelerating sparsity prior in the
            'VBEM' mode, when `prior_shape < 1`
        learn_prior_alpha: bool, optional, default False
            Whether the shape hyperparameter of the prior distribution should be
            learned or not (only in the 'VBEM' mode, only available for
            LeafDirichlet class for now).
        """
        self.prior_shape = glob.xp.array(prior_shape, dtype=glob.float)
        self.constraint_coeffs = constraint_coeffs
        self.constraint_type = constraint_type
        self.constraint_max_iter = constraint_max_iter
        self.prior_accelerator = prior_accelerator
        self.learn_prior_alpha = learn_prior_alpha
        if self.constraint_coeffs is not None:
            self.constraint_coeffs = glob.xp.array(
                self.constraint_coeffs, dtype=glob.float
            )
        self.min_val = 1e-100
        self._set_inference_mode()
        super().__init__(**kwargs)

    def _clip_tensor_min_value(self):
        utils.clip_inplace(self.tensor, a_min=self.min_val, backend=glob.backend)

    def _initialization(self):

        if self.update_period != 0:
            self.tensor_update = glob.xp.ones_like(self.tensor)
            if self._inference_mode == "VBEM":
                # to be sure prior_shape has good dimension
                # (useful for hyperparameters learning)
                if self.tensor.ndim != 0:
                    self.prior_shape = (
                        glob.xp.zeros((1,) * self.tensor.ndim, dtype=glob.float)
                        + self.prior_shape
                    )
                self.alpha_estim = glob.xp.zeros_like(self.tensor)

    def _update_tensor(self, update_type="regular", update_param=None):
        # update rules are the same in VBEM and EM mode
        if update_type == "no_update_for_leaves":
            pass
        elif update_type == "just_normalize":
            self._normalize_tensor()
        elif update_type == "parabolic":
            self._parabolic_update(parabolic_param=update_param)
        elif update_type == "regular":
            self._regular_update_tensor()
        else:
            raise ValueError("Unknown `update_type`")

    def _normalize_tensor(self):
        raise NotImplementedError()

    def _regular_update_tensor(self):
        if self.prior_accelerator is not None:
            self.tensor[...] = self.tensor / self.prior_accelerator

        if self._inference_mode == "EM":
            self.tensor *= self.tensor_update
            if not self._prior_alpha_all_one:
                self.tensor += self._prior_shape_minus_one
            if self._might_need_clipping:
                self._clip_tensor_min_value()

        elif self._inference_mode == "VBEM":
            self.alpha_estim[...] = self.tensor * self.tensor_update
            self.alpha_estim += self.prior_shape

        self._normalize_tensor()

    def _reinit_tensor_values(self):
        # initialization corresponds to regular update when
        # self.tensor_update[...] = 0
        self.tensor_update[...] = 0
        self._regular_update_tensor()

    def _set_bezier_point(self, param):
        if glob.processor == glob.GPU:
            utils.xp_utils.get_cupy_utils(glob.backend)._set_bezier_point(
                self._past_tensor[0],
                self._past_tensor[1],
                self._past_tensor[2],
                param,
                self.tensor,
            )
        else:
            self.tensor[...] = (
                (1 - param) ** 2 * self._past_tensor[0]
                + 2 * (1 - param) * param * self._past_tensor[1]
                + (param ** 2) * self._past_tensor[2]
            )

    def _parabolic_update(self, parabolic_param):
        raise NotImplementedError

    def _update_past_tensors(self):
        if not hasattr(self, "_past_tensor"):
            self._past_tensor = [glob.xp.empty_like(self.tensor) for __ in range(3)]
        self._past_tensor[0][...] = self.tensor.copy()
        self._past_tensor = self._past_tensor[1:] + self._past_tensor[:1]

    @cached_property
    def _might_need_clipping(self):
        return (self.prior_shape < 1).any() or self._prior_alpha_all_one

    @cached_property
    def _prior_shape_minus_one(self):
        return self.prior_shape - 1

    @cached_property
    def _prior_alpha_all_one(self):
        return (self.prior_shape == 1).all()


class LeafGamma(_Leaf):
    """
    Class for gamma leaves of a graphical model, i.e. non-normalized tensors to
    estimate in a factorization model.
    """

    def __init__(
        self,
        prior_rate=None,
        max_energy=None,
        total_max_energy=None,
        learn_prior_beta=False,
        **kwargs
    ):
        """
        Returns a LeafGamma object, corresponding to non-normalized tensors.

        Parameters
        ----------
        prior_rate: array_like or float or None, optional, default None
            Rate hyperparameter of the prior Gamma distribution. Must be
            strictly greater than 0. If None, no prior on the inner tensor.
        max_energy: float or None, optional, default None
            Maximum value that each coefficient of inner tensor can take. If
            None, no maximum value.
        total_max_energy: float or None, optional, default None
            Maximum value that the sum of all coefficient of inner tensor can
            take. If None, no maximum value.
        learn_prior_beta: bool, optional, default False
            Whether the rate parameter of the prior Gamma distribution should be
            learned or not (only in the 'VBEM' mode, not yet available for now).
        """
        prior_rate = 0.0 if prior_rate is None else prior_rate
        self.prior_rate = glob.xp.array(prior_rate, dtype=glob.float)
        if self.prior_rate.size > 1 and not (self.prior_rate > 0).all():
            raise ValueError("prior_rate, if not None, must be > 0")
        self.max_energy = max_energy
        self.total_max_energy = total_max_energy
        self.learn_prior_beta = learn_prior_beta
        super().__init__(**kwargs)

    def _initialization(self):
        super()._initialization()
        if self.update_period != 0 and self._inference_mode == "VBEM":
            # to be sure hyperparameters has good dimension (usefull for learning):
            self.prior_rate = (
                glob.xp.zeros((1,) * self.tensor.ndim, dtype=glob.float)
                + self.prior_rate
            )
        if self.update_period != 0:
            self._reinit_tensor_values()

    def _update_tensor(self, update_type="regular", update_param=None):
        super()._update_tensor(update_type=update_type, update_param=update_param)

    def _parabolic_update(self, parabolic_param):
        self._set_bezier_point(parabolic_param)

        self._clip_tensor_min_value()

        self._clip_max_energy()

    def _clip_max_energy(self):
        if self.max_energy is not None:
            utils.clip_inplace(self.tensor, a_max=self.max_energy, backend=glob.backend)
        if self.total_max_energy is not None:
            total_energy = self.tensor.sum()
            if total_energy > self.total_max_energy:
                self.tensor *= self.total_max_energy / total_energy

    def _normalize_tensor(self):
        den = 1 + self.prior_rate
        if self._inference_mode == "VBEM":
            self.tensor[...] = utils.exp_digamma(self.alpha_estim)
        if self.constraint_coeffs is not None:
            sigma = utils._find_equality_root(
                self.tensor,
                den,
                self.constraint_coeffs,
                self.constraint_max_iter,
                type=self.constraint_type,
                atol=1e-10,
            )
            self.tensor /= den - sigma * self.constraint_coeffs
        else:
            self.tensor /= den
        if self.prior_accelerator is not None:
            self.tensor *= self.prior_accelerator
        self._clip_max_energy()

    @cached_property
    def _cst_prior_value(self):
        if self._inference_mode == "EM":
            if self.prior_shape.any() and self.prior_rate.any():
                prior_shape = (
                    glob.xp.zeros(self.tensor.shape, dtype=glob.float)
                    + self.prior_shape
                )
                cst_prior = utils.xlogy(
                    prior_shape, self.prior_rate
                ) - glob.sps.gammaln(prior_shape)
            else:
                return 0
        elif self._inference_mode == "VBEM":
            prior_shape = (
                glob.xp.zeros(self.tensor.shape, dtype=glob.float) + self.prior_shape
            )
            cst_prior = utils.xlogy(
                prior_shape, self.prior_rate / (self.prior_rate + 1)
            ) - glob.sps.gammaln(prior_shape)
        if self.prior_accelerator is not None:
            cst_prior *= self.prior_accelerator
        return float(cst_prior.sum())  # float is when cupy is used

    def _prior_value(self):
        if self.update_period == 0:
            return 0
        if self._inference_mode == "EM":
            if self._prior_alpha_all_one and (self.prior_rate == 0).all():  # no prior
                return 0
            tensor = self.tensor
            if self.prior_accelerator is not None:
                tensor = tensor / self.prior_accelerator
            prior_val = (
                utils.xlogy(self._prior_shape_minus_one, tensor)
                - self.prior_rate * tensor
            )
        elif self._inference_mode == "VBEM":
            prior_val = (
                -(
                    glob.xp.log(utils.exp_digamma(self.alpha_estim))
                    * (self.alpha_estim - self.prior_shape)
                )
                + glob.sps.gammaln(self.alpha_estim)
                + self.alpha_estim / (self.prior_rate + 1)
            )
        if self.prior_accelerator is not None:
            prior_val *= self.prior_accelerator
        return (
            float(prior_val.sum()) + self._cst_prior_value
        )  # float is when cupy is used

    def _bump(self):
        if self._inference_mode == "EM":
            posterior_alpha = self.tensor * self.tensor_update + self.prior_shape
            posterior_beta = (
                glob.xp.ones(self.tensor.shape, dtype=glob.float) + self.prior_rate
            ).clip(1e-10, None)
        elif self._inference_mode == "VBEM":
            posterior_alpha = self.alpha_estim
            posterior_beta = (
                glob.xp.ones(self.tensor.shape, dtype=glob.float) + self.prior_rate
            )
        self.tensor[...] = glob.xp.random.gamma(posterior_alpha, 1.0 / posterior_beta)

    def _prior_alpha_update(self, n_iter=1):
        if self.learn_prior_alpha:
            raise NotImplementedError

    def _prior_beta_update(self, n_iter=1):
        if self.learn_prior_beta:
            raise NotImplementedError

    @property
    def tensor_has_energy(self):
        return True

    @cached_property
    def norm_axis(self):
        return None

    @lru_cache(maxsize=1)
    def _get_raw_mean_tensor_for_VBEM(self, current_iter):
        if self.update_period == 0:
            raw_tensor = self.tensor
        else:
            raw_tensor = self.alpha_estim / (self.prior_rate + 1)
        return raw_tensor


class LeafDirichlet(_Leaf):
    """
    Class for the Dirichlet leaves of a graphical model, i.e. normalized
    tensors to estimate in a factorization model.
    """

    def __init__(self, norm_axis=(), **kwargs):
        """
        Returns a LeafDirichlet object, corresponding to a normalized tensor.

        Parameters
        ----------
        norm_axis: sequence of int, optional, default ()
            Normalization axis for inner tensor, such that
            `self.tensor.sum(norm_axis) == 1` is all True. Must be the last axes
            of tensor.
        prior_shape: array_like or float, optional, default 1
            Shape hyperparameter of the Dirichlet prior distribution.
        """
        self._norm_axis = norm_axis
        super().__init__(**kwargs)

    @property
    def norm_axis(self):
        return self._norm_axis

    def _initialization(self):
        super()._initialization()
        if self.update_period != 0:
            self._reinit_tensor_values()
        else:
            if not glob.xp.allclose(self.tensor.sum(axis=self.norm_axis), 1):
                raise ValueError(
                    "Please provide a well normalized tensor if update_period is 0"
                )

    def _normalize_tensor(self, **kwargs):
        if self._inference_mode == "EM" or self.update_period == 0:
            norm_tensor = self.tensor.sum(axis=self.norm_axis, keepdims=True)
            if self.constraint_coeffs is not None:
                sigma = utils._find_equality_root(
                    self.tensor,
                    norm_tensor,
                    self.constraint_coeffs,
                    self.constraint_max_iter,
                    type=self.constraint_type,
                    atol=1e-10,
                )
                self.tensor /= norm_tensor - sigma * self.constraint_coeffs
            else:
                self.tensor /= norm_tensor
        elif self._inference_mode == "VBEM":
            if self.constraint_coeffs is not None:
                raise NotImplementedError
            else:
                norm_tensor = self.alpha_estim.sum(axis=self.norm_axis, keepdims=True)
                self.tensor[...] = utils.exp_digamma(
                    self.alpha_estim
                ) / utils.exp_digamma(norm_tensor)

    def _parabolic_update(self, parabolic_param):
        self._set_bezier_point(parabolic_param)
        if (self.tensor <= self.min_val).any():
            self._clip_tensor_min_value()
            self._normalize_tensor()

    def _prior_alpha_update(self, n_iter=1):
        if self.learn_prior_alpha:
            size_norm_axis = np.array(
                [self.tensor.shape[axis] for axis in self.norm_axis]
            ).prod()
            if size_norm_axis > 1:
                sum_axis = tuple(
                    ii
                    for ii in range(self.tensor.ndim - len(self.norm_axis))
                    if self.tensor.shape[ii] != self.prior_shape.shape[ii]
                )

                denominator = -(
                    glob.xp.log(self.tensor).mean(axis=self.norm_axis, keepdims=True)
                ).mean(axis=sum_axis, keepdims=True)
                for __ in range(n_iter):
                    update = (
                        glob.xp.log(
                            utils.exp_digamma(size_norm_axis * self.prior_shape)
                        )
                        - glob.xp.log(utils.exp_digamma(self.prior_shape))
                    ) / denominator
                    self.prior_shape *= update
                    self.prior_shape = self.prior_shape.clip(0.01, None)
                    # TODO: eventually compute directly from self.alpha_estim

    def _prior_alpha_online_update(
        self, memory_weight=0, memory_sufficient_stat=None, n_iter=100, max_weight=None
    ):
        # TODO : check for the new way prior_shape is implemented
        if memory_sufficient_stat is None:
            memory_sufficient_stat = glob.xp.zeros(
                self.prior_shape.shape, dtype=glob.float
            )

        current_obs_weight = self.tensor.size - self.prior_shape.size
        memory_sufficient_stat *= memory_weight
        memory_weight += current_obs_weight

        sum_axis = [
            ii
            for ii in range(self.tensor.ndim)
            if self.tensor.shape[ii] != self.prior_shape.shape[ii]
        ]

        sufficient_stat = (
            memory_sufficient_stat
            - glob.xp.log(self.tensor).sum(axis=sum_axis, keepdims=True)
        ) / memory_weight

        for __ in range(n_iter):
            numerator = glob.xp.log(
                utils.exp_digamma(
                    self.prior_shape.sum(axis=self.norm_axis, keepdims=True)
                )
                / utils.exp_digamma(self.prior_shape)
            )
            self.prior_shape *= numerator / sufficient_stat

        memory_weight = min(memory_weight, max_weight or memory_weight)

        return memory_weight, memory_sufficient_stat

    @cached_property
    def _cst_prior_value(self):
        if self._inference_mode == "EM":
            if self._prior_alpha_all_one:  # no prior
                return 0
            prior_shape = (
                glob.xp.zeros(self.tensor.shape, dtype=glob.float) + self.prior_shape
            )
            cst_prior = (
                glob.sps.gammaln(prior_shape)
                - glob.sps.gammaln((prior_shape).mean(self.norm_axis, keepdims=True))
            ).sum()
        elif self._inference_mode == "VBEM":
            prior_shape = (
                glob.xp.zeros(self.tensor.shape, dtype=glob.float) + self.prior_shape
            )
            cst_prior = (
                glob.sps.gammaln(prior_shape.mean(self.norm_axis, keepdims=True)).sum()
                - glob.sps.gammaln(prior_shape).sum()
            )
        if self.prior_accelerator is not None:
            cst_prior *= self.prior_accelerator
        # return cst_prior.sum()
        return float(cst_prior)

    def _prior_value(self):
        if self.update_period == 0 or self.norm_axis == ():
            return 0
        if self._inference_mode == "EM":
            if self._prior_alpha_all_one:
                return 0
            prior_val = utils.xlogy(self._prior_shape_minus_one, self.tensor).sum()
        elif self._inference_mode == "VBEM":
            prior_shape = (
                glob.xp.zeros(self.tensor.shape, dtype=glob.float) + self.prior_shape
            )
            prior_val = (
                glob.sps.gammaln(self.alpha_estim).sum()
                - (
                    glob.sps.gammaln(
                        self.alpha_estim.sum(self.norm_axis, keepdims=True)
                    )
                ).sum()
            )
            prior_val -= (
                (self.alpha_estim - prior_shape)
                * (
                    glob.xp.log(utils.exp_digamma(self.alpha_estim))
                    - glob.xp.log(
                        utils.exp_digamma(
                            self.alpha_estim.sum(self.norm_axis, keepdims=True)
                        )
                    )
                )
            ).sum()
        if self.prior_accelerator is not None:
            prior_val *= self.prior_accelerator
        return float(prior_val) + self._cst_prior_value

    def _bump(self):
        if self._inference_mode == "EM":
            posterior_alpha = self.tensor * self.tensor_update + self.prior_shape
        elif self._inference_mode == "VBEM":
            posterior_alpha = self.alpha_estim
        self.tensor[...] = glob.xp.random.gamma(
            posterior_alpha, glob.xp.ones(self.tensor.shape, dtype=glob.float)
        )
        norm_tensor = self.tensor.sum(axis=self.norm_axis, keepdims=True)
        self.tensor /= norm_tensor

    def get_l2_norm(self, **kwargs):
        """
        Returns the l2 norm of each distribution contained in the tensor.
        """
        tensor = self.tensor.reshape(self.tensor.shape[: self.norm_axis[0]] + (-1,))
        return glob.xp.linalg.norm(tensor, ord=2, axis=-1, **kwargs)

    @cached_property
    def tensor_has_energy(self):
        return False

    @lru_cache(maxsize=1)
    def _get_raw_mean_tensor_for_VBEM(self, current_iter):
        if self.update_period == 0:
            raw_tensor = self.tensor
        else:
            norm_tensor = self.alpha_estim.sum(axis=self.norm_axis, keepdims=True)
            raw_tensor = self.alpha_estim / norm_tensor
        return raw_tensor


class LeafGammaNorm(LeafGamma):
    """
    Special leaf with non-normalized inner tensor with an ad hoc prior whose goal
    is to minimize mix between l1 and l2 norm. Only works in 'EM' mode.
    """

    def __init__(self, l2_norm_axis=Ellipsis, n_iter_norm2=10, **kwargs):
        """
        Returns a LeafGammaNorm object, for non-normalized tensor with ad hoc prior
        based on a mix l1/l2 norm.

        Parameters
        ----------
        l2_norm_axis: sequence of int or Ellipsis, optional, default Ellispsis
            Axes of inner tensor on which the l2 norm is computed. Must be the
            last axe(s) of inner tensor.
        n_iter_norm2: int, optional, default 10
            Fixed number of iteration for the nested fixed-point algorithm used
            to resolve the maximization step of the EM algorithm.

        Notes
        -----
        `prior_shape` cannot be different from 1 for this class (it should work
        but it has not be be proven).
        """
        self.l2_norm_axis = l2_norm_axis
        self.n_iter_norm2 = n_iter_norm2
        super().__init__(**kwargs)
        self.previous_tensor_update = None

    def _update_tensor(self, update_type="regular", update_param=None):
        super()._update_tensor(update_type=update_type, update_param=update_param)

    def _normalize_tensor(self):
        if self._inference_mode == "EM":
            if self.constraint_coeffs is not None:
                raise NotImplementedError
            if glob.processor == glob.CPU:
                self._normalize_l1_l2_tensor(
                    self.tensor,
                    self.l2_norm_axis,
                    self.n_iter_norm2,
                    self.prior_rate_as_float,
                )
                # tensor_init = self.tensor.copy()
                # for __ in range(self.n_iter_norm2):
                #     norm22 = (self.tensor ** 2).sum(self.l2_norm_axis, keepdims=True)
                #     norm2 = norm22 ** 0.5
                #     delta = norm22 + 4 * self.prior_rate * tensor_init * norm2
                #     self.tensor[...] = (-norm2 + delta ** 0.5) / (2 * self.prior_rate)
            elif glob.processor == glob.GPU:
                tensor = self.tensor.reshape(
                    (-1, np.prod([self.tensor.shape[ii] for ii in self.l2_norm_axis]))
                )
                # pylint: disable=unsubscriptable-object
                utils.xp_utils.get_cupy_utils(
                    glob.backend
                ).normalize_l1_l2_tensor_numba_core[
                    tensor.shape[0], (tensor.shape[1] + 1) // 2
                ](
                    tensor, self.n_iter_norm2, self.prior_rate_as_float
                )
                # pylint: enable=unsubscriptable-object
        elif self._inference_mode == "VBEM":
            raise NotImplementedError
        if self.prior_accelerator is not None:
            self.tensor[...] = self.tensor * self.prior_accelerator
        self._clip_max_energy()

    @staticmethod
    def _normalize_l1_l2_tensor(tensor, l2_norm_axis, n_iter, prior_rate):
        tensor_init = tensor.copy()
        for __ in range(n_iter):
            norm22 = (tensor ** 2).sum(l2_norm_axis, keepdims=True)
            norm2 = norm22 ** 0.5
            delta = norm22 + 4 * prior_rate * tensor_init * norm2
            tensor[...] = (-norm2 + delta ** 0.5) / (2 * prior_rate)

    @cached_property
    def prior_rate_as_float(self):
        if self.prior_rate.size != 1:
            raise NotImplementedError
        return float(self.prior_rate)

    @cached_property
    def _cst_prior_value(self):
        # The normalization constant of such an ad hoc prior is unknown. We then
        # return 0.
        return 0.0

    def _prior_value(self):
        if self.update_period == 0:
            return 0
        if self._inference_mode == "EM":
            if self._prior_alpha_all_one and (self.prior_rate == 0).all():  # no prior
                return 0
            tensor = self.tensor
            if self.prior_accelerator is not None:
                tensor = tensor / self.prior_accelerator
            prior_val = utils.xlogy(self._prior_shape_minus_one, tensor)
            norm2 = (tensor ** 2).sum(self.l2_norm_axis, keepdims=True) ** 0.5
            prior_val -= (
                self.prior_rate
                * norm2
                / np.prod([self.tensor.shape[ii] for ii in self.l2_norm_axis])
            )
        elif self._inference_mode == "VBEM":
            raise NotImplementedError
        if self.prior_accelerator is not None:
            prior_val *= self.prior_accelerator
        return float(prior_val.sum()) + self._cst_prior_value
