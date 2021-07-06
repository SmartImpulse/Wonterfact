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
from . import utils, core_nodes, buds
from .glob_var_manager import glob


class _Leaf(core_nodes._DynNodeData, core_nodes._ChildNode):
    """
    Mother class for the parameter leaves of a graphical model, i.e. tensors to
    estimate in a factorization model.
    """

    def __init__(
        self,
        init_type="custom",
        prior_shape=None,
        constraint_coeffs=None,
        constraint_type="inequality",
        constraint_max_iter=10,
        prior_accelerator=None,
        **kwargs
    ):
        """
        Parameters
        ----------
        init_type: 'custom', 'prior', or 'random', default 'custom'
            If 'prior', initialization is defined by the prior distribution
            (mode of prior in EM mode and 'exp(mean of sufficient statistic)' in
            VBEM mode); if 'custom', inner tensor is initialized by the user via
            the 'tensor' attribute; if 'random', inner tensor is randomly
            initialized.
        prior_shape: array_like or float or None, optional, default None
            Shape hyperparameter of the prior distribution. If not None, automatically
            creates a BudShape node and links it to the returned leaf.
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
        """
        self.init_type = init_type
        self.constraint_coeffs = constraint_coeffs
        self.constraint_type = constraint_type
        self.constraint_max_iter = constraint_max_iter
        self.prior_accelerator = prior_accelerator
        if self.constraint_coeffs is not None:
            self.constraint_coeffs = glob.xp.array(
                self.constraint_coeffs, dtype=glob.float
            )
        self._set_inference_mode()
        super().__init__(**kwargs)
        if prior_shape is not None:
            self._create_bud_parent(prior_shape, type="shape")

    @property
    def min_val(self):
        if self._inference_mode == "EM":
            return 1e-20
        elif self._inference_mode == "VBEM":
            return 0.02

    @property
    def level(self):
        """
        Returns 1 , which is the default level for of a leaf.

        Returns
        ------
        int
        """
        return 1

    def _create_bud_parent(self, prior_val, type="shape"):
        """
        Automatically creates a BudShape (if type="shape") or BudRate (if type=
        "rate") parent and link to self.
        """
        prior_val = glob.xp.array(prior_val, dtype=glob.float)
        ndim = prior_val.ndim
        idx = []
        shape = []
        for num_dim in range(ndim):
            if (
                prior_val.shape[-num_dim - 1] != 1
                or prior_val.shape[-num_dim - 1] == self.tensor.shape[-num_dim - 1]
            ):
                idx.append(self.index_id[-num_dim - 1])
                shape.append(self.tensor.shape[-num_dim - 1])
        idx = tuple(idx[::-1])
        shape = tuple(shape[::-1])
        if isinstance(self.index_id, str):
            idx = "".join(idx)
        update_period = 1 if self.update_period else 0
        if type == "shape":
            bud = buds.BudShape(
                name="{}_{}".format(self.name, type),
                index_id=idx,
                tensor=prior_val.reshape(shape),
                update_period=update_period,
            )
        elif type == "rate":
            bud = buds.BudRate(
                name="{}_{}".format(self.name, type),
                index_id=idx,
                tensor=prior_val.reshape(shape),
                update_period=update_period,
            )
        else:
            raise ValueError("type must be either 'rate' or 'shape'")
        bud.new_child(self)

    @property
    def prior_shape(self):
        if self.shape_parent is not None:
            return self._get_prior_arr(self.shape_parent)
        else:
            return glob.xp.array(1.0)

    def _get_prior_arr(self, prior_parent):
        transpose, sl = utils.get_transpose_and_slice(
            prior_parent.get_index_id_for_children(self), self.index_id
        )
        return prior_parent.get_tensor_for_children(self).transpose(transpose)[sl]

    def _clip_tensor_min_value(self):
        if self._inference_mode == "VBEM":
            utils.clip_inplace(
                self.posterior_shape, a_min=self.min_val, backend=glob.backend
            )
        else:
            utils.clip_inplace(self.tensor, a_min=self.min_val, backend=glob.backend)

    def _initialization(self):

        if self.update_period != 0:
            self.tensor_update = glob.xp.empty_like(self.tensor)
            if self._inference_mode == "VBEM":
                self.posterior_shape = glob.xp.zeros_like(self.tensor)

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
            self.posterior_shape[...] = self.tensor * self.tensor_update
            self.posterior_shape += self.prior_shape

        self._normalize_tensor()

    def _reinit_tensor_values(self, init_type=None):
        init_type = init_type or self.init_type
        if init_type == "random":
            if self._inference_mode == "EM":
                tensor = self.tensor
            elif self._inference_mode == "VBEM":
                tensor = self.posterior_shape
            tensor[...] = 1 + glob.xp.random.rand(*tensor.shape).astype(tensor.dtype)
            self._normalize_tensor()
        elif init_type == "prior":
            if self._inference_mode == "EM":
                # one takes self.prior_shape / (1 + self.prior_rate)
                self.tensor[...] = 0
                self.tensor += self.prior_shape
                self._normalize_tensor()
            if self._inference_mode == "VBEM":
                # initialization corresponds to regular update when
                # self.tensor_update[...] = 0
                self.tensor_update[...] = 0
                self._regular_update_tensor()
        elif init_type == "custom":
            if self._inference_mode == "VBEM":
                self.posterior_shape[...] = self.tensor * 1e5

    def _set_bezier_point(self, param):
        if self._inference_mode == "EM":
            tensor = self.tensor
        elif self._inference_mode == "VBEM":
            tensor = self.posterior_shape
        if glob.processor == glob.GPU:
            utils.xp_utils.get_cupy_utils(glob.backend)._set_bezier_point(
                self._past_tensor[0],
                self._past_tensor[1],
                self._past_tensor[2],
                param,
                tensor,
            )
        else:
            tensor[...] = (
                (1 - param) ** 2 * self._past_tensor[0]
                + 2 * (1 - param) * param * self._past_tensor[1]
                + (param ** 2) * self._past_tensor[2]
            )

    def _parabolic_update(self, parabolic_param):
        raise NotImplementedError

    def _update_past_tensors(self):
        if not hasattr(self, "_past_tensor"):
            self._past_tensor = [glob.xp.empty_like(self.tensor) for __ in range(3)]
        if self._inference_mode == "EM":
            self._past_tensor[0][...] = self.tensor.copy()
        elif self._inference_mode == "VBEM":
            self._past_tensor[0][...] = self.posterior_shape.copy()
        else:
            raise ValueError("unknown inference mode")
        self._past_tensor = self._past_tensor[1:] + self._past_tensor[:1]

    @cached_property
    def _might_need_clipping(self):
        return (self.prior_shape <= 1).any()

    @cached_property
    def _prior_shape_minus_one(self):
        return self.prior_shape - 1

    @cached_property
    def _prior_alpha_all_one(self):
        return (self.shape_parent is None) or (self.prior_shape == 1).all()

    @cached_property
    def shape_parent(self):
        return next(
            (
                parent
                for parent in self.list_of_parents
                if isinstance(parent, buds.BudShape)
            ),
            None,
        )

    def _give_update_alpha(self, parent, tensor, out=None):
        parent_idx_id = parent.get_index_id_for_children(self)
        update_tensor = utils.einsum(
            glob.xp.log(tensor), self.index_id, parent_idx_id, out=out
        )
        if out is None:
            return update_tensor

    def _give_update(self, parent, out):
        if isinstance(parent, buds.BudShape):
            ## returns quantity e_d (cf technical report)
            return self._give_update_alpha(parent, self.tensor, out=out)
        raise ValueError(
            "Class of parent argument must be either wonterfact.bubs.BudShape or wonterfact.bubs.BudRate"
        )

    def _give_number_of_users(self, parent, out=None):
        """
        Gives the number of parameters that share a same hyperparameter for each
        hyperparameter (corresponds to $|\\phi^{-1}(d)|$ in tech report)
        """
        parent_idx_id = parent.get_index_id_for_children(self)
        number_or_users = utils.einsum(
            glob.xp.ones_like(self.tensor), self.index_id, parent_idx_id, out=out
        )
        if out is None:
            return number_or_users

    def get_posterior_shape(self, force_numpy=False):
        """
        Return the posterior shape array. If force_numpy is true, the tensor is casted to
        numpy ndarray if needed (if cupy backend is used)
        """
        return self.cast_array(self.posterior_shape, force_numpy=force_numpy)

    def get_posterior_rate(self, force_numpy=False):
        """
        Return the posterior rate array. If force_numpy is true, the tensor is casted to
        numpy ndarray if needed (if cupy backend is used)
        """
        return self.cast_array(self.posterior_rate, force_numpy=force_numpy)


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
            Rate hyperparameter of the prior Gamma distribution. If None, no
            prior on the inner tensor. If not None, automatically creates a
            BudRate node and links it to the returned leaf. In this case, must
            be strictly greater than 0.
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
        self.max_energy = max_energy
        self.total_max_energy = total_max_energy
        self.learn_prior_beta = learn_prior_beta
        super().__init__(**kwargs)
        if prior_rate is not None:
            self._create_bud_parent(prior_rate, type="rate")

    def _initialization(self):
        super()._initialization()
        if self.update_period != 0:
            self._reinit_tensor_values()

    def _parabolic_update(self, parabolic_param):
        if self.constraint_coeffs is not None:
            raise NotImplementedError(
                "Parabolic update is not allowed yet with equality or inequality constraints"
            )
        # filling self.tensor (EM) or self.alpha_estim (VBEM) with new values
        self._set_bezier_point(parabolic_param)

        self._clip_tensor_min_value()

        self._clip_max_energy()

        # if VBEM, one need to recompute self.tensor
        if self._inference_mode == "VBEM":
            self._normalize_tensor()

    def _clip_max_energy(self):
        if self._inference_mode == "EM":
            tensor = self.tensor
        elif self._inference_mode == "VBEM":
            tensor = self.posterior_shape
        if self.max_energy is not None:
            utils.clip_inplace(tensor, a_max=self.max_energy, backend=glob.backend)
        if self.total_max_energy is not None:
            total_energy = tensor.sum()
            if total_energy > self.total_max_energy:
                tensor *= self.total_max_energy / total_energy

    def _normalize_tensor(self):
        if self._inference_mode == "VBEM":
            self.tensor[...] = utils.exp_digamma(self.posterior_shape)
        if self.constraint_coeffs is not None:
            sigma = utils._find_equality_root(
                self.tensor,
                self.posterior_rate,
                self.constraint_coeffs,
                self.constraint_max_iter,
                type=self.constraint_type,
                atol=1e-10,
            )
            self.tensor /= self.posterior_rate - sigma * self.constraint_coeffs
        else:
            self.tensor /= self.posterior_rate
        if self.prior_accelerator is not None:
            self.tensor *= self.prior_accelerator
        self._clip_max_energy()

    @property
    def posterior_rate(self):
        return 1 + self.prior_rate

    @cached_property
    def _cst_prior_value(self):
        if self._inference_mode == "EM":
            prior_shape = glob.xp.zeros_like(self.tensor) + self.prior_shape
            prior_rate = glob.xp.zeros_like(self.tensor) + self.prior_rate
            cst_prior = utils.xlogy(prior_shape, prior_rate)
            cst_prior = glob.xp.array(cst_prior)  # necessary in case cst_prior is float
            prior_rate = glob.xp.array(prior_rate)  # idem
            cst_prior[prior_rate == 0] = 0
            cst_prior -= glob.sps.gammaln(prior_shape)
        elif self._inference_mode == "VBEM":
            prior_shape = glob.xp.zeros_like(self.tensor) + self.prior_shape
            cst_prior = utils.xlogy(
                prior_shape, self.prior_rate / self.posterior_rate
            ) - glob.sps.gammaln(prior_shape)
        if self.prior_accelerator is not None:
            cst_prior *= self.prior_accelerator
        return cst_prior.sum().item()

    def _prior_value(self):
        if self.update_period == 0:
            return 0
        if self._inference_mode == "EM":
            if self._prior_alpha_all_one and self.rate_parent is None:  # no prior
                return 0
            tensor = self.tensor
            if self.prior_accelerator is not None:
                tensor = tensor / self.prior_accelerator
            if self.shape_parent is not None:
                prior_val = utils.xlogy(self._prior_shape_minus_one, tensor)
            else:
                prior_val = glob.xp.zeros_like(tensor)
            if self.rate_parent is not None:
                prior_val -= self.prior_rate * tensor
        elif self._inference_mode == "VBEM":
            prior_val = (
                -(
                    glob.xp.log(utils.exp_digamma(self.posterior_shape))
                    * (self.posterior_shape - self.prior_shape)
                )
                + glob.sps.gammaln(self.posterior_shape)
                # + self.alpha_estim * (1 - self.prior_rate / self.prior_rate_estim)
                # this last part is canceled by one of the fitting_data components of the observers
            )
        if self.prior_accelerator is not None:
            prior_val *= self.prior_accelerator
        return prior_val.sum().item() + self._cst_prior_value

    def _bump(self):
        if self._inference_mode == "EM":
            posterior_shape = self.tensor * self.tensor_update + self.prior_shape
        elif self._inference_mode == "VBEM":
            posterior_shape = self.posterior_shape
        posterior_rate = glob.xp.ones_like(self.tensor) + self.prior_rate
        self.tensor[...] = glob.xp.random.gamma(posterior_shape, 1.0 / posterior_rate)

    @property
    def tensor_has_energy(self):
        return True

    @cached_property
    def norm_axis(self):
        return None

    def _give_update_first_iteration(self, parent, out=None):
        if isinstance(parent, buds.BudRate):
            shape = np.zeros_like(self.posterior_shape) + self.prior_shape
            rate = self.prior_rate
            return self._give_update_beta(parent, shape, rate, out=out)
        if isinstance(parent, buds.BudShape):
            alpha = np.zeros_like(self.posterior_shape) + self.prior_shape
            tensor = utils.exp_digamma(alpha) / self.prior_rate
            return self._give_update_alpha(parent, tensor, out=out)
        raise ValueError(
            "Class of parent argument must be either wonterfact.bubs.BudShape or wonterfact.bubs.BudRate"
        )

    def _give_update(self, parent, out=None):
        if isinstance(parent, buds.BudRate):
            ## returns quantity m_e (cf technical report)
            return self._give_update_beta(
                parent, self.posterior_shape, self.posterior_rate, out=out
            )
        return super()._give_update(parent, out=out)

    def _give_update_beta(self, parent, shape, rate, out=None):
        parent_idx_id = parent.get_index_id_for_children(self)
        post_mean_tensor = shape / rate
        update_tensor = utils.einsum(
            post_mean_tensor, self.index_id, parent_idx_id, out=out
        )
        if out is None:
            return update_tensor

    def _give_update_bis(self, parent, out=None):
        parent_idx_id = parent.get_index_id_for_children(self)
        if isinstance(parent, buds.BudShape):
            prior_tensor = glob.xp.zeros_like(self.tensor) + glob.xp.log(
                self.prior_rate
            )
        elif isinstance(parent, buds.BudRate):
            prior_tensor = glob.xp.zeros_like(self.tensor) + self.prior_shape
        else:
            raise ValueError(
                "Class of parent argument must be either wonterfact.bubs.BudShape or wonterfact.bubs.BudRate"
            )
        update_tensor = utils.einsum(
            prior_tensor, self.index_id, parent_idx_id, out=out
        )
        if out is None:
            return update_tensor

    @cached_property
    def rate_parent(self):
        return next(
            (
                parent
                for parent in self.list_of_parents
                if isinstance(parent, buds.BudRate)
            ),
            None,
        )

    @property
    def prior_rate(self):
        if self.rate_parent is not None:
            return self._get_prior_arr(self.rate_parent)
        else:
            return glob.xp.array(0.0)


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

    def _reinit_tensor_values(self, init_type=None):
        init_type = init_type or self.init_type
        if init_type == "custom":
            if not glob.xp.allclose(self.tensor.sum(axis=self.norm_axis), 1):
                raise ValueError(
                    "Please provide a well normalized tensor for custom initialization"
                )
        super()._reinit_tensor_values(init_type=init_type)

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
                norm_tensor = self.posterior_shape.sum(
                    axis=self.norm_axis, keepdims=True
                )
                self.tensor[...] = utils.exp_digamma(
                    self.posterior_shape
                ) / utils.exp_digamma(norm_tensor)

    def _parabolic_update(self, parabolic_param):
        self._set_bezier_point(parabolic_param)
        if self._inference_mode == "EM":
            if (self.tensor <= self.min_val).any():
                self._clip_tensor_min_value()
                self._normalize_tensor()

        # if VBEM, one need to recompute self.tensor
        if self._inference_mode == "VBEM":
            self._clip_tensor_min_value()
            self._normalize_tensor()

    @cached_property
    def _cst_prior_value(self):
        if self._inference_mode == "EM":
            prior_shape = glob.xp.zeros_like(self.tensor) + self.prior_shape
            cst_prior = (
                glob.sps.gammaln((prior_shape).sum(self.norm_axis)).sum()
                - glob.sps.gammaln(prior_shape).sum()
            )
        elif self._inference_mode == "VBEM":
            prior_shape = (
                glob.xp.zeros(self.tensor.shape, dtype=glob.float) + self.prior_shape
            )
            cst_prior = (
                glob.sps.gammaln(prior_shape.sum(self.norm_axis, keepdims=True)).sum()
                - glob.sps.gammaln(prior_shape).sum()
            )
        if self.prior_accelerator is not None:
            cst_prior *= self.prior_accelerator
        return cst_prior.item()

    def _prior_value(self):
        if self.update_period == 0 or self.norm_axis == ():
            return 0
        if self._inference_mode == "EM":
            if self._prior_alpha_all_one:
                prior_val = np.array(0)
            else:
                prior_val = utils.xlogy(self._prior_shape_minus_one, self.tensor).sum()
        elif self._inference_mode == "VBEM":
            prior_shape = (
                glob.xp.zeros(self.tensor.shape, dtype=glob.float) + self.prior_shape
            )
            prior_val = (
                glob.sps.gammaln(self.posterior_shape).sum()
                - (
                    glob.sps.gammaln(
                        self.posterior_shape.sum(self.norm_axis, keepdims=True)
                    )
                ).sum()
            )
            prior_val -= (
                (self.posterior_shape - prior_shape)
                * (
                    glob.xp.log(utils.exp_digamma(self.posterior_shape))
                    - glob.xp.log(
                        utils.exp_digamma(
                            self.posterior_shape.sum(self.norm_axis, keepdims=True)
                        )
                    )
                )
            ).sum()
        if self.prior_accelerator is not None:
            prior_val *= self.prior_accelerator
        return prior_val.item() + self._cst_prior_value

    def _bump(self):
        if self._inference_mode == "EM":
            posterior_shape = self.tensor * self.tensor_update + self.prior_shape
        elif self._inference_mode == "VBEM":
            posterior_shape = self.posterior_shape
        self.tensor[...] = glob.xp.random.gamma(
            posterior_shape, glob.xp.ones_like(self.tensor)
        )
        self.tensor /= self.tensor.sum(axis=self.norm_axis, keepdims=True)

    def get_l2_norm(self, **kwargs):
        """
        Returns the l2 norm of each distribution contained in the tensor.
        """
        tensor = self.tensor.reshape(self.tensor.shape[: self.norm_axis[0]] + (-1,))
        return glob.xp.linalg.norm(tensor, ord=2, axis=-1, **kwargs)

    @cached_property
    def tensor_has_energy(self):
        return False

    def _give_update_bis(self, parent, out=None):
        if isinstance(parent, buds.BudShape):
            parent_idx_id = parent.get_index_id_for_children(self)
            prior_tensor = glob.xp.zeros_like(self.tensor) + self.prior_shape
            prior_tensor = glob.xp.zeros_like(self.tensor) + glob.sps.digamma(
                prior_tensor.sum(self.norm_axis, keepdims=True)
            )

            update_tensor = utils.einsum(
                prior_tensor, self.index_id, parent_idx_id, out=out
            )
            if out is None:
                return update_tensor
        else:
            raise ValueError(
                "Class of parent argument must be wonterfact.bubs.BudShape"
            )

    def _give_update_first_iteration(self, parent, out=None):
        if isinstance(parent, buds.BudShape):
            tensor = np.zeros_like(self.tensor) + self.prior_shape
            tensor = utils.exp_digamma(tensor) / utils.exp_digamma(
                tensor.sum(self.norm_axis, keepdims=True)
            )
            return self._give_update_alpha(parent, tensor, out=out)
        raise ValueError("'parent' must be an instance of wonterfact.bubs.BudShape")

    def compute_alpha_estim(self, n_iter=10):  # very slow: need research to be faster
        alpha_estim = np.ones_like(self.tensor)
        for __ in range(n_iter):
            alpha_sum = alpha_estim.sum(self.norm_axis, keepdims=True)
            utils.inverse_digamma(
                glob.sps.digamma(alpha_sum) + glob.xp.log(self.tensor), out=alpha_estim
            )
        return alpha_estim


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
        `prior_shape` cannot be different from None for this class and no BudShape
        should be linked (it should work but it has not be be proven).
        """
        self.l2_norm_axis = l2_norm_axis
        self.n_iter_norm2 = n_iter_norm2
        super().__init__(**kwargs)
        self.previous_tensor_update = None

    def _normalize_tensor(self):
        if self._inference_mode == "EM":
            if self.constraint_coeffs is not None:
                raise NotImplementedError
            if glob.processor == glob.CPU:
                self._normalize_l1_l2_tensor(
                    self.tensor,
                    self.l2_norm_axis,
                    self.n_iter_norm2,
                    self.prior_rate,
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
        return self.prior_rate.item()

    @cached_property
    def _cst_prior_value(self):
        # The normalization constant of such an ad hoc prior is unknown. We then
        # return 0.
        return 0.0

    def _prior_value(self):
        if self.update_period == 0:
            return 0
        if self._inference_mode == "EM":
            if self._prior_alpha_all_one and self.rate_parent is None:  # no prior
                return 0
            tensor = self.tensor
            if self.prior_accelerator is not None:
                tensor = tensor / self.prior_accelerator
            if not self._prior_alpha_all_one:
                prior_val = utils.xlogy(self._prior_shape_minus_one, tensor)
            else:
                prior_val = glob.xp.zeros_like(tensor)
            norm2 = (tensor ** 2).sum(self.l2_norm_axis, keepdims=True) ** 0.5
            if self.rate_parent is not None:
                prior_val -= (
                    self.prior_rate
                    * norm2
                    / np.prod([self.tensor.shape[ii] for ii in self.l2_norm_axis])
                )
        elif self._inference_mode == "VBEM":
            raise NotImplementedError
        if self.prior_accelerator is not None:
            prior_val *= self.prior_accelerator
        return prior_val.sum().item() + self._cst_prior_value
