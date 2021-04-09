# ----------------------------------------------------------------------------
# Copyright 2021 Benoit Fuentes <bf@benoit-fuentes.fr>
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

"""Module for all buds (i.e. hyperparameter nodes) classes"""

# Python System imports
from functools import cached_property

# Third-party imports
import numpy as np
from methodtools import lru_cache

# Relative imports
from . import utils, core_nodes
from .glob_var_manager import glob


class _Bud(core_nodes._DynNodeData0):
    """
    Mother class for the hyperparameters buds of a graphical model
    """

    def __init__(self, **kwargs):
        self.processed_data_counter = 0
        super().__init__(**kwargs)

    @property
    def level(self):
        """
        Returns 0 , which is the default level for of a bud.

        Returns
        ------
        int
        """
        return 0

    def _check_model_validity(self):
        super()._check_model_validity()
        if not self.are_all_tensor_coefs_linked_to_at_least_one_child:
            raise ValueError(
                "Model invalid at {} level. Please make sure that each"
                "hyperparameters is linked to at least one child.".format(self)
            )

    def _set_inference_mode(self, mode="EM"):
        super()._set_inference_mode(mode=mode)
        if mode == "EM":
            self.update_period = 0

    def _initialization(self):
        # this one is for e_d or m_e (cf technical report)
        self.tensor_update = glob.xp.empty_like(self.tensor)
        # this one is for hyperparameter optimization algorithm
        self.tensor_update_bis = glob.xp.empty_like(self.tensor)

    @cached_property
    def number_of_users(self):
        """
        Gives the number of parameters that share a same hyperparameter for each
        hyperparameter (corresponds to $|\\phi^{-1}(d)|$ in tech report)
        """
        number = glob.xp.zeros_like(self.tensor)
        self._compute_tensor_update_aux2(
            tensor_to_fill=number, method_to_call="_give_number_of_users",
        )
        return number

    def get_update_bis(self, tensor_to_fill):
        self._compute_tensor_update_aux2(
            tensor_to_fill=tensor_to_fill, method_to_call="_give_update_bis",
        )

    def compute_tensor_update_online(self, counter_max=0):
        if counter_max == 0:
            self.compute_tensor_update()
        else:
            counter = max(self.processed_data_counter, counter_max)
            past_tensor_update = self.tensor_update * counter / (counter + 1)
            self.compute_tensor_update()  # new values in self.tensor_update
            self.tensor_update /= counter + 1
            self.tensor_update += past_tensor_update
        self.processed_data_counter += 1


class BudShape(_Bud):
    @property
    def tensor_has_energy(self):
        return False

    def update_tensor(self):
        self.get_update_bis(tensor_to_fill=self.tensor_update_bis)
        utils.inverse_digamma(
            (self.tensor_update_bis + self.tensor_update) / self.number_of_users,
            out=self.tensor,
        )


class BudRate(_Bud):
    def init(self, prior_rate):
        prior_rate = glob.xp.array(prior_rate, dtype=glob.float)
        self.prior_rate = prior_rate
        if self.prior_rate.size > 1 and not (self.prior_rate > 0).all():
            raise ValueError("prior_rate, if not None, must be > 0")

    @property
    def tensor_has_energy(self):
        return True

    def update_tensor(self):
        self.get_update_bis(tensor_to_fill=self.tensor_update_bis)
        self.tensor[...] = self.tensor_update_bis / self.tensor_update

