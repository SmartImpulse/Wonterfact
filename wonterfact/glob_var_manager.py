# ----------------------------------------------------------------------------
# Copyright 2019 Smart Impulse SAS
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

"""Module that deals with global variables of the program"""

# Python Future imports
from __future__ import division, unicode_literals, print_function, absolute_import

# Python System imports

# Third party import
from functools import cached_property

# Relative imports
from . import utils


class GlobalVarManager:
    CPU = "cpu"
    GPU = "gpu"
    FLOAT64 = "float64"
    FLOAT32 = "float32"
    CUPY = "cupy"
    NUMPY = "numpy"

    def __init__(self, processor=None, float_precision=None):
        self._processor = processor or self.CPU
        self._float = float_precision or self.FLOAT64
        self._can_change_backend = True
        self._can_change_float_precision = True

    @property
    def processor(self):
        """
        Returns the current backend processor (gpu or cpu)
        """
        return self._processor

    @property
    def backend(self):
        """
        Returns the current backend python package for array manipulation and
        operation (numpy or cupy)
        """
        if self.processor == self.GPU:
            backend = self.CUPY
        elif self.processor == self.CPU:
            backend = self.NUMPY
        return backend

    @cached_property
    def float(self):
        """
        Returns the current float precision (32 or 64). Cautious once this
        property is called, float precision cannot no longer be changed.
        """
        self._forbid_float_precision_change()
        return self._float

    def _forbid_backend_change(self):
        self._can_change_backend = False

    def _forbid_float_precision_change(self):
        self._can_change_float_precision = False

    def set_float_precision(self, float_precision):
        """
        Set the float precision for all tensors in wonterfact's nodes. By
        default the float precision is `float64`. It can be changed to `float32`
        only once and should be just after importing wonterfact package.
        """
        if float_precision != self.FLOAT64:
            raise NotImplementedError
        if float_precision not in [self.FLOAT64, self.FLOAT32]:
            raise ValueError(
                "float_precision can be '{}' or '{}'".format(self.FLOAT64, self.FLOAT32)
            )
        if float_precision != self._float:
            if self._can_change_float_precision:
                self._float = float_precision
            else:
                raise ValueError(
                    """
                    This call to set_float_precision has no effect because the precision has already been set.
                    You can change float precision only just after importing wonterfact
                    """
                )

    def _force_backend_reinit(self):
        for name_attr in ["xp", "sps", "as_strided"]:
            try:
                self.__delattr__(name_attr)
            except AttributeError:
                pass
        self._can_change_backend = True

    def set_backend_processor(self, processor, force=False):
        """
        Set the backend processor. It should be set just after importing
        wonterfact package. By default, wonterfact uses cpu.

        Parameters
        ----------
        processor: 'cpu' or 'gpu'
        force: bool, optional, default False
            In order to prevent unexpected errors, the backend can be changed
            only once, right after wonterfact package import. If True, user can
            force the backend setup any time, but at its own risk.
        """
        if force:
            self._force_backend_reinit()
        if processor not in ["cpu", "gpu"]:
            raise ValueError("processor can be 'gpu' or 'cpu'")
        if processor != self.processor:
            if self._can_change_backend:
                self._processor = processor
            else:
                raise ValueError(
                    """
                    This call to set_backend_processor has no effect because the processor has already been chosen.
                    You can change processor engine only just after importing wonterfact
                    """
                )

    @cached_property
    def xp(self):
        """
        Return either cupy or numpy depending on the backend processor used.
        """
        self._forbid_backend_change()
        if self.processor == "cpu":
            import numpy

            return numpy
        elif self.processor == "gpu":
            import cupy  # pylint: disable=E0401

            return cupy

    @cached_property
    def sps(self):
        """
        Returns either scipy.special or cupyx.scipy.special depending on the
        backend processor used.
        """
        self._forbid_backend_change()
        if self.processor == "cpu":
            import scipy.special

            return scipy.special
        elif self.processor == "gpu":
            import cupyx.scipy.special  # pylint: disable=E0401

            return cupyx.scipy.special

    @cached_property
    def as_strided(self):
        """
        Returns either numpy.lib.stride_tricks.as_strided or
        cupy.lib.stride_tricks.as_strided depending on the backend processor
        used.
        """
        self._forbid_backend_change()
        if self.processor == "cpu":
            from numpy.lib.stride_tricks import as_strided

            return as_strided
        elif self.processor == "gpu":
            from cupy.lib.stride_tricks import as_strided  # pylint: disable=E0401

            return as_strided


glob = GlobalVarManager()
