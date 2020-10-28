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

"""Initialization file for the wonterfact package"""

# Python Future imports
from __future__ import division, unicode_literals, print_function, absolute_import

# Python System imports

# Third-party imports

# Smart impulse common modules

# Relative imports
from . import utils
from .utils import create_filiation
from .root import Root
from .leaves import LeafGamma, LeafDirichlet, LeafGammaNorm
from .operators import Multiplier, Multiplexer, Integrator, Adder, Proxy
from .observers import PosObserver, RealObserver, BlindObs
from .glob_var_manager import glob

# Django imports only if possible.

__all__ = [
    "LeafGamma",
    "LeafDirichlet",
    "Multiplier",
    "Multiplexer",
    "PosObserver",
    "RealObserver",
    "Root",
    "Integrator",
    "Adder",
    "Proxy",
    "BlindObs",
    "LeafGammaNorm",
    "utils",
    "glob",
    "create_filiation",
]
