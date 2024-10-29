# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

__version__ = "0.0.1"

from .mojocell import MojoMagic


def load_ipython_extension(ipython):
    ipython.register_magics(MojoMagic)
