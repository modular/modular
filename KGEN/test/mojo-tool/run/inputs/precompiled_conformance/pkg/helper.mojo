# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from .impl import DefaultHelper


trait Helper:
    def help(self):
        ...


comptime default_helper = DefaultHelper[4]
