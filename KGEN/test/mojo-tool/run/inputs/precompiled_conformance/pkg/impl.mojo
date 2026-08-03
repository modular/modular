# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Referenced only through the `default_helper` alias, so a consumer that
# imports `Container` never materializes `DefaultHelper`'s declaration.

from .helper import Helper


@fieldwise_init
struct DefaultHelper[n: Int](Copyable, Helper, Movable):
    var x: Int

    def help(self):
        pass
