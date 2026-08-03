# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from .helper import Helper, default_helper


trait Marked:
    def marked(self):
        ...


struct Container[T: Marked, H: Helper = default_helper]:
    var value: Int

    def __init__(out self):
        self.value = 0
