# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from .wrapper import Wrapper


trait Marked:
    def marked(self):
        ...


struct Container[T: Marked, W: Wrapper = Int(4)]:
    var value: Int

    def __init__(out self):
        self.value = 0
