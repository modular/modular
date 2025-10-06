# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

alias int = __mlir_type.index


trait Flying:
    fn fly_to(mut self, new_location: int):
        ...
