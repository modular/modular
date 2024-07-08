# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# https://linear.app/modularml/issue/MOCO-773/error-init-result-type-must-be-elided-or-none-does-not-have-source
# The bug is that the location that should be in this file, but ends up on the package __init__.mojo instead.


struct LocTest:
    fn __init__(inout self) -> LocTest:
        pass
