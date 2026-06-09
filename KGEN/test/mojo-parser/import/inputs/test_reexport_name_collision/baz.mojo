# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Defines a struct with the same name as the module it lives in.
# Used to test that re-exporting this struct from __init__.mojo
# resolves to the struct, not the module.


struct baz:
    var x: Int

    def __init__(out self):
        self.x = 42
