# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug -D MLIRC_DYLIB=.graph_lib %s


import _mlir
from testing import assert_equal


def main():
    with _mlir.Context() as ctx:
        var module = _mlir.Module(_mlir.Location.unknown(ctx))
        assert_equal("module {\n}\n", String(module))

        # Right now the lifetime of `module` is poorly defined.
        # This `destroy()` is just a temp. workaround so
        # ASAN does not complain (and can therefore catch realer bugs)
        module.destroy()
