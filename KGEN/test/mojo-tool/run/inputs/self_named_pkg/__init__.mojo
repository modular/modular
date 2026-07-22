# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# A submodule file sharing the package's own name, mirroring the `layout`
# kernel package (`layout/__init__.mojo` + `layout/layout.mojo`).
from .self_named_pkg import greet


def run():
    greet()
