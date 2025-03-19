# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from .._mlir._mlir_libs._mlir import *
from .._mlir._mlir_libs._mlir.ir import *


def site_initialize():
    from max._core import graph  # type: ignore

    from .._mlir._mlir_libs import get_dialect_registry

    graph.load_modular_dialects(get_dialect_registry())


site_initialize()
