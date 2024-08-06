# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: windows
# RUN: MODULAR_MAX_TEMPS_DIR=$(dirname %t) mojo %s
# RUN: grep mo.graph $(dirname %t)/temp.mo.mlir
# RUN: grep mgp.model $(dirname %t)/temp.mgp.mlir

from pathlib.path import Path
from testing import *

from max.graph import Type, Graph, TensorType, Symbol
from max._driver import cpu_device, Tensor


fn test_identity_graph() raises:
    var g = Graph(
        name="identity_graph",
        in_types=List[Type](TensorType(DType.int32)),
    )
    g.output(g[0])

    var cpu = cpu_device()
    cpu.compile(g)


fn main() raises:
    test_identity_graph()
