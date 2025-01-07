# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# FIXME: KERN-1377
# UNSUPPORTED: AMD-GPU
# RUN: %mojo-no-debug-no-assert %s

from pathlib import Path
from sys._assembly import inlined_assembly

from gpu.host import DeviceContext


def test__dump_elaborated_mlir():
    # CHECK: kgen.func export @"{{.*}}_kernel_inlined_assembly()"
    fn kernel_inlined_assembly():
        inlined_assembly["nanosleep.u32 $0;", NoneType, constraints="r"](
            UInt32(100)
        )

    with DeviceContext() as ctx:
        _ = ctx.compile_function[kernel_inlined_assembly, _dump_elabmlir=True]()


def main():
    test__dump_elaborated_mlir()
