# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s | FileCheck %s

from compile import compile_info
from gpu.host._compile import _get_gpu_target
from layout import Layout, LayoutTensor
from layout.int_tuple import UNKNOWN_VALUE


# CHECK-LABEL: test_no_alloca_fill
fn test_no_alloca_fill():
    print("== test_no_alloca_fill")

    fn layout_tensor_kernel(
        outout: LayoutTensor[
            DType.float32,
            Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE),
            MutableAnyOrigin,
        ],
        i: Int,
        j: Int,
    ):
        var reg_tile = LayoutTensor[
            DType.float32, Layout.row_major(4, 4), MutableAnyOrigin
        ].stack_allocation().fill(0)

        outout.tile[4, 4](i, j).copy_from(reg_tile)

    # CHECK-NOT: alloca float, i64 16, align 4
    print(
        compile_info[
            layout_tensor_kernel,
            emission_kind="llvm",
            target = _get_gpu_target(),
        ]()
    )


fn main():
    test_no_alloca_fill()
