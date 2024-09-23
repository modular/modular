# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s | FileCheck %s

from layout import LayoutTensor, Layout
from layout.int_tuple import UNKNOWN_VALUE

from compile import compile_code
from gpu.host._compile import _get_nvptx_target


# CHECK-LABEL: test_no_alloca_fill
fn test_no_alloca_fill():
    print("== test_no_alloca_fill")

    fn layout_tensor_kernel(
        outout: LayoutTensor[
            DType.float32, Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)
        ],
        i: Int,
        j: Int,
    ):
        var reg_tile = LayoutTensor[
            DType.float32, Layout.row_major(4, 4)
        ].stack_allocation().fill(0)

        outout.tile[4, 4](i, j).copy_from(reg_tile)

    # CHECK-NOT: alloca float, i64 16, align 4
    print(
        compile_code[
            layout_tensor_kernel,
            emission_kind="llvm",
            target = _get_nvptx_target(),
        ]()
    )


fn main():
    test_no_alloca_fill()
