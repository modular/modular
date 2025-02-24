# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: AMD-GPU
# RUN: %mojo-no-debug-no-assert %s | FileCheck %s

from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, grid_dim, thread_idx
from builtin.io import _get_stdout_stream
from sys._libc import fflush


# CHECK-LABEL: test_amd_dims
# CHECK: 14 15 16
# CHECK: 2 3 4
fn test_amd_dims(ctx: DeviceContext) raises:
    print("== test_amd_dims")

    fn test_dims_kernel():
        if (
            block_idx.x == 0
            and block_idx.y == 0
            and block_idx.z == 0
            and thread_idx.x == 0
            and thread_idx.y == 0
            and thread_idx.z == 0
        ):
            print(grid_dim.x, grid_dim.y, grid_dim.z)
            print(block_dim.x, block_dim.y, block_dim.z)

    ctx.enqueue_function[test_dims_kernel](
        grid_dim=(14, 15, 16),
        block_dim=(2, 3, 4),
    )


def main():
    var stdout_stream = _get_stdout_stream()
    with DeviceContext() as ctx:
        test_amd_dims(ctx)
        _ = fflush(stdout_stream)
