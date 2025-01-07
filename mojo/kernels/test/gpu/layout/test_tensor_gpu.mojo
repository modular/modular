# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s | FileCheck %s

from builtin.io import _printf
from gpu import BlockIdx, GridDim, ThreadIdx, barrier
from gpu.host import DeviceContext
from gpu.host.memory_v1 import _make_ctx_current
from gpu.host.nvidia_cuda import CUDA
from gpu.memory import (
    _GPUAddressSpace,
    async_copy_commit_group,
    async_copy_wait_group,
)
from layout._utils import ManagedLayoutGPUTensor
from layout.fillers import arange
from layout.layout_tensor import Layout, LayoutTensor
from memory import UnsafePointer
from testing import assert_true


# CHECK-LABEL: test_copy_dram_to_sram_async
def test_copy_dram_to_sram_async(ctx: DeviceContext):
    print("== test_copy_dram_to_sram_async")
    alias tensor_layout = Layout.row_major(4, 16)
    var tensor = ManagedLayoutGPUTensor[DType.float32, tensor_layout]()
    arange(tensor.tensor)

    var check_state = True

    fn copy_to_sram_test_kernel[
        layout: Layout,
    ](
        dram_tensor: LayoutTensor[DType.float32, layout],
        flag: UnsafePointer[Bool],
    ):
        var dram_tile = dram_tensor.tile[4, 4](0, BlockIdx.x)
        var sram_tensor = LayoutTensor[
            DType.float32,
            Layout.row_major(4, 4),
            address_space = _GPUAddressSpace.SHARED,
        ].stack_allocation()
        sram_tensor.copy_from_async(dram_tile)

        async_copy_commit_group()
        async_copy_wait_group(0)

        var col_offset = BlockIdx.x * 4

        for r in range(4):
            for c in range(4):
                if sram_tensor[r, c] != r * 16 + col_offset + c:
                    flag[] = False

    var copy_to_sram_test_launch = ctx.compile_function[
        copy_to_sram_test_kernel[tensor_layout]
    ]()

    ctx.enqueue_function(
        copy_to_sram_test_launch,
        tensor.tensor,
        UnsafePointer.address_of(check_state),
        grid_dim=(4),
        block_dim=(1),
    )
    assert_true(check_state, "Inconsistent values in shared memory")


def main():
    with DeviceContext() as ctx:
        var prev_ctx = _make_ctx_current(CUDA(ctx))
        test_copy_dram_to_sram_async(ctx)
        _ = _make_ctx_current(prev_ctx)
