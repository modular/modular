# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo-no-debug %s | FileCheck %s

from layout.layout_tensor import LayoutTensor, Layout
from layout._utils import ManagedLayoutGPUTensor

from gpu.host import Function
from gpu.host.memory import _malloc_managed, _free
from gpu.host.device_context import DeviceContext
from gpu.memory import (
    _GPUAddressSpace,
    async_copy_commit_group,
    async_copy_wait_group,
)
from gpu import BlockIdx, GridDim, ThreadIdx, barrier
from testing import assert_true


from builtin.io import _printf


# CHECK-LABEL: test_copy_dram_to_sram_async
def test_copy_dram_to_sram_async():
    print("== test_copy_dram_to_sram_async")
    alias tensor_layout = Layout.row_major(4, 16)
    var tensor = ManagedLayoutGPUTensor[DType.float32, tensor_layout]()
    tensor.tensor.linspace()

    fn copy_to_sram_test_kernel[
        layout: Layout,
    ](dram_tensor: LayoutTensor[DType.float32, layout]):
        var dram_tile = dram_tensor.tile[4, 4](0, BlockIdx.x().value)
        var sram_tensor = LayoutTensor[
            DType.float32,
            Layout.row_major(4, 4),
            address_space = _GPUAddressSpace.SHARED,
        ].stack_allocation()
        sram_tensor.copy_from_async(dram_tile)

        async_copy_commit_group()
        async_copy_wait_group(0)

        _printf[
            "Block : %lu [%g %g %g %g ; %g %g %g %g ; %g %g %g %g ; %g %g %g"
            " %g]\n"
        ](
            BlockIdx.x(),
            sram_tensor[0, 0].cast[DType.float64](),
            sram_tensor[0, 1].cast[DType.float64](),
            sram_tensor[0, 2].cast[DType.float64](),
            sram_tensor[0, 3].cast[DType.float64](),
            sram_tensor[1, 0].cast[DType.float64](),
            sram_tensor[1, 1].cast[DType.float64](),
            sram_tensor[1, 2].cast[DType.float64](),
            sram_tensor[1, 3].cast[DType.float64](),
            sram_tensor[2, 0].cast[DType.float64](),
            sram_tensor[2, 1].cast[DType.float64](),
            sram_tensor[2, 2].cast[DType.float64](),
            sram_tensor[2, 3].cast[DType.float64](),
            sram_tensor[3, 0].cast[DType.float64](),
            sram_tensor[3, 1].cast[DType.float64](),
            sram_tensor[3, 2].cast[DType.float64](),
            sram_tensor[3, 3].cast[DType.float64](),
        )

    var copy_to_sram_test_launch = Function[
        copy_to_sram_test_kernel[tensor_layout]
    ]()

    # CHECK-DAG: Block : 2 [8 9 10 11 ; 24 25 26 27 ; 40 41 42 43 ; 56 57 58 59]
    # CHECK-DAG: Block : 3 [12 13 14 15 ; 28 29 30 31 ; 44 45 46 47 ; 60 61 62 63]
    # CHECK-DAG: Block : 1 [4 5 6 7 ; 20 21 22 23 ; 36 37 38 39 ; 52 53 54 55]
    # CHECK-DAG: Block : 0 [0 1 2 3 ; 16 17 18 19 ; 32 33 34 35 ; 48 49 50 51]
    copy_to_sram_test_launch(tensor.tensor, grid_dim=(4), block_dim=(1))


# CHECK-LABEL: test_copy_from_async_masked_src
def test_copy_from_async_masked_src():
    print("== test_copy_from_async_masked_src")
    alias tensor_layout = Layout.row_major(31, 32)
    var tensor = ManagedLayoutGPUTensor[DType.int32, tensor_layout]()
    tensor.tensor.linspace()
    var check_state = _malloc_managed[Bool](1)

    check_state[] = True

    fn copy_to_sram_masked_src_test_kernel[
        layout: Layout,
    ](
        dram_tensor: LayoutTensor[DType.int32, layout],
        flag: UnsafePointer[Bool],
    ):
        var dram_tile = dram_tensor.tile[16, 16](BlockIdx.y(), BlockIdx.x())
        var smem_tensor = LayoutTensor[
            DType.int32,
            Layout.row_major(16, 16),
            address_space = _GPUAddressSpace.SHARED,
        ].stack_allocation()

        alias thrd_layout = Layout.row_major(8, 8)
        var smem_frag = smem_tensor.distribute[thrd_layout](ThreadIdx.x())

        # Zero Init
        smem_frag.fill(0)
        barrier()

        var dram_frag = dram_tile.distribute[thrd_layout](ThreadIdx.x())
        var offset = dram_frag.distance(dram_tensor.ptr)
        smem_frag.copy_from_async_masked_src(dram_frag, offset, 31, 32)

        async_copy_commit_group()
        async_copy_wait_group(0)

        var row_offset = BlockIdx.y() * 16
        var col_offset = BlockIdx.x() * 16

        if ThreadIdx.x() == 0:
            for sub_r in range(16):
                for sub_c in range(16):
                    var row_id = row_offset + sub_r
                    var col_id = col_offset + sub_c
                    if (
                        row_id < 31
                        and smem_tensor[sub_r, sub_c] != row_id * 32 + col_id
                    ):
                        flag[] = False

        barrier()

    var copy_to_sram_test_launch = Function[
        copy_to_sram_masked_src_test_kernel[tensor_layout]
    ]()

    copy_to_sram_test_launch(
        tensor.tensor, check_state, grid_dim=(2, 2, 1), block_dim=(64, 1, 1)
    )
    assert_true(check_state[], "Shared memory value mismatches")
    _free(check_state)


def main():
    with DeviceContext() as ctx:
        test_copy_dram_to_sram_async()
        test_copy_from_async_masked_src()
