# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: H100-GPU
# RUN: %mojo-no-debug-no-assert %s | FileCheck %s

from builtin.io import _printf
from gpu.host import DeviceContext
from gpu.host._compile import _get_gpu_target
from gpu.host.memory_v1 import _make_ctx_current
from gpu.host.nvidia_cuda import CUDA
from gpu.id import BlockIdx, ThreadIdx
from layout import Layout, LayoutTensor
from layout.tma_async import TMATensorTile, create_tma_tile, TMABarrier
from layout._utils import ManagedLayoutGPUTensor
from layout.fillers import arange
from memory.pointer import _GPUAddressSpace

from utils.static_tuple import StaticTuple


@__llvm_metadata(`nvvm.grid_constant`=StaticTuple[Int, 1](0))
fn test_tma_async_load[
    dtype: DType, layout: Layout
](tma_tile: TMATensorTile[dtype, layout]):
    # FIXME(KERN-1365): This test case fails if the barrier is created before the memory!
    tile = LayoutTensor[
        dtype, layout, address_space = _GPUAddressSpace.SHARED
    ].stack_allocation()
    barrier = TMABarrier()
    tma_tile.async_copy(tile, barrier, (BlockIdx.x * 4, BlockIdx.y * 4))
    barrier.wait()

    _printf[
        "(%lu, %lu) : %g %g %g %g; %g %g %g %g; %g %g %g %g; %g %g %g %g\n"
    ](
        BlockIdx.x,
        BlockIdx.y,
        tile.ptr[0].cast[DType.float64](),
        tile.ptr[1].cast[DType.float64](),
        tile.ptr[2].cast[DType.float64](),
        tile.ptr[3].cast[DType.float64](),
        tile.ptr[4].cast[DType.float64](),
        tile.ptr[5].cast[DType.float64](),
        tile.ptr[6].cast[DType.float64](),
        tile.ptr[7].cast[DType.float64](),
        tile.ptr[8].cast[DType.float64](),
        tile.ptr[9].cast[DType.float64](),
        tile.ptr[10].cast[DType.float64](),
        tile.ptr[11].cast[DType.float64](),
        tile.ptr[12].cast[DType.float64](),
        tile.ptr[13].cast[DType.float64](),
        tile.ptr[14].cast[DType.float64](),
        tile.ptr[15].cast[DType.float64](),
    )


def test_tma_async_copy(ctx: DeviceContext):
    print("== test_tma_async_copy")
    var tensor = ManagedLayoutGPUTensor[DType.float32, Layout.row_major(8, 8)]()
    arange(tensor.tensor)
    var tma_tensor = create_tma_tile[4, 4](ctx, tensor.tensor)
    ctx.synchronize()

    var kernel_copy_async = ctx.compile_function[
        test_tma_async_load[
            __type_of(tma_tensor).dtype,
            __type_of(tma_tensor).layout,
        ],
        _target = _get_gpu_target["sm_90"](),
    ]()
    ctx.enqueue_function(
        kernel_copy_async, tma_tensor, grid_dim=(2, 2), block_dim=(1)
    )
    ctx.synchronize()
    _ = kernel_copy_async^
    _ = tensor^


@__llvm_metadata(`nvvm.grid_constant`=StaticTuple[Int, 1](0))
fn test_tma_async_load_multiple_threads[
    dtype: DType, layout: Layout
](tma_tile: TMATensorTile[dtype, layout]):
    # FIXME(KERN-1365): This test case fails if the barrier is created before the memory!
    tile = LayoutTensor[
        dtype, layout, address_space = _GPUAddressSpace.SHARED
    ].stack_allocation()
    barrier = TMABarrier()
    if ThreadIdx.x == 0:
        tma_tile.async_copy(tile, barrier, (BlockIdx.x * 4, BlockIdx.y * 4))
    barrier.wait()

    _printf[
        "(%lu, %lu) (%lu): %g %g %g %g; %g %g %g %g; %g %g %g %g; %g %g %g %g\n"
    ](
        BlockIdx.x,
        BlockIdx.y,
        ThreadIdx.x,
        tile.ptr[0].cast[DType.float64](),
        tile.ptr[1].cast[DType.float64](),
        tile.ptr[2].cast[DType.float64](),
        tile.ptr[3].cast[DType.float64](),
        tile.ptr[4].cast[DType.float64](),
        tile.ptr[5].cast[DType.float64](),
        tile.ptr[6].cast[DType.float64](),
        tile.ptr[7].cast[DType.float64](),
        tile.ptr[8].cast[DType.float64](),
        tile.ptr[9].cast[DType.float64](),
        tile.ptr[10].cast[DType.float64](),
        tile.ptr[11].cast[DType.float64](),
        tile.ptr[12].cast[DType.float64](),
        tile.ptr[13].cast[DType.float64](),
        tile.ptr[14].cast[DType.float64](),
        tile.ptr[15].cast[DType.float64](),
    )


def test_tma_async_copy_multiple_threads(ctx: DeviceContext):
    print("== test_tma_async_copy_multiple_threads")
    var tensor = ManagedLayoutGPUTensor[DType.float32, Layout.row_major(8, 8)]()
    arange(tensor.tensor, 1)
    var tma_tensor = create_tma_tile[4, 4](ctx, tensor.tensor)
    ctx.synchronize()

    var kernel_copy_async = ctx.compile_function[
        test_tma_async_load_multiple_threads[
            __type_of(tma_tensor).dtype,
            __type_of(tma_tensor).layout,
        ],
        _target = _get_gpu_target["sm_90"](),
    ]()
    ctx.enqueue_function(
        kernel_copy_async, tma_tensor, grid_dim=(2, 2), block_dim=(4)
    )
    ctx.synchronize()
    _ = kernel_copy_async^
    _ = tensor^


def main():
    with DeviceContext() as ctx:
        var prev_ctx = _make_ctx_current(CUDA(ctx))
        # CHECK-LABLE: test_tma_async_copy
        # CHECK-DAG: (0, 0) : 0 1 2 3; 8 9 10 11; 16 17 18 19; 24 25 26 27
        # CHECK-DAG: (1, 0) : 4 5 6 7; 12 13 14 15; 20 21 22 23; 28 29 30 31
        # CHECK-DAG: (0, 1) : 32 33 34 35; 40 41 42 43; 48 49 50 51; 56 57 58 59
        # CHECK-DAG: (1, 1) : 36 37 38 39; 44 45 46 47; 52 53 54 55; 60 61 62 63
        test_tma_async_copy(ctx)
        # CHECK-LABEL: == test_tma_async_copy_multiple_threads
        # CHECK-DAG: (0, 1) (0): 33 34 35 36; 41 42 43 44; 49 50 51 52; 57 58 59 60
        # CHECK-DAG: (0, 1) (1): 33 34 35 36; 41 42 43 44; 49 50 51 52; 57 58 59 60
        # CHECK-DAG: (0, 1) (2): 33 34 35 36; 41 42 43 44; 49 50 51 52; 57 58 59 60
        # CHECK-DAG: (0, 1) (3): 33 34 35 36; 41 42 43 44; 49 50 51 52; 57 58 59 60
        # CHECK-DAG: (1, 1) (0): 37 38 39 40; 45 46 47 48; 53 54 55 56; 61 62 63 64
        # CHECK-DAG: (1, 1) (1): 37 38 39 40; 45 46 47 48; 53 54 55 56; 61 62 63 64
        # CHECK-DAG: (1, 1) (2): 37 38 39 40; 45 46 47 48; 53 54 55 56; 61 62 63 64
        # CHECK-DAG: (1, 1) (3): 37 38 39 40; 45 46 47 48; 53 54 55 56; 61 62 63 64
        # CHECK-DAG: (0, 0) (0): 1 2 3 4; 9 10 11 12; 17 18 19 20; 25 26 27 28
        # CHECK-DAG: (0, 0) (1): 1 2 3 4; 9 10 11 12; 17 18 19 20; 25 26 27 28
        # CHECK-DAG: (0, 0) (2): 1 2 3 4; 9 10 11 12; 17 18 19 20; 25 26 27 28
        # CHECK-DAG: (0, 0) (3): 1 2 3 4; 9 10 11 12; 17 18 19 20; 25 26 27 28
        # CHECK-DAG: (1, 0) (0): 5 6 7 8; 13 14 15 16; 21 22 23 24; 29 30 31 32
        # CHECK-DAG: (1, 0) (1): 5 6 7 8; 13 14 15 16; 21 22 23 24; 29 30 31 32
        # CHECK-DAG: (1, 0) (2): 5 6 7 8; 13 14 15 16; 21 22 23 24; 29 30 31 32
        # CHECK-DAG: (1, 0) (3): 5 6 7 8; 13 14 15 16; 21 22 23 24; 29 30 31 32
        test_tma_async_copy_multiple_threads(ctx)
        _ = _make_ctx_current(prev_ctx)
