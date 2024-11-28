# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: GPU-H100
# RUN: %mojo-no-debug %s | FileCheck %s

from builtin.io import _printf
from gpu.host import DeviceContext
from gpu.host._compile import _get_gpu_target
from gpu.id import BlockIdx
from layout import Layout, LayoutTensor
from layout.tma_async import TMATensorTile, create_tma_tile
from memory import UnsafePointer, stack_allocation

from utils.static_tuple import StaticTuple


@__llvm_metadata(`nvvm.grid_constant`=StaticTuple[Int, 1](0))
fn test_tma_async_load[
    dtype: DType, layout: Layout
](tma_tile: TMATensorTile[dtype, layout]):
    tile, barrier = tma_tile.async_load(BlockIdx.x(), BlockIdx.y())
    barrier.wait()

    _printf[
        "(%lu, %lu) : %g %g %g %g; %g %g %g %g; %g %g %g %g; %g %g %g %g\n"
    ](
        BlockIdx.x(),
        BlockIdx.y(),
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
    var gmem_host = UnsafePointer[Float32].alloc(8 * 8)
    for i in range(64):
        gmem_host[i] = i

    var gmem_dev = ctx.create_buffer_sync[DType.float32](8 * 8)
    tensor = LayoutTensor[DType.float32, Layout.row_major(8, 8)](gmem_dev.ptr)

    ctx.enqueue_copy_to_device(gmem_dev, gmem_host)

    var tma_tensor = create_tma_tile[4, 4](tensor)
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
    gmem_host.free()


def main():
    with DeviceContext() as ctx:
        # CHECK-LABLE: test_tma_async_copy
        # CHECK-DAG: (0, 0) : 0 1 2 3; 8 9 10 11; 16 17 18 19; 24 25 26 27
        # CHECK-DAG: (1, 0) : 4 5 6 7; 12 13 14 15; 20 21 22 23; 28 29 30 31
        # CHECK-DAG: (0, 1) : 32 33 34 35; 40 41 42 43; 48 49 50 51; 56 57 58 59
        # CHECK-DAG: (1, 1) : 36 37 38 39; 44 45 46 47; 52 53 54 55; 60 61 62 63
        test_tma_async_copy(ctx)
