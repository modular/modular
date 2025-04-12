# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug --debug-level full %s | FileCheck %s

from collections import OptionalReg
from math import ceildiv
from pathlib import Path
from sys import bitwidthof, simdwidthof

from gpu import barrier
from gpu.host import DeviceContext
from gpu.id import block_idx, thread_idx
from gpu.memory import (
    AddressSpace,
    async_copy_commit_group,
    async_copy_wait_all,
    async_copy_wait_group,
)
from layout import *
from layout._fillers import arange
from layout._utils import ManagedLayoutTensor
from layout.layout_tensor import (
    UNKNOWN_VALUE,
    LayoutTensor,
    binary_op_type,
    copy,
    copy_dram_to_sram,
    copy_dram_to_sram_async,
    copy_local_to_dram,
    copy_local_to_local,
    copy_sram_to_dram,
    copy_sram_to_local,
)
from memory import UnsafePointer
from testing import assert_almost_equal

from utils import IndexList


@always_inline
fn add_op[
    type: DType, width: Int
](lhs: SIMD[type, width], rhs: SIMD[type, width]) -> SIMD[type, width]:
    return lhs + rhs


# ----------------------------------------------------------------------
# async copy tests
# ----------------------------------------------------------------------
fn async_copy_kernel[
    input_layout: Layout,
    BM: Int,
    BN: Int,
](input: LayoutTensor[DType.float32, input_layout, MutableAnyOrigin]):
    var input_tile = input.tile[BM, BN](block_idx.y, block_idx.x)

    var smem_tile = LayoutTensor[
        DType.float32,
        Layout(IntTuple(BM, BN)),
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    smem_tile.copy_from_async(input_tile)
    async_copy_wait_all()

    var tx = thread_idx.x
    var ty = thread_idx.y
    smem_tile[tx, ty] += ty

    input_tile.copy_from(smem_tile)


fn test_async_copy[
    layout: Layout, M: Int, N: Int, BM: Int, BN: Int
](ctx: DeviceContext) raises:
    print("=== test_async_copy")

    alias managed_layout_tensor_type = ManagedLayoutTensor[
        DType.float32,
        layout,
    ]

    alias element_type = managed_layout_tensor_type.element_type
    alias idx_type = managed_layout_tensor_type.index_type

    alias runtime_layout = RuntimeLayout[
        layout, element_type=element_type, linear_idx_type=idx_type
    ].row_major(IndexList[2, element_type=element_type](M, N))

    var input = ManagedLayoutTensor[DType.float32, layout](runtime_layout, ctx)

    arange(input.tensor())

    ctx.enqueue_function[async_copy_kernel[layout, BM, BN]](
        input.device_tensor(), grid_dim=(N // BN, M // BM), block_dim=(BM, BN)
    )

    ctx.synchronize()

    print(input.tensor())
    _ = input^


def run_async_copy_tests(ctx: DeviceContext):
    # CHECK: === test_async_copy
    # CHECK: 0.0   2.0   4.0   3.0   5.0   7.0
    # CHECK: 6.0   8.0   10.0   9.0   11.0   13.0
    # CHECK: 12.0   14.0   16.0   15.0   17.0   19.0
    # CHECK: 18.0   20.0   22.0   21.0   23.0   25.0
    # CHECK: 24.0   26.0   28.0   27.0   29.0   31.0
    # CHECK: 30.0   32.0   34.0   33.0   35.0   37.0
    test_async_copy[
        Layout.row_major(6, 6),
        M=6,
        N=6,
        BM=2,
        BN=3,
    ](ctx)

    # CHECK: === test_async_copy
    # CHECK: 0.0   2.0   4.0   3.0   5.0   7.0
    # CHECK: 6.0   8.0   10.0   9.0   11.0   13.0
    # CHECK: 12.0   14.0   16.0   15.0   17.0   19.0
    # CHECK: 18.0   20.0   22.0   21.0   23.0   25.0
    # CHECK: 24.0   26.0   28.0   27.0   29.0   31.0
    # CHECK: 30.0   32.0   34.0   33.0   35.0   37.0
    test_async_copy[
        Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE),
        M=6,
        N=6,
        BM=2,
        BN=3,
    ](ctx)


# ----------------------------------------------------------------------
# dynamic async copy tests
# ----------------------------------------------------------------------


fn async_dynamic_copy_kernel[
    input_layout: Layout,
    output_layout: Layout,
    BM: Int,
    BN: Int,
    num_rows: Int,
](
    input: LayoutTensor[DType.float32, input_layout, MutableAnyOrigin],
    output: LayoutTensor[DType.float32, output_layout, MutableAnyOrigin],
):
    var masked_input = LayoutTensor[
        DType.float32,
        input_layout,
        MutableAnyOrigin,
        masked=True,
    ](
        input.ptr,
        __type_of(input.runtime_layout)(
            __type_of(input.runtime_layout.shape)(num_rows, input.dim(1)),
            input.runtime_layout.stride,
        ),
    )

    var input_tile = masked_input.tile[BM, BN](block_idx.x, block_idx.y)
    var output_tile = output.tile[BM, BN](block_idx.x, block_idx.y)

    var smem_tile = LayoutTensor[
        DType.float32,
        Layout(IntTuple(BM, BN)),
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    smem_tile.copy_from_async[is_masked=True](input_tile)
    async_copy_wait_all()

    output_tile.copy_from(smem_tile)


fn test_dynamic_async_copy[
    M: Int, N: Int, BM: Int, BN: Int, num_rows: Int
](ctx: DeviceContext) raises:
    print("=== test_dynamic_async_copy")

    alias unknown_layout = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)

    alias input_runtime_layout = RuntimeLayout[
        unknown_layout,
        element_type = DType.int64,
        linear_idx_type = DType.int64,
    ].row_major(IndexList[2, element_type = DType.int64](M, N))

    alias output_runtime_layout = RuntimeLayout[
        unknown_layout,
        element_type = DType.int64,
        linear_idx_type = DType.int64,
    ].row_major(IndexList[2, element_type = DType.int64](num_rows, N))

    var input = ManagedLayoutTensor[
        DType.float32,
        unknown_layout,
    ](input_runtime_layout, ctx)
    arange(input.tensor())

    var output = ManagedLayoutTensor[
        DType.float32,
        unknown_layout,
    ](output_runtime_layout, ctx)

    alias kernel_type = async_dynamic_copy_kernel[
        unknown_layout,
        unknown_layout,
        BM,
        BN,
        num_rows,
    ]

    ctx.enqueue_function[kernel_type](
        input.device_tensor(),
        output.device_tensor(),
        grid_dim=(ceildiv(M, BM), ceildiv(M, BN)),
        block_dim=(1, 1),
    )

    ctx.synchronize()

    print(output.tensor())

    _ = input^
    _ = output^


def run_dynamic_async_copy_tests(ctx: DeviceContext):
    # CHECK: === test_dynamic_async_copy
    # CHECK: 0.0 1.0 2.0 3.0 4.0 5.0
    # CHECK: 6.0 7.0 8.0 9.0 10.0 11.0
    # CHECK: 12.0 13.0 14.0 15.0 16.0 17.0
    # CHECK: 18.0 19.0 20.0 21.0 22.0 23.0
    # CHECK: 24.0 25.0 26.0 27.0 28.0 29.0
    test_dynamic_async_copy[
        M=6,
        N=6,
        BM=2,
        BN=3,
        num_rows=5,
    ](ctx)


# ----------------------------------------------------------------------
# swizzle copy tests
# ----------------------------------------------------------------------


fn swizzle_copy[
    type: DType,
    layout: Layout,
    BM: Int,
    BK: Int,
    num_threads: Int,
](
    a: LayoutTensor[type, layout, MutableAnyOrigin],
    b: LayoutTensor[type, layout, MutableAnyOrigin],
):
    alias simd_size = simdwidthof[type]()

    # Double buffer in shared memory.
    var a_smem_tile = LayoutTensor[
        type,
        Layout.row_major(BM, BK),
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation().fill(0)

    alias thread_layout = Layout.row_major(
        num_threads * simd_size // BK, BK // simd_size
    )

    copy_dram_to_sram_async[thread_layout=thread_layout, swizzle=True](
        a_smem_tile.vectorize[1, simd_size](),
        a.tile[BM, BK](block_idx.x, 0).vectorize[1, simd_size](),
    )

    async_copy_wait_all()
    barrier()

    # Write current stage to global memory.
    var b_gmem_tile = b.tile[BM, BK](block_idx.x, 0)
    var b_gmem_frag = b_gmem_tile.vectorize[1, simd_size]().distribute[
        thread_layout
    ](thread_idx.x)
    var a_smem_frag = a_smem_tile.vectorize[1, simd_size]().distribute[
        thread_layout
    ](thread_idx.x)
    b_gmem_frag.copy_from(a_smem_frag)


fn test_swizzle_copy[
    layout: Layout,
    M: Int,
    K: Int,
    BM: Int,
    BK: Int,
    num_threads: Int,
    skew_M: Int = 0,
](ctx: DeviceContext) raises:
    print("=== test_swizzle_copy")

    alias managed_layout_tensor_type = ManagedLayoutTensor[
        DType.float32,
        layout,
    ]

    alias element_type = managed_layout_tensor_type.element_type
    alias idx_type = managed_layout_tensor_type.index_type

    alias a_runtime_layout = RuntimeLayout[
        layout, element_type=element_type, linear_idx_type=idx_type
    ].row_major(IndexList[2, element_type=element_type](M - skew_M, K))

    alias b_runtime_layout = RuntimeLayout[
        layout, element_type=element_type, linear_idx_type=idx_type
    ].row_major(IndexList[2, element_type=element_type](M, K))

    var a_tensor = ManagedLayoutTensor[
        DType.float32,
        layout,
    ](a_runtime_layout, ctx)
    arange(a_tensor.tensor())

    var b_tensor = ManagedLayoutTensor[
        DType.float32,
        layout,
    ](b_runtime_layout, ctx)

    alias copy = swizzle_copy[
        DType.float32,
        layout,
        BM,
        BK,
        num_threads,
    ]

    ctx.enqueue_function[copy](
        a_tensor.device_tensor(),
        b_tensor.device_tensor(),
        grid_dim=(ceildiv(M, BM), 1, 1),
        block_dim=(num_threads, 1, 1),
    )

    ctx.synchronize()
    print(b_tensor.tensor())

    _ = a_tensor^
    _ = b_tensor^


def run_swizzle_copy_tests(ctx: DeviceContext):
    # CHECK: === test_swizzle_copy
    # CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
    # CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
    # CHECK: 36.0 37.0 38.0 39.0 32.0 33.0 34.0 35.0 44.0 45.0 46.0 47.0 40.0 41.0 42.0 43.0
    # CHECK: 52.0 53.0 54.0 55.0 48.0 49.0 50.0 51.0 60.0 61.0 62.0 63.0 56.0 57.0 58.0 59.0
    # CHECK: 72.0 73.0 74.0 75.0 76.0 77.0 78.0 79.0 64.0 65.0 66.0 67.0 68.0 69.0 70.0 71.0
    # CHECK: 88.0 89.0 90.0 91.0 92.0 93.0 94.0 95.0 80.0 81.0 82.0 83.0 84.0 85.0 86.0 87.0
    # CHECK: 108.0 109.0 110.0 111.0 104.0 105.0 106.0 107.0 100.0 101.0 102.0 103.0 96.0 97.0 98.0 99.0
    # CHECK: 124.0 125.0 126.0 127.0 120.0 121.0 122.0 123.0 116.0 117.0 118.0 119.0 112.0 113.0 114.0 115.0
    test_swizzle_copy[
        Layout.row_major(8, 16),
        M=8,
        K=16,
        BM=8,
        BK=16,
        num_threads=32,
    ](ctx)

    # CHECK: == test_swizzle_copy
    # CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
    # CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
    # CHECK: 36.0 37.0 38.0 39.0 32.0 33.0 34.0 35.0 44.0 45.0 46.0 47.0 40.0 41.0 42.0 43.0
    # CHECK: 52.0 53.0 54.0 55.0 48.0 49.0 50.0 51.0 60.0 61.0 62.0 63.0 56.0 57.0 58.0 59.0
    # CHECK: 72.0 73.0 74.0 75.0 76.0 77.0 78.0 79.0 64.0 65.0 66.0 67.0 68.0 69.0 70.0 71.0
    # CHECK: 88.0 89.0 90.0 91.0 92.0 93.0 94.0 95.0 80.0 81.0 82.0 83.0 84.0 85.0 86.0 87.0
    # CHECK: 108.0 109.0 110.0 111.0 104.0 105.0 106.0 107.0 100.0 101.0 102.0 103.0 96.0 97.0 98.0 99.0
    # CHECK: 124.0 125.0 126.0 127.0 120.0 121.0 122.0 123.0 116.0 117.0 118.0 119.0 112.0 113.0 114.0 115.0
    test_swizzle_copy[
        Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE),
        M=8,
        K=16,
        BM=8,
        BK=16,
        num_threads=32,
    ](ctx)

    # CHECK: === test_swizzle_copy
    # CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
    # CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
    # CHECK: 36.0 37.0 38.0 39.0 32.0 33.0 34.0 35.0 44.0 45.0 46.0 47.0 40.0 41.0 42.0 43.0
    # CHECK: 52.0 53.0 54.0 55.0 48.0 49.0 50.0 51.0 60.0 61.0 62.0 63.0 56.0 57.0 58.0 59.0
    # CHECK: 72.0 73.0 74.0 75.0 76.0 77.0 78.0 79.0 64.0 65.0 66.0 67.0 68.0 69.0 70.0 71.0
    # CHECK: 88.0 89.0 90.0 91.0 92.0 93.0 94.0 95.0 80.0 81.0 82.0 83.0 84.0 85.0 86.0 87.0
    # CHECK: 108.0 109.0 110.0 111.0 104.0 105.0 106.0 107.0 100.0 101.0 102.0 103.0 96.0 97.0 98.0 99.0
    # CHECK: 124.0 125.0 126.0 127.0 120.0 121.0 122.0 123.0 116.0 117.0 118.0 119.0 112.0 113.0 114.0 115.0
    test_swizzle_copy[
        Layout.row_major(UNKNOWN_VALUE, 16),
        M=8,
        K=16,
        BM=8,
        BK=16,
        num_threads=32,
    ](ctx)

    # CHECK: === test_swizzle_copy
    # CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
    # CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
    # CHECK: 36.0 37.0 38.0 39.0 32.0 33.0 34.0 35.0 44.0 45.0 46.0 47.0 40.0 41.0 42.0 43.0
    # CHECK: 52.0 53.0 54.0 55.0 48.0 49.0 50.0 51.0 60.0 61.0 62.0 63.0 56.0 57.0 58.0 59.0
    # CHECK: 72.0 73.0 74.0 75.0 76.0 77.0 78.0 79.0 64.0 65.0 66.0 67.0 68.0 69.0 70.0 71.0
    # CHECK: 88.0 89.0 90.0 91.0 92.0 93.0 94.0 95.0 80.0 81.0 82.0 83.0 84.0 85.0 86.0 87.0
    # CHECK: 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
    # CHECK: 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
    test_swizzle_copy[
        Layout.row_major(UNKNOWN_VALUE, 16),
        M=8,
        K=16,
        BM=8,
        BK=16,
        num_threads=32,
        skew_M=2,
    ](ctx)


# ----------------------------------------------------------------------
# masked async copy tests
# ----------------------------------------------------------------------


@always_inline
fn masked_async_copy_kernel[
    layout: Layout, num_rows: Int
](input: LayoutTensor[DType.float32, layout, MutableAnyOrigin]):
    alias thread_layout = Layout.row_major(4, 2)

    var masked_input = LayoutTensor[
        DType.float32,
        layout,
        MutableAnyOrigin,
        masked=True,
    ](
        input.ptr,
        __type_of(input.runtime_layout)(
            __type_of(input.runtime_layout.shape)(num_rows, input.dim(1)),
            input.runtime_layout.stride,
        ),
    )

    var smem_tile = LayoutTensor[
        DType.float32,
        layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation().fill(-1.0)

    copy_dram_to_sram_async[thread_layout=thread_layout](
        smem_tile.vectorize[1, 4](), masked_input.vectorize[1, 4]()
    )

    async_copy_commit_group()
    async_copy_wait_all()

    copy_sram_to_dram[thread_layout=thread_layout](
        input.vectorize[1, 4](),
        smem_tile.vectorize[1, 4](),
    )


fn test_masked_async_copy[
    layout: Layout, M: Int, N: Int, skew_rows: Int
](ctx: DeviceContext) raises:
    print("=== test_masked_async_copy")

    alias managed_layout_tensor_type = ManagedLayoutTensor[
        DType.float32,
        layout,
    ]

    alias element_type = managed_layout_tensor_type.element_type
    alias idx_type = managed_layout_tensor_type.index_type

    alias runtime_layout = RuntimeLayout[
        layout, element_type=element_type, linear_idx_type=idx_type
    ].row_major(IndexList[2, element_type=element_type](M, N))

    var input = ManagedLayoutTensor[
        DType.float32,
        layout,
    ](runtime_layout, ctx)

    arange(input.tensor())

    ctx.enqueue_function[
        masked_async_copy_kernel[Layout.row_major(M, N), M - skew_rows]
    ](
        input.device_tensor(),
        grid_dim=(1,),
        block_dim=(8,),
    )

    ctx.synchronize()

    print(input.tensor())

    _ = input^


def run_masked_async_copy_tests(ctx: DeviceContext):
    # CHECK: === test_masked_async_copy
    # CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0
    # CHECK: 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
    # CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0
    # CHECK: 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
    # CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0
    # CHECK: 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0
    # CHECK: 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0
    # CHECK: 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
    test_masked_async_copy[
        Layout.row_major(8, 8),
        M=8,
        N=8,
        skew_rows=1,
    ](ctx)

    # CHECK: === test_masked_async_copy
    # CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0
    # CHECK: 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
    # CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0
    # CHECK: 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
    # CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0
    # CHECK: 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0
    # CHECK: 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0
    # CHECK: 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
    test_masked_async_copy[
        Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE),
        M=8,
        N=8,
        skew_rows=1,
    ](ctx)


# ----------------------------------------------------------------------
# masked copy tests
# ----------------------------------------------------------------------


@always_inline
fn masked_copy_kernel[
    layout: Layout, num_rows: Int
](input: LayoutTensor[DType.float32, layout, MutableAnyOrigin]):
    alias thread_layout = Layout.row_major(4, 2)

    var masked_input = LayoutTensor[
        DType.float32,
        layout,
        MutableAnyOrigin,
        masked=True,
    ](
        input.ptr,
        __type_of(input.runtime_layout)(
            __type_of(input.runtime_layout.shape)(num_rows, input.dim(1)),
            input.runtime_layout.stride,
        ),
    )

    var smem_tile = LayoutTensor[
        DType.float32,
        layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation().fill(0)

    copy_dram_to_sram[thread_layout=thread_layout](
        smem_tile.vectorize[1, 4](), masked_input.vectorize[1, 4]()
    )

    barrier()

    copy_sram_to_dram[thread_layout=thread_layout](
        input.vectorize[1, 4](),
        smem_tile.vectorize[1, 4](),
    )


fn test_masked_copy[
    layout: Layout, M: Int, N: Int, skew_rows: Int
](ctx: DeviceContext) raises:
    print("=== test_masked_copy")

    alias managed_layout_tensor_type = ManagedLayoutTensor[
        DType.float32,
        layout,
    ]

    alias element_type = managed_layout_tensor_type.element_type
    alias idx_type = managed_layout_tensor_type.index_type

    alias runtime_layout = RuntimeLayout[
        layout, element_type=element_type, linear_idx_type=idx_type
    ].row_major(IndexList[2, element_type=element_type](M, N))

    var input = ManagedLayoutTensor[
        DType.float32,
        layout,
    ](runtime_layout, ctx)

    arange(input.tensor())

    alias kernel_type = masked_copy_kernel[
        Layout.row_major(M, N), M - skew_rows
    ]
    ctx.enqueue_function[kernel_type](
        input.device_tensor(), grid_dim=(1,), block_dim=(8,)
    )

    ctx.synchronize()

    print(input.tensor())

    _ = input^


def run_masked_copy_tests(ctx: DeviceContext):
    # CHECK: === test_masked_copy
    # CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0
    # CHECK: 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
    # CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0
    # CHECK: 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
    # CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0
    # CHECK: 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0
    # CHECK: 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0
    # CHECK: 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
    test_masked_copy[
        Layout.row_major(8, 8),
        M=8,
        N=8,
        skew_rows=1,
    ](ctx)

    # CHECK: === test_masked_copy
    # CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0
    # CHECK: 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
    # CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0
    # CHECK: 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
    # CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0
    # CHECK: 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0
    # CHECK: 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0
    # CHECK: 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
    test_masked_copy[
        Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE),
        M=8,
        N=8,
        skew_rows=1,
    ](ctx)


# ----------------------------------------------------------------------
# partial copy_dram_to_sram tests
# ----------------------------------------------------------------------


@always_inline
fn partial_copy_dram_to_sram_async_kernel[
    layout: Layout,
    thread_layout: Layout,
    num_threads: Int,
](input: LayoutTensor[DType.float32, layout, MutableAnyOrigin]):
    var smem_tile = LayoutTensor[
        DType.float32,
        layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation().fill(-1.0)

    copy_dram_to_sram_async[
        thread_layout=thread_layout, num_threads=num_threads
    ](smem_tile.vectorize[1, 4](), input.vectorize[1, 4]())

    async_copy_commit_group()
    async_copy_wait_all()

    copy_sram_to_dram[thread_layout=thread_layout](
        input.vectorize[1, 4](),
        smem_tile.vectorize[1, 4](),
    )


fn test_partial_copy_dram_to_sram_async[
    layout: Layout,
    thread_layout: Layout,
    num_threads: Int,
](ctx: DeviceContext) raises:
    print("=== test_partial_copy_dram_to_sram_async")

    var input = ManagedLayoutTensor[
        DType.float32,
        layout,
    ](ctx)

    arange(input.tensor())

    alias kernel_type = partial_copy_dram_to_sram_async_kernel[
        layout,
        thread_layout,
        num_threads,
    ]
    ctx.enqueue_function[kernel_type](
        input.device_tensor(), grid_dim=(1,), block_dim=(num_threads,)
    )

    ctx.synchronize()

    print(input.tensor())

    _ = input^


def run_partial_copy_dram_to_sram_async(ctx: DeviceContext):
    # CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
    # CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
    test_partial_copy_dram_to_sram_async[
        layout = Layout.row_major(2, 16),
        thread_layout = Layout.row_major(2, 4),
        num_threads=32,
    ](ctx)


# ----------------------------------------------------------------------
# copy_sram_to_dram tests
# ----------------------------------------------------------------------


@always_inline
fn copy_sram_to_dram_kernel[
    type: DType,
    layout: Layout,
    M: Int,
    N: Int,
    binary_op: OptionalReg[binary_op_type] = None,
](input: LayoutTensor[type, layout, MutableAnyOrigin]):
    alias simd_size = simdwidthof[type]()
    alias thread_layout = Layout.row_major(simd_size, N // simd_size)

    var smem_tile = LayoutTensor[
        DType.float32,
        Layout.row_major(M, N),
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()
    arange(smem_tile)

    copy_sram_to_dram[thread_layout=thread_layout, binary_op=binary_op](
        input.vectorize[1, simd_size](),
        smem_tile.vectorize[1, simd_size](),
    )


fn test_copy_sram_to_dram[
    type: DType,
    layout: Layout,
    M: Int,
    N: Int,
    skew_M: Int = 0,
    binary_op: OptionalReg[binary_op_type] = None,
](ctx: DeviceContext) raises:
    print("=== test_copy_sram_to_dram")

    alias managed_layout_tensor_type = ManagedLayoutTensor[
        type,
        layout,
    ]

    alias element_type = managed_layout_tensor_type.element_type
    alias idx_type = managed_layout_tensor_type.index_type

    var runtime_layout = RuntimeLayout[
        layout,
        element_type=element_type,
        linear_idx_type=idx_type,
    ].row_major(
        IndexList[
            2,
            element_type=element_type,
        ](M - skew_M, N)
    )

    var input = managed_layout_tensor_type(runtime_layout, ctx)
    _ = input.tensor().fill(-1.0)

    alias tile_layout = Layout.row_major(M - skew_M, N)

    var tile_tensor = input.device_tensor().tile[M - skew_M, N](0, 0)

    alias kernel_type = copy_sram_to_dram_kernel[
        type, tile_layout, M, N, binary_op
    ]
    ctx.enqueue_function[kernel_type](
        tile_tensor, grid_dim=(1,), block_dim=(8,)
    )

    ctx.synchronize()

    print(input.tensor().tile[M - skew_M, N](0, 0))

    _ = input^


def run_copy_sram_to_dram_tests(ctx: DeviceContext):
    # CHECK: == test_copy_sram_to_dram
    # CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0
    # CHECK: 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
    # CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0
    # CHECK: 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
    # CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0
    # CHECK: 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0
    # CHECK: 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0
    # CHECK: 56.0 57.0 58.0 59.0 60.0 61.0 62.0 63.0
    test_copy_sram_to_dram[
        DType.float32,
        Layout.row_major(8, 8),
        M=8,
        N=8,
    ](ctx)

    # CHECK: == test_copy_sram_to_dram
    # CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0
    # CHECK: 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
    # CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0
    # CHECK: 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
    # CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0
    # CHECK: 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0
    # CHECK: 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0
    # CHECK: 56.0 57.0 58.0 59.0 60.0 61.0 62.0 63.0
    test_copy_sram_to_dram[
        DType.bfloat16,
        Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE),
        M=8,
        N=8,
    ](ctx)


# ----------------------------------------------------------------------
# copy_local_to_local tests
# ----------------------------------------------------------------------


@always_inline
fn copy_local_to_local_kernel[
    type: DType, layout: Layout, WM: Int, WN: Int, MMA_M: Int, MMA_N: Int
](output: LayoutTensor[type, layout, MutableAnyOrigin]):
    alias simd_size = 2

    var reg_tile0 = LayoutTensor[
        DType.float32,
        Layout.row_major(MMA_M, MMA_N * simd_size),
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
    ].stack_allocation()
    arange(reg_tile0)

    var reg_tile1 = LayoutTensor[
        DType.bfloat16,
        Layout.row_major(MMA_M, MMA_N * simd_size),
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
    ].stack_allocation().fill(0)

    copy_local_to_local(
        reg_tile1,
        reg_tile0,
    )

    copy_local_to_dram[
        dst_thread_layout = Layout.row_major(
            WM // MMA_M, WN // simd_size // MMA_N
        )
    ](
        output.vectorize[1, simd_size](),
        reg_tile1.vectorize[1, simd_size](),
    )


fn test_copy_local_to_local[
    type: DType,
    WM: Int,
    WN: Int,
    MMA_M: Int,
    MMA_N: Int,
](ctx: DeviceContext) raises:
    print("=== test_copy_local_to_local")

    alias layout = Layout.row_major(WM, WN)
    var output = ManagedLayoutTensor[
        type,
        layout,
    ](ctx)

    alias kernel_type = copy_local_to_local_kernel[
        type, layout, WM, WN, MMA_M, MMA_N
    ]
    ctx.enqueue_function[kernel_type](
        output.device_tensor(), grid_dim=(1, 1), block_dim=(8, 1)
    )

    ctx.synchronize()

    print(output.tensor())

    _ = output^


def run_copy_local_to_local_tests(ctx: DeviceContext):
    # CHECK: === test_copy_local_to_local
    # CHECK: 0.0 1.0 0.0 1.0 2.0 3.0 2.0 3.0 4.0 5.0 4.0 5.0 6.0 7.0 6.0 7.0
    # CHECK: 0.0 1.0 0.0 1.0 2.0 3.0 2.0 3.0 4.0 5.0 4.0 5.0 6.0 7.0 6.0 7.0
    # CHECK: 16.0 17.0 16.0 17.0 18.0 19.0 18.0 19.0 20.0 21.0 20.0 21.0 22.0 23.0 22.0 23.0
    # CHECK: 16.0 17.0 16.0 17.0 18.0 19.0 18.0 19.0 20.0 21.0 20.0 21.0 22.0 23.0 22.0 23.0
    # CHECK: 8.0 9.0 8.0 9.0 10.0 11.0 10.0 11.0 12.0 13.0 12.0 13.0 14.0 15.0 14.0 15.0
    # CHECK: 8.0 9.0 8.0 9.0 10.0 11.0 10.0 11.0 12.0 13.0 12.0 13.0 14.0 15.0 14.0 15.0
    # CHECK: 24.0 25.0 24.0 25.0 26.0 27.0 26.0 27.0 28.0 29.0 28.0 29.0 30.0 31.0 30.0 31.0
    # CHECK: 24.0 25.0 24.0 25.0 26.0 27.0 26.0 27.0 28.0 29.0 28.0 29.0 30.0 31.0 30.0 31.0
    test_copy_local_to_local[
        DType.bfloat16,
        WM=8,
        WN=16,
        MMA_M=4,
        MMA_N=4,
    ](ctx)


# ----------------------------------------------------------------------
# copy_local_to_dram tests
# ----------------------------------------------------------------------


@always_inline
fn copy_local_to_sram_kernel[
    type: DType,
    layout: Layout,
    WM: Int,
    WN: Int,
    MMA_M: Int,
    MMA_N: Int,
    simd_size_row: Int,
    simd_size_col: Int,
](output: LayoutTensor[type, layout, MutableAnyOrigin]):
    var reg_tile0 = LayoutTensor[
        DType.float32,
        Layout.row_major(MMA_M * simd_size_row, MMA_N * simd_size_col),
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
    ].stack_allocation()
    arange(reg_tile0)

    var smem_warp_tile = LayoutTensor[
        type,
        Layout.row_major(WM, WN),
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation().fill(0)

    copy[
        thread_layout = Layout.row_major(
            WM // simd_size_row // MMA_M, WN // simd_size_col // MMA_N
        )
    ](
        smem_warp_tile.vectorize[simd_size_row, simd_size_col](),
        reg_tile0.vectorize[simd_size_row, simd_size_col](),
    )

    copy_sram_to_dram[
        thread_layout = Layout.row_major(
            WM // simd_size_row // MMA_M, WN // simd_size_col // MMA_N
        )
    ](
        output.vectorize[simd_size_row, simd_size_col](),
        smem_warp_tile.vectorize[simd_size_row, simd_size_col](),
    )


fn test_copy_local_to_sram[
    type: DType,
    WM: Int,
    WN: Int,
    MMA_M: Int,
    MMA_N: Int,
    simd_size_row: Int,
    simd_size_col: Int,
](ctx: DeviceContext) raises:
    print(
        "=== test_copy_local_to_sram_",
        type,
        "_simd_size_",
        simd_size_row,
        simd_size_col,
        sep="",
    )

    alias layout = Layout.row_major(WM, WN)
    var output = ManagedLayoutTensor[
        type,
        layout,
    ](ctx)

    alias kernel_type = copy_local_to_sram_kernel[
        type, layout, WM, WN, MMA_M, MMA_N, simd_size_row, simd_size_col
    ]
    ctx.enqueue_function[kernel_type](
        output.device_tensor(), grid_dim=(1, 1), block_dim=(8, 1)
    )

    ctx.synchronize()

    print(output.tensor())

    _ = output^


def run_copy_local_to_sram_tests_float32_simd_size_12(ctx: DeviceContext):
    # CHECK: === test_copy_local_to_sram_float32_simd_size_12
    # CHECK: 0.0 1.0 0.0 1.0 2.0 3.0 2.0 3.0 4.0 5.0 4.0 5.0 6.0 7.0 6.0 7.0
    # CHECK: 0.0 1.0 0.0 1.0 2.0 3.0 2.0 3.0 4.0 5.0 4.0 5.0 6.0 7.0 6.0 7.0
    # CHECK: 8.0 9.0 8.0 9.0 10.0 11.0 10.0 11.0 12.0 13.0 12.0 13.0 14.0 15.0 14.0 15.0
    # CHECK: 8.0 9.0 8.0 9.0 10.0 11.0 10.0 11.0 12.0 13.0 12.0 13.0 14.0 15.0 14.0 15.0
    # CHECK: 16.0 17.0 16.0 17.0 18.0 19.0 18.0 19.0 20.0 21.0 20.0 21.0 22.0 23.0 22.0 23.0
    # CHECK: 16.0 17.0 16.0 17.0 18.0 19.0 18.0 19.0 20.0 21.0 20.0 21.0 22.0 23.0 22.0 23.0
    # CHECK: 24.0 25.0 24.0 25.0 26.0 27.0 26.0 27.0 28.0 29.0 28.0 29.0 30.0 31.0 30.0 31.0
    # CHECK: 24.0 25.0 24.0 25.0 26.0 27.0 26.0 27.0 28.0 29.0 28.0 29.0 30.0 31.0 30.0 31.0
    test_copy_local_to_sram[
        DType.float32,
        WM=8,
        WN=16,
        MMA_M=4,
        MMA_N=4,
        simd_size_row=1,
        simd_size_col=2,
    ](ctx)


def run_copy_local_to_sram_tests_float32_simd_size_21(ctx: DeviceContext):
    # CHECK: === test_copy_local_to_sram_float32_simd_size_21
    # CHECK: 0.0 0.0 0.0 0.0 1.0 1.0 1.0 1.0 2.0 2.0 2.0 2.0 3.0 3.0 3.0 3.0
    # CHECK: 4.0 4.0 4.0 4.0 5.0 5.0 5.0 5.0 6.0 6.0 6.0 6.0 7.0 7.0 7.0 7.0
    # CHECK: 8.0 8.0 8.0 8.0 9.0 9.0 9.0 9.0 10.0 10.0 10.0 10.0 11.0 11.0 11.0 11.0
    # CHECK: 12.0 12.0 12.0 12.0 13.0 13.0 13.0 13.0 14.0 14.0 14.0 14.0 15.0 15.0 15.0 15.0
    # CHECK: 16.0 16.0 16.0 16.0 17.0 17.0 17.0 17.0 18.0 18.0 18.0 18.0 19.0 19.0 19.0 19.0
    # CHECK: 20.0 20.0 20.0 20.0 21.0 21.0 21.0 21.0 22.0 22.0 22.0 22.0 23.0 23.0 23.0 23.0
    # CHECK: 24.0 24.0 24.0 24.0 25.0 25.0 25.0 25.0 26.0 26.0 26.0 26.0 27.0 27.0 27.0 27.0
    # CHECK: 28.0 28.0 28.0 28.0 29.0 29.0 29.0 29.0 30.0 30.0 30.0 30.0 31.0 31.0 31.0 31.0
    test_copy_local_to_sram[
        DType.float32,
        WM=8,
        WN=16,
        MMA_M=4,
        MMA_N=4,
        simd_size_row=2,
        simd_size_col=1,
    ](ctx)


def run_copy_local_to_sram_tests_bfloat16_simd_size_12(ctx: DeviceContext):
    # CHECK: === test_copy_local_to_sram_bfloat16_simd_size_12
    # CHECK: 0.0 1.0 0.0 1.0 2.0 3.0 2.0 3.0 4.0 5.0 4.0 5.0 6.0 7.0 6.0 7.0
    # CHECK: 0.0 1.0 0.0 1.0 2.0 3.0 2.0 3.0 4.0 5.0 4.0 5.0 6.0 7.0 6.0 7.0
    # CHECK: 8.0 9.0 8.0 9.0 10.0 11.0 10.0 11.0 12.0 13.0 12.0 13.0 14.0 15.0 14.0 15.0
    # CHECK: 8.0 9.0 8.0 9.0 10.0 11.0 10.0 11.0 12.0 13.0 12.0 13.0 14.0 15.0 14.0 15.0
    # CHECK: 16.0 17.0 16.0 17.0 18.0 19.0 18.0 19.0 20.0 21.0 20.0 21.0 22.0 23.0 22.0 23.0
    # CHECK: 16.0 17.0 16.0 17.0 18.0 19.0 18.0 19.0 20.0 21.0 20.0 21.0 22.0 23.0 22.0 23.0
    # CHECK: 24.0 25.0 24.0 25.0 26.0 27.0 26.0 27.0 28.0 29.0 28.0 29.0 30.0 31.0 30.0 31.0
    # CHECK: 24.0 25.0 24.0 25.0 26.0 27.0 26.0 27.0 28.0 29.0 28.0 29.0 30.0 31.0 30.0 31.0
    test_copy_local_to_sram[
        DType.bfloat16,
        WM=8,
        WN=16,
        MMA_M=4,
        MMA_N=4,
        simd_size_row=1,
        simd_size_col=2,
    ](ctx)


def run_copy_local_to_sram_tests_bfloat16_simd_size_21(ctx: DeviceContext):
    # CHECK: === test_copy_local_to_sram_bfloat16_simd_size_21
    # CHECK: 0.0 0.0 0.0 0.0 1.0 1.0 1.0 1.0 2.0 2.0 2.0 2.0 3.0 3.0 3.0 3.0
    # CHECK: 4.0 4.0 4.0 4.0 5.0 5.0 5.0 5.0 6.0 6.0 6.0 6.0 7.0 7.0 7.0 7.0
    # CHECK: 8.0 8.0 8.0 8.0 9.0 9.0 9.0 9.0 10.0 10.0 10.0 10.0 11.0 11.0 11.0 11.0
    # CHECK: 12.0 12.0 12.0 12.0 13.0 13.0 13.0 13.0 14.0 14.0 14.0 14.0 15.0 15.0 15.0 15.0
    # CHECK: 16.0 16.0 16.0 16.0 17.0 17.0 17.0 17.0 18.0 18.0 18.0 18.0 19.0 19.0 19.0 19.0
    # CHECK: 20.0 20.0 20.0 20.0 21.0 21.0 21.0 21.0 22.0 22.0 22.0 22.0 23.0 23.0 23.0 23.0
    # CHECK: 24.0 24.0 24.0 24.0 25.0 25.0 25.0 25.0 26.0 26.0 26.0 26.0 27.0 27.0 27.0 27.0
    # CHECK: 28.0 28.0 28.0 28.0 29.0 29.0 29.0 29.0 30.0 30.0 30.0 30.0 31.0 31.0 31.0 31.0
    test_copy_local_to_sram[
        DType.bfloat16,
        WM=8,
        WN=16,
        MMA_M=4,
        MMA_N=4,
        simd_size_row=2,
        simd_size_col=1,
    ](ctx)


fn main() raises:
    with DeviceContext() as ctx:
        run_async_copy_tests(ctx)
        run_dynamic_async_copy_tests(ctx)
        run_swizzle_copy_tests(ctx)
        run_masked_async_copy_tests(ctx)
        run_masked_copy_tests(ctx)
        run_partial_copy_dram_to_sram_async(ctx)
        run_copy_sram_to_dram_tests(ctx)
        run_copy_local_to_local_tests(ctx)
        run_copy_local_to_sram_tests_float32_simd_size_12(ctx)
        run_copy_local_to_sram_tests_float32_simd_size_21(ctx)
        run_copy_local_to_sram_tests_bfloat16_simd_size_12(ctx)
        run_copy_local_to_sram_tests_bfloat16_simd_size_21(ctx)
