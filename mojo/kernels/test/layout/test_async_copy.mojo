# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug --debug-level full %s | FileCheck %s

from math import ceildiv
from pathlib import Path
from sys import simdwidthof, bitwidthof

from gpu import barrier
from gpu.host import DeviceContext
from gpu.id import BlockIdx, ThreadIdx
from gpu.memory import (
    AddressSpace,
    async_copy_commit_group,
    async_copy_wait_all,
    async_copy_wait_group,
)
from layout import *
from layout._utils import (
    ManagedLayoutTensor,
    gpu_free,
    gpu_managed_alloc,
    gpu_managed_alloc_runtime,
)
from layout.fillers import arange
from layout.layout_tensor import (
    UNKNOWN_VALUE,
    LayoutTensor,
    copy_dram_to_sram_async,
    copy_sram_to_dram,
)
from memory import UnsafePointer
from testing import assert_almost_equal

from utils import IndexList


fn async_copy_kernel[
    input_layout: Layout,
    BM: Int,
    BN: Int,
](input: LayoutTensor[DType.float32, input_layout]):
    var input_tile = input.tile[BM, BN](BlockIdx.y(), BlockIdx.x())

    var smem_tile = LayoutTensor[
        DType.float32,
        Layout(IntTuple(BM, BN)),
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    smem_tile.copy_from_async(input_tile)
    async_copy_wait_all()

    var tx = ThreadIdx.x()
    var ty = ThreadIdx.y()
    smem_tile[tx, ty] += ty

    input_tile.copy_from(smem_tile)


fn test_async_copy[
    layout: Layout, M: Int, N: Int, BM: Int, BN: Int
](ctx: DeviceContext) raises:
    print("=== test_async_copy")

    alias runtime_layout = RuntimeLayout[
        layout, bitwidth = bitwidthof[Int]()
    ].row_major(IndexList[2](M, N))

    var input = ManagedLayoutTensor[
        DType.float32,
        layout,
        gpu_managed_alloc,
        gpu_free,
        gpu_managed_alloc_runtime,
    ](runtime_layout)

    arange(input.tensor)

    alias kernel_type = async_copy_kernel[layout, BM, BN]

    var kernel = ctx.compile_function[kernel_type]()

    ctx.enqueue_function(
        kernel,
        input,
        grid_dim=(M // BM, N // BN),
        block_dim=(BM, BN),
    )

    ctx.synchronize()
    print(input.tensor)

    _ = input^


fn async_dynamic_copy_kernel[
    input_layout: Layout,
    output_layout: Layout,
    BM: Int,
    BN: Int,
    /,
    _is_homogeneous: Bool = False,
](
    input: LayoutTensor[
        DType.float32,
        input_layout,
        __experimental_non_homogeneous_tile=_is_homogeneous,
    ],
    output: LayoutTensor[
        DType.float32,
        output_layout,
        __experimental_non_homogeneous_tile=_is_homogeneous,
    ],
):
    var input_tile = input.tile[BM, BN](BlockIdx.x(), BlockIdx.y())
    var output_tile = output.tile[BM, BN](BlockIdx.x(), BlockIdx.y())

    var smem_tile = LayoutTensor[
        DType.float32,
        Layout(IntTuple(BM, BN)),
        address_space = AddressSpace.SHARED,
        __experimental_non_homogeneous_tile=True,
    ].stack_allocation()

    smem_tile.copy_from_async(input_tile)
    async_copy_wait_all()

    output_tile.copy_from(smem_tile)


fn test_dynamic_async_copy[
    M: Int, N: Int, BM: Int, BN: Int, /, skew_M: Int = 0, skew_N: Int = 0
](ctx: DeviceContext) raises:
    print("=== test_dynamic_async_copy")

    alias unknown_layout = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)

    alias input_runtime_layout = RuntimeLayout[unknown_layout].row_major(
        IndexList[2](M, N)
    )

    alias output_runtime_layout = RuntimeLayout[unknown_layout].row_major(
        IndexList[2](M - skew_M, N - skew_N)
    )

    var input = ManagedLayoutTensor[
        DType.float32,
        unknown_layout,
        gpu_managed_alloc,
        gpu_free,
        gpu_managed_alloc_runtime,
        __experimental_non_homogeneous_tile=True,
    ](input_runtime_layout)
    arange(input.tensor)

    var output = ManagedLayoutTensor[
        DType.float32,
        unknown_layout,
        gpu_managed_alloc,
        gpu_free,
        gpu_managed_alloc_runtime,
        __experimental_non_homogeneous_tile=True,
    ](output_runtime_layout)

    alias kernel_type = async_dynamic_copy_kernel[
        unknown_layout, unknown_layout, BM, BN, _is_homogeneous=True
    ]

    var kernel = ctx.compile_function[kernel_type]()

    ctx.enqueue_function(
        kernel,
        input.tensor,
        output.tensor,
        grid_dim=(ceildiv(M, BM), ceildiv(M, BN)),
        block_dim=(1, 1),
    )

    ctx.synchronize()
    print(output.tensor)

    _ = input^
    _ = output^


fn multistage_copy[
    type: DType,
    a_layout: Layout,
    b_layout: Layout,
    BM: Int,
    BK: Int,
    num_threads: Int,
    num_pipeline_stages: Int,
](a: LayoutTensor[type, a_layout], b: LayoutTensor[type, b_layout]):
    constrained[num_pipeline_stages >= 2, "Require at least 2 stages."]()

    alias simd_size = simdwidthof[type]()

    var M = a.shape[0]()
    var K = a.shape[1]()

    # Double buffer in shared memory.
    var a_smem_tiles = LayoutTensor[
        type,
        Layout.row_major((num_pipeline_stages - 1) * BM, BK),
        address_space = AddressSpace.SHARED,
    ].stack_allocation().split[num_pipeline_stages - 1]()

    alias thread_layout = Layout.row_major(
        num_threads * simd_size // BK, BK // simd_size
    )

    # Prefetch (num_pipeline_stages - 1) stages.
    @parameter
    for stage in range(num_pipeline_stages - 1):
        copy_dram_to_sram_async[
            src_thread_layout=thread_layout,
            dst_thread_layout=thread_layout,
        ](
            a_smem_tiles[stage].vectorize[1, simd_size](),
            a.tile[BM, BK](BlockIdx.x(), stage).vectorize[1, simd_size](),
        )

        async_copy_commit_group()

    # Guard stage 0.
    async_copy_wait_group(num_pipeline_stages - 2)
    barrier()

    var num_k_tiles = ceildiv(K, BK)

    for k_tile_id in range(num_k_tiles):
        var stage = k_tile_id % (num_pipeline_stages - 1)

        # Write current stage to global memory.
        var b_gmem_tile = b.tile[BM, BK](BlockIdx.x(), k_tile_id)
        var b_gmem_frag = b_gmem_tile.vectorize[1, simd_size]().distribute[
            thread_layout
        ](ThreadIdx.x())
        var a_smem_frag = a_smem_tiles[stage].vectorize[
            1, simd_size
        ]().distribute[thread_layout](ThreadIdx.x())
        b_gmem_frag.copy_from(a_smem_frag)

        # Prefetch stage $(current + num_pipeline_stages - 1)
        # When the prefetch goes OOB, Cutlass sets src_in_bytes to 0 and does
        # zero fill (zfill) for dst. We circulate the global address for now
        # because llvm instrinsic doesn't have src_in_bytes.
        var prefetch_tile_id = k_tile_id + num_pipeline_stages - 1
        var prefetch_stage = prefetch_tile_id % (num_pipeline_stages - 1)

        copy_dram_to_sram_async[
            src_thread_layout=thread_layout,
            dst_thread_layout=thread_layout,
        ](
            a_smem_tiles[prefetch_stage].vectorize[1, simd_size](),
            a.tile[BM, BK](
                BlockIdx.x(), prefetch_tile_id % num_k_tiles
            ).vectorize[1, simd_size](),
        )

        async_copy_commit_group()

        async_copy_wait_group(num_pipeline_stages - 2)
        barrier()


fn test_multistage_copy[
    a_layout: Layout,
    b_layout: Layout,
    M: Int,
    K: Int,
    BM: Int,
    BK: Int,
    num_threads: Int,
    num_pipeline_stages: Int,
](ctx: DeviceContext) raises:
    print("=== test_multistage_copy")

    constrained[
        K // BK >= num_pipeline_stages,
        "Require more k tiles than pipeline stages.",
    ]()

    var a_host = UnsafePointer[Float32].alloc(M * K)
    var b_host = UnsafePointer[Float32].alloc(M * K)

    for i in range(M * K):
        a_host[i] = i
        b_host[i] = 0

    var a_device = ctx.create_buffer[DType.float32](M * K)
    var b_device = ctx.create_buffer[DType.float32](M * K)

    ctx.enqueue_copy_to_device(a_device, a_host)

    alias a_runtime_layout = RuntimeLayout[a_layout].row_major(
        IndexList[2](M, K)
    )
    alias b_runtime_layout = RuntimeLayout[b_layout].row_major(
        IndexList[2](M, K)
    )

    var a_tensor = LayoutTensor[DType.float32, a_layout](
        a_device.ptr, a_runtime_layout
    )
    var b_tensor = LayoutTensor[DType.float32, b_layout](
        b_device.ptr, b_runtime_layout
    )

    alias copy = multistage_copy[
        DType.float32,
        a_layout,
        b_layout,
        BM,
        BK,
        num_threads,
        num_pipeline_stages,
    ]
    var func = ctx.compile_function[copy](threads_per_block=num_threads)

    ctx.enqueue_function(
        func,
        a_tensor,
        b_tensor,
        grid_dim=(ceildiv(M, BM), 1, 1),
        block_dim=(num_threads, 1, 1),
    )

    ctx.synchronize()

    ctx.enqueue_copy_from_device(b_host, b_device)

    for i in range(M * K):
        assert_almost_equal(a_host[i], b_host[i])

    _ = a_device
    _ = b_device

    a_host.free()
    b_host.free()


fn swizzle_copy[
    type: DType,
    a_layout: Layout,
    b_layout: Layout,
    BM: Int,
    BK: Int,
    num_threads: Int,
    /,
    _is_homogeneous: Bool = False,
](
    a: LayoutTensor[
        type, a_layout, __experimental_non_homogeneous_tile=_is_homogeneous
    ],
    b: LayoutTensor[
        type, b_layout, __experimental_non_homogeneous_tile=_is_homogeneous
    ],
):
    alias simd_size = simdwidthof[type]()

    var M = a.shape[0]()
    var K = a.shape[1]()

    # Double buffer in shared memory.
    var a_smem_tile = LayoutTensor[
        type,
        Layout.row_major(BM, BK),
        address_space = AddressSpace.SHARED,
    ].stack_allocation().fill(0)

    alias thread_layout = Layout.row_major(
        num_threads * simd_size // BK, BK // simd_size
    )

    copy_dram_to_sram_async[
        src_thread_layout=thread_layout,
        dst_thread_layout=thread_layout,
        swizzle=True,
    ](
        a_smem_tile.vectorize[1, simd_size](),
        a.tile[BM, BK](BlockIdx.x(), 0).vectorize[1, simd_size](),
    )

    async_copy_wait_all()
    barrier()

    # Write current stage to global memory.
    var b_gmem_tile = b.tile[BM, BK](BlockIdx.x(), 0)
    var b_gmem_frag = b_gmem_tile.vectorize[1, simd_size]().distribute[
        thread_layout
    ](ThreadIdx.x())
    var a_smem_frag = a_smem_tile.vectorize[1, simd_size]().distribute[
        thread_layout
    ](ThreadIdx.x())
    b_gmem_frag.copy_from(a_smem_frag)


fn test_swizzle_copy[
    a_layout: Layout,
    b_layout: Layout,
    M: Int,
    K: Int,
    BM: Int,
    BK: Int,
    num_threads: Int,
    /,
    skew_M: Int = 0,
    _is_homogeneous: Bool = False,
](ctx: DeviceContext) raises:
    print("=== test_swizzle_copy")

    alias a_runtime_layout = RuntimeLayout[a_layout].row_major(
        IndexList[2]((M - skew_M), K)
    )
    alias b_runtime_layout = RuntimeLayout[b_layout].row_major(
        IndexList[2](M, K)
    )

    var a_tensor = ManagedLayoutTensor[
        DType.float32,
        a_layout,
        gpu_managed_alloc,
        gpu_free,
        gpu_managed_alloc_runtime,
    ](a_runtime_layout)
    arange(a_tensor.tensor)

    var b_tensor = ManagedLayoutTensor[
        DType.float32,
        b_layout,
        gpu_managed_alloc,
        gpu_free,
        gpu_managed_alloc_runtime,
    ](b_runtime_layout)

    alias copy = swizzle_copy[
        DType.float32,
        a_layout,
        b_layout,
        BM,
        BK,
        num_threads,
        _is_homogeneous=_is_homogeneous,
    ]
    var func = ctx.compile_function[copy](threads_per_block=num_threads)

    ctx.enqueue_function(
        func,
        a_tensor.tensor,
        b_tensor.tensor,
        grid_dim=(ceildiv(M, BM), 1, 1),
        block_dim=(num_threads, 1, 1),
    )

    ctx.synchronize()

    print(b_tensor.tensor)

    _ = a_tensor^
    _ = b_tensor^


@always_inline
fn masked_copy_kernel[
    layout: Layout, num_rows: Int
](input: LayoutTensor[DType.float32, layout]):
    alias thread_layout = Layout.row_major(4, 2)

    var smem_tile = LayoutTensor[
        DType.float32, layout, address_space = AddressSpace.SHARED
    ].stack_allocation().fill(-1.0)

    copy_dram_to_sram_async[thread_layout=thread_layout, masked=True](
        smem_tile.vectorize[1, 4](), input.vectorize[1, 4](), num_rows
    )

    async_copy_commit_group()
    async_copy_wait_all()

    copy_sram_to_dram[thread_layout=thread_layout](
        input.vectorize[1, 4]().bitcast[
            DType.float32, address_space = AddressSpace.GENERIC
        ](),
        smem_tile.vectorize[1, 4](),
    )


fn test_masked_async_copy[
    layout: Layout, M: Int, N: Int, num_rows: Int
](ctx: DeviceContext) raises:
    print("=== test_masked_async_copy")

    # alias num_threads = thread_layout.size()

    alias runtime_layout = RuntimeLayout[layout].row_major(IndexList[2](M, N))

    var input = ManagedLayoutTensor[
        DType.float32,
        layout,
        gpu_managed_alloc,
        gpu_free,
        gpu_managed_alloc_runtime,
    ](runtime_layout)

    arange(input.tensor)

    alias kernel_type = masked_copy_kernel[Layout.row_major(M, N), num_rows]
    var kernel = ctx.compile_function[kernel_type]()

    ctx.enqueue_function(
        kernel,
        input,
        grid_dim=(1,),
        block_dim=(8,),
    )

    ctx.synchronize()
    print(input.tensor)

    _ = input^


@always_inline
fn copy_sram_to_dram_kernel[
    type: DType,
    layout: Layout,
    M: Int,
    N: Int,
    /,
    __non_homogeneous_tile: Bool = False,
](
    input: LayoutTensor[
        type, layout, __experimental_non_homogeneous_tile=__non_homogeneous_tile
    ]
):
    alias simd_size = simdwidthof[type]()
    alias thread_layout = Layout.row_major(simd_size, N // simd_size)

    var smem_tile = LayoutTensor[
        DType.float32,
        Layout.row_major(M, N),
        address_space = AddressSpace.SHARED,
    ].stack_allocation()
    arange(smem_tile)

    copy_sram_to_dram[thread_layout=thread_layout](
        input.vectorize[1, simd_size](),
        smem_tile.vectorize[1, simd_size](),
    )


fn test_copy_sram_to_dram[
    type: DType,
    layout: Layout,
    M: Int,
    N: Int,
    /,
    *,
    skew_M: Int = 0,
    skew_N: Int = 0,
    __non_homogeneous_tile: Bool = False,
](ctx: DeviceContext) raises:
    print("=== test_copy_sram_to_dram")

    alias runtime_layout = RuntimeLayout[layout].row_major(
        IndexList[2](M - skew_M, N - skew_N)
    )

    var input = ManagedLayoutTensor[
        type,
        layout,
        gpu_managed_alloc,
        gpu_free,
        gpu_managed_alloc_runtime,
        __experimental_non_homogeneous_tile=__non_homogeneous_tile,
    ](runtime_layout)

    alias tile_layout = Layout.row_major(M - skew_M, N - skew_N)

    var tile_tensor = input.tensor.tile[M - skew_M, N - skew_N](0, 0)

    alias kernel_type = copy_sram_to_dram_kernel[
        type, tile_layout, M, N, __non_homogeneous_tile
    ]
    var kernel = ctx.compile_function[kernel_type]()

    ctx.enqueue_function(
        kernel,
        tile_tensor,
        grid_dim=(1,),
        block_dim=(8,),
    )

    ctx.synchronize()
    print(tile_tensor)

    _ = input^


fn main() raises:
    alias unknown_layout = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)

    # TODO: clean up these tests
    with DeviceContext() as ctx:
        # Matrix dimension
        alias M = 6
        alias N = 6
        # Block dimension
        alias BM = 2
        alias BN = 3
        # CHECK: === test_async_copy
        # CHECK: 0.0   2.0   4.0   3.0   5.0   7.0
        # CHECK: 6.0   8.0   10.0   9.0   11.0   13.0
        # CHECK: 12.0   14.0   16.0   15.0   17.0   19.0
        # CHECK: 18.0   20.0   22.0   21.0   23.0   25.0
        # CHECK: 24.0   26.0   28.0   27.0   28.0   29.0
        # CHECK: 30.0   31.0   32.0   33.0   34.0   35.0
        alias layout = Layout.row_major(M, N)
        test_async_copy[layout, M, N, BM, BN](ctx)

        # CHECK: === test_async_copy
        # CHECK: 0.0   2.0   4.0   3.0   5.0   7.0
        # CHECK: 6.0   8.0   10.0   9.0   11.0   13.0
        # CHECK: 12.0   14.0   16.0   15.0   17.0   19.0
        # CHECK: 18.0   20.0   22.0   21.0   23.0   25.0
        # CHECK: 24.0   26.0   28.0   27.0   28.0   29.0
        # CHECK: 30.0   31.0   32.0   33.0   34.0   35.0
        test_async_copy[unknown_layout, M, N, BM, BN](ctx)

        # CHECK: === test_dynamic_async_copy
        # CHECK: 0.0 1.0 2.0 3.0 4.0 5.0
        # CHECK: 6.0 7.0 8.0 9.0 10.0 11.0
        # CHECK: 12.0 13.0 14.0 15.0 16.0 17.0
        # CHECK: 18.0 19.0 20.0 21.0 22.0 23.0
        # CHECK: 24.0 25.0 26.0 27.0 28.0 29.0
        test_dynamic_async_copy[M, N, BM, BN, skew_M=1, skew_N=0](ctx)

        alias num_threads = 256
        alias num_pipeline_stages = 4
        alias M_multi = 128
        alias K_multi = 128
        alias BM_multi = 128
        alias BK_multi = 16

        # CHECK: === test_multistage_copy
        alias a_layout = Layout.row_major(M_multi, K_multi)
        alias b_layout = Layout.row_major(M_multi, K_multi)
        test_multistage_copy[
            a_layout,
            b_layout,
            M_multi,
            K_multi,
            BM_multi,
            BK_multi,
            num_threads,
            num_pipeline_stages,
        ](ctx)

        # CHECK: === test_multistage_copy
        test_multistage_copy[
            unknown_layout,
            unknown_layout,
            M_multi,
            K_multi,
            BM_multi,
            BK_multi,
            num_threads,
            num_pipeline_stages,
        ](ctx)

        alias num_threads_swizz = 32
        alias M_swizz = 8
        alias K_swizz = 16
        alias BM_swizz = 8
        alias BK_swizz = 16
        # CHECK: === test_swizzle_copy
        # CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
        # CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
        # CHECK: 36.0 37.0 38.0 39.0 32.0 33.0 34.0 35.0 44.0 45.0 46.0 47.0 40.0 41.0 42.0 43.0
        # CHECK: 52.0 53.0 54.0 55.0 48.0 49.0 50.0 51.0 60.0 61.0 62.0 63.0 56.0 57.0 58.0 59.0
        # CHECK: 72.0 73.0 74.0 75.0 76.0 77.0 78.0 79.0 64.0 65.0 66.0 67.0 68.0 69.0 70.0 71.0
        # CHECK: 88.0 89.0 90.0 91.0 92.0 93.0 94.0 95.0 80.0 81.0 82.0 83.0 84.0 85.0 86.0 87.0
        # CHECK: 108.0 109.0 110.0 111.0 104.0 105.0 106.0 107.0 100.0 101.0 102.0 103.0 96.0 97.0 98.0 99.0
        # CHECK: 124.0 125.0 126.0 127.0 120.0 121.0 122.0 123.0 116.0 117.0 118.0 119.0 112.0 113.0 114.0 115.0
        alias a_layout_swizz = Layout.row_major(M_swizz, K_swizz)
        alias b_layout_swizz = Layout.row_major(M_swizz, K_swizz)
        test_swizzle_copy[
            a_layout_swizz,
            b_layout_swizz,
            M_swizz,
            K_swizz,
            BM_swizz,
            BK_swizz,
            num_threads_swizz,
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
            unknown_layout,
            unknown_layout,
            M_swizz,
            K_swizz,
            BM_swizz,
            BK_swizz,
            num_threads_swizz,
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
            Layout.row_major(UNKNOWN_VALUE, K_swizz),
            Layout.row_major(UNKNOWN_VALUE, K_swizz),
            M_swizz,
            K_swizz,
            BM_swizz,
            BK_swizz,
            num_threads_swizz,
            _is_homogeneous=True,
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
            Layout.row_major(UNKNOWN_VALUE, K_swizz),
            Layout.row_major(UNKNOWN_VALUE, K_swizz),
            M_swizz,
            K_swizz,
            BM_swizz,
            BK_swizz,
            num_threads_swizz,
            skew_M=2,
            _is_homogeneous=True,
        ](ctx)

        alias M_masked = 8
        alias N_masked = 8
        alias num_rows = 7
        # CHECK: === test_masked_async_copy
        # CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0
        # CHECK: 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
        # CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0
        # CHECK: 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
        # CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0
        # CHECK: 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0
        # CHECK: 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0
        # CHECK: 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
        alias layout_masked = Layout.row_major(M_masked, N_masked)
        test_masked_async_copy[layout_masked, M_masked, N_masked, num_rows](ctx)

        # CHECK: === test_masked_async_copy
        # CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0
        # CHECK: 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
        # CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0
        # CHECK: 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
        # CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0
        # CHECK: 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0
        # CHECK: 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0
        # CHECK: 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
        test_masked_async_copy[unknown_layout, M_masked, N_masked, num_rows](
            ctx
        )

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
            DType.float32, layout_masked, M_masked, N_masked
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
            DType.bfloat16, unknown_layout, M_masked, N_masked
        ](ctx)

        # === test_copy_sram_to_dram
        # 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0
        # 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
        # 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0
        # 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
        # 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0
        # 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0
        # 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0
        test_copy_sram_to_dram[
            DType.bfloat16,
            Layout.row_major(UNKNOWN_VALUE, 8),
            8,
            8,
            skew_M=1,
            skew_N=0,
            __non_homogeneous_tile=True,
        ](ctx)
