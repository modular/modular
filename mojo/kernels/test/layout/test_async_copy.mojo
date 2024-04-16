# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo-no-debug %s | FileCheck %s

from gpu import AddressSpace, barrier
from gpu.host import Context, Function, synchronize, Stream
from gpu.host.memory import (
    _copy_device_to_host,
    _copy_host_to_device,
    _free,
    _malloc,
)
from gpu.id import BlockDim, BlockIdx, ThreadIdx
from gpu.memory import (
    async_copy_wait_all,
    async_copy_commit_group,
    async_copy_wait_group,
)
from layout import *
from layout._utils import ManagedLayoutTensor, gpu_free, gpu_managed_alloc
from layout.layout_tensor import LayoutTensor, copy_dram_to_sram_async
from layout.int_tuple import int
from math import div_ceil, isclose, max
from memory.unsafe import DTypePointer
from pathlib import Path
from testing import assert_almost_equal


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

    input_tile.copy_from_numa(smem_tile)


fn test_async_copy() raises:
    print("=== test_async_copy")
    # Matrix dimension
    alias M = 6
    alias N = 6
    # Block dimension
    alias BM = 2
    alias BN = 3

    alias input_layout = Layout(IntTuple(M, N), IntTuple(N, 1))
    var input = ManagedLayoutTensor[
        DType.float32, input_layout, gpu_managed_alloc, gpu_free
    ]()

    input.tensor.linspace()

    alias kernel_type = async_copy_kernel[input_layout, BM, BN]

    var kernel = Function[__type_of(kernel_type), kernel_type]()

    kernel(
        input,
        grid_dim=(M // BM, N // BN),
        block_dim=(BM, BN),
    )

    synchronize()
    input.tensor.print()

    _ = input^


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

    var M = a.dim[0]()
    var K = a.dim[1]()

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
    @unroll
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

    var num_k_tiles = div_ceil(K, BK)

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
        b_gmem_frag.copy_from_numa(a_smem_frag)

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


fn test_multistage_copy() raises:
    alias num_threads = 256
    alias num_pipeline_stages = 4
    alias M = 128
    alias K = 128
    alias BM = 128
    alias BK = 16

    constrained[
        K // BK >= num_pipeline_stages,
        "Require more k tiles than pipeline stages.",
    ]()

    var stream = Stream()

    alias a_layout = Layout.row_major(M, K)
    alias b_layout = Layout.row_major(M, K)

    var a_host = DTypePointer[DType.float32].alloc(M * K)
    var b_host = DTypePointer[DType.float32].alloc(M * K)

    for i in range(M * K):
        a_host[i] = i
        b_host[i] = 0

    var a_device = _malloc[Float32](M * K)
    var b_device = _malloc[Float32](M * K)

    _copy_host_to_device(a_device, a_host, M * K)

    var a_tensor = LayoutTensor[DType.float32, a_layout](a_device)
    var b_tensor = LayoutTensor[DType.float32, b_layout](b_device)

    alias copy = multistage_copy[
        DType.float32,
        a_layout,
        b_layout,
        BM,
        BK,
        num_threads,
        num_pipeline_stages,
    ]
    var func = Function[__type_of(copy), copy](
        threads_per_block=num_threads, dump_ptx=Path("./copy.ptx")
    )

    func(
        a_tensor,
        b_tensor,
        grid_dim=(div_ceil(M, BM), 1, 1),
        block_dim=(num_threads, 1, 1),
        stream=stream,
    )

    synchronize()

    _copy_device_to_host(b_host, b_device, M * K)

    for i in range(M * K):
        assert_almost_equal(a_host[i], b_host[i])

    _free(a_device)
    _free(b_device)

    a_host.free()
    b_host.free()

    _ = func^
    _ = stream^


fn main() raises:
    with Context() as ctx:
        # CHECK: === test_async_copy
        # CHECK: 0.0   2.0   4.0   3.0   5.0   7.0
        # CHECK: 6.0   8.0   10.0   9.0   11.0   13.0
        # CHECK: 12.0   14.0   16.0   15.0   17.0   19.0
        # CHECK: 18.0   20.0   22.0   21.0   23.0   25.0
        # CHECK: 24.0   26.0   28.0   27.0   28.0   29.0
        # CHECK: 30.0   31.0   32.0   33.0   34.0   35.0
        test_async_copy()
        test_multistage_copy()
