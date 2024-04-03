# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo %s

from math import div_ceil, isclose, isnan
from buffer import NDBuffer
from buffer.list import DimList
from memory.unsafe import DTypePointer
from memory.unsafe import _GPUAddressSpace as AddressSpace
from Matmul import matmul_kernel_naive
from gpu import (
    WARP_SIZE,
    BlockDim,
    BlockIdx,
    ThreadIdx,
    barrier,
    lane_id,
)
from gpu.host import Context, Function, synchronize, Stream
from gpu.host.event import time_function
from gpu.host.memory import (
    _copy_device_to_host,
    _copy_host_to_device,
    _free,
    _malloc,
)
from gpu.memory import async_copy, async_copy_wait_all
from testing import assert_almost_equal
from sys import argv
from layout._utils import ManagedLayoutTensor, gpu_free, gpu_managed_alloc
from layout.int_tuple import IntTuple
from layout.layout import *
from layout.layout_tensor import LayoutTensor, outer_product_acc
from gpu.device_print import _printf
from pathlib import Path


fn is_benchmark() -> Bool:
    for arg in argv():
        if arg == "--benchmark" or arg == "-benchmark":
            return True
    return False


@always_inline
fn swap_ptr(
    inout tensor0: LayoutTensor,
    inout tensor1: LayoutTensor[
        tensor0.layout, tensor0.dtype, address_space = tensor0.address_space
    ],
):
    var tmp = tensor0.ptr
    tensor0.ptr = tensor1.ptr
    tensor1.ptr = tmp


fn sgemm_double_buffer[
    c_type: DType,
    c_layout: Layout,
    a_type: DType,
    a_layout: Layout,
    b_type: DType,
    b_layout: Layout,
    itype: DType,
    BM: Int,
    BN: Int,
    BK: Int,
    WM: Int,
    WN: Int,
    TM: Int,
    TN: Int,
    NUM_THREADS: Int,
](
    c: LayoutTensor[c_layout, c_type],
    a: LayoutTensor[a_layout, a_type],
    b: LayoutTensor[b_layout, b_type],
):
    alias _uint = Scalar[itype]

    alias simd_size = simdwidthof[c_type]()
    alias a_align = alignof[SIMD[a_type, simd_size]]()
    alias b_align = alignof[SIMD[b_type, simd_size]]()
    alias c_align = alignof[SIMD[c_type, simd_size]]()

    var M = c.dim[0]()
    var N = c.dim[1]()
    var K = a.dim[1]()

    alias num_warps_m = BM // WM
    alias num_warps_n = BN // WN

    var tid = ThreadIdx.x()
    var warp_id = tid // WARP_SIZE
    var lane_id = tid % WARP_SIZE

    # Coordinates of the current warp.
    var warp_x = warp_id % num_warps_n
    var warp_y = warp_id // num_warps_n

    # Warp shape in 2D.
    alias warp_dim_x = WN // TN
    alias warp_dim_y = WM // TM
    constrained[
        warp_dim_x * warp_dim_y == WARP_SIZE,
        "Warp 2d shape doesn't match 32 threads",
    ]()

    # Pad BM to avoid back conflict
    alias pad_avoid_bank_conflict = 4
    alias BM_padded = BM + pad_avoid_bank_conflict

    # Double buffer in shared memory.
    alias a_smem_size = BK * BM_padded
    var a_smem_base = LayoutTensor[
        Layout.row_major(2 * BK, BM_padded),
        a_type,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()
    var a_smem_tile = StaticTuple[_, 2](
        a_smem_base.tile[BK, BM](0, 0), a_smem_base.tile[BK, BM](1, 0)
    )

    # Align the address by the maximum async copy size (16 bytes).
    alias b_smem_size = BK * BN
    var b_smem_base = LayoutTensor[
        Layout.row_major(2 * BK, BN),
        b_type,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()
    var b_smem_tile = StaticTuple[_, 2](
        b_smem_base.tile[BK, BN](0, 0), b_smem_base.tile[BK, BN](1, 0)
    )

    # Global memory tile.
    var a_gmem_tile = a.tile[BM, BK](BlockIdx.y(), 0)
    var b_gmem_tile = b.tile[BK, BN](0, BlockIdx.x())

    # Load A tile from global memory to shared.
    # Row major thread layout for coalesced access.
    alias thread_loada_gmem_layout = Layout.row_major(NUM_THREADS // BK, BK)
    alias thread_storea_smem_layout = Layout.col_major(BK, NUM_THREADS // BK)
    var thread_loada_gmem_frags = a_gmem_tile.distribute[
        thread_loada_gmem_layout
    ](ThreadIdx.x())
    var thread_storea_smem_frags = a_smem_tile[0].distribute[
        thread_storea_smem_layout
    ](ThreadIdx.x())
    thread_storea_smem_frags.copy_from_async(thread_loada_gmem_frags)

    # Load B tile from global memory to shared.
    # Row major thread layout for coalesced access.
    alias thread_layout_loadb = Layout.row_major(
        (NUM_THREADS // BN) * simd_size, BN // simd_size
    )
    var thread_loadb_gmem_frags = b_gmem_tile.vectorize[
        1, simd_size
    ]().distribute[thread_layout_loadb](ThreadIdx.x())
    var thread_storeb_smem_frags = b_smem_tile[0].vectorize[
        1, simd_size
    ]().distribute[thread_layout_loadb](ThreadIdx.x())
    thread_storeb_smem_frags.copy_from_async(thread_loadb_gmem_frags)

    async_copy_wait_all()
    barrier()

    # Advance A and B to next k tile.
    a_gmem_tile = a.tile[BM, BK](BlockIdx.y(), 1)
    b_gmem_tile = b.tile[BK, BN](1, BlockIdx.x())

    # Double buffer in registers (fragments in nvidia terms).
    var a_reg0 = LayoutTensor[Layout(TM), a_type].stack_allocation()
    var a_reg1 = LayoutTensor[Layout(TM), a_type].stack_allocation()
    var b_reg0 = LayoutTensor[Layout(TN), b_type].stack_allocation()
    var b_reg1 = LayoutTensor[Layout(TN), b_type].stack_allocation()
    var c_reg = LayoutTensor[
        Layout.row_major(TM, TN), c_type
    ].stack_allocation()
    c_reg.fill(0)

    # Thread swizzling
    # Warp has 2D Layout [warp_dim_x, warp_dim_y]. Current thread is mapped to
    # (mma_x, mma_y) in this layout as follow (the number is thread id).
    # 0  2  4  6  8  10 12 14
    # 1  3  5  7  9  11 13 15
    # 16 18 20 22 24 26 28 30
    # 17 19 21 23 25 27 29 31
    alias thread_layout = Layout(
        IntTuple(IntTuple(2, 2), 8), IntTuple(IntTuple(1, 16), 2)
    )

    # Load A fragments to the first buffer.
    var a_smem_warp_tile = a_smem_tile[0].tile[BK, WM](0, warp_y)
    var a_smem_warp_row = a_smem_warp_tile.tile[1, WM](0, 0).coalesce()
    var thread_loada_smem_frags = a_smem_warp_row.distribute[
        thread_layout, tile_size=simd_size, axis=0
    ](lane_id).vectorize[simd_size]()
    a_reg0.vectorize[simd_size]().copy_from_numa[a_align](
        thread_loada_smem_frags
    )

    # Load B fragments to the first buffer.
    var b_smem_warp_tile = b_smem_tile[0].tile[BK, WN](0, warp_x)
    var b_smem_warp_row = b_smem_warp_tile.tile[1, WN](0, 0).coalesce()
    var thread_loadb_smem_frags = b_smem_warp_row.distribute[
        thread_layout, tile_size=simd_size, axis=1
    ](lane_id).vectorize[simd_size]()
    b_reg0.vectorize[simd_size]().copy_from_numa[b_align](
        thread_loadb_smem_frags
    )

    var num_k_tiles = div_ceil(K, BK)

    # Update (num_k_tile - 1) tiles while switching buffers.
    for k_tile_id in range(num_k_tiles - 1):
        # The shared memory buffer to be prefetched
        var prefetch_id = 1 if k_tile_id % 2 == 0 else 0

        @unroll
        for k in range(BK):
            var next_k = (k + 1) % BK

            if k == BK - 1:
                async_copy_wait_all()
                barrier()

                a_smem_warp_tile = a_smem_tile[prefetch_id].tile[BK, WM](
                    0, warp_y
                )
                b_smem_warp_tile = b_smem_tile[prefetch_id].tile[BK, WN](
                    0, warp_x
                )

            # Fill the other A fragments buffer using the next row in A.
            a_smem_warp_row = a_smem_warp_tile.tile[1, WM](next_k, 0).coalesce()
            thread_loada_smem_frags = a_smem_warp_row.distribute[
                thread_layout, tile_size=simd_size, axis=0
            ](lane_id).vectorize[simd_size]()
            a_reg1.vectorize[simd_size]().copy_from_numa[a_align](
                thread_loada_smem_frags
            )

            b_smem_warp_row = b_smem_warp_tile.tile[1, WN](next_k, 0).coalesce()
            thread_loadb_smem_frags = b_smem_warp_row.distribute[
                thread_layout, tile_size=simd_size, axis=1
            ](lane_id).vectorize[simd_size]()
            b_reg1.vectorize[simd_size]().copy_from_numa[b_align](
                thread_loadb_smem_frags
            )

            # Load next k tile from global memory to shared memory.
            if k == 0:
                a_gmem_tile = a.tile[BM, BK](BlockIdx.y(), k_tile_id + 1)
                b_gmem_tile = b.tile[BK, BN](k_tile_id + 1, BlockIdx.x())
                var thread_loada_gmem_frags = a_gmem_tile.distribute[
                    thread_loada_gmem_layout
                ](ThreadIdx.x())
                var thread_loada_smem_frags = a_smem_tile[
                    prefetch_id
                ].distribute[thread_storea_smem_layout](ThreadIdx.x())
                thread_loada_smem_frags.copy_from_async(thread_loada_gmem_frags)

                var thread_loadb_gmem_frags = b_gmem_tile.vectorize[
                    1, simd_size
                ]().distribute[thread_layout_loadb](ThreadIdx.x())
                var thread_storeb_smem_frags = b_smem_tile[
                    prefetch_id
                ].vectorize[1, simd_size]().distribute[thread_layout_loadb](
                    ThreadIdx.x()
                )
                thread_storeb_smem_frags.copy_from_async(
                    thread_loadb_gmem_frags
                )

            outer_product_acc(c_reg, a_reg0, b_reg0)

            # Alternate buffer
            swap_ptr(a_reg0, a_reg1)
            swap_ptr(b_reg0, b_reg1)

    # Last k tile.
    @unroll
    for k in range(BK):
        var next_k = (k + 1) % BK

        if k < BK - 1:
            # Fill the other A fragments buffer.
            a_smem_warp_row = a_smem_warp_tile.tile[1, WM](next_k, 0).coalesce()
            thread_loada_smem_frags = a_smem_warp_row.distribute[
                thread_layout, tile_size=simd_size, axis=0
            ](lane_id).vectorize[simd_size]()
            a_reg1.vectorize[simd_size]().copy_from_numa[a_align](
                thread_loada_smem_frags
            )

            # Fill the other B fragments buffer.
            b_smem_warp_row = b_smem_warp_tile.tile[1, WN](next_k, 0).coalesce()
            thread_loadb_smem_frags = b_smem_warp_row.distribute[
                thread_layout, tile_size=simd_size, axis=1
            ](lane_id).vectorize[simd_size]()
            b_reg1.vectorize[simd_size]().copy_from_numa[b_align](
                thread_loadb_smem_frags
            )

        outer_product_acc(c_reg, a_reg0, b_reg0)

        swap_ptr(a_reg0, a_reg1)
        swap_ptr(b_reg0, b_reg1)

    # Map global memory tile down to thread.
    var c_gmem_tile = c.tile[BM, BN](BlockIdx.y(), BlockIdx.x())
    var c_gmem_warp_tile = c_gmem_tile.tile[WM, WN](warp_y, warp_x)
    var c_gmem_thread_tile = c_gmem_warp_tile.distribute[
        thread_layout, tile_size = IntTuple(simd_size, simd_size)
    ](ThreadIdx.x()).coalesce().vectorize[1, simd_size]()
    # Reshape thread register tile to match its global memory tile.
    alias c_tiled_layout = c_reg._compute_tile_layout[simd_size, simd_size]()
    var c_reg_reshaped = LayoutTensor[
        c_tiled_layout, c_type, address_space = c_reg.address_space
    ](c_reg.ptr).coalesce().vectorize[1, simd_size]()
    # vectorized store to global memory
    c_gmem_thread_tile.copy_from_numa[c_align](c_reg_reshaped)


fn test() raises:
    alias NUM_THREADS = 256
    alias M = 8192
    alias N = 8192
    alias K = 128
    alias BM = 128
    alias BN = 128
    alias BK = 16
    alias WM = 32
    alias WN = 64
    alias TM = 8
    alias TN = 8

    var stream = Stream()

    alias a_layout = Layout(IntTuple(M, K), IntTuple(K, 1))
    alias b_layout = Layout(IntTuple(K, N), IntTuple(N, 1))
    alias c_layout = Layout(IntTuple(M, N), IntTuple(N, 1))

    var a_host = DTypePointer[DType.float32].alloc(M * K)
    var b_host = DTypePointer[DType.float32].alloc(K * N)
    var c_host = DTypePointer[DType.float32].alloc(M * N)
    var c_host_ref = DTypePointer[DType.float32].alloc(M * N)

    for i in range(M * K):
        a_host[i] = i

    for i in range(K * N):
        b_host[i] = i

    var a_device = _malloc[Float32](M * K)
    var b_device = _malloc[Float32](K * N)
    var c_device = _malloc[Float32](M * N)
    var c_device_ref = _malloc[Float32](M * N)

    _copy_host_to_device(a_device, a_host, M * K)
    _copy_host_to_device(b_device, b_host, K * N)

    var c_buffer = NDBuffer[DType.float32, 2, DimList(M, N)](c_device)
    var a_buffer = NDBuffer[DType.float32, 2, DimList(M, K)](a_device)
    var b_buffer = NDBuffer[DType.float32, 2, DimList(K, N)](b_device)

    var c_tensor = LayoutTensor[c_layout, DType.float32](c_device)
    var a_tensor = LayoutTensor[a_layout, DType.float32](a_device)
    var b_tensor = LayoutTensor[b_layout, DType.float32](b_device)

    alias gemm = sgemm_double_buffer[
        DType.float32,
        c_layout,
        DType.float32,
        a_layout,
        DType.float32,
        b_layout,
        DType.uint32,
        BM,
        BN,
        BK,
        WM,
        WN,
        TM,
        TN,
        NUM_THREADS,
    ]
    var func = Function[__type_of(gemm), gemm](threads_per_block=NUM_THREADS)

    if is_benchmark():
        alias nrun = 200
        alias nwarmup = 2

        @always_inline
        @parameter
        fn run_func(stream: Stream) raises:
            for i in range(nrun):
                func(
                    c_tensor,
                    a_tensor,
                    b_tensor,
                    grid_dim=(div_ceil(N, BN), div_ceil(M, BM), 1),
                    block_dim=(NUM_THREADS, 1, 1),
                    stream=stream,
                )

        # Warmup
        for i in range(nwarmup):
            run_func(stream)

        var nstime = time_function[run_func](stream) / nrun
        var sectime = nstime * 1e-9
        var TFlog = 2.0 * M * N * K * 1e-12
        print(nrun, "runs avg(s)", sectime, "TFlogs/s", TFlog / sectime)

    func(
        c_tensor,
        a_tensor,
        b_tensor,
        grid_dim=(div_ceil(N, BN), div_ceil(M, BM), 1),
        block_dim=(NUM_THREADS, 1, 1),
        stream=stream,
    )

    synchronize()

    _copy_device_to_host(c_host, c_device, M * N)

    # Naive gemm.
    alias BLOCK_DIM = 16
    alias gemm_naive = matmul_kernel_naive[
        DType.float32, DType.float32, DType.float32, BLOCK_DIM
    ]
    var func_naive = Function[__type_of(gemm_naive), gemm_naive](
        threads_per_block=NUM_THREADS
    )
    var c_buffer_ref = NDBuffer[DType.float32, 2, DimList(M, N)](c_device_ref)
    func_naive(
        c_buffer_ref,
        a_buffer,
        b_buffer,
        M,
        N,
        K,
        grid_dim=(div_ceil(M, BLOCK_DIM), div_ceil(N, BLOCK_DIM), 1),
        block_dim=(BLOCK_DIM, BLOCK_DIM, 1),
    )

    synchronize()
    _copy_device_to_host(c_host_ref, c_device_ref, M * N)

    for i in range(M * N):
        if not isclose(c_host[i], c_host_ref[i]):
            print(i, c_host[i], c_host_ref[i])
        assert_almost_equal(c_host[i], c_host_ref[i])

    _free(c_device)
    _free(c_device_ref)
    _free(a_device)
    _free(b_device)

    c_host.free()
    c_host_ref.free()
    a_host.free()
    b_host.free()

    _ = func^
    _ = func_naive^
    _ = stream^


def main():
    try:
        with Context() as ctx:
            test()
    except e:
        print("ERROR:", e)
