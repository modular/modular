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
from Matmul import matmul_kernel_naive
from gpu import (
    WARP_SIZE,
    BlockDim,
    BlockIdx,
    ThreadIdx,
    barrier,
    lane_id,
    AddressSpace,
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
from layout.layout_tensor import LayoutTensor
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
    BM: Scalar[itype],
    BN: Scalar[itype],
    BK: Scalar[itype],
    WM: Scalar[itype],
    WN: Scalar[itype],
    TM: Scalar[itype],
    TN: Scalar[itype],
    NUM_THREADS: Scalar[itype],
](
    c: LayoutTensor[c_layout, c_type],
    a: LayoutTensor[a_layout, a_type],
    b: LayoutTensor[b_layout, b_type],
):
    alias _uint = Scalar[itype]

    alias simd_size_int = simdwidthof[c_type]()
    alias simd_size = Scalar[itype](simd_size_int)

    var M = Scalar[itype](c.dim[0]())
    var N = Scalar[itype](c.dim[1]())
    var K = Scalar[itype](a.dim[1]())

    alias num_warps_m = BM // WM
    alias num_warps_n = BN // WN

    var tid = Scalar[itype](ThreadIdx.x())
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
    alias pad_avoid_bank_conflict = Scalar[itype](4)
    alias BM_padded = BM + pad_avoid_bank_conflict

    # Double buffer in shared memory.
    alias a_smem_size = int(BK * BM_padded)
    var a_smem_ptr = stack_allocation[
        2 * a_smem_size, a_type, address_space = AddressSpace.SHARED
    ]()
    alias b_smem_size = int(BK * BN)
    var b_smem_ptr = stack_allocation[
        2 * b_smem_size, b_type, address_space = AddressSpace.SHARED
    ]()
    var a_smem_tile0 = LayoutTensor[
        Layout(IntTuple(int(BK), int(BM)), IntTuple(int(BM_padded), 1)),
        a_type,
        address_space = AddressSpace.SHARED,
    ](a_smem_ptr)
    var a_smem_tile1 = LayoutTensor[
        Layout(IntTuple(int(BK), int(BM)), IntTuple(int(BM_padded), 1)),
        a_type,
        address_space = AddressSpace.SHARED,
    ](a_smem_ptr + a_smem_size)
    var b_smem_tile0 = LayoutTensor[
        Layout(IntTuple(int(BK), int(BN)), IntTuple(int(BN), 1)),
        b_type,
        address_space = AddressSpace.SHARED,
    ](b_smem_ptr)
    var b_smem_tile1 = LayoutTensor[
        Layout(IntTuple(int(BK), int(BN)), IntTuple(int(BN), 1)),
        b_type,
        address_space = AddressSpace.SHARED,
    ](b_smem_ptr + b_smem_size)

    # Global memory tile.
    var a_gmem_tile = a.tile[int(BM), int(BK)](BlockIdx.y(), 0)
    var b_gmem_tile = b.tile[int(BK), int(BN)](0, BlockIdx.x())

    # Load A tile from global memory to shared.
    # Row major thread layout for coalesced access.
    alias thread_loada_gmem_layout = Layout(
        IntTuple(int(NUM_THREADS // BK), int(BK)), IntTuple(int(BK), 1)
    )
    alias thread_storea_smem_layout = Layout(
        IntTuple(int(BK), int(NUM_THREADS // BK)), IntTuple(1, int(BK))
    )
    var thread_loada_gmem_frags = a_gmem_tile.distribute[
        thread_loada_gmem_layout
    ](ThreadIdx.x())
    var thread_storea_smem_frags = a_smem_tile0.distribute[
        thread_storea_smem_layout
    ](ThreadIdx.x())
    thread_storea_smem_frags.copy_from_async(thread_loada_gmem_frags)

    # Load B tile from global memory to shared.
    # Row major thread layout for coalesced access.
    alias thread_layout_loadb = Layout(
        IntTuple(int(NUM_THREADS // BN), int(BN)), IntTuple(int(BN), 1)
    )
    var thread_loadb_gmem_frags = b_gmem_tile.distribute[thread_layout_loadb](
        ThreadIdx.x()
    )
    var thread_storeb_smem_frags = b_smem_tile0.distribute[thread_layout_loadb](
        ThreadIdx.x()
    )
    thread_storeb_smem_frags.copy_from_async(thread_loadb_gmem_frags)

    async_copy_wait_all()
    barrier()

    # Advance A and B to next k tile.
    a_gmem_tile = a.tile[int(BM), int(BK)](BlockIdx.y(), 1)
    b_gmem_tile = b.tile[int(BK), int(BN)](1, BlockIdx.x())

    # Double buffer in registers (fragments in nvidia terms).
    var a_reg0 = LayoutTensor[Layout(int(TM)), a_type].stack_allocation()
    var a_reg1 = LayoutTensor[Layout(int(TM)), a_type].stack_allocation()
    var b_reg0 = LayoutTensor[Layout(int(TN)), b_type].stack_allocation()
    var b_reg1 = LayoutTensor[Layout(int(TN)), b_type].stack_allocation()
    var c_reg = LayoutTensor[
        Layout(IntTuple(int(TM), int(TN)), IntTuple(int(TN), 1)), c_type
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
    var a_smem_warp_tile = a_smem_tile0.tile[int(BK), int(WM)](0, int(warp_y))
    var a_smem_warp_row = a_smem_warp_tile.tile[1, int(WM)](0, 0).coalesce()
    var thread_loada_smem_frags = a_smem_warp_row.distribute[
        thread_layout, tile_size=simd_size_int, axis=0
    ](int(lane_id))
    a_reg0.copy_from_numa(thread_loada_smem_frags)

    # Load B fragments to the first buffer.
    var b_smem_warp_tile = b_smem_tile0.tile[int(BK), int(WN)](0, int(warp_x))
    var b_smem_warp_row = b_smem_warp_tile.tile[1, int(WN)](0, 0).coalesce()
    var thread_loadb_smem_frags = b_smem_warp_row.distribute[
        thread_layout, tile_size=simd_size_int, axis=1
    ](int(lane_id))
    b_reg0.copy_from_numa(thread_loadb_smem_frags)

    var num_k_tiles = Scalar[itype](div_ceil(int(K), int(BK)))

    # Update (num_k_tile - 1) tiles while switching buffers.
    for k_tile_id in range(num_k_tiles - 1):

        @unroll
        for k in range(BK):
            var next_k = (k + 1) % int(BK)

            if k == int(BK - 1):
                async_copy_wait_all()
                barrier()

                # Switch shared memory buffer.
                var shift = a_smem_size if k_tile_id % 2 == 0 else -a_smem_size
                a_smem_tile0.offset(shift)
                a_smem_tile1.offset(-shift)
                shift = b_smem_size if k_tile_id % 2 == 0 else -b_smem_size
                b_smem_tile0.offset(shift)
                b_smem_tile1.offset(-shift)

                a_smem_warp_tile = a_smem_tile0.tile[int(BK), int(WM)](
                    0, int(warp_y)
                )
                b_smem_warp_tile = b_smem_tile0.tile[int(BK), int(WN)](
                    0, int(warp_x)
                )

            # Fill the other A fragments buffer using the next row in A.
            a_smem_warp_row = a_smem_warp_tile.tile[1, int(WM)](
                next_k, 0
            ).coalesce()
            thread_loada_smem_frags = a_smem_warp_row.distribute[
                thread_layout, tile_size=simd_size_int, axis=0
            ](int(lane_id))
            a_reg1.copy_from_numa(thread_loada_smem_frags)

            b_smem_warp_row = b_smem_warp_tile.tile[1, int(WN)](
                next_k, 0
            ).coalesce()
            thread_loadb_smem_frags = b_smem_warp_row.distribute[
                thread_layout, tile_size=simd_size_int, axis=1
            ](int(lane_id))
            b_reg1.copy_from_numa(thread_loadb_smem_frags)

            # Load next k tile from global memory to shared memory.
            if k == 0:
                a_gmem_tile = a.tile[int(BM), int(BK)](
                    BlockIdx.y(), k_tile_id + 1
                )
                b_gmem_tile = b.tile[int(BK), int(BN)](
                    k_tile_id + 1, BlockIdx.x()
                )
                var thread_loada_gmem_frags = a_gmem_tile.distribute[
                    thread_loada_gmem_layout
                ](ThreadIdx.x())
                var thread_loada_smem_frags = a_smem_tile1.distribute[
                    thread_storea_smem_layout
                ](ThreadIdx.x())
                thread_loada_smem_frags.copy_from_async(thread_loada_gmem_frags)

                var thread_loadb_gmem_frags = b_gmem_tile.distribute[
                    thread_layout_loadb
                ](ThreadIdx.x())
                var thread_loadb_smem_frags = b_smem_tile1.distribute[
                    thread_layout_loadb
                ](ThreadIdx.x())
                thread_loadb_smem_frags.copy_from_async(thread_loadb_gmem_frags)

            # FFMA loop
            @unroll
            for i in range(TM):

                @unroll
                for j in range(TN):
                    c_reg[i, j] += (
                        a_reg0[i].cast[c_type]() * b_reg0[j].cast[c_type]()
                    )

            # Alternate buffer
            swap_ptr(a_reg0, a_reg1)
            swap_ptr(b_reg0, b_reg1)

    # Last k tile.
    @unroll
    for k in range(BK):
        var next_k = (k + 1) % int(BK)

        if k < int(BK - 1):
            # Fill the other A fragments buffer.
            a_smem_warp_row = a_smem_warp_tile.tile[1, int(WM)](
                next_k, 0
            ).coalesce()
            thread_loada_smem_frags = a_smem_warp_row.distribute[
                thread_layout, tile_size=simd_size_int, axis=0
            ](int(lane_id))
            a_reg1.copy_from_numa(thread_loada_smem_frags)

            # Fill the other B fragments buffer.
            b_smem_warp_row = b_smem_warp_tile.tile[1, int(WN)](
                next_k, 0
            ).coalesce()
            thread_loadb_smem_frags = b_smem_warp_row.distribute[
                thread_layout, tile_size=simd_size_int, axis=1
            ](int(lane_id))
            b_reg1.copy_from_numa(thread_loadb_smem_frags)

        # FFMA loop
        @unroll
        for i in range(TM):

            @unroll
            for j in range(TN):
                c_reg[i, j] += (
                    a_reg0[i].cast[c_type]() * b_reg0[j].cast[c_type]()
                )

        swap_ptr(a_reg0, a_reg1)
        swap_ptr(b_reg0, b_reg1)

    var c_gmem_tile = c.tile[int(BM), int(BN)](BlockIdx.y(), BlockIdx.x())
    var c_gmem_warp_tile = c_gmem_tile.tile[int(WM), int(WN)](
        int(warp_y), int(warp_x)
    )
    var c_gmem_thread_tile = c_gmem_warp_tile.distribute[
        thread_layout, tile_size = IntTuple(simd_size_int, simd_size_int)
    ](ThreadIdx.x())
    alias c_tiled_layout = c_reg._compute_tile_layout[
        simd_size_int, simd_size_int
    ]()
    var c_reg_reshaped = LayoutTensor[
        c_tiled_layout, c_type, address_space = c_reg.address_space
    ](c_reg.ptr)
    c_gmem_thread_tile.copy_from_numa(c_reg_reshaped)


fn test() raises:
    alias NUM_THREADS = 256
    alias M = 128
    alias N = 128
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
    var func = Function[__type_of(gemm), gemm](
        threads_per_block=NUM_THREADS, dump_ptx=Path("./mm.ptx")
    )

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
