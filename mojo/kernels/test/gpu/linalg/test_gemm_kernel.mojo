# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo-no-debug %s

from math import ceildiv, isclose
from sys import argv

from buffer import DimList, NDBuffer
from gpu import WARP_SIZE
from gpu.host import Context, Function, Stream, synchronize
from gpu.host.event import time_function
from gpu.host.memory import (
    _copy_device_to_host,
    _copy_host_to_device,
    _free,
    _malloc,
)
from gpu.id import BlockIdx, ThreadIdx
from gpu.memory import AddressSpace, async_copy_wait_all
from gpu.sync import barrier
from layout import Layout, LayoutTensor
from layout.layout_tensor import copy_sram_to_local, outer_product_acc
from layout.nd_buffer_stub import copy_from_nd_buffer, copy_to_nd_buffer
from LinAlg.MatmulGPU import matmul_kernel_naive
from testing import assert_almost_equal

from utils import Index


fn is_benchmark() -> Bool:
    for arg in argv():
        if arg == "--benchmark" or arg == "-benchmark":
            return True
    return False


fn dump_ptx() -> Bool:
    for arg in argv():
        if arg == "--dump_ptx" or arg == "--dump_ptx":
            return True
    return False


fn dump_llvm() -> Bool:
    for arg in argv():
        if arg == "--dump_llvm" or arg == "--dump_llvm":
            return True
    return False


fn gemm_kernel[
    c_type: DType,
    c_shape: DimList,
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList,
    NUM_THREADS: Int,
    BM: Int,
    BN: Int,
    BK: Int,
    WM: Int,
    WN: Int,
    TM: Int,
    TN: Int,
](
    mat_c: NDBuffer[c_type, 2, c_shape],
    mat_a: NDBuffer[a_type, 2, a_shape],
    mat_b: NDBuffer[b_type, 2, b_shape],
):
    var M = mat_c.dim(0)
    var N = mat_c.dim(1)
    var K = mat_a.dim(1)

    var a_tile_sram = LayoutTensor[
        a_type, Layout.row_major(BM, BK), address_space = AddressSpace.SHARED
    ].stack_allocation()

    var b_tile_sram = LayoutTensor[
        b_type, Layout.row_major(BK, BN), address_space = AddressSpace.SHARED
    ].stack_allocation()

    var num_warps = NUM_THREADS // WARP_SIZE
    var n_warp_n = BN // WN
    var n_warp_m = BM // WM
    var warp_id = ThreadIdx.x() // WARP_SIZE
    var warp_m = warp_id // n_warp_n
    var warp_n = warp_id % n_warp_n

    # Allocate register tiles.

    var a_reg = LayoutTensor[a_type, Layout(TM)].stack_allocation()
    var b_reg = LayoutTensor[b_type, Layout(TN)].stack_allocation()
    var c_reg = LayoutTensor[
        c_type, Layout.row_major(TM, TN)
    ].stack_allocation()
    c_reg.fill(0)

    alias warp_layout = Layout.row_major(8, 4)

    for k_i in range(ceildiv(K, BK)):
        var a_tile_dram = mat_a.tile[BM, BK]((BlockIdx.y(), k_i))
        var a_tile_sram_local = a_tile_sram.distribute[
            Layout.row_major(NUM_THREADS // BK, BK)
        ](ThreadIdx.x())
        copy_from_nd_buffer[
            thread_layout = Layout.row_major(NUM_THREADS // BK, BK),
            is_async=True,
        ](a_tile_sram_local, a_tile_dram, ThreadIdx.x())

        var b_tile_dram = mat_b.tile[BK, BN]((k_i, BlockIdx.x()))
        var b_tile_sram_local = b_tile_sram.distribute[
            Layout.row_major(NUM_THREADS // BN, BN)
        ](ThreadIdx.x())
        copy_from_nd_buffer[
            thread_layout = Layout.row_major(NUM_THREADS // BN, BN),
            is_async=True,
        ](b_tile_sram_local, b_tile_dram, ThreadIdx.x())
        async_copy_wait_all()
        barrier()

        @parameter
        for k_i in range(BK):
            var a_smem_warp_row = a_tile_sram.tile[WM, BK](warp_m, 0).slice[
                :, k_i : k_i + 1
            ]()

            var b_smem_warp_row = b_tile_sram.tile[BK, WN](0, warp_n).slice[
                k_i : k_i + 1, :
            ]()
            copy_sram_to_local[src_warp_layout=warp_layout, axis=0](
                a_reg, a_smem_warp_row
            )
            copy_sram_to_local[src_warp_layout=warp_layout, axis=1](
                b_reg, b_smem_warp_row
            )
            outer_product_acc(c_reg, a_reg, b_reg)

        # Otherwise a data race, faster threads will modify shared memory.
        barrier()

    var c_warp_tile = mat_c.tile[BM, BN]((BlockIdx.y(), BlockIdx.x())).tile[
        WM, WN
    ]((warp_m, warp_n))

    copy_to_nd_buffer[thread_layout=warp_layout](
        c_warp_tile, c_reg, ThreadIdx.x()
    )


fn test_gemm_kernel_dynamic() raises:
    alias NUM_THREADS = 256
    alias BM = 64
    alias BN = 64
    alias BK = 16
    alias WM = 32
    alias WN = 16
    alias TM = 4
    alias TN = 4

    alias M = 1024
    alias N = 1024
    alias K = 128

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

    alias gemm_kernel_func_t = gemm_kernel[
        DType.float32,
        DimList.create_unknown[2](),
        DType.float32,
        DimList.create_unknown[2](),
        DType.float32,
        DimList.create_unknown[2](),
        NUM_THREADS,
        BM,
        BN,
        BK,
        WM,
        WN,
        TM,
        TN,
    ]

    var stream = Stream()

    _copy_host_to_device(a_device, a_host, M * K)
    _copy_host_to_device(b_device, b_host, K * N)

    synchronize()

    var gemm_kernel_func = Function[gemm_kernel_func_t](
        dump_ptx=dump_ptx(), dump_llvm=dump_llvm()
    )

    var mat_a = NDBuffer[DType.float32, 2, DimList.create_unknown[2]()](
        a_device, dynamic_shape=Index(M, K)
    )
    var mat_b = NDBuffer[DType.float32, 2, DimList.create_unknown[2]()](
        b_device, dynamic_shape=Index(K, M)
    )
    var mat_c = NDBuffer[DType.float32, 2, DimList.create_unknown[2]()](
        c_device, dynamic_shape=Index(N, M)
    )

    gemm_kernel_func(
        mat_c,
        mat_a,
        mat_b,
        grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
        block_dim=(NUM_THREADS),
    )
    synchronize()

    _copy_device_to_host(c_host, c_device, M * N)

    # Naive gemm.
    alias BLOCK_DIM = 16
    alias gemm_naive = matmul_kernel_naive[
        DType.float32, DType.float32, DType.float32, BLOCK_DIM
    ]
    var func_naive = Function[gemm_naive](threads_per_block=NUM_THREADS)
    var c_buffer_ref = NDBuffer[DType.float32, 2, DimList(M, N)](c_device_ref)
    func_naive(
        c_buffer_ref,
        mat_a,
        mat_b,
        M,
        N,
        K,
        grid_dim=(ceildiv(M, BLOCK_DIM), ceildiv(N, BLOCK_DIM), 1),
        block_dim=(BLOCK_DIM, BLOCK_DIM, 1),
    )
    synchronize()
    _copy_device_to_host(c_host_ref, c_device_ref, M * N)
    for i in range(M * N):
        if not isclose(c_host[i], c_host_ref[i]):
            print(i, c_host[i], c_host_ref[i])
        assert_almost_equal(c_host[i], c_host_ref[i])

    if is_benchmark():
        alias nrun = 200
        alias nwarmup = 2

        @always_inline
        @parameter
        fn run_func(stream: Stream) raises:
            for i in range(nrun):
                gemm_kernel_func(
                    mat_c,
                    mat_a,
                    mat_b,
                    grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
                    block_dim=(NUM_THREADS),
                    stream=stream,
                )

        # Warmup
        for i in range(nwarmup):
            run_func(stream)
        var nstime = time_function[run_func](stream) / nrun
        var sectime = nstime * 1e-9
        var TFlop = 2.0 * M * N * K * 1e-12
        print(nrun, "runs avg(s)", sectime, "TFlops/s", TFlop / sectime)

    _free(c_device)
    _free(c_device_ref)
    _free(a_device)
    _free(b_device)

    c_host.free()
    c_host_ref.free()
    a_host.free()
    b_host.free()

    _ = gemm_kernel_func^
    _ = stream^


fn main():
    try:
        with Context() as ctx:
            test_gemm_kernel_dynamic()
    except e:
        print("CUDA err:", e)
