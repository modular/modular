# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# FIXME: KERN-1377
# UNSUPPORTED: AMD-GPU
# RUN: %mojo-no-debug %s

from math import ceildiv, isclose
from sys import argv

from buffer import DimList, NDBuffer
from gpu import WARP_SIZE
from gpu.host import DeviceContext
from gpu.id import block_idx, thread_idx
from gpu.memory import AddressSpace, async_copy_wait_all
from gpu.sync import barrier
from layout import Layout, LayoutTensor
from layout.layout_tensor import copy_sram_to_local
from layout.math import outer_product_acc
from layout._ndbuffer_stub import copy_from_nd_buffer, copy_to_nd_buffer
from layout.tensor_builder import LayoutTensorBuild as tb
from linalg.matmul_gpu import matmul_kernel_naive
from memory import UnsafePointer
from testing import assert_almost_equal

from utils import Index


fn is_benchmark() -> Bool:
    for arg in argv():
        if arg == "--benchmark" or arg == "-benchmark":
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
    mat_c: NDBuffer[c_type, 2, MutableAnyOrigin, c_shape],
    mat_a: NDBuffer[a_type, 2, MutableAnyOrigin, a_shape],
    mat_b: NDBuffer[b_type, 2, MutableAnyOrigin, b_shape],
):
    var M = mat_c.dim(0)
    var N = mat_c.dim(1)
    var K = mat_a.dim(1)

    var a_tile_sram = LayoutTensor[
        a_type,
        Layout.row_major(BM, BK),
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var b_tile_sram = LayoutTensor[
        b_type,
        Layout.row_major(BK, BN),
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var num_warps = NUM_THREADS // WARP_SIZE
    var n_warp_n = BN // WN
    var n_warp_m = BM // WM
    var warp_id = thread_idx.x // WARP_SIZE
    var warp_m = Int(warp_id) // n_warp_n
    var warp_n = Int(warp_id) % n_warp_n

    # Allocate register tiles.
    var a_reg = tb[a_type]().row_major[TM]().local().alloc()
    var b_reg = tb[b_type]().row_major[TN]().local().alloc()
    var c_reg = tb[c_type]().row_major[TM, TN]().local().alloc().fill(0)

    alias warp_layout = Layout.row_major(8, 4)

    for k_i in range(ceildiv(K, BK)):
        var a_tile_dram = mat_a.tile[BM, BK](Index(Int(block_idx.y), k_i))
        var a_tile_sram_local = a_tile_sram.distribute[
            Layout.row_major(NUM_THREADS // BK, BK)
        ](thread_idx.x)
        copy_from_nd_buffer[
            thread_layout = Layout.row_major(NUM_THREADS // BK, BK),
            is_async=True,
        ](a_tile_sram_local, a_tile_dram, thread_idx.x)

        var b_tile_dram = mat_b.tile[BK, BN](Index(k_i, Int(block_idx.x)))
        var b_tile_sram_local = b_tile_sram.distribute[
            Layout.row_major(NUM_THREADS // BN, BN)
        ](thread_idx.x)
        copy_from_nd_buffer[
            thread_layout = Layout.row_major(NUM_THREADS // BN, BN),
            is_async=True,
        ](b_tile_sram_local, b_tile_dram, thread_idx.x)
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

    var c_warp_tile = mat_c.tile[BM, BN](
        Index(Int(block_idx.y), Int(block_idx.x))
    ).tile[WM, WN](Index(warp_m, warp_n))

    copy_to_nd_buffer[thread_layout=warp_layout](
        c_warp_tile, c_reg, thread_idx.x
    )


fn test_gemm_kernel_dynamic(ctx: DeviceContext) raises:
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

    var a_host = UnsafePointer[Float32].alloc(M * K)
    var b_host = UnsafePointer[Float32].alloc(K * N)
    var c_host = UnsafePointer[Float32].alloc(M * N)
    var c_host_ref = UnsafePointer[Float32].alloc(M * N)

    for i in range(M * K):
        a_host[i] = i

    for i in range(K * N):
        b_host[i] = i

    var a_device = ctx.enqueue_create_buffer[DType.float32](M * K)
    var b_device = ctx.enqueue_create_buffer[DType.float32](K * N)
    var c_device = ctx.enqueue_create_buffer[DType.float32](M * N)
    var c_device_ref = ctx.enqueue_create_buffer[DType.float32](M * N)

    ctx.enqueue_copy(a_device, a_host)
    ctx.enqueue_copy(b_device, b_host)

    var mat_a = NDBuffer[
        DType.float32, 2, MutableAnyOrigin, DimList.create_unknown[2]()
    ](a_device._unsafe_ptr(), dynamic_shape=Index(M, K))
    var mat_b = NDBuffer[
        DType.float32, 2, MutableAnyOrigin, DimList.create_unknown[2]()
    ](b_device._unsafe_ptr(), dynamic_shape=Index(K, M))
    var mat_c = NDBuffer[
        DType.float32, 2, MutableAnyOrigin, DimList.create_unknown[2]()
    ](c_device._unsafe_ptr(), dynamic_shape=Index(N, M))

    alias kernel = gemm_kernel[
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

    ctx.enqueue_function[kernel](
        mat_c,
        mat_a,
        mat_b,
        grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
        block_dim=(NUM_THREADS),
    )

    ctx.enqueue_copy(c_host, c_device)

    # Naive gemm.
    alias BLOCK_DIM = 16
    alias gemm_naive = matmul_kernel_naive[
        DType.float32, DType.float32, DType.float32, BLOCK_DIM
    ]
    var c_buffer_ref = NDBuffer[
        DType.float32, 2, MutableAnyOrigin, DimList(M, N)
    ](c_device_ref._unsafe_ptr())
    ctx.enqueue_function[gemm_naive](
        c_buffer_ref,
        mat_a,
        mat_b,
        M,
        N,
        K,
        grid_dim=(ceildiv(M, BLOCK_DIM), ceildiv(N, BLOCK_DIM), 1),
        block_dim=(BLOCK_DIM, BLOCK_DIM, 1),
    )

    ctx.enqueue_copy(c_host_ref, c_device_ref)
    ctx.synchronize()
    for i in range(M * N):
        if not isclose(c_host[i], c_host_ref[i]):
            print(i, c_host[i], c_host_ref[i])
        assert_almost_equal(c_host[i], c_host_ref[i])

    if is_benchmark():
        alias nrun = 200
        alias nwarmup = 2

        @always_inline
        @parameter
        fn run_func(ctx: DeviceContext) raises:
            ctx.enqueue_function[kernel](
                mat_c,
                mat_a,
                mat_b,
                grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
                block_dim=(NUM_THREADS),
            )

        # Warmup
        for i in range(nwarmup):
            ctx.enqueue_function[kernel](
                mat_c,
                mat_a,
                mat_b,
                grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
                block_dim=(NUM_THREADS),
            )
        var nstime = ctx.execution_time[run_func](nrun) / nrun
        var sectime = nstime * 1e-9
        var TFlop = 2.0 * M * N * K * 1e-12
        print(nrun, "runs avg(s)", sectime, "TFlops/s", TFlop / sectime)

    _ = c_device
    _ = c_device_ref
    _ = a_device
    _ = b_device

    c_host.free()
    c_host_ref.free()
    a_host.free()
    b_host.free()


def main():
    with DeviceContext() as ctx:
        test_gemm_kernel_dynamic(ctx)
