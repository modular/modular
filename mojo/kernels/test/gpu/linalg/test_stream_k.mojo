# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug-no-assert %s

from math import ceildiv, isclose
from random import random_float64
from sys import bitwidthof

from buffer import NDBuffer
from buffer.dimlist import DimList
from gpu.host import DeviceBuffer, DeviceContext
from memory import UnsafePointer
from testing import assert_almost_equal
from gpu import thread_idx, block_idx
from os.atomic import Atomic
from sys import _RegisterPackType, sizeof
from utils import Index, IndexList
from linalg.matmul_gpu import matmul_kernel_naive
from memory import bitcast


fn compare_and_swap(
    ptr: UnsafePointer[Int32],
    e: Int,
    d: Int,
) -> Int32:
    var old = Atomic._fetch_add(ptr, 0)
    while old == e:
        if Atomic._xchg(ptr, d) == old:
            break
        old = Atomic._fetch_add(ptr, 0)
    return old


fn swizzle_tile(
    tile_id: Int,
    M: Int,
    N: Int,
    K: Int,
    BLOCK_M: Int,
    BLOCK_N: Int,
    BLOCK_K: Int,
    GROUP_M: Int,
) -> IndexList[2]:
    var grid_m = (M + BLOCK_M - 1) // BLOCK_M
    var grid_n = (N + BLOCK_N - 1) // BLOCK_N
    var width = GROUP_M * grid_n
    var group_id = tile_id // width
    var group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    var pid_m = group_id * GROUP_M + (tile_id % group_size)
    var pid_n = (tile_id % width) // group_size
    return IndexList[2](pid_m, pid_n)


fn linear_tile(
    tile_id: Int,
    M: Int,
    N: Int,
    K: Int,
    BLOCK_M: Int,
    BLOCK_N: Int,
    BLOCK_K: Int,
    GROUP_M: Int,
) -> IndexList[2]:
    var pid_m = tile_id // ((N + BLOCK_N - 1) // BLOCK_N)
    var pid_n = tile_id % ((N + BLOCK_N - 1) // BLOCK_N)
    return IndexList[2](pid_m, pid_n)


fn mac_loop[
    c_type: DType,
    a_type: DType,
    b_type: DType,
](
    C: UnsafePointer[Scalar[c_type]],
    A: UnsafePointer[Scalar[a_type]],
    B: UnsafePointer[Scalar[b_type]],
    M: Int,
    N: Int,
    K: Int,
    locks: UnsafePointer[Int32],
    stride_am: Int,
    stride_ak: Int,
    stride_bk: Int,
    stride_bn: Int,
    stride_cm: Int,
    stride_cn: Int,
    iters_per_tile: Int,
    start_iter: Int,
    end_iter: Int,
    BLOCK_M: Int,
    BLOCK_N: Int,
    BLOCK_K: Int,
    GROUP_M: Int,
):
    var tile_id = start_iter // iters_per_tile
    var pid = IndexList[2](0, 0)
    if GROUP_M > 0:
        pid = swizzle_tile(tile_id, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M)
    else:
        pid = linear_tile(tile_id, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M)

    var rm_base = pid[0] * BLOCK_M
    var rn_base = pid[1] * BLOCK_N

    var tx = thread_idx.x
    var ty = thread_idx.y
    var partial_offset_k = (start_iter % iters_per_tile) * BLOCK_K
    var global_r = rm_base + ty
    var global_c = rn_base + tx
    var accum = Scalar[c_type](0)

    for iter in range(start_iter, end_iter):
        var k_offset = (iter % iters_per_tile) * BLOCK_K

        for kk in range(BLOCK_K):
            var actual_k = k_offset + kk
            if global_r < M and actual_k < K and global_c < N:
                var a_val = A.load(global_r * stride_am + actual_k * stride_ak)
                var b_val = B.load(actual_k * stride_bk + global_c * stride_bn)
                accum += a_val.cast[c_type]() * b_val.cast[c_type]()

    if (end_iter % iters_per_tile) == 0:
        if global_r < M and global_c < N:
            C[global_r * stride_cm + global_c * stride_cn] = accum

        if (start_iter % iters_per_tile) != 0:
            if thread_idx.x == 0 and thread_idx.y == 0:
                _ = Atomic._xchg(locks.offset(tile_id), 1)
    else:
        if global_r < M and global_c < N:
            while compare_and_swap(locks.offset(tile_id), 1, 1) != 1:
                pass
            _ = Atomic._fetch_add(
                C.offset(global_r * stride_cm + global_c * stride_cn), accum
            )


fn first_wave_kernel[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    BLOCK_M: Int,
    BLOCK_N: Int,
    BLOCK_K: Int,
    GROUP_M: Int,
](
    C: UnsafePointer[Scalar[c_type]],
    A: UnsafePointer[Scalar[a_type]],
    B: UnsafePointer[Scalar[b_type]],
    M: Int,
    N: Int,
    K: Int,
    locks: UnsafePointer[Int32],
    stride_am: Int,
    stride_ak: Int,
    stride_bk: Int,
    stride_bn: Int,
    stride_cm: Int,
    stride_cn: Int,
    total_full_tiles_streamk: UInt,
    total_partial_tiles_streamk: UInt,
    iters_per_tile: UInt,
):
    var pid = block_idx.x

    var start_iter = pid * total_full_tiles_streamk + (
        pid if pid
        < total_partial_tiles_streamk else total_partial_tiles_streamk
    )
    var last_iter = (pid + 1) * total_full_tiles_streamk + (
        (pid + 1) if (pid + 1)
        < total_partial_tiles_streamk else total_partial_tiles_streamk
    )

    while start_iter < last_iter:
        var remainder = (iters_per_tile - (start_iter % iters_per_tile))
        var boundary = start_iter + remainder
        var end_iter = boundary if (boundary < last_iter) else last_iter
        mac_loop(
            C,
            A,
            B,
            M,
            N,
            K,
            locks,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            stride_cm,
            stride_cn,
            iters_per_tile,
            start_iter,
            end_iter,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            GROUP_M,
        )
        start_iter = end_iter


fn full_tiles_kernel[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    BLOCK_M: Int,
    BLOCK_N: Int,
    BLOCK_K: Int,
    GROUP_M: Int,
](
    C: UnsafePointer[Scalar[c_type]],
    A: UnsafePointer[Scalar[a_type]],
    B: UnsafePointer[Scalar[b_type]],
    M: Int,
    N: Int,
    K: Int,
    locks: UnsafePointer[Int32],
    stride_am: Int,
    stride_ak: Int,
    stride_bk: Int,
    stride_bn: Int,
    stride_cm: Int,
    stride_cn: Int,
    total_tiles_streamk: UInt,
):
    var tile_id = block_idx.x + total_tiles_streamk
    var pid = IndexList[2](0, 0)
    if GROUP_M > 0:
        pid = swizzle_tile(tile_id, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M)
    else:
        pid = linear_tile(tile_id, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M)

    var rm_base = pid[0] * BLOCK_M
    var rn_base = pid[1] * BLOCK_N

    var tx = thread_idx.x
    var ty = thread_idx.y

    var global_r = rm_base + ty
    var global_c = rn_base + tx
    var accum = Scalar[c_type](0)

    var steps = (K + BLOCK_K - 1) // BLOCK_K
    for s in range(steps):
        var k_offset = s * BLOCK_K
        for kk in range(BLOCK_K):
            var actual_k = k_offset + kk
            if global_r < M and actual_k < K and global_c < N:
                var a_val = A.load(global_r * stride_am + actual_k * stride_ak)
                var b_val = B.load(actual_k * stride_bk + global_c * stride_bn)
                accum += a_val.cast[c_type]() * b_val.cast[c_type]()

    if global_r < M and global_c < N:
        C[global_r * stride_cm + global_c * stride_cn] = accum


"""
(1) Stream-K wave:  multiple blocks subdivide K across some subset of tiles
       +-----------+    +-----------+     +-----------+    +-----------+
       | Block 0   |    | Block 1   |     | Block 2   |    | Block 3   |
       +-----+-----+    +-----+-----+     +-----+-----+    +-----+-----+
       |T0,K0 |T0,K1|    |T0,K2 |T1,K0|     |T1,K1 |T1,K2|    |T2,K0 |T2,K1| ...
        partial  partial  partial  partial  partial  partial  partial  partial

    - The tile T0 is computed in 3 partial K-chunks by Blocks 0,1,...  
    - The tile T1 is also subdivided, etc.
    - M <-> tile dimension,  N <-> tile dimension,  K <-> subdivided dimension
    - Atomic merges & locks coordinate partial sums.

(2) Full Tiles wave:   each remaining tile is handled by 1 block fully
       +-----------+  <--- Block 10 covers tile T3 entirely, no partial sums
       |   T3      |
       +-----------+

       +-----------+  <--- Block 11 covers tile T4 entirely, no partial sums
       |   T4      |
       +-----------+

       +-----------+  <--- Block 12 covers tile T5 entirely, no partial sums
       |   T5      |
       +-----------+
"""


fn matmul_stream_k[
    c_type: DType,
    c_shape: DimList,
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList, //,
    *,
    total_programs_streamk: Int,
](
    c: NDBuffer[c_type, 2, c_shape],
    a: NDBuffer[a_type, 2, a_shape],
    b: NDBuffer[b_type, 2, b_shape],
    M: Int,
    N: Int,
    K: Int,
    ctx: DeviceContext,
) raises:
    alias BLK_M = 16
    alias BLK_N = 16
    alias BLK_K = 16

    var total_blocks_M = (M + BLK_M - 1) // BLK_M
    var total_blocks_N = (N + BLK_N - 1) // BLK_N
    var iters_per_tile = (K + BLK_K - 1) // BLK_K
    alias GROUP_M = 8

    var total_tiles = total_blocks_M * total_blocks_N
    var total_tiles_streamk = total_tiles % total_programs_streamk
    var total_blocking_tiles = total_tiles - total_tiles_streamk
    var total_iters_streamk = total_tiles_streamk * iters_per_tile
    var total_full_tiles_streamk = 0 if (total_iters_streamk == 0) else (
        total_iters_streamk // total_programs_streamk
    )
    var total_partial_tiles_streamk = 0 if (total_iters_streamk == 0) else (
        total_iters_streamk % total_programs_streamk
    )

    var locks_data = ctx.enqueue_create_buffer[DType.int32](total_tiles_streamk)
    ctx.enqueue_memset(locks_data, 0)

    print("M=", M, ", N=", N, ", K=", K)
    print(
        "Total tiles=",
        total_tiles,
        "  (M-tiles=",
        total_blocks_M,
        ", N-tiles=",
        total_blocks_N,
        ")",
    )
    print("iters_per_tile=", iters_per_tile)
    print(
        "total_tiles_streamk=",
        total_tiles_streamk,
        ", total_blocking_tiles=",
        total_blocking_tiles,
    )
    print(
        "total_full_tiles_streamk=",
        total_full_tiles_streamk,
        ", total_partial_tiles_streamk=",
        total_partial_tiles_streamk,
    )

    if total_programs_streamk > 0:
        alias first_wave = first_wave_kernel[
            c_type,
            a_type,
            b_type,
            BLK_M,
            BLK_N,
            BLK_K,
            GROUP_M,
        ]

        ctx.enqueue_function[first_wave, dump_asm=False](
            c.data,
            a.data,
            b.data,
            M,
            N,
            K,
            locks_data.unsafe_ptr(),
            K,
            1,
            N,
            1,
            N,
            1,
            total_full_tiles_streamk,
            total_partial_tiles_streamk,
            iters_per_tile,
            grid_dim=total_programs_streamk,
            block_dim=(BLK_N, BLK_M),
        )
        ctx.synchronize()

    if total_blocking_tiles > 0:
        alias full_tiles = full_tiles_kernel[
            c_type,
            a_type,
            b_type,
            BLK_M,
            BLK_N,
            BLK_K,
            GROUP_M,
        ]
        ctx.enqueue_function[full_tiles](
            c.data,
            a.data,
            b.data,
            M,
            N,
            K,
            locks_data.unsafe_ptr(),
            K,
            1,
            N,
            1,
            N,
            1,
            total_tiles_streamk,
            grid_dim=total_blocking_tiles,
            block_dim=(BLK_N, BLK_M),
        )
        ctx.synchronize()

    _ = locks_data^
    return


fn run_matmul_stream_k[
    type: DType,
    M: Int,
    N: Int,
    K: Int,
](ctx: DeviceContext,) raises:
    print("== run_matmul kernel stream_k")

    var a_host = UnsafePointer[Scalar[type]].alloc(M * K)
    var b_host = UnsafePointer[Scalar[type]].alloc(K * N)
    var c_host = UnsafePointer[Scalar[type]].alloc(M * N)
    var c_host_n = UnsafePointer[Scalar[type]].alloc(M * N)

    var rng_width = 2
    var rand_min = -1 * rng_width
    var rand_max = rng_width

    for i in range(M * K):
        var val = Scalar[DType.float32](i % 20)
        a_host[i] = val.cast[type]()

    for i in range(K * N):
        var val = Scalar[DType.float32](i % 20)
        b_host[i] = val.cast[type]()

    for i in range(M * N):
        var val = Float32(0)
        c_host[i] = val.cast[type]()
        c_host_n[i] = c_host[i]

    alias a_shape = DimList(M, K)
    alias b_shape = DimList(K, N)
    alias c_shape = DimList(M, N)

    var a_device = ctx.enqueue_create_buffer[type](M * K)
    var b_device = ctx.enqueue_create_buffer[type](K * N)
    var c_device = ctx.enqueue_create_buffer[type](M * N)
    var a_buf = NDBuffer[type, 2, a_shape](a_device.unsafe_ptr(), Index(M, K))
    var b_buf = NDBuffer[type, 2, b_shape](b_device.unsafe_ptr(), Index(K, N))
    var c_buf = NDBuffer[type, 2, c_shape](c_device.unsafe_ptr(), Index(M, N))

    var a_device_n = ctx.enqueue_create_buffer[type](M * K)
    var b_device_n = ctx.enqueue_create_buffer[type](K * N)
    var c_device_n = ctx.enqueue_create_buffer[type](M * N)

    ctx.enqueue_copy(a_device, a_host)
    ctx.enqueue_copy(b_device, b_host)

    alias sm_count = ctx.device_info.sm_count

    matmul_stream_k[total_programs_streamk=sm_count](
        rebind[NDBuffer[type, 2, c_shape]](c_buf),
        rebind[NDBuffer[type, 2, a_shape]](a_buf),
        rebind[NDBuffer[type, 2, b_shape]](b_buf),
        M,
        N,
        K,
        ctx,
    )

    ctx.enqueue_copy(c_host, c_device)
    ctx.synchronize()

    ctx.enqueue_copy(a_device_n, a_host)
    ctx.enqueue_copy(b_device_n, b_host)

    alias BLOCK_DIM = 16

    ctx.enqueue_function[matmul_kernel_naive[type, type, type, BLOCK_DIM]](
        c_device_n,
        a_device_n,
        b_device_n,
        M,
        N,
        K,
        grid_dim=(ceildiv(M, BLOCK_DIM), ceildiv(N, BLOCK_DIM)),
        block_dim=(BLOCK_DIM, BLOCK_DIM),
    )

    ctx.enqueue_copy(c_host_n, c_device_n)
    ctx.synchronize()

    var rtol = 0.01

    for i in range(M * N):
        var out_val = c_host[i]
        var out_ref = c_host_n[i]
        assert_almost_equal(out_val, out_ref, rtol=rtol)

    _ = a_device
    _ = b_device
    _ = c_device

    _ = a_device_n
    _ = b_device_n
    _ = c_device_n

    _ = a_host
    _ = b_host
    _ = c_host
    _ = c_host_n


def main():
    with DeviceContext() as ctx:
        run_matmul_stream_k[DType.float32, 128, 128, 128](ctx)
        run_matmul_stream_k[DType.float32, 512, 2560, 8192](ctx)
        run_matmul_stream_k[DType.float32, 256, 256, 1024](ctx)
        run_matmul_stream_k[DType.float32, 128, 128, 1024](ctx)
