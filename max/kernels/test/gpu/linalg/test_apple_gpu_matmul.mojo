# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Unit tests for the Apple M5 simdgroup-tiled matmul kernel."""

from std.random import random_si64
from std.gpu.host import DeviceContext
from std.sys.info import _accelerator_arch

from layout import TileTensor, Idx
from layout.tile_layout import row_major

from linalg.matmul.gpu.apple.matmul_kernel import (
    apple_matmul_kernel,
    enqueue_apple_matmul,
    morton_decode_2d,
    morton_decode_2d_rect,
)


def _host_matmul_nn[
    a_type: DType, b_type: DType
](
    a_ptr: UnsafePointer[Scalar[a_type], ...],
    b_ptr: UnsafePointer[Scalar[b_type], ...],
    M: Int,
    N: Int,
    K: Int,
    i: Int,
    j: Int,
) -> Float32:
    """Compute one element D[i,j] = sum_k A[i,k] * B[k,j] on the host (fp32 accum).
    """
    var acc = Float32(0)
    for k in range(K):
        acc += Float32(a_ptr[i * K + k]) * Float32(b_ptr[k * N + j])
    return acc


def test_morton_decode_2d() raises:
    """Verify Morton bit-interleave decode on a 4x4 virtual grid.

    Canonical Z-order on 2-bit x 2-bit -> 4-bit flat index:
        flat=0  -> (0, 0)
        flat=1  -> (0, 1)     even bits (0,2,...)  = col
        flat=2  -> (1, 0)     odd bits  (1,3,...)  = row
        flat=3  -> (1, 1)
        flat=4  -> (0, 2)
        flat=5  -> (0, 3)
        flat=6  -> (1, 2)
        flat=7  -> (1, 3)
        flat=8  -> (2, 0)
        ...
    """
    print("== test_morton_decode_2d")
    var pass_ = True

    # Expected (tile_m, tile_n) pairs for flat indices 0..15.
    var exp_m: InlineArray[UInt32, 16] = [
        UInt32(0),
        0,
        1,
        1,
        0,
        0,
        1,
        1,
        2,
        2,
        3,
        3,
        2,
        2,
        3,
        3,
    ]
    var exp_n: InlineArray[UInt32, 16] = [
        UInt32(0),
        1,
        0,
        1,
        2,
        3,
        2,
        3,
        0,
        1,
        0,
        1,
        2,
        3,
        2,
        3,
    ]

    for i in range(16):
        var got = morton_decode_2d(UInt32(i))
        if got[0] != exp_m[i] or got[1] != exp_n[i]:
            print(
                "FAIL: flat",
                i,
                "got",
                got[0],
                got[1],
                "expected",
                exp_m[i],
                exp_n[i],
            )
            pass_ = False
    if pass_:
        print("PASS")


def test_morton_decode_2d_rect() raises:
    """Verify rectangular Morton decode covers each (m, n) exactly once.

    Tests three regimes:
      - Square: log2_m == log2_n -> reduces to square Morton (already
        tested by test_morton_decode_2d).
      - Tall (log2_m > log2_n): high bits sweep along M.
      - Wide (log2_n > log2_m): high bits sweep along N.
      - Degenerate (log2_m == 0): pure linear sweep along N.
    """
    print("== test_morton_decode_2d_rect")
    var pass_ = True

    # 2x16 grid (log2_m=1, log2_n=4): square core 2x2, sweep 4 hi-bit
    # chunks along N. Every flat in [0, 32) must hit a unique (m, n) in
    # [0, 2) x [0, 16).
    var seen_2x16 = InlineArray[Bool, 32](fill=False)
    for i in range(32):
        var got = morton_decode_2d_rect(UInt32(i), UInt32(1), UInt32(4))
        var m = Int(got[0])
        var n = Int(got[1])
        if m < 0 or m >= 2 or n < 0 or n >= 16:
            print(
                "FAIL: 2x16 flat=",
                i,
                "decoded m=",
                m,
                "n=",
                n,
                "out of range",
            )
            pass_ = False
            continue
        var slot = m * 16 + n
        if seen_2x16[slot]:
            print(
                "FAIL: 2x16 flat=",
                i,
                "decoded m=",
                m,
                "n=",
                n,
                "(duplicate)",
            )
            pass_ = False
        seen_2x16[slot] = True

    # 16x2 grid (log2_m=4, log2_n=1): tall analogue. Should cover all
    # (m, n) in [0, 16) x [0, 2).
    var seen_16x2 = InlineArray[Bool, 32](fill=False)
    for i in range(32):
        var got = morton_decode_2d_rect(UInt32(i), UInt32(4), UInt32(1))
        var m = Int(got[0])
        var n = Int(got[1])
        if m < 0 or m >= 16 or n < 0 or n >= 2:
            print(
                "FAIL: 16x2 flat=",
                i,
                "decoded m=",
                m,
                "n=",
                n,
                "out of range",
            )
            pass_ = False
            continue
        var slot = m * 2 + n
        if seen_16x2[slot]:
            print(
                "FAIL: 16x2 flat=",
                i,
                "decoded m=",
                m,
                "n=",
                n,
                "(duplicate)",
            )
            pass_ = False
        seen_16x2[slot] = True

    # 4x4 square: must agree with morton_decode_2d on every flat.
    for i in range(16):
        var got_rect = morton_decode_2d_rect(UInt32(i), UInt32(2), UInt32(2))
        var got_sq = morton_decode_2d(UInt32(i))
        if got_rect[0] != got_sq[0] or got_rect[1] != got_sq[1]:
            print(
                "FAIL: 4x4 flat=",
                i,
                "rect=(",
                got_rect[0],
                ",",
                got_rect[1],
                ") square=(",
                got_sq[0],
                ",",
                got_sq[1],
                ")",
            )
            pass_ = False

    # 1x16 degenerate (log2_m=0): rect should produce (0, i) for
    # i in [0, 16).
    for i in range(16):
        var got = morton_decode_2d_rect(UInt32(i), UInt32(0), UInt32(4))
        if Int(got[0]) != 0 or Int(got[1]) != i:
            print(
                "FAIL: 1x16 flat=",
                i,
                "got=(",
                got[0],
                ",",
                got[1],
                ") expected=(0,",
                i,
                ")",
            )
            pass_ = False

    # 16x1 degenerate (log2_n=0): rect should produce (i, 0) for
    # i in [0, 16). Symmetric counterpart of the 1x16 case above.
    for i in range(16):
        var got = morton_decode_2d_rect(UInt32(i), UInt32(4), UInt32(0))
        if Int(got[0]) != i or Int(got[1]) != 0:
            print(
                "FAIL: 16x1 flat=",
                i,
                "got=(",
                got[0],
                ",",
                got[1],
                ") expected=(",
                i,
                ",0)",
            )
            pass_ = False

    if pass_:
        print("PASS")


def test_kernel_single_tile_nn_fp16(ctx: DeviceContext) raises:
    """D[64,64] = A[64,16] @ B[16,64], NN, fp16->fp32, single threadgroup."""
    print("== test_kernel_single_tile_nn_fp16")
    comptime M = 64
    comptime N = 64
    comptime K = 16

    var a_host = ctx.enqueue_create_host_buffer[DType.float16](M * K)
    var b_host = ctx.enqueue_create_host_buffer[DType.float16](K * N)
    for i in range(M * K):
        a_host[i] = Scalar[DType.float16](
            random_si64(Int64(-2), Int64(2)).cast[DType.float16]()
        )
    for i in range(K * N):
        b_host[i] = Scalar[DType.float16](
            random_si64(Int64(-2), Int64(2)).cast[DType.float16]()
        )

    var a_dev = ctx.enqueue_create_buffer[DType.float16](M * K)
    var b_dev = ctx.enqueue_create_buffer[DType.float16](K * N)
    var d_dev = ctx.enqueue_create_buffer[DType.float32](M * N)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)

    comptime kernel = apple_matmul_kernel[
        in_type=DType.float16,
        transpose_b=False,
    ]
    ctx.enqueue_function[kernel](
        d_dev.unsafe_ptr(),
        a_dev.unsafe_ptr(),
        b_dev.unsafe_ptr(),
        M,
        N,
        K,
        UInt32(0),
        UInt32(0),
        grid_dim=(1),
        block_dim=(128),
    )

    var d_host = ctx.enqueue_create_host_buffer[DType.float32](M * N)
    ctx.enqueue_copy(d_host, d_dev)
    ctx.synchronize()

    var pass_ = True
    for i in range(M):
        for j in range(N):
            var exp = _host_matmul_nn[DType.float16, DType.float16](
                a_host.unsafe_ptr(), b_host.unsafe_ptr(), M, N, K, i, j
            )
            var got = d_host[i * N + j]
            if abs(got - exp) > Float32(0.5):
                print("FAIL:", i, j, "got", got, "expected", exp)
                pass_ = False
    if pass_:
        print("PASS")


def test_kernel_single_tile_k128_nn_fp16(ctx: DeviceContext) raises:
    """D[64,64] = A[64,128] @ B[128,64], NN, fp16->fp32."""
    print("== test_kernel_single_tile_k128_nn_fp16")
    comptime M = 64
    comptime N = 64
    comptime K = 128

    var a_host = ctx.enqueue_create_host_buffer[DType.float16](M * K)
    var b_host = ctx.enqueue_create_host_buffer[DType.float16](K * N)
    for i in range(M * K):
        a_host[i] = Scalar[DType.float16](
            random_si64(Int64(-2), Int64(2)).cast[DType.float16]()
        )
    for i in range(K * N):
        b_host[i] = Scalar[DType.float16](
            random_si64(Int64(-2), Int64(2)).cast[DType.float16]()
        )

    var a_dev = ctx.enqueue_create_buffer[DType.float16](M * K)
    var b_dev = ctx.enqueue_create_buffer[DType.float16](K * N)
    var d_dev = ctx.enqueue_create_buffer[DType.float32](M * N)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)

    comptime kernel = apple_matmul_kernel[
        in_type=DType.float16,
        transpose_b=False,
    ]
    ctx.enqueue_function[kernel](
        d_dev.unsafe_ptr(),
        a_dev.unsafe_ptr(),
        b_dev.unsafe_ptr(),
        M,
        N,
        K,
        UInt32(0),
        UInt32(0),
        grid_dim=(1),
        block_dim=(128),
    )

    var d_host = ctx.enqueue_create_host_buffer[DType.float32](M * N)
    ctx.enqueue_copy(d_host, d_dev)
    ctx.synchronize()

    var pass_ = True
    for i in range(M):
        for j in range(N):
            var exp = _host_matmul_nn[DType.float16, DType.float16](
                a_host.unsafe_ptr(), b_host.unsafe_ptr(), M, N, K, i, j
            )
            var got = d_host[i * N + j]
            if abs(got - exp) > Float32(1.0):
                print("FAIL:", i, j, "got", got, "expected", exp)
                pass_ = False
    if pass_:
        print("PASS")


def _ceil_pow2(x: Int) -> Int:
    """Smallest power of 2 >= x, for x >= 1."""
    var r = 1
    while r < x:
        r *= 2
    return r


def _grid_for(
    m: Int, n: Int, bm: Int = 64, bn: Int = 64
) -> Tuple[Int, UInt32, UInt32]:
    """Compute (grid_dim, log2_grid_m, log2_grid_n) for apple_matmul_kernel.

    Returns side_m * side_n (per-axis next-pow2) and the log2 of each.
    Matches the semantics of enqueue_apple_matmul.
    """
    var grid_m = (m + bm - 1) // bm
    var grid_n = (n + bn - 1) // bn

    var side_m = 1
    var log2_m: UInt32 = 0
    while side_m < grid_m:
        side_m *= 2
        log2_m += 1
    var side_n = 1
    var log2_n: UInt32 = 0
    while side_n < grid_n:
        side_n *= 2
        log2_n += 1

    return (side_m * side_n, log2_m, log2_n)


def test_kernel_64x64x17_nn_fp16(ctx: DeviceContext) raises:
    """K=17 with clean M/N: exercises the fast-strip + bounded-tail K split.

    M=64 (one 64x64 threadgroup tile, clean), N=64 (clean), K=17 (= BK + 1
    so one full BK=16 strip via fast MMA, then one bounded MMA with
    k_valid=1).
    """
    print("== test_kernel_64x64x17_nn_fp16")
    comptime M = 64
    comptime N = 64
    comptime K = 17

    var a_host = ctx.enqueue_create_host_buffer[DType.float16](M * K)
    var b_host = ctx.enqueue_create_host_buffer[DType.float16](K * N)
    for i in range(M * K):
        a_host[i] = Scalar[DType.float16](
            random_si64(Int64(-2), Int64(2)).cast[DType.float16]()
        )
    for i in range(K * N):
        b_host[i] = Scalar[DType.float16](
            random_si64(Int64(-2), Int64(2)).cast[DType.float16]()
        )

    var a_dev = ctx.enqueue_create_buffer[DType.float16](M * K)
    var b_dev = ctx.enqueue_create_buffer[DType.float16](K * N)
    var d_dev = ctx.enqueue_create_buffer[DType.float32](M * N)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)

    comptime kernel = apple_matmul_kernel[
        in_type=DType.float16,
        transpose_b=False,
    ]
    var grid_dim, log2_grid_m, log2_grid_n = _grid_for(M, N)
    ctx.enqueue_function[kernel](
        d_dev.unsafe_ptr(),
        a_dev.unsafe_ptr(),
        b_dev.unsafe_ptr(),
        M,
        N,
        K,
        log2_grid_m,
        log2_grid_n,
        grid_dim=(grid_dim),
        block_dim=(128),
    )

    var d_host = ctx.enqueue_create_host_buffer[DType.float32](M * N)
    ctx.enqueue_copy(d_host, d_dev)
    ctx.synchronize()

    var pass_ = True
    for i in range(M):
        for j in range(N):
            var exp = _host_matmul_nn[DType.float16, DType.float16](
                a_host.unsafe_ptr(), b_host.unsafe_ptr(), M, N, K, i, j
            )
            var got = d_host[i * N + j]
            if abs(got - exp) > Float32(0.5):
                print("FAIL:", i, j, "got", got, "expected", exp)
                pass_ = False
    if pass_:
        print("PASS")


def test_kernel_256x256x16_nn_fp16(ctx: DeviceContext) raises:
    """D[256,256] = A[256,16] @ B[16,256] -- 16 threadgroups (4x4 tiles)."""
    print("== test_kernel_256x256x16_nn_fp16")
    comptime M = 256
    comptime N = 256
    comptime K = 16

    var a_host = ctx.enqueue_create_host_buffer[DType.float16](M * K)
    var b_host = ctx.enqueue_create_host_buffer[DType.float16](K * N)
    for i in range(M * K):
        a_host[i] = Scalar[DType.float16](
            random_si64(Int64(-2), Int64(2)).cast[DType.float16]()
        )
    for i in range(K * N):
        b_host[i] = Scalar[DType.float16](
            random_si64(Int64(-2), Int64(2)).cast[DType.float16]()
        )

    var a_dev = ctx.enqueue_create_buffer[DType.float16](M * K)
    var b_dev = ctx.enqueue_create_buffer[DType.float16](K * N)
    var d_dev = ctx.enqueue_create_buffer[DType.float32](M * N)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)

    comptime kernel = apple_matmul_kernel[
        in_type=DType.float16,
        transpose_b=False,
    ]
    var grid_dim, log2_grid_m, log2_grid_n = _grid_for(M, N)
    ctx.enqueue_function[kernel](
        d_dev.unsafe_ptr(),
        a_dev.unsafe_ptr(),
        b_dev.unsafe_ptr(),
        M,
        N,
        K,
        log2_grid_m,
        log2_grid_n,
        grid_dim=(grid_dim),
        block_dim=(128),
    )

    var d_host = ctx.enqueue_create_host_buffer[DType.float32](M * N)
    ctx.enqueue_copy(d_host, d_dev)
    ctx.synchronize()

    var pass_ = True
    for i in range(M):
        for j in range(N):
            var exp = _host_matmul_nn[DType.float16, DType.float16](
                a_host.unsafe_ptr(), b_host.unsafe_ptr(), M, N, K, i, j
            )
            var got = d_host[i * N + j]
            if abs(got - exp) > Float32(0.5):
                print("FAIL:", i, j, "got", got, "expected", exp)
                pass_ = False
    if pass_:
        print("PASS")


def _host_matmul_nt[
    a_type: DType, b_type: DType
](
    a_ptr: UnsafePointer[Scalar[a_type], ...],
    b_ptr: UnsafePointer[Scalar[b_type], ...],
    M: Int,
    N: Int,
    K: Int,
    i: Int,
    j: Int,
) -> Float32:
    """D[i,j] = sum_k A[i,k] * B[j,k]  (transpose_b=True, B stored as (N, K)).
    """
    var acc = Float32(0)
    for k in range(K):
        acc += Float32(a_ptr[i * K + k]) * Float32(b_ptr[j * K + k])
    return acc


def test_kernel_128x128x32_nt_fp16(ctx: DeviceContext) raises:
    """D[128,128] = A[128,32] @ B[128,32]^T, NT, fp16->fp32."""
    print("== test_kernel_128x128x32_nt_fp16")
    comptime M = 128
    comptime N = 128
    comptime K = 32

    var a_host = ctx.enqueue_create_host_buffer[DType.float16](M * K)
    # B is stored as (N, K) for transpose_b=True.
    var b_host = ctx.enqueue_create_host_buffer[DType.float16](N * K)
    for i in range(M * K):
        a_host[i] = Scalar[DType.float16](
            random_si64(Int64(-2), Int64(2)).cast[DType.float16]()
        )
    for i in range(N * K):
        b_host[i] = Scalar[DType.float16](
            random_si64(Int64(-2), Int64(2)).cast[DType.float16]()
        )

    var a_dev = ctx.enqueue_create_buffer[DType.float16](M * K)
    var b_dev = ctx.enqueue_create_buffer[DType.float16](N * K)
    var d_dev = ctx.enqueue_create_buffer[DType.float32](M * N)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)

    comptime kernel = apple_matmul_kernel[
        in_type=DType.float16,
        transpose_b=True,
    ]
    var grid_dim, log2_grid_m, log2_grid_n = _grid_for(M, N)
    ctx.enqueue_function[kernel](
        d_dev.unsafe_ptr(),
        a_dev.unsafe_ptr(),
        b_dev.unsafe_ptr(),
        M,
        N,
        K,
        log2_grid_m,
        log2_grid_n,
        grid_dim=(grid_dim),
        block_dim=(128),
    )

    var d_host = ctx.enqueue_create_host_buffer[DType.float32](M * N)
    ctx.enqueue_copy(d_host, d_dev)
    ctx.synchronize()

    var pass_ = True
    for i in range(M):
        for j in range(N):
            var exp = _host_matmul_nt[DType.float16, DType.float16](
                a_host.unsafe_ptr(), b_host.unsafe_ptr(), M, N, K, i, j
            )
            var got = d_host[i * N + j]
            if abs(got - exp) > Float32(0.5):
                print("FAIL:", i, j, "got", got, "expected", exp)
                pass_ = False
    if pass_:
        print("PASS")


def test_kernel_ragged_100x200x33_nn_fp16(ctx: DeviceContext) raises:
    """Ragged shape: M=100 (< 2*64), N=200 (< 4*64), K=33 (not 16-aligned)."""
    print("== test_kernel_ragged_100x200x33_nn_fp16")
    comptime M = 100
    comptime N = 200
    comptime K = 33

    var a_host = ctx.enqueue_create_host_buffer[DType.float16](M * K)
    var b_host = ctx.enqueue_create_host_buffer[DType.float16](K * N)
    for i in range(M * K):
        a_host[i] = Scalar[DType.float16](
            random_si64(Int64(-2), Int64(2)).cast[DType.float16]()
        )
    for i in range(K * N):
        b_host[i] = Scalar[DType.float16](
            random_si64(Int64(-2), Int64(2)).cast[DType.float16]()
        )

    var a_dev = ctx.enqueue_create_buffer[DType.float16](M * K)
    var b_dev = ctx.enqueue_create_buffer[DType.float16](K * N)
    var d_dev = ctx.enqueue_create_buffer[DType.float32](M * N)
    # Sentinel: initialize output to -1e30 so untouched elements are visible.
    var d_init = ctx.enqueue_create_host_buffer[DType.float32](M * N)
    for i in range(M * N):
        d_init[i] = Float32(-1.0e30)
    ctx.enqueue_copy(d_dev, d_init)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)

    comptime kernel = apple_matmul_kernel[
        in_type=DType.float16,
        transpose_b=False,
    ]
    var grid_dim, log2_grid_m, log2_grid_n = _grid_for(M, N)
    ctx.enqueue_function[kernel](
        d_dev.unsafe_ptr(),
        a_dev.unsafe_ptr(),
        b_dev.unsafe_ptr(),
        M,
        N,
        K,
        log2_grid_m,
        log2_grid_n,
        grid_dim=(grid_dim),
        block_dim=(128),
    )

    var d_host = ctx.enqueue_create_host_buffer[DType.float32](M * N)
    ctx.enqueue_copy(d_host, d_dev)
    ctx.synchronize()

    var pass_ = True
    for i in range(M):
        for j in range(N):
            var exp = _host_matmul_nn[DType.float16, DType.float16](
                a_host.unsafe_ptr(), b_host.unsafe_ptr(), M, N, K, i, j
            )
            var got = d_host[i * N + j]
            if abs(got - exp) > Float32(0.5):
                print("FAIL:", i, j, "got", got, "expected", exp)
                pass_ = False
    if pass_:
        print("PASS")


def test_kernel_ragged_100x200x32_nn_fp16(ctx: DeviceContext) raises:
    """Bounded path with K-clean: M=100, N=200, K=32 (BK-aligned).

    Exercises m_n_edge=True && has_k_tail=False -- the bounded for-loop
    runs k_full_strips iterations with tail_count=0, then store_bounded.
    The existing K=33 ragged test always sets has_k_tail=True so this
    branch was previously unreached.
    """
    print("== test_kernel_ragged_100x200x32_nn_fp16")
    comptime M = 100
    comptime N = 200
    comptime K = 32

    var a_host = ctx.enqueue_create_host_buffer[DType.float16](M * K)
    var b_host = ctx.enqueue_create_host_buffer[DType.float16](K * N)
    for i in range(M * K):
        a_host[i] = Scalar[DType.float16](
            random_si64(Int64(-2), Int64(2)).cast[DType.float16]()
        )
    for i in range(K * N):
        b_host[i] = Scalar[DType.float16](
            random_si64(Int64(-2), Int64(2)).cast[DType.float16]()
        )

    var a_dev = ctx.enqueue_create_buffer[DType.float16](M * K)
    var b_dev = ctx.enqueue_create_buffer[DType.float16](K * N)
    var d_dev = ctx.enqueue_create_buffer[DType.float32](M * N)
    # Sentinel: detect spurious stores past M/N edge in the bounded path.
    var d_init = ctx.enqueue_create_host_buffer[DType.float32](M * N)
    for i in range(M * N):
        d_init[i] = Float32(-1.0e30)
    ctx.enqueue_copy(d_dev, d_init)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)

    comptime kernel = apple_matmul_kernel[
        in_type=DType.float16,
        transpose_b=False,
    ]
    var grid_dim, log2_grid_m, log2_grid_n = _grid_for(M, N)
    ctx.enqueue_function[kernel](
        d_dev.unsafe_ptr(),
        a_dev.unsafe_ptr(),
        b_dev.unsafe_ptr(),
        M,
        N,
        K,
        log2_grid_m,
        log2_grid_n,
        grid_dim=(grid_dim),
        block_dim=(128),
    )

    var d_host = ctx.enqueue_create_host_buffer[DType.float32](M * N)
    ctx.enqueue_copy(d_host, d_dev)
    ctx.synchronize()

    var pass_ = True
    for i in range(M):
        for j in range(N):
            var exp = _host_matmul_nn[DType.float16, DType.float16](
                a_host.unsafe_ptr(), b_host.unsafe_ptr(), M, N, K, i, j
            )
            var got = d_host[i * N + j]
            if abs(got - exp) > Float32(0.5):
                print("FAIL:", i, j, "got", got, "expected", exp)
                pass_ = False
    if pass_:
        print("PASS")


def test_kernel_ragged_100x200x32_nt_fp16(ctx: DeviceContext) raises:
    """Bounded path with K-clean and transpose_b=True.

    Exercises the NT bounded-K-clean branch (the equivalent of the NN
    bounded path tested above, but with B presented as col_major(K, N)
    over a (N, K) row-major buffer). None of the existing ragged tests
    use transpose_b, so this is the first transpose_b ragged coverage.
    """
    print("== test_kernel_ragged_100x200x32_nt_fp16")
    comptime M = 100
    comptime N = 200
    comptime K = 32

    var a_host = ctx.enqueue_create_host_buffer[DType.float16](M * K)
    # B stored as (N, K) for transpose_b=True.
    var b_host = ctx.enqueue_create_host_buffer[DType.float16](N * K)
    for i in range(M * K):
        a_host[i] = Scalar[DType.float16](
            random_si64(Int64(-2), Int64(2)).cast[DType.float16]()
        )
    for i in range(N * K):
        b_host[i] = Scalar[DType.float16](
            random_si64(Int64(-2), Int64(2)).cast[DType.float16]()
        )

    var a_dev = ctx.enqueue_create_buffer[DType.float16](M * K)
    var b_dev = ctx.enqueue_create_buffer[DType.float16](N * K)
    var d_dev = ctx.enqueue_create_buffer[DType.float32](M * N)
    # Sentinel: detect spurious stores past M/N edge in the bounded path.
    var d_init = ctx.enqueue_create_host_buffer[DType.float32](M * N)
    for i in range(M * N):
        d_init[i] = Float32(-1.0e30)
    ctx.enqueue_copy(d_dev, d_init)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)

    comptime kernel = apple_matmul_kernel[
        in_type=DType.float16,
        transpose_b=True,
    ]
    var grid_dim, log2_grid_m, log2_grid_n = _grid_for(M, N)
    ctx.enqueue_function[kernel](
        d_dev.unsafe_ptr(),
        a_dev.unsafe_ptr(),
        b_dev.unsafe_ptr(),
        M,
        N,
        K,
        log2_grid_m,
        log2_grid_n,
        grid_dim=(grid_dim),
        block_dim=(128),
    )

    var d_host = ctx.enqueue_create_host_buffer[DType.float32](M * N)
    ctx.enqueue_copy(d_host, d_dev)
    ctx.synchronize()

    var pass_ = True
    for i in range(M):
        for j in range(N):
            var exp = _host_matmul_nt[DType.float16, DType.float16](
                a_host.unsafe_ptr(), b_host.unsafe_ptr(), M, N, K, i, j
            )
            var got = d_host[i * N + j]
            if abs(got - exp) > Float32(0.5):
                print("FAIL:", i, j, "got", got, "expected", exp)
                pass_ = False
    if pass_:
        print("PASS")


def test_kernel_M20_N80_K16_nn_fp16(ctx: DeviceContext) raises:
    """M < SG_M edge: triggers the per-simdgroup OOB early return.

    M=20 < SG_M=32. With grid_m=1, grid_n=2 -> side_m=1, side_n=2 -> 2
    threadgroups launched. Each threadgroup has 4 simdgroups; for those
    at sg_m_idx=1, row_base=32 >= m=20 and the new early return fires
    without issuing any loads or stores (Task 3). Sentinel-fills D
    with -1e30 first so any spurious write past M=20 would be visible.
    """
    print("== test_kernel_M20_N80_K16_nn_fp16")
    comptime M = 20
    comptime N = 80
    comptime K = 16

    var a_host = ctx.enqueue_create_host_buffer[DType.float16](M * K)
    var b_host = ctx.enqueue_create_host_buffer[DType.float16](K * N)
    for i in range(M * K):
        a_host[i] = Scalar[DType.float16](
            random_si64(Int64(-2), Int64(2)).cast[DType.float16]()
        )
    for i in range(K * N):
        b_host[i] = Scalar[DType.float16](
            random_si64(Int64(-2), Int64(2)).cast[DType.float16]()
        )

    var a_dev = ctx.enqueue_create_buffer[DType.float16](M * K)
    var b_dev = ctx.enqueue_create_buffer[DType.float16](K * N)
    var d_dev = ctx.enqueue_create_buffer[DType.float32](M * N)
    var d_init = ctx.enqueue_create_host_buffer[DType.float32](M * N)
    for i in range(M * N):
        d_init[i] = Float32(-1.0e30)
    ctx.enqueue_copy(d_dev, d_init)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)

    comptime kernel = apple_matmul_kernel[
        in_type=DType.float16,
        transpose_b=False,
    ]
    var grid_dim, log2_grid_m, log2_grid_n = _grid_for(M, N)
    ctx.enqueue_function[kernel](
        d_dev.unsafe_ptr(),
        a_dev.unsafe_ptr(),
        b_dev.unsafe_ptr(),
        M,
        N,
        K,
        log2_grid_m,
        log2_grid_n,
        grid_dim=(grid_dim),
        block_dim=(128),
    )

    var d_host = ctx.enqueue_create_host_buffer[DType.float32](M * N)
    ctx.enqueue_copy(d_host, d_dev)
    ctx.synchronize()

    var pass_ = True
    for i in range(M):
        for j in range(N):
            var exp = _host_matmul_nn[DType.float16, DType.float16](
                a_host.unsafe_ptr(), b_host.unsafe_ptr(), M, N, K, i, j
            )
            var got = d_host[i * N + j]
            if abs(got - exp) > Float32(0.5):
                print("FAIL:", i, j, "got", got, "expected", exp)
                pass_ = False
    if pass_:
        print("PASS")


def test_kernel_128x128x32_nn_bf16(ctx: DeviceContext) raises:
    """D[128,128] = A[128,32] @ B[32,128], NN, bf16->fp32."""
    print("== test_kernel_128x128x32_nn_bf16")
    comptime M = 128
    comptime N = 128
    comptime K = 32

    var a_host = ctx.enqueue_create_host_buffer[DType.bfloat16](M * K)
    var b_host = ctx.enqueue_create_host_buffer[DType.bfloat16](K * N)
    for i in range(M * K):
        a_host[i] = Scalar[DType.bfloat16](
            random_si64(Int64(-2), Int64(2)).cast[DType.bfloat16]()
        )
    for i in range(K * N):
        b_host[i] = Scalar[DType.bfloat16](
            random_si64(Int64(-2), Int64(2)).cast[DType.bfloat16]()
        )

    var a_dev = ctx.enqueue_create_buffer[DType.bfloat16](M * K)
    var b_dev = ctx.enqueue_create_buffer[DType.bfloat16](K * N)
    var d_dev = ctx.enqueue_create_buffer[DType.float32](M * N)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)

    comptime kernel = apple_matmul_kernel[
        in_type=DType.bfloat16,
        transpose_b=False,
    ]
    var grid_dim, log2_grid_m, log2_grid_n = _grid_for(M, N)
    ctx.enqueue_function[kernel](
        d_dev.unsafe_ptr(),
        a_dev.unsafe_ptr(),
        b_dev.unsafe_ptr(),
        M,
        N,
        K,
        log2_grid_m,
        log2_grid_n,
        grid_dim=(grid_dim),
        block_dim=(128),
    )

    var d_host = ctx.enqueue_create_host_buffer[DType.float32](M * N)
    ctx.enqueue_copy(d_host, d_dev)
    ctx.synchronize()

    var pass_ = True
    for i in range(M):
        for j in range(N):
            var exp = _host_matmul_nn[DType.bfloat16, DType.bfloat16](
                a_host.unsafe_ptr(), b_host.unsafe_ptr(), M, N, K, i, j
            )
            var got = d_host[i * N + j]
            if abs(got - exp) > Float32(0.5):
                print("FAIL:", i, j, "got", got, "expected", exp)
                pass_ = False
    if pass_:
        print("PASS")


def test_kernel_128x128x32_nn_fp32(ctx: DeviceContext) raises:
    """D[128,128] = A[128,32] @ B[32,128], NN, fp32 input + accum."""
    print("== test_kernel_128x128x32_nn_fp32")
    comptime M = 128
    comptime N = 128
    comptime K = 32

    var a_host = ctx.enqueue_create_host_buffer[DType.float32](M * K)
    var b_host = ctx.enqueue_create_host_buffer[DType.float32](K * N)
    for i in range(M * K):
        a_host[i] = Scalar[DType.float32](
            random_si64(Int64(-2), Int64(2)).cast[DType.float32]()
        )
    for i in range(K * N):
        b_host[i] = Scalar[DType.float32](
            random_si64(Int64(-2), Int64(2)).cast[DType.float32]()
        )

    var a_dev = ctx.enqueue_create_buffer[DType.float32](M * K)
    var b_dev = ctx.enqueue_create_buffer[DType.float32](K * N)
    var d_dev = ctx.enqueue_create_buffer[DType.float32](M * N)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)

    comptime kernel = apple_matmul_kernel[
        in_type=DType.float32,
        transpose_b=False,
    ]
    var grid_dim, log2_grid_m, log2_grid_n = _grid_for(M, N)
    ctx.enqueue_function[kernel](
        d_dev.unsafe_ptr(),
        a_dev.unsafe_ptr(),
        b_dev.unsafe_ptr(),
        M,
        N,
        K,
        log2_grid_m,
        log2_grid_n,
        grid_dim=(grid_dim),
        block_dim=(128),
    )

    var d_host = ctx.enqueue_create_host_buffer[DType.float32](M * N)
    ctx.enqueue_copy(d_host, d_dev)
    ctx.synchronize()

    var pass_ = True
    for i in range(M):
        for j in range(N):
            var exp = _host_matmul_nn[DType.float32, DType.float32](
                a_host.unsafe_ptr(), b_host.unsafe_ptr(), M, N, K, i, j
            )
            var got = d_host[i * N + j]
            if abs(got - exp) > Float32(0.01):
                print("FAIL:", i, j, "got", got, "expected", exp)
                pass_ = False
    if pass_:
        print("PASS")


def test_enqueue_helper_fp16(ctx: DeviceContext) raises:
    """Smoke-test the host-side helper: 128x128x16 fp16 NN."""
    print("== test_enqueue_helper_fp16")
    comptime M = 128
    comptime N = 128
    comptime K = 16
    var a_host = ctx.enqueue_create_host_buffer[DType.float16](M * K)
    var b_host = ctx.enqueue_create_host_buffer[DType.float16](K * N)
    for i in range(M * K):
        a_host[i] = Scalar[DType.float16](
            random_si64(Int64(-2), Int64(2)).cast[DType.float16]()
        )
    for i in range(K * N):
        b_host[i] = Scalar[DType.float16](
            random_si64(Int64(-2), Int64(2)).cast[DType.float16]()
        )
    var a_dev = ctx.enqueue_create_buffer[DType.float16](M * K)
    var b_dev = ctx.enqueue_create_buffer[DType.float16](K * N)
    var d_dev = ctx.enqueue_create_buffer[DType.float32](M * N)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)

    var a_tt = TileTensor(a_dev.unsafe_ptr(), row_major(Idx(M), Idx(K)))
    var b_tt = TileTensor(b_dev.unsafe_ptr(), row_major(Idx(K), Idx(N)))
    var d_tt = TileTensor(d_dev.unsafe_ptr(), row_major(Idx(M), Idx(N)))

    enqueue_apple_matmul[in_type=DType.float16, transpose_b=False](
        d_tt, a_tt, b_tt, ctx
    )

    var d_host = ctx.enqueue_create_host_buffer[DType.float32](M * N)
    ctx.enqueue_copy(d_host, d_dev)
    ctx.synchronize()
    var pass_ = True
    for i in range(M):
        for j in range(N):
            var exp = _host_matmul_nn[DType.float16, DType.float16](
                a_host.unsafe_ptr(), b_host.unsafe_ptr(), M, N, K, i, j
            )
            var got = d_host[i * N + j]
            if abs(got - exp) > Float32(0.5):
                print("FAIL:", i, j, "got", got, "expected", exp)
                pass_ = False
    if pass_:
        print("PASS")


def main() raises:
    test_morton_decode_2d()
    test_morton_decode_2d_rect()
    comptime if "metal" not in _accelerator_arch():
        print("SKIP: apple_gpu_matmul tests require Apple GPU")
        return
    var ctx = DeviceContext()
    if ctx.compute_capability() != 5:
        print(
            "SKIP: apple_gpu_matmul tests require Apple M5"
            " (compute_capability == 5)"
        )
        return
    test_kernel_single_tile_nn_fp16(ctx)
    test_kernel_single_tile_k128_nn_fp16(ctx)
    test_kernel_64x64x17_nn_fp16(ctx)
    test_kernel_256x256x16_nn_fp16(ctx)
    test_kernel_ragged_100x200x33_nn_fp16(ctx)
    test_kernel_ragged_100x200x32_nn_fp16(ctx)
    test_kernel_ragged_100x200x32_nt_fp16(ctx)
    test_kernel_M20_N80_K16_nn_fp16(ctx)
    test_kernel_128x128x32_nt_fp16(ctx)
    test_kernel_128x128x32_nn_bf16(ctx)
    test_kernel_128x128x32_nn_fp32(ctx)
    test_enqueue_helper_fp16(ctx)
