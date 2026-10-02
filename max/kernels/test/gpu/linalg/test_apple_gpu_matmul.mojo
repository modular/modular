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
"""Unit tests for the Apple GPU matmul kernels.

Covers three paths:

- The 8x8 `simdgroup_matrix` GEMM (`gemm_kernel_apple_8x8`), the M1-M4 dispatch
  path in `_matmul_gpu`. These tests run on any Apple GPU.
- The M5 hardware-MMA simdgroup-tiled kernel (`AppleM5MatMul.run` /
  `enqueue_apple_matmul`), which requires `compute_capability() == 5`. This
  includes the split-K path (`enqueue_apple_matmul_split_k` and the
  `force_split_k` flag), folded in here from the former `test_apple_split_k`.
- The M5 small-batch GEMV (`apple/gemv.mojo`), which `_matmul_gpu` routes
  `transpose_b` bf16/fp16 matmuls with 2 <= M <= 8 to. Its reference is an
  fp32 host sum over the same 16-bit inputs, so only summation order and the
  output rounding differ; the tolerance is the output type's half-ulp plus an
  order term proportional to `sum_k |x * w|`.
"""

from std.collections import Optional
from std.random import random_float64, random_si64, seed
from max.gpu import WARP_SIZE
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from std.sys.info import _accelerator_arch
from std.utils import IndexList

from layout import Coord, TileTensor, Idx
from layout.tile_layout import row_major

from linalg.matmul.gpu import _matmul_gpu
from linalg.matmul.gpu.apple import gemm_kernel_apple_8x8
from linalg.matmul.gpu.apple.gemv import enqueue_apple_gemv_config
from linalg.matmul.gpu.apple.matmul_kernel import (
    AppleM5MatMul,
    enqueue_apple_matmul,
    enqueue_apple_matmul_split_k,
)


# Morton decode is a static method of the struct; bind a canonical
# instantiation (the helpers are parameter-independent) for the unit tests.
comptime _MM = AppleM5MatMul[.float16]


def _launch[
    a_type: DType, transpose_b: Bool, c_type: DType = .float32
](
    ctx: DeviceContext,
    mut d_dev: DeviceBuffer[c_type],
    a_dev: DeviceBuffer[a_type],
    b_dev: DeviceBuffer[a_type],
    M: Int,
    N: Int,
    K: Int,
) raises:
    """Wrap device buffers as TileTensors and launch via the standalone
    `enqueue_apple_matmul` (the host entry to `AppleM5MatMul.run`)."""
    var b_rows = N if transpose_b else K
    var b_cols = K if transpose_b else N
    var c_tt = TileTensor(d_dev.unsafe_ptr(), row_major(M, N))
    var a_tt = TileTensor(a_dev.unsafe_ptr(), row_major(M, K)).as_imm()
    var b_tt = TileTensor(
        b_dev.unsafe_ptr(), row_major(b_rows, b_cols)
    ).as_imm()
    enqueue_apple_matmul[
        in_type=a_type, c_type=c_type, transpose_b=transpose_b
    ](c_tt, a_tt, b_tt, ctx)


from linalg.utils import elementwise_epilogue_type


def _host_matmul_nn[
    a_type: DType, b_type: DType
](
    a_ptr: ImmPointer[Scalar[a_type], ...],
    b_ptr: ImmPointer[Scalar[b_type], ...],
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


@inline(.always)
def _within_tol[c_type: DType](got: Float32, exp: Float32) -> Bool:
    """Standard mixed tolerance: ``|got - exp| <= atol + rtol * |exp|``.

    Per-c_type bounds:
        fp16:  rtol=1e-3,   atol=1e-5
        bf16:  rtol=1.6e-2, atol=1e-5
        fp32:  rtol=1e-4,   atol=1e-5
    """
    comptime if c_type == .float16:
        return abs(got - exp) <= Float32(1e-5) + Float32(1e-3) * abs(exp)
    elif c_type == .bfloat16:
        return abs(got - exp) <= Float32(1e-5) + Float32(1.6e-2) * abs(exp)
    else:
        return abs(got - exp) <= Float32(1e-5) + Float32(1e-4) * abs(exp)


# ===----------------------------------------------------------------------=== #
# 8x8 simdgroup-matrix GEMM (M1-M4 dispatch path; runs on any Apple GPU)
# ===----------------------------------------------------------------------=== #


def _run_8x8_case[
    a_type: DType, c_type: DType, transpose_b: Bool
](ctx: DeviceContext, M: Int, N: Int, K: Int, name: String) raises:
    """One launch + forced readback of `gemm_kernel_apple_8x8`, checked against
    an fp32 host reference.

    Uses the same 64x64 / 4-simdgroup tiling the dispatcher enqueues. Exercises
    clean and ragged M/N (the kernel bounds-checks edge subtiles); K is a
    multiple of 16, matching the dispatch gate.
    """
    print("==", name, M, "x", N, "x", K, "NT=" + String(transpose_b))
    comptime BM = 64
    comptime BN = 64
    comptime NSG = 4
    var b_rows = N if transpose_b else K
    var b_cols = K if transpose_b else N

    var a_host = ctx.enqueue_create_host_buffer[a_type](M * K)
    var b_host = ctx.enqueue_create_host_buffer[a_type](b_rows * b_cols)
    for i in range(M * K):
        a_host[i] = random_si64(Int64(-2), Int64(2)).cast[a_type]()
    for i in range(b_rows * b_cols):
        b_host[i] = random_si64(Int64(-2), Int64(2)).cast[a_type]()

    var a_dev = ctx.enqueue_create_buffer[a_type](M * K)
    var b_dev = ctx.enqueue_create_buffer[a_type](b_rows * b_cols)
    var d_dev = ctx.enqueue_create_buffer[c_type](M * N)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)

    var a_tt = TileTensor(a_dev.unsafe_ptr(), row_major(M, K)).as_imm()
    var b_tt = TileTensor(
        b_dev.unsafe_ptr(), row_major(b_rows, b_cols)
    ).as_imm()
    var d_tt = TileTensor(d_dev.unsafe_ptr(), row_major(M, N))

    comptime kernel = gemm_kernel_apple_8x8[
        c_type,
        a_type,
        a_type,
        type_of(d_tt).LayoutType,
        type_of(a_tt).LayoutType,
        type_of(b_tt).LayoutType,
        type_of(d_tt).Engine,
        type_of(a_tt).Engine,
        type_of(b_tt).Engine,
        transpose_b,
        BLOCK_M=BM,
        BLOCK_N=BN,
        NUM_SIMDGROUPS=NSG,
    ]
    ctx.enqueue_function[kernel](
        d_tt,
        a_tt,
        b_tt,
        Int32(M),
        Int32(N),
        Int32(K),
        grid_dim=((N + BN - 1) // BN, (M + BM - 1) // BM),
        block_dim=(NSG * WARP_SIZE,),
    )

    var d_host = ctx.enqueue_create_host_buffer[c_type](M * N)
    ctx.enqueue_copy(d_host, d_dev)
    ctx.synchronize()

    # DRIV-199 workaround: keep device buffers alive past `synchronize`, else
    # ASAP destruction frees them mid-kernel and the suite flakes.
    _ = a_dev^
    _ = b_dev^
    _ = d_dev^

    var pass_ = True
    for i in range(M):
        for j in range(N):
            var exp = Float32(0)
            for kk in range(K):
                var bv = Float32(
                    b_host[j * K + kk]
                ) if transpose_b else Float32(b_host[kk * N + j])
                exp += Float32(a_host[i * K + kk]) * bv
            var got = Float32(d_host[i * N + j])
            if not _within_tol[c_type](got, exp):
                if pass_:
                    print("FAIL:", i, j, "got", got, "expected", exp)
                pass_ = False
    if not pass_:
        raise Error("FAILED (see FAIL lines above)")
    print("PASS")


def _run_8x8_bias_case[
    a_type: DType, c_type: DType, transpose_b: Bool
](ctx: DeviceContext, M: Int, N: Int, K: Int, name: String) raises:
    """`gemm_kernel_apple_8x8` with a bias-add `elementwise_lambda_fn`, checked
    against an fp32 host reference.

    Exercises the epilogue store path (`elementwise_lambda_fn` branch), and
    uses odd/ragged shapes to cover the epilogue at edge subtiles as well as
    the interior.
    """
    print("==", name, M, "x", N, "x", K, "NT=" + String(transpose_b))
    comptime BM = 64
    comptime BN = 64
    comptime NSG = 4
    var b_rows = N if transpose_b else K
    var b_cols = K if transpose_b else N

    var a_host = ctx.enqueue_create_host_buffer[a_type](M * K)
    var b_host = ctx.enqueue_create_host_buffer[a_type](b_rows * b_cols)
    var bias_host = ctx.enqueue_create_host_buffer[c_type](N)
    for i in range(M * K):
        a_host[i] = random_si64(Int64(-2), Int64(2)).cast[a_type]()
    for i in range(b_rows * b_cols):
        b_host[i] = random_si64(Int64(-2), Int64(2)).cast[a_type]()
    for j in range(N):
        bias_host[j] = random_si64(Int64(-2), Int64(2)).cast[c_type]()

    var a_dev = ctx.enqueue_create_buffer[a_type](M * K)
    var b_dev = ctx.enqueue_create_buffer[a_type](b_rows * b_cols)
    var bias_dev = ctx.enqueue_create_buffer[c_type](N)
    var d_dev = ctx.enqueue_create_buffer[c_type](M * N)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)
    ctx.enqueue_copy(bias_dev, bias_host)

    var a_tt = TileTensor(a_dev.unsafe_ptr(), row_major(M, K)).as_imm()
    var b_tt = TileTensor(
        b_dev.unsafe_ptr(), row_major(b_rows, b_cols)
    ).as_imm()
    var d_tt = TileTensor(d_dev.unsafe_ptr(), row_major(M, N))

    var d_ptr = d_dev.unsafe_ptr()
    var bias_ptr = bias_dev.unsafe_ptr()
    var row_stride = N  # output is row_major(M, N)

    @__parameter
    @inline(.always)
    @__copy_capture(d_ptr, bias_ptr, row_stride)
    def bias_epilogue[
        dt: DType, w: SIMDLength, *, alignment: Int = 1
    ](coords: IndexList[2], val: SIMD[dt, w]) capturing -> None:
        # Kernel invokes with `dt == c_type`; rebind so the store matches d_ptr.
        var bias = (bias_ptr + coords[1]).load[width=w]()
        var v_c = rebind[SIMD[c_type, w]](val)
        (d_ptr + coords[0] * row_stride + coords[1]).store[alignment=alignment](
            v_c + bias
        )

    comptime kernel = gemm_kernel_apple_8x8[
        c_type,
        a_type,
        a_type,
        type_of(d_tt).LayoutType,
        type_of(a_tt).LayoutType,
        type_of(b_tt).LayoutType,
        type_of(d_tt).Engine,
        type_of(a_tt).Engine,
        type_of(b_tt).Engine,
        transpose_b,
        elementwise_lambda_fn=Optional[elementwise_epilogue_type](
            bias_epilogue
        ),
        BLOCK_M=BM,
        BLOCK_N=BN,
        NUM_SIMDGROUPS=NSG,
    ]
    ctx.enqueue_function[kernel](
        d_tt,
        a_tt,
        b_tt,
        Int32(M),
        Int32(N),
        Int32(K),
        grid_dim=((N + BN - 1) // BN, (M + BM - 1) // BM),
        block_dim=(NSG * WARP_SIZE,),
    )

    var d_host = ctx.enqueue_create_host_buffer[c_type](M * N)
    ctx.enqueue_copy(d_host, d_dev)
    ctx.synchronize()

    # DRIV-199 workaround: keep device buffers alive past `synchronize`, else
    # ASAP destruction frees them mid-kernel and the suite flakes.
    _ = a_dev^
    _ = b_dev^
    _ = bias_dev^
    _ = d_dev^

    var pass_ = True
    for i in range(M):
        for j in range(N):
            var acc = Float32(0)
            for kk in range(K):
                var bv = Float32(
                    b_host[j * K + kk]
                ) if transpose_b else Float32(b_host[kk * N + j])
                acc += Float32(a_host[i * K + kk]) * bv
            var exp = acc + Float32(bias_host[j])
            var got = Float32(d_host[i * N + j])
            if not _within_tol[c_type](got, exp):
                if pass_:
                    print("FAIL:", i, j, "got", got, "expected", exp)
                pass_ = False
    if not pass_:
        raise Error("FAILED (see FAIL lines above)")
    print("PASS")


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
    var exp_m: Array[UInt32, 16] = [
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
    var exp_n: Array[UInt32, 16] = [
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
        var got = _MM.morton_decode_2d(UInt32(i))
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
    if not pass_:
        raise Error("FAILED (see FAIL lines above)")
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
    var seen_2x16 = Array[Bool, 32](fill=False)
    for i in range(32):
        var got = _MM.morton_decode_2d_rect(UInt32(i), UInt32(1), UInt32(4))
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
    var seen_16x2 = Array[Bool, 32](fill=False)
    for i in range(32):
        var got = _MM.morton_decode_2d_rect(UInt32(i), UInt32(4), UInt32(1))
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
        var got_rect = _MM.morton_decode_2d_rect(
            UInt32(i), UInt32(2), UInt32(2)
        )
        var got_sq = _MM.morton_decode_2d(UInt32(i))
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
        var got = _MM.morton_decode_2d_rect(UInt32(i), UInt32(0), UInt32(4))
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
        var got = _MM.morton_decode_2d_rect(UInt32(i), UInt32(4), UInt32(0))
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

    if not pass_:
        raise Error("FAILED (see FAIL lines above)")
    print("PASS")


def test_kernel_single_tile_nn_fp16(ctx: DeviceContext) raises:
    """D[64,64] = A[64,16] @ B[16,64], NN, fp16->fp32, single threadgroup."""
    print("== test_kernel_single_tile_nn_fp16")
    comptime M = 64
    comptime N = 64
    comptime K = 16

    var a_host = ctx.enqueue_create_host_buffer[.float16](M * K)
    var b_host = ctx.enqueue_create_host_buffer[.float16](K * N)
    for i in range(M * K):
        a_host[i] = Float16(random_si64(Int64(-2), Int64(2)).cast[.float16]())
    for i in range(K * N):
        b_host[i] = Float16(random_si64(Int64(-2), Int64(2)).cast[.float16]())

    var a_dev = ctx.enqueue_create_buffer[.float16](M * K)
    var b_dev = ctx.enqueue_create_buffer[.float16](K * N)
    var d_dev = ctx.enqueue_create_buffer[.float32](M * N)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)

    _launch[.float16, False](
        ctx,
        d_dev,
        a_dev,
        b_dev,
        M,
        N,
        K,
    )

    var d_host = ctx.enqueue_create_host_buffer[.float32](M * N)
    ctx.enqueue_copy(d_host, d_dev)
    ctx.synchronize()

    # DRIV-199 workaround: keep device buffers alive past `synchronize`, else
    # ASAP destruction frees them mid-kernel and the suite flakes.
    _ = a_dev^
    _ = b_dev^
    _ = d_dev^

    var pass_ = True
    for i in range(M):
        for j in range(N):
            var exp = _host_matmul_nn[.float16, .float16](
                a_host.unsafe_ptr(), b_host.unsafe_ptr(), M, N, K, i, j
            )
            var got = d_host[i * N + j]
            if abs(got - exp) > Float32(0.5):
                print("FAIL:", i, j, "got", got, "expected", exp)
                pass_ = False
    if not pass_:
        raise Error("FAILED (see FAIL lines above)")
    print("PASS")


def test_kernel_single_tile_k128_nn_fp16(ctx: DeviceContext) raises:
    """D[64,64] = A[64,128] @ B[128,64], NN, fp16->fp32."""
    print("== test_kernel_single_tile_k128_nn_fp16")
    comptime M = 64
    comptime N = 64
    comptime K = 128

    var a_host = ctx.enqueue_create_host_buffer[.float16](M * K)
    var b_host = ctx.enqueue_create_host_buffer[.float16](K * N)
    for i in range(M * K):
        a_host[i] = Float16(random_si64(Int64(-2), Int64(2)).cast[.float16]())
    for i in range(K * N):
        b_host[i] = Float16(random_si64(Int64(-2), Int64(2)).cast[.float16]())

    var a_dev = ctx.enqueue_create_buffer[.float16](M * K)
    var b_dev = ctx.enqueue_create_buffer[.float16](K * N)
    var d_dev = ctx.enqueue_create_buffer[.float32](M * N)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)

    _launch[.float16, False](
        ctx,
        d_dev,
        a_dev,
        b_dev,
        M,
        N,
        K,
    )

    var d_host = ctx.enqueue_create_host_buffer[.float32](M * N)
    ctx.enqueue_copy(d_host, d_dev)
    ctx.synchronize()

    # DRIV-199 workaround: keep device buffers alive past `synchronize`, else
    # ASAP destruction frees them mid-kernel and the suite flakes.
    _ = a_dev^
    _ = b_dev^
    _ = d_dev^

    var pass_ = True
    for i in range(M):
        for j in range(N):
            var exp = _host_matmul_nn[.float16, .float16](
                a_host.unsafe_ptr(), b_host.unsafe_ptr(), M, N, K, i, j
            )
            var got = d_host[i * N + j]
            if abs(got - exp) > Float32(1.0):
                print("FAIL:", i, j, "got", got, "expected", exp)
                pass_ = False
    if not pass_:
        raise Error("FAILED (see FAIL lines above)")
    print("PASS")


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

    var a_host = ctx.enqueue_create_host_buffer[.float16](M * K)
    var b_host = ctx.enqueue_create_host_buffer[.float16](K * N)
    for i in range(M * K):
        a_host[i] = Float16(random_si64(Int64(-2), Int64(2)).cast[.float16]())
    for i in range(K * N):
        b_host[i] = Float16(random_si64(Int64(-2), Int64(2)).cast[.float16]())

    var a_dev = ctx.enqueue_create_buffer[.float16](M * K)
    var b_dev = ctx.enqueue_create_buffer[.float16](K * N)
    var d_dev = ctx.enqueue_create_buffer[.float32](M * N)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)

    _launch[.float16, False](
        ctx,
        d_dev,
        a_dev,
        b_dev,
        M,
        N,
        K,
    )

    var d_host = ctx.enqueue_create_host_buffer[.float32](M * N)
    ctx.enqueue_copy(d_host, d_dev)
    ctx.synchronize()

    # DRIV-199 workaround: keep device buffers alive past `synchronize`, else
    # ASAP destruction frees them mid-kernel and the suite flakes.
    _ = a_dev^
    _ = b_dev^
    _ = d_dev^

    var pass_ = True
    for i in range(M):
        for j in range(N):
            var exp = _host_matmul_nn[.float16, .float16](
                a_host.unsafe_ptr(), b_host.unsafe_ptr(), M, N, K, i, j
            )
            var got = d_host[i * N + j]
            if abs(got - exp) > Float32(0.5):
                print("FAIL:", i, j, "got", got, "expected", exp)
                pass_ = False
    if not pass_:
        raise Error("FAILED (see FAIL lines above)")
    print("PASS")


def test_kernel_256x256x16_nn_fp16(ctx: DeviceContext) raises:
    """D[256,256] = A[256,16] @ B[16,256] -- 16 threadgroups (4x4 tiles)."""
    print("== test_kernel_256x256x16_nn_fp16")
    comptime M = 256
    comptime N = 256
    comptime K = 16

    var a_host = ctx.enqueue_create_host_buffer[.float16](M * K)
    var b_host = ctx.enqueue_create_host_buffer[.float16](K * N)
    for i in range(M * K):
        a_host[i] = Float16(random_si64(Int64(-2), Int64(2)).cast[.float16]())
    for i in range(K * N):
        b_host[i] = Float16(random_si64(Int64(-2), Int64(2)).cast[.float16]())

    var a_dev = ctx.enqueue_create_buffer[.float16](M * K)
    var b_dev = ctx.enqueue_create_buffer[.float16](K * N)
    var d_dev = ctx.enqueue_create_buffer[.float32](M * N)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)

    _launch[.float16, False](
        ctx,
        d_dev,
        a_dev,
        b_dev,
        M,
        N,
        K,
    )

    var d_host = ctx.enqueue_create_host_buffer[.float32](M * N)
    ctx.enqueue_copy(d_host, d_dev)
    ctx.synchronize()

    # DRIV-199 workaround: keep device buffers alive past `synchronize`, else
    # ASAP destruction frees them mid-kernel and the suite flakes.
    _ = a_dev^
    _ = b_dev^
    _ = d_dev^

    var pass_ = True
    for i in range(M):
        for j in range(N):
            var exp = _host_matmul_nn[.float16, .float16](
                a_host.unsafe_ptr(), b_host.unsafe_ptr(), M, N, K, i, j
            )
            var got = d_host[i * N + j]
            if abs(got - exp) > Float32(0.5):
                print("FAIL:", i, j, "got", got, "expected", exp)
                pass_ = False
    if not pass_:
        raise Error("FAILED (see FAIL lines above)")
    print("PASS")


def _host_matmul_nt[
    a_type: DType, b_type: DType
](
    a_ptr: ImmPointer[Scalar[a_type], ...],
    b_ptr: ImmPointer[Scalar[b_type], ...],
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


def _run_split_k_case[
    in_type: DType, c_type: DType, transpose_b: Bool
](
    ctx: DeviceContext,
    M: Int,
    N: Int,
    K: Int,
    name: String,
    *,
    splits: Int = 0,
    force_split_k: Bool = False,
) raises:
    """One split-K matmul launch, checked against an fp32 host reference.

    Folds in the former standalone `test_apple_split_k.mojo`. `splits > 0`
    launches `enqueue_apple_matmul_split_k` with that explicit split count;
    otherwise launches `enqueue_apple_matmul` with `force_split_k` (the single
    unified entry forcing the split-K route, even on shapes that would not
    auto-route). Flat 1.0 abs tolerance (inputs in [-2, 2]).
    """
    if splits > 0:
        print("==", name, M, "x", N, "x", K, "split=", splits)
    else:
        print("==", name, M, "x", N, "x", K, "force_split_k")
    var b_rows = N if transpose_b else K
    var b_cols = K if transpose_b else N

    var a_host = ctx.enqueue_create_host_buffer[in_type](M * K)
    var b_host = ctx.enqueue_create_host_buffer[in_type](b_rows * b_cols)
    for i in range(M * K):
        a_host[i] = random_si64(Int64(-2), Int64(2)).cast[in_type]()
    for i in range(b_rows * b_cols):
        b_host[i] = random_si64(Int64(-2), Int64(2)).cast[in_type]()

    var a_dev = ctx.enqueue_create_buffer[in_type](M * K)
    var b_dev = ctx.enqueue_create_buffer[in_type](b_rows * b_cols)
    var d_dev = ctx.enqueue_create_buffer[c_type](M * N)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)

    var a_tt = TileTensor(a_dev.unsafe_ptr(), row_major(M, K)).as_imm()
    var b_tt = TileTensor(
        b_dev.unsafe_ptr(), row_major(b_rows, b_cols)
    ).as_imm()
    var d_tt = TileTensor(d_dev.unsafe_ptr(), row_major(M, N))

    if splits > 0:
        enqueue_apple_matmul_split_k[
            in_type=in_type, c_type=c_type, transpose_b=transpose_b
        ](d_tt, a_tt, b_tt, ctx, splits)
    else:
        enqueue_apple_matmul[
            in_type=in_type, c_type=c_type, transpose_b=transpose_b
        ](d_tt, a_tt, b_tt, ctx, force_split_k)

    var d_host = ctx.enqueue_create_host_buffer[c_type](M * N)
    ctx.enqueue_copy(d_host, d_dev)
    ctx.synchronize()

    # DRIV-199 workaround: keep device buffers alive past `synchronize`, else
    # ASAP destruction frees them mid-kernel and the suite flakes.
    _ = a_dev^
    _ = b_dev^
    _ = d_dev^

    var pass_ = True
    for i in range(M):
        for j in range(N):
            var exp = _host_matmul_nt[in_type, in_type](
                a_host.unsafe_ptr(), b_host.unsafe_ptr(), M, N, K, i, j
            ) if transpose_b else _host_matmul_nn[in_type, in_type](
                a_host.unsafe_ptr(), b_host.unsafe_ptr(), M, N, K, i, j
            )
            var got = Float32(d_host[i * N + j])
            if abs(got - exp) > Float32(1.0):
                if pass_:
                    print("FAIL:", i, j, "got", got, "exp", exp)
                pass_ = False
    if not pass_:
        raise Error("FAILED (see FAIL lines above)")
    print("PASS")


def test_kernel_128x128x32_nt_fp16(ctx: DeviceContext) raises:
    """D[128,128] = A[128,32] @ B[128,32]^T, NT, fp16->fp32."""
    print("== test_kernel_128x128x32_nt_fp16")
    comptime M = 128
    comptime N = 128
    comptime K = 32

    var a_host = ctx.enqueue_create_host_buffer[.float16](M * K)
    # B is stored as (N, K) for transpose_b=True.
    var b_host = ctx.enqueue_create_host_buffer[.float16](N * K)
    for i in range(M * K):
        a_host[i] = Float16(random_si64(Int64(-2), Int64(2)).cast[.float16]())
    for i in range(N * K):
        b_host[i] = Float16(random_si64(Int64(-2), Int64(2)).cast[.float16]())

    var a_dev = ctx.enqueue_create_buffer[.float16](M * K)
    var b_dev = ctx.enqueue_create_buffer[.float16](N * K)
    var d_dev = ctx.enqueue_create_buffer[.float32](M * N)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)

    _launch[.float16, True](
        ctx,
        d_dev,
        a_dev,
        b_dev,
        M,
        N,
        K,
    )

    var d_host = ctx.enqueue_create_host_buffer[.float32](M * N)
    ctx.enqueue_copy(d_host, d_dev)
    ctx.synchronize()

    # DRIV-199 workaround: keep device buffers alive past `synchronize`, else
    # ASAP destruction frees them mid-kernel and the suite flakes.
    _ = a_dev^
    _ = b_dev^
    _ = d_dev^

    var pass_ = True
    for i in range(M):
        for j in range(N):
            var exp = _host_matmul_nt[.float16, .float16](
                a_host.unsafe_ptr(), b_host.unsafe_ptr(), M, N, K, i, j
            )
            var got = d_host[i * N + j]
            if abs(got - exp) > Float32(0.5):
                print("FAIL:", i, j, "got", got, "expected", exp)
                pass_ = False
    if not pass_:
        raise Error("FAILED (see FAIL lines above)")
    print("PASS")


def test_kernel_ragged_100x200x33_nn_fp16(ctx: DeviceContext) raises:
    """Ragged shape: M=100 (< 2*64), N=200 (< 4*64), K=33 (not 16-aligned)."""
    print("== test_kernel_ragged_100x200x33_nn_fp16")
    comptime M = 100
    comptime N = 200
    comptime K = 33

    var a_host = ctx.enqueue_create_host_buffer[.float16](M * K)
    var b_host = ctx.enqueue_create_host_buffer[.float16](K * N)
    for i in range(M * K):
        a_host[i] = Float16(random_si64(Int64(-2), Int64(2)).cast[.float16]())
    for i in range(K * N):
        b_host[i] = Float16(random_si64(Int64(-2), Int64(2)).cast[.float16]())

    var a_dev = ctx.enqueue_create_buffer[.float16](M * K)
    var b_dev = ctx.enqueue_create_buffer[.float16](K * N)
    var d_dev = ctx.enqueue_create_buffer[.float32](M * N)
    # Sentinel: initialize output to -1e30 so untouched elements are visible.
    var d_init = ctx.enqueue_create_host_buffer[.float32](M * N)
    for i in range(M * N):
        d_init[i] = Float32(-1.0e30)
    ctx.enqueue_copy(d_dev, d_init)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)

    _launch[.float16, False](
        ctx,
        d_dev,
        a_dev,
        b_dev,
        M,
        N,
        K,
    )

    var d_host = ctx.enqueue_create_host_buffer[.float32](M * N)
    ctx.enqueue_copy(d_host, d_dev)
    ctx.synchronize()

    # DRIV-199 workaround: keep device buffers alive past `synchronize`, else
    # ASAP destruction frees them mid-kernel and the suite flakes.
    _ = a_dev^
    _ = b_dev^
    _ = d_dev^

    var pass_ = True
    for i in range(M):
        for j in range(N):
            var exp = _host_matmul_nn[.float16, .float16](
                a_host.unsafe_ptr(), b_host.unsafe_ptr(), M, N, K, i, j
            )
            var got = d_host[i * N + j]
            if abs(got - exp) > Float32(0.5):
                print("FAIL:", i, j, "got", got, "expected", exp)
                pass_ = False
    if not pass_:
        raise Error("FAILED (see FAIL lines above)")
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

    var a_host = ctx.enqueue_create_host_buffer[.float16](M * K)
    var b_host = ctx.enqueue_create_host_buffer[.float16](K * N)
    for i in range(M * K):
        a_host[i] = Float16(random_si64(Int64(-2), Int64(2)).cast[.float16]())
    for i in range(K * N):
        b_host[i] = Float16(random_si64(Int64(-2), Int64(2)).cast[.float16]())

    var a_dev = ctx.enqueue_create_buffer[.float16](M * K)
    var b_dev = ctx.enqueue_create_buffer[.float16](K * N)
    var d_dev = ctx.enqueue_create_buffer[.float32](M * N)
    # Sentinel: detect spurious stores past M/N edge in the bounded path.
    var d_init = ctx.enqueue_create_host_buffer[.float32](M * N)
    for i in range(M * N):
        d_init[i] = Float32(-1.0e30)
    ctx.enqueue_copy(d_dev, d_init)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)

    _launch[.float16, False](
        ctx,
        d_dev,
        a_dev,
        b_dev,
        M,
        N,
        K,
    )

    var d_host = ctx.enqueue_create_host_buffer[.float32](M * N)
    ctx.enqueue_copy(d_host, d_dev)
    ctx.synchronize()

    # DRIV-199 workaround: keep device buffers alive past `synchronize`, else
    # ASAP destruction frees them mid-kernel and the suite flakes.
    _ = a_dev^
    _ = b_dev^
    _ = d_dev^

    var pass_ = True
    for i in range(M):
        for j in range(N):
            var exp = _host_matmul_nn[.float16, .float16](
                a_host.unsafe_ptr(), b_host.unsafe_ptr(), M, N, K, i, j
            )
            var got = d_host[i * N + j]
            if abs(got - exp) > Float32(0.5):
                print("FAIL:", i, j, "got", got, "expected", exp)
                pass_ = False
    if not pass_:
        raise Error("FAILED (see FAIL lines above)")
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

    var a_host = ctx.enqueue_create_host_buffer[.float16](M * K)
    # B stored as (N, K) for transpose_b=True.
    var b_host = ctx.enqueue_create_host_buffer[.float16](N * K)
    for i in range(M * K):
        a_host[i] = Float16(random_si64(Int64(-2), Int64(2)).cast[.float16]())
    for i in range(N * K):
        b_host[i] = Float16(random_si64(Int64(-2), Int64(2)).cast[.float16]())

    var a_dev = ctx.enqueue_create_buffer[.float16](M * K)
    var b_dev = ctx.enqueue_create_buffer[.float16](N * K)
    var d_dev = ctx.enqueue_create_buffer[.float32](M * N)
    # Sentinel: detect spurious stores past M/N edge in the bounded path.
    var d_init = ctx.enqueue_create_host_buffer[.float32](M * N)
    for i in range(M * N):
        d_init[i] = Float32(-1.0e30)
    ctx.enqueue_copy(d_dev, d_init)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)

    _launch[.float16, True](
        ctx,
        d_dev,
        a_dev,
        b_dev,
        M,
        N,
        K,
    )

    var d_host = ctx.enqueue_create_host_buffer[.float32](M * N)
    ctx.enqueue_copy(d_host, d_dev)
    ctx.synchronize()

    # DRIV-199 workaround: keep device buffers alive past `synchronize`, else
    # ASAP destruction frees them mid-kernel and the suite flakes.
    _ = a_dev^
    _ = b_dev^
    _ = d_dev^

    var pass_ = True
    for i in range(M):
        for j in range(N):
            var exp = _host_matmul_nt[.float16, .float16](
                a_host.unsafe_ptr(), b_host.unsafe_ptr(), M, N, K, i, j
            )
            var got = d_host[i * N + j]
            if abs(got - exp) > Float32(0.5):
                print("FAIL:", i, j, "got", got, "expected", exp)
                pass_ = False
    if not pass_:
        raise Error("FAILED (see FAIL lines above)")
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

    var a_host = ctx.enqueue_create_host_buffer[.float16](M * K)
    var b_host = ctx.enqueue_create_host_buffer[.float16](K * N)
    for i in range(M * K):
        a_host[i] = Float16(random_si64(Int64(-2), Int64(2)).cast[.float16]())
    for i in range(K * N):
        b_host[i] = Float16(random_si64(Int64(-2), Int64(2)).cast[.float16]())

    var a_dev = ctx.enqueue_create_buffer[.float16](M * K)
    var b_dev = ctx.enqueue_create_buffer[.float16](K * N)
    var d_dev = ctx.enqueue_create_buffer[.float32](M * N)
    var d_init = ctx.enqueue_create_host_buffer[.float32](M * N)
    for i in range(M * N):
        d_init[i] = Float32(-1.0e30)
    ctx.enqueue_copy(d_dev, d_init)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)

    _launch[.float16, False](
        ctx,
        d_dev,
        a_dev,
        b_dev,
        M,
        N,
        K,
    )

    var d_host = ctx.enqueue_create_host_buffer[.float32](M * N)
    ctx.enqueue_copy(d_host, d_dev)
    ctx.synchronize()

    # DRIV-199 workaround: keep device buffers alive past `synchronize`, else
    # ASAP destruction frees them mid-kernel and the suite flakes.
    _ = a_dev^
    _ = b_dev^
    _ = d_dev^

    var pass_ = True
    for i in range(M):
        for j in range(N):
            var exp = _host_matmul_nn[.float16, .float16](
                a_host.unsafe_ptr(), b_host.unsafe_ptr(), M, N, K, i, j
            )
            var got = d_host[i * N + j]
            if abs(got - exp) > Float32(0.5):
                print("FAIL:", i, j, "got", got, "expected", exp)
                pass_ = False
    if not pass_:
        raise Error("FAILED (see FAIL lines above)")
    print("PASS")


def _run_partial_m_nt_case[
    in_type: DType
](ctx: DeviceContext, M: Int, N: Int, K: Int, name: String) raises:
    """Concurrent-decode partial-M co-batched GEMM (NT) vs an fp32 CPU reference.

    For `1 < M < 64` (the batch widths the `m > 1` dispatch guard now routes to
    `enqueue_apple_matmul`), `grid_m = ceildiv(M, 64) = 1`: one 64-row M tile of
    which only `M` rows are valid. Verifies both halves of the partial-M
    contract:

    - (a) valid rows `[0, M)` match `A @ B.T` (fp32 accum). Inputs are small
      ints `{-2..2}`, so the bf16->fp32 result is exact -- the `> 0.5` gap check
      catches any deviation.
    - (b) rows `[M, 64)` are NEVER written. C is backed by a GUARD-row-padded
      device buffer sentinel-filled with `-1e30`; the kernel is handed only an
      `(M, N)` view, so any store to a row `>= M` corrupts the sentinel guard
      region and is caught. This is the OOB-write risk the guard relaxation
      hinges on (`_bounded_store` row-gating + the `row_base >= M` early return).
    """
    print("==", name, M, "x", N, "x", K, "NT")
    comptime GUARD = 64  # >= the full [M, 64) partial-tile extent for any M<64.
    var rows = M + GUARD

    var a_host = ctx.enqueue_create_host_buffer[in_type](M * K)
    var b_host = ctx.enqueue_create_host_buffer[in_type](N * K)  # NT: (N, K)
    for i in range(M * K):
        a_host[i] = Scalar[in_type](
            random_si64(Int64(-2), Int64(2)).cast[in_type]()
        )
    for i in range(N * K):
        b_host[i] = Scalar[in_type](
            random_si64(Int64(-2), Int64(2)).cast[in_type]()
        )

    var a_dev = ctx.enqueue_create_buffer[in_type](M * K)
    var b_dev = ctx.enqueue_create_buffer[in_type](N * K)
    var d_dev = ctx.enqueue_create_buffer[.float32](rows * N)
    # Sentinel-fill the WHOLE (valid + guard) buffer so a spurious write into
    # rows [M, rows) is visible on readback.
    var d_init = ctx.enqueue_create_host_buffer[.float32](rows * N)
    for i in range(rows * N):
        d_init[i] = Float32(-1.0e30)
    ctx.enqueue_copy(d_dev, d_init)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)

    # C view is (M, N): the kernel only knows M rows, so a write to row >= M
    # lands in the sentinel guard region rather than a legal C slot.
    _launch[in_type, True](ctx, d_dev, a_dev, b_dev, M, N, K)

    var d_host = ctx.enqueue_create_host_buffer[.float32](rows * N)
    ctx.enqueue_copy(d_host, d_dev)
    ctx.synchronize()

    # DRIV-199 workaround: keep device buffers alive past `synchronize`.
    _ = a_dev^
    _ = b_dev^
    _ = d_dev^

    var pass_ = True
    # (a) valid rows [0, M) match the reference (exact for small-int inputs).
    for i in range(M):
        for j in range(N):
            var exp = _host_matmul_nt[in_type, in_type](
                a_host.unsafe_ptr(), b_host.unsafe_ptr(), M, N, K, i, j
            )
            var got = d_host[i * N + j]
            if abs(got - exp) > Float32(0.5):
                print("FAIL valid:", i, j, "got", got, "expected", exp)
                pass_ = False
    # (b) guard rows [M, rows) untouched (still the -1e30 sentinel).
    for i in range(M, rows):
        for j in range(N):
            var got = d_host[i * N + j]
            if got != Float32(-1.0e30):
                print("FAIL guard-write: row", i, "col", j, "got", got)
                pass_ = False
    if not pass_:
        raise Error("FAILED (see FAIL lines above)")
    print("PASS")


def test_partial_m_decode_nt_bf16(ctx: DeviceContext) raises:
    """Partial-M sweep at real Llama-3.1-8B decode weight shapes (NT bf16).

    Covers the concurrent-decode batch widths that the relaxed `m > 1` guard now
    routes to the co-batched GEMM instead of the naive per-row fallback. Each
    case asserts the valid rows are numerically correct AND that no row in
    [M, 64) is written (see `_run_partial_m_nt_case`).
    """
    print("== test_partial_m_decode_nt_bf16")
    # k/v_proj (GQA KV projection): the smallest real decode weight; all widths.
    _run_partial_m_nt_case[.bfloat16](ctx, 2, 1024, 4096, "kproj m2")
    _run_partial_m_nt_case[.bfloat16](ctx, 4, 1024, 4096, "kproj m4")
    _run_partial_m_nt_case[.bfloat16](ctx, 16, 1024, 4096, "kproj m16")
    _run_partial_m_nt_case[.bfloat16](ctx, 31, 1024, 4096, "kproj m31")
    _run_partial_m_nt_case[.bfloat16](ctx, 32, 1024, 4096, "kproj m32")
    _run_partial_m_nt_case[.bfloat16](ctx, 63, 1024, 4096, "kproj m63")
    # o_proj / q_proj (larger real weight): boundary widths, second shape.
    _run_partial_m_nt_case[.bfloat16](ctx, 16, 4096, 4096, "oproj m16")
    _run_partial_m_nt_case[.bfloat16](ctx, 63, 4096, 4096, "oproj m63")


def test_kernel_128x128x32_nn_bf16(ctx: DeviceContext) raises:
    """D[128,128] = A[128,32] @ B[32,128], NN, bf16->fp32."""
    print("== test_kernel_128x128x32_nn_bf16")
    comptime M = 128
    comptime N = 128
    comptime K = 32

    var a_host = ctx.enqueue_create_host_buffer[.bfloat16](M * K)
    var b_host = ctx.enqueue_create_host_buffer[.bfloat16](K * N)
    for i in range(M * K):
        a_host[i] = BFloat16(random_si64(Int64(-2), Int64(2)).cast[.bfloat16]())
    for i in range(K * N):
        b_host[i] = BFloat16(random_si64(Int64(-2), Int64(2)).cast[.bfloat16]())

    var a_dev = ctx.enqueue_create_buffer[.bfloat16](M * K)
    var b_dev = ctx.enqueue_create_buffer[.bfloat16](K * N)
    var d_dev = ctx.enqueue_create_buffer[.float32](M * N)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)

    _launch[.bfloat16, False](
        ctx,
        d_dev,
        a_dev,
        b_dev,
        M,
        N,
        K,
    )

    var d_host = ctx.enqueue_create_host_buffer[.float32](M * N)
    ctx.enqueue_copy(d_host, d_dev)
    ctx.synchronize()

    # DRIV-199 workaround: keep device buffers alive past `synchronize`, else
    # ASAP destruction frees them mid-kernel and the suite flakes.
    _ = a_dev^
    _ = b_dev^
    _ = d_dev^

    var pass_ = True
    for i in range(M):
        for j in range(N):
            var exp = _host_matmul_nn[.bfloat16, .bfloat16](
                a_host.unsafe_ptr(), b_host.unsafe_ptr(), M, N, K, i, j
            )
            var got = d_host[i * N + j]
            if abs(got - exp) > Float32(0.5):
                print("FAIL:", i, j, "got", got, "expected", exp)
                pass_ = False
    if not pass_:
        raise Error("FAILED (see FAIL lines above)")
    print("PASS")


def test_kernel_ragged_100x200x64_nn_bf16_clamp_chain(
    ctx: DeviceContext,
) raises:
    """D[100,200] = A[100,64] @ B[64,200], NN, bf16->fp32, via `enqueue_apple_matmul`.

    Shape hits the `clamp_v2` + chained-K route: M and N are both ragged, so
    both axes clamp, and grid_m=2/grid_n=4 cover every clamp/neighbor/interior
    tile combination in one shape. K=64 gives a real 2-strip-per-pass split
    without tripping the split-K heuristic.
    """
    print("== test_kernel_ragged_100x200x64_nn_bf16_clamp_chain")
    comptime M = 100
    comptime N = 200
    comptime K = 64

    var a_host = ctx.enqueue_create_host_buffer[.bfloat16](M * K)
    var b_host = ctx.enqueue_create_host_buffer[.bfloat16](K * N)
    for i in range(M * K):
        a_host[i] = BFloat16(random_si64(Int64(-2), Int64(2)).cast[.bfloat16]())
    for i in range(K * N):
        b_host[i] = BFloat16(random_si64(Int64(-2), Int64(2)).cast[.bfloat16]())

    var a_dev = ctx.enqueue_create_buffer[.bfloat16](M * K)
    var b_dev = ctx.enqueue_create_buffer[.bfloat16](K * N)
    var d_dev = ctx.enqueue_create_buffer[.float32](M * N)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)

    _launch[.bfloat16, False](
        ctx,
        d_dev,
        a_dev,
        b_dev,
        M,
        N,
        K,
    )

    var d_host = ctx.enqueue_create_host_buffer[.float32](M * N)
    ctx.enqueue_copy(d_host, d_dev)
    ctx.synchronize()

    # DRIV-199 workaround: keep device buffers alive past `synchronize`, else
    # ASAP destruction frees them mid-kernel and the suite flakes.
    _ = a_dev^
    _ = b_dev^
    _ = d_dev^

    var pass_ = True
    for i in range(M):
        for j in range(N):
            var exp = _host_matmul_nn[.bfloat16, .bfloat16](
                a_host.unsafe_ptr(), b_host.unsafe_ptr(), M, N, K, i, j
            )
            var got = d_host[i * N + j]
            if abs(got - exp) > Float32(0.5):
                print("FAIL:", i, j, "got", got, "expected", exp)
                pass_ = False
    if not pass_:
        raise Error("FAILED (see FAIL lines above)")
    print("PASS")


def test_kernel_128x128x32_nn_fp32(ctx: DeviceContext) raises:
    """D[128,128] = A[128,32] @ B[32,128], NN, fp32 input + accum."""
    print("== test_kernel_128x128x32_nn_fp32")
    comptime M = 128
    comptime N = 128
    comptime K = 32

    var a_host = ctx.enqueue_create_host_buffer[.float32](M * K)
    var b_host = ctx.enqueue_create_host_buffer[.float32](K * N)
    for i in range(M * K):
        a_host[i] = Float32(random_si64(Int64(-2), Int64(2)).cast[.float32]())
    for i in range(K * N):
        b_host[i] = Float32(random_si64(Int64(-2), Int64(2)).cast[.float32]())

    var a_dev = ctx.enqueue_create_buffer[.float32](M * K)
    var b_dev = ctx.enqueue_create_buffer[.float32](K * N)
    var d_dev = ctx.enqueue_create_buffer[.float32](M * N)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)

    _launch[.float32, False](
        ctx,
        d_dev,
        a_dev,
        b_dev,
        M,
        N,
        K,
    )

    var d_host = ctx.enqueue_create_host_buffer[.float32](M * N)
    ctx.enqueue_copy(d_host, d_dev)
    ctx.synchronize()

    # DRIV-199 workaround: keep device buffers alive past `synchronize`, else
    # ASAP destruction frees them mid-kernel and the suite flakes.
    _ = a_dev^
    _ = b_dev^
    _ = d_dev^

    var pass_ = True
    for i in range(M):
        for j in range(N):
            var exp = _host_matmul_nn[.float32, .float32](
                a_host.unsafe_ptr(), b_host.unsafe_ptr(), M, N, K, i, j
            )
            var got = d_host[i * N + j]
            if abs(got - exp) > Float32(0.01):
                print("FAIL:", i, j, "got", got, "expected", exp)
                pass_ = False
    if not pass_:
        raise Error("FAILED (see FAIL lines above)")
    print("PASS")


def test_enqueue_helper_fp16(ctx: DeviceContext) raises:
    """Smoke-test the host-side helper: 128x128x16 fp16 NN."""
    print("== test_enqueue_helper_fp16")
    comptime M = 128
    comptime N = 128
    comptime K = 16
    var a_host = ctx.enqueue_create_host_buffer[.float16](M * K)
    var b_host = ctx.enqueue_create_host_buffer[.float16](K * N)
    for i in range(M * K):
        a_host[i] = Float16(random_si64(Int64(-2), Int64(2)).cast[.float16]())
    for i in range(K * N):
        b_host[i] = Float16(random_si64(Int64(-2), Int64(2)).cast[.float16]())
    var a_dev = ctx.enqueue_create_buffer[.float16](M * K)
    var b_dev = ctx.enqueue_create_buffer[.float16](K * N)
    var d_dev = ctx.enqueue_create_buffer[.float32](M * N)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)

    var a_tt = TileTensor(a_dev.unsafe_ptr(), row_major(M, K))
    var b_tt = TileTensor(b_dev.unsafe_ptr(), row_major(K, N))
    var d_tt = TileTensor(d_dev.unsafe_ptr(), row_major(M, N))

    enqueue_apple_matmul[in_type=.float16, transpose_b=False](
        d_tt, a_tt, b_tt, ctx
    )

    var d_host = ctx.enqueue_create_host_buffer[.float32](M * N)
    ctx.enqueue_copy(d_host, d_dev)
    ctx.synchronize()

    # DRIV-199 workaround: keep device buffers alive past `synchronize`, else
    # ASAP destruction frees them mid-kernel and the suite flakes.
    _ = a_dev^
    _ = b_dev^
    _ = d_dev^

    var pass_ = True
    for i in range(M):
        for j in range(N):
            var exp = _host_matmul_nn[.float16, .float16](
                a_host.unsafe_ptr(), b_host.unsafe_ptr(), M, N, K, i, j
            )
            var got = d_host[i * N + j]
            if abs(got - exp) > Float32(0.5):
                print("FAIL:", i, j, "got", got, "expected", exp)
                pass_ = False
    if not pass_:
        raise Error("FAILED (see FAIL lines above)")
    print("PASS")


def test_kernel_128_nn_fp16_fp16_no_lambda(ctx: DeviceContext) raises:
    """Cast-only epilogue: c_type triggers `use_epilogue_path`, no lambda."""
    print("== test_kernel_128_nn_fp16_fp16_no_lambda")
    comptime M = 128
    comptime N = 128
    comptime K = 128

    var a_host = ctx.enqueue_create_host_buffer[.float16](M * K)
    var b_host = ctx.enqueue_create_host_buffer[.float16](K * N)
    for i in range(M * K):
        a_host[i] = Float16(random_si64(Int64(-2), Int64(2)).cast[.float16]())
    for i in range(K * N):
        b_host[i] = Float16(random_si64(Int64(-2), Int64(2)).cast[.float16]())

    var a_dev = ctx.enqueue_create_buffer[.float16](M * K)
    var b_dev = ctx.enqueue_create_buffer[.float16](K * N)
    var d_dev = ctx.enqueue_create_buffer[.float16](M * N)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)

    var a_tt = TileTensor(a_dev.unsafe_ptr(), row_major(M, K))
    var b_tt = TileTensor(b_dev.unsafe_ptr(), row_major(K, N))
    var d_tt = TileTensor(d_dev.unsafe_ptr(), row_major(M, N))

    enqueue_apple_matmul[
        in_type=.float16,
        c_type=.float16,
        transpose_b=False,
    ](d_tt, a_tt, b_tt, ctx)

    var d_host = ctx.enqueue_create_host_buffer[.float16](M * N)
    ctx.enqueue_copy(d_host, d_dev)
    ctx.synchronize()

    # DRIV-199 workaround: keep device buffers alive past `synchronize`, else
    # ASAP destruction frees them mid-kernel and the suite flakes.
    _ = a_dev^
    _ = b_dev^
    _ = d_dev^

    var pass_ = True
    for i in range(M):
        for j in range(N):
            var exp = _host_matmul_nn[.float16, .float16](
                a_host.unsafe_ptr(), b_host.unsafe_ptr(), M, N, K, i, j
            )
            # Compare in fp32 space; tolerance allows for fp16 downcast.
            var got = Float32(d_host[i * N + j])
            if not _within_tol[.float16](got, exp):
                print("FAIL:", i, j, "got", got, "expected", exp)
                pass_ = False
    if not pass_:
        raise Error("FAILED (see FAIL lines above)")
    print("PASS")


def test_kernel_128_nn_fp16_bf16_no_lambda(ctx: DeviceContext) raises:
    """D[128,128,128] = A @ B, NN, fp16 in / bf16 out, no lambda."""
    print("== test_kernel_128_nn_fp16_bf16_no_lambda")
    comptime M = 128
    comptime N = 128
    comptime K = 128

    var a_host = ctx.enqueue_create_host_buffer[.float16](M * K)
    var b_host = ctx.enqueue_create_host_buffer[.float16](K * N)
    for i in range(M * K):
        a_host[i] = Float16(random_si64(Int64(-2), Int64(2)).cast[.float16]())
    for i in range(K * N):
        b_host[i] = Float16(random_si64(Int64(-2), Int64(2)).cast[.float16]())

    var a_dev = ctx.enqueue_create_buffer[.float16](M * K)
    var b_dev = ctx.enqueue_create_buffer[.float16](K * N)
    var d_dev = ctx.enqueue_create_buffer[.bfloat16](M * N)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)

    var a_tt = TileTensor(a_dev.unsafe_ptr(), row_major(M, K))
    var b_tt = TileTensor(b_dev.unsafe_ptr(), row_major(K, N))
    var d_tt = TileTensor(d_dev.unsafe_ptr(), row_major(M, N))

    enqueue_apple_matmul[
        in_type=.float16,
        c_type=.bfloat16,
        transpose_b=False,
    ](d_tt, a_tt, b_tt, ctx)

    var d_host = ctx.enqueue_create_host_buffer[.bfloat16](M * N)
    ctx.enqueue_copy(d_host, d_dev)
    ctx.synchronize()

    # DRIV-199 workaround: keep device buffers alive past `synchronize`, else
    # ASAP destruction frees them mid-kernel and the suite flakes.
    _ = a_dev^
    _ = b_dev^
    _ = d_dev^

    var pass_ = True
    for i in range(M):
        for j in range(N):
            var exp = _host_matmul_nn[.float16, .float16](
                a_host.unsafe_ptr(), b_host.unsafe_ptr(), M, N, K, i, j
            )
            var got = Float32(d_host[i * N + j])
            if not _within_tol[.bfloat16](got, exp):
                print("FAIL:", i, j, "got", got, "expected", exp)
                pass_ = False
    if not pass_:
        raise Error("FAILED (see FAIL lines above)")
    print("PASS")


@inline(.always)
def _run_bias_epilogue_test[
    in_type: DType,
    c_type: DType,
    transpose_b: Bool,
](ctx: DeviceContext, test_name: String) raises:
    """128x128x128 bias-epilogue matmul, verified via `_within_tol[c_type]`."""
    print("== ", test_name)
    comptime M = 128
    comptime N = 128
    comptime K = 128

    var a_host = ctx.enqueue_create_host_buffer[in_type](M * K)
    comptime b_count = (N * K) if transpose_b else (K * N)
    var b_host = ctx.enqueue_create_host_buffer[in_type](b_count)
    var bias_host = ctx.enqueue_create_host_buffer[c_type](N)
    for i in range(M * K):
        a_host[i] = Scalar[in_type](
            random_si64(Int64(-2), Int64(2)).cast[in_type]()
        )
    for i in range(b_count):
        b_host[i] = Scalar[in_type](
            random_si64(Int64(-2), Int64(2)).cast[in_type]()
        )
    for j in range(N):
        bias_host[j] = Scalar[c_type](
            random_si64(Int64(-2), Int64(2)).cast[c_type]()
        )

    var a_dev = ctx.enqueue_create_buffer[in_type](M * K)
    var b_dev = ctx.enqueue_create_buffer[in_type](b_count)
    var bias_dev = ctx.enqueue_create_buffer[c_type](N)
    var d_dev = ctx.enqueue_create_buffer[c_type](M * N)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)
    ctx.enqueue_copy(bias_dev, bias_host)

    var a_tt = TileTensor(a_dev.unsafe_ptr(), row_major(M, K))
    var d_tt = TileTensor(d_dev.unsafe_ptr(), row_major(M, N))

    var d_ptr = d_dev.unsafe_ptr()
    var bias_ptr = bias_dev.unsafe_ptr()

    @__parameter
    @inline(.always)
    @__copy_capture(d_ptr, bias_ptr)
    def bias_epilogue[
        dt: DType, w: SIMDLength, *, alignment: Int = 1
    ](coords: IndexList[2], val: SIMD[dt, w]) capturing -> None:
        # Kernel invokes with `dt == c_type`; rebind so the store matches d_ptr.
        var b = (bias_ptr + coords[1]).load[width=w]()
        var v_c = rebind[SIMD[c_type, w]](val)
        (d_ptr + coords[0] * N + coords[1]).store[alignment=alignment](v_c + b)

    # Mojo `comptime if` does not lift `var` bindings out of branches, so
    # inline both enqueue calls.
    comptime if transpose_b:
        var b_tt = TileTensor(b_dev.unsafe_ptr(), row_major(N, K))
        enqueue_apple_matmul[
            in_type=in_type,
            c_type=c_type,
            transpose_b=True,
            elementwise_lambda_fn=Optional[elementwise_epilogue_type](
                bias_epilogue
            ),
        ](d_tt, a_tt, b_tt, ctx)
    else:
        var b_tt = TileTensor(b_dev.unsafe_ptr(), row_major(K, N))
        enqueue_apple_matmul[
            in_type=in_type,
            c_type=c_type,
            transpose_b=False,
            elementwise_lambda_fn=Optional[elementwise_epilogue_type](
                bias_epilogue
            ),
        ](d_tt, a_tt, b_tt, ctx)

    var d_host = ctx.enqueue_create_host_buffer[c_type](M * N)
    ctx.enqueue_copy(d_host, d_dev)
    ctx.synchronize()

    # DRIV-199 workaround: keep device buffers alive past `synchronize`, else
    # ASAP destruction frees them mid-kernel and the suite flakes.
    _ = a_dev^
    _ = b_dev^
    _ = bias_dev^
    _ = d_dev^

    var pass_ = True
    for i in range(M):
        for j in range(N):
            comptime if transpose_b:
                var acc = _host_matmul_nt[in_type, in_type](
                    a_host.unsafe_ptr(),
                    b_host.unsafe_ptr(),
                    M,
                    N,
                    K,
                    i,
                    j,
                )
                var exp = acc + Float32(bias_host[j])
                var got = Float32(d_host[i * N + j])
                if not _within_tol[c_type](got, exp):
                    print("FAIL:", i, j, "got", got, "expected", exp)
                    pass_ = False
            else:
                var acc = _host_matmul_nn[in_type, in_type](
                    a_host.unsafe_ptr(),
                    b_host.unsafe_ptr(),
                    M,
                    N,
                    K,
                    i,
                    j,
                )
                var exp = acc + Float32(bias_host[j])
                var got = Float32(d_host[i * N + j])
                if not _within_tol[c_type](got, exp):
                    print("FAIL:", i, j, "got", got, "expected", exp)
                    pass_ = False
    if not pass_:
        raise Error("FAILED (see FAIL lines above)")
    print("PASS")


def test_kernel_128_nt_fp16_fp16_bias_epilogue(ctx: DeviceContext) raises:
    """Bias-add via `elementwise_lambda_fn` — exercises column-coord propagation.
    """
    _run_bias_epilogue_test[.float16, .float16, True](
        ctx, "test_kernel_128_nt_fp16_fp16_bias_epilogue"
    )


def test_kernel_128_nn_fp16_fp16_bias_epilogue(ctx: DeviceContext) raises:
    _run_bias_epilogue_test[.float16, .float16, False](
        ctx, "test_kernel_128_nn_fp16_fp16_bias_epilogue"
    )


def test_kernel_128_nn_fp16_bf16_bias_epilogue(ctx: DeviceContext) raises:
    _run_bias_epilogue_test[.float16, .bfloat16, False](
        ctx, "test_kernel_128_nn_fp16_bf16_bias_epilogue"
    )


def test_kernel_128_nn_fp16_fp32_bias_epilogue(ctx: DeviceContext) raises:
    _run_bias_epilogue_test[.float16, .float32, False](
        ctx, "test_kernel_128_nn_fp16_fp32_bias_epilogue"
    )


def test_kernel_128_nt_fp16_bf16_bias_epilogue(ctx: DeviceContext) raises:
    _run_bias_epilogue_test[.float16, .bfloat16, True](
        ctx, "test_kernel_128_nt_fp16_bf16_bias_epilogue"
    )


def test_kernel_128_nt_fp16_fp32_bias_epilogue(ctx: DeviceContext) raises:
    _run_bias_epilogue_test[.float16, .float32, True](
        ctx, "test_kernel_128_nt_fp16_fp32_bias_epilogue"
    )


def test_kernel_128_nn_bf16_fp16_bias_epilogue(ctx: DeviceContext) raises:
    _run_bias_epilogue_test[.bfloat16, .float16, False](
        ctx, "test_kernel_128_nn_bf16_fp16_bias_epilogue"
    )


def test_kernel_128_nn_bf16_bf16_bias_epilogue(ctx: DeviceContext) raises:
    _run_bias_epilogue_test[.bfloat16, .bfloat16, False](
        ctx, "test_kernel_128_nn_bf16_bf16_bias_epilogue"
    )


def test_kernel_128_nn_bf16_fp32_bias_epilogue(ctx: DeviceContext) raises:
    _run_bias_epilogue_test[.bfloat16, .float32, False](
        ctx, "test_kernel_128_nn_bf16_fp32_bias_epilogue"
    )


def test_kernel_128_nt_bf16_fp16_bias_epilogue(ctx: DeviceContext) raises:
    _run_bias_epilogue_test[.bfloat16, .float16, True](
        ctx, "test_kernel_128_nt_bf16_fp16_bias_epilogue"
    )


def test_kernel_128_nt_bf16_bf16_bias_epilogue(ctx: DeviceContext) raises:
    _run_bias_epilogue_test[.bfloat16, .bfloat16, True](
        ctx, "test_kernel_128_nt_bf16_bf16_bias_epilogue"
    )


def test_kernel_128_nt_bf16_fp32_bias_epilogue(ctx: DeviceContext) raises:
    _run_bias_epilogue_test[.bfloat16, .float32, True](
        ctx, "test_kernel_128_nt_bf16_fp32_bias_epilogue"
    )


def test_kernel_128_nn_fp32_fp16_bias_epilogue(ctx: DeviceContext) raises:
    _run_bias_epilogue_test[.float32, .float16, False](
        ctx, "test_kernel_128_nn_fp32_fp16_bias_epilogue"
    )


def test_kernel_128_nn_fp32_bf16_bias_epilogue(ctx: DeviceContext) raises:
    _run_bias_epilogue_test[.float32, .bfloat16, False](
        ctx, "test_kernel_128_nn_fp32_bf16_bias_epilogue"
    )


def test_kernel_128_nn_fp32_fp32_bias_epilogue(ctx: DeviceContext) raises:
    _run_bias_epilogue_test[.float32, .float32, False](
        ctx, "test_kernel_128_nn_fp32_fp32_bias_epilogue"
    )


def test_kernel_128_nt_fp32_fp16_bias_epilogue(ctx: DeviceContext) raises:
    _run_bias_epilogue_test[.float32, .float16, True](
        ctx, "test_kernel_128_nt_fp32_fp16_bias_epilogue"
    )


def test_kernel_128_nt_fp32_bf16_bias_epilogue(ctx: DeviceContext) raises:
    _run_bias_epilogue_test[.float32, .bfloat16, True](
        ctx, "test_kernel_128_nt_fp32_bf16_bias_epilogue"
    )


def test_kernel_128_nt_fp32_fp32_bias_epilogue(ctx: DeviceContext) raises:
    _run_bias_epilogue_test[.float32, .float32, True](
        ctx, "test_kernel_128_nt_fp32_fp32_bias_epilogue"
    )


def test_kernel_128_nt_fp16_fp16_relu_compose_epilogue(
    ctx: DeviceContext,
) raises:
    """ReLU composed into `elementwise_lambda_fn` — mirrors MXF-369's
    `compute_lambda_wrapper` composition.
    """
    print("== test_kernel_128_nt_fp16_fp16_relu_compose_epilogue")
    comptime M = 128
    comptime N = 128
    comptime K = 128

    var a_host = ctx.enqueue_create_host_buffer[.float16](M * K)
    var b_host = ctx.enqueue_create_host_buffer[.float16](N * K)
    for i in range(M * K):
        a_host[i] = Float16(random_si64(Int64(-2), Int64(2)).cast[.float16]())
    for i in range(N * K):
        b_host[i] = Float16(random_si64(Int64(-2), Int64(2)).cast[.float16]())

    var a_dev = ctx.enqueue_create_buffer[.float16](M * K)
    var b_dev = ctx.enqueue_create_buffer[.float16](N * K)
    var d_dev = ctx.enqueue_create_buffer[.float16](M * N)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)

    var a_tt = TileTensor(a_dev.unsafe_ptr(), row_major(M, K))
    var b_tt = TileTensor(b_dev.unsafe_ptr(), row_major(N, K))
    var d_tt = TileTensor(d_dev.unsafe_ptr(), row_major(M, N))

    var d_ptr = d_dev.unsafe_ptr()

    @__parameter
    @inline(.always)
    @__copy_capture(d_ptr)
    def relu_compose_epilogue[
        dt: DType, w: SIMDLength, *, alignment: Int = 1
    ](coords: IndexList[2], val: SIMD[dt, w]) capturing -> None:
        var v_fp16 = rebind[SIMD[.float16, w]](val)
        var relu_val = max(v_fp16, SIMD[.float16, w](0))
        (d_ptr + coords[0] * N + coords[1]).store[alignment=alignment](relu_val)

    enqueue_apple_matmul[
        in_type=.float16,
        c_type=.float16,
        transpose_b=True,
        elementwise_lambda_fn=Optional[elementwise_epilogue_type](
            relu_compose_epilogue
        ),
    ](d_tt, a_tt, b_tt, ctx)

    var d_host = ctx.enqueue_create_host_buffer[.float16](M * N)
    ctx.enqueue_copy(d_host, d_dev)
    ctx.synchronize()

    # DRIV-199 workaround: keep device buffers alive past `synchronize`, else
    # ASAP destruction frees them mid-kernel and the suite flakes.
    _ = a_dev^
    _ = b_dev^
    _ = d_dev^

    var pass_ = True
    for i in range(M):
        for j in range(N):
            var acc = Float32(0)
            for k in range(K):
                acc += Float32(a_host[i * K + k]) * Float32(b_host[j * K + k])
            var exp = max(acc, Float32(0))
            var got = Float32(d_host[i * N + j])
            if not _within_tol[.float16](got, exp):
                print("FAIL:", i, j, "got", got, "expected", exp)
                pass_ = False
    if not pass_:
        raise Error("FAILED (see FAIL lines above)")
    print("PASS")


def test_kernel_128_nt_fp16_fp16_bias_relu_compose_epilogue(
    ctx: DeviceContext,
) raises:
    """Chained bias-then-ReLU composed into one `elementwise_lambda_fn`."""
    print("== test_kernel_128_nt_fp16_fp16_bias_relu_compose_epilogue")
    comptime M = 128
    comptime N = 128
    comptime K = 128

    var a_host = ctx.enqueue_create_host_buffer[.float16](M * K)
    var b_host = ctx.enqueue_create_host_buffer[.float16](N * K)
    var bias_host = ctx.enqueue_create_host_buffer[.float16](N)
    for i in range(M * K):
        a_host[i] = Float16(random_si64(Int64(-2), Int64(2)).cast[.float16]())
    for i in range(N * K):
        b_host[i] = Float16(random_si64(Int64(-2), Int64(2)).cast[.float16]())
    for j in range(N):
        bias_host[j] = Float16(
            random_si64(Int64(-2), Int64(2)).cast[.float16]()
        )

    var a_dev = ctx.enqueue_create_buffer[.float16](M * K)
    var b_dev = ctx.enqueue_create_buffer[.float16](N * K)
    var bias_dev = ctx.enqueue_create_buffer[.float16](N)
    var d_dev = ctx.enqueue_create_buffer[.float16](M * N)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)
    ctx.enqueue_copy(bias_dev, bias_host)

    var a_tt = TileTensor(a_dev.unsafe_ptr(), row_major(M, K))
    var b_tt = TileTensor(b_dev.unsafe_ptr(), row_major(N, K))
    var d_tt = TileTensor(d_dev.unsafe_ptr(), row_major(M, N))

    var d_ptr = d_dev.unsafe_ptr()
    var bias_ptr = bias_dev.unsafe_ptr()

    @__parameter
    @inline(.always)
    @__copy_capture(d_ptr, bias_ptr)
    def bias_relu_compose_epilogue[
        dt: DType, w: SIMDLength, *, alignment: Int = 1
    ](coords: IndexList[2], val: SIMD[dt, w]) capturing -> None:
        var v_fp16 = rebind[SIMD[.float16, w]](val)
        var b = (bias_ptr + coords[1]).load[width=w]()
        var biased = v_fp16 + b
        var activated = max(biased, SIMD[.float16, w](0))
        (d_ptr + coords[0] * N + coords[1]).store[alignment=alignment](
            activated
        )

    enqueue_apple_matmul[
        in_type=.float16,
        c_type=.float16,
        transpose_b=True,
        elementwise_lambda_fn=Optional[elementwise_epilogue_type](
            bias_relu_compose_epilogue
        ),
    ](d_tt, a_tt, b_tt, ctx)

    var d_host = ctx.enqueue_create_host_buffer[.float16](M * N)
    ctx.enqueue_copy(d_host, d_dev)
    ctx.synchronize()

    # DRIV-199 workaround: keep device buffers alive past `synchronize`, else
    # ASAP destruction frees them mid-kernel and the suite flakes.
    _ = a_dev^
    _ = b_dev^
    _ = bias_dev^
    _ = d_dev^

    var pass_ = True
    for i in range(M):
        for j in range(N):
            var acc = Float32(0)
            for k in range(K):
                acc += Float32(a_host[i * K + k]) * Float32(b_host[j * K + k])
            var exp = max(acc + Float32(bias_host[j]), Float32(0))
            var got = Float32(d_host[i * N + j])
            if not _within_tol[.float16](got, exp):
                print("FAIL:", i, j, "got", got, "expected", exp)
                pass_ = False
    if not pass_:
        raise Error("FAILED (see FAIL lines above)")
    print("PASS")


def test_kernel_ragged_100x100x97_nt_fp16_fp16_bias_epilogue(
    ctx: DeviceContext,
) raises:
    """Bounded epilogue + lambda — proves OOB rows/cols never invoke the lambda.
    """
    print("== test_kernel_ragged_100x100x97_nt_fp16_fp16_bias_epilogue")
    comptime M = 100
    comptime N = 100
    comptime K = 97

    var a_host = ctx.enqueue_create_host_buffer[.float16](M * K)
    var b_host = ctx.enqueue_create_host_buffer[.float16](N * K)
    var bias_host = ctx.enqueue_create_host_buffer[.float16](N)
    for i in range(M * K):
        a_host[i] = Float16(random_si64(Int64(-2), Int64(2)).cast[.float16]())
    for i in range(N * K):
        b_host[i] = Float16(random_si64(Int64(-2), Int64(2)).cast[.float16]())
    for j in range(N):
        bias_host[j] = Float16(
            random_si64(Int64(-2), Int64(2)).cast[.float16]()
        )

    var a_dev = ctx.enqueue_create_buffer[.float16](M * K)
    var b_dev = ctx.enqueue_create_buffer[.float16](N * K)
    var bias_dev = ctx.enqueue_create_buffer[.float16](N)
    var d_dev = ctx.enqueue_create_buffer[.float16](M * N)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)
    ctx.enqueue_copy(bias_dev, bias_host)

    var a_tt = TileTensor(a_dev.unsafe_ptr(), row_major(M, K))
    var b_tt = TileTensor(b_dev.unsafe_ptr(), row_major(N, K))
    var d_tt = TileTensor(d_dev.unsafe_ptr(), row_major(M, N))

    var d_ptr = d_dev.unsafe_ptr()
    var bias_ptr = bias_dev.unsafe_ptr()

    @__parameter
    @inline(.always)
    @__copy_capture(d_ptr, bias_ptr)
    def bias_epilogue[
        dt: DType, w: SIMDLength, *, alignment: Int = 1
    ](coords: IndexList[2], val: SIMD[dt, w]) capturing -> None:
        var v_fp16 = rebind[SIMD[.float16, w]](val)
        var b = (bias_ptr + coords[1]).load[width=w]()
        (d_ptr + coords[0] * N + coords[1]).store[alignment=alignment](
            v_fp16 + b
        )

    enqueue_apple_matmul[
        in_type=.float16,
        c_type=.float16,
        transpose_b=True,
        elementwise_lambda_fn=Optional[elementwise_epilogue_type](
            bias_epilogue
        ),
    ](d_tt, a_tt, b_tt, ctx)

    var d_host = ctx.enqueue_create_host_buffer[.float16](M * N)
    ctx.enqueue_copy(d_host, d_dev)
    ctx.synchronize()

    # DRIV-199 workaround: keep device buffers alive past `synchronize`, else
    # ASAP destruction frees them mid-kernel and the suite flakes.
    _ = a_dev^
    _ = b_dev^
    _ = bias_dev^
    _ = d_dev^

    var pass_ = True
    for i in range(M):
        for j in range(N):
            var acc = Float32(0)
            for k in range(K):
                acc += Float32(a_host[i * K + k]) * Float32(b_host[j * K + k])
            var exp = acc + Float32(bias_host[j])
            var got = Float32(d_host[i * N + j])
            if not _within_tol[.float16](got, exp):
                print("FAIL:", i, j, "got", got, "expected", exp)
                pass_ = False
    if not pass_:
        raise Error("FAILED (see FAIL lines above)")
    print("PASS")


def test_kernel_64x130x64_nn_fp16_fp16_oddn(ctx: DeviceContext) raises:
    """Cast epilogue, non-mult-of-4 N (130): the width-4 store stride `row*N`
    is element- but not vector-aligned. Guards the unaligned store path.
    """
    print("== test_kernel_64x130x64_nn_fp16_fp16_oddn")
    comptime M = 64
    comptime N = 130
    comptime K = 64

    var a_host = ctx.enqueue_create_host_buffer[.float16](M * K)
    var b_host = ctx.enqueue_create_host_buffer[.float16](K * N)
    for i in range(M * K):
        a_host[i] = Float16(random_si64(Int64(-2), Int64(2)).cast[.float16]())
    for i in range(K * N):
        b_host[i] = Float16(random_si64(Int64(-2), Int64(2)).cast[.float16]())

    var a_dev = ctx.enqueue_create_buffer[.float16](M * K)
    var b_dev = ctx.enqueue_create_buffer[.float16](K * N)
    var d_dev = ctx.enqueue_create_buffer[.float16](M * N)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)

    var a_tt = TileTensor(a_dev.unsafe_ptr(), row_major(M, K))
    var b_tt = TileTensor(b_dev.unsafe_ptr(), row_major(K, N))
    var d_tt = TileTensor(d_dev.unsafe_ptr(), row_major(M, N))

    enqueue_apple_matmul[
        in_type=.float16,
        c_type=.float16,
        transpose_b=False,
    ](d_tt, a_tt, b_tt, ctx)

    var d_host = ctx.enqueue_create_host_buffer[.float16](M * N)
    ctx.enqueue_copy(d_host, d_dev)
    ctx.synchronize()

    # DRIV-199 workaround: keep device buffers alive past `synchronize`, else
    # ASAP destruction frees them mid-kernel and the suite flakes.
    _ = a_dev^
    _ = b_dev^
    _ = d_dev^

    var pass_ = True
    for i in range(M):
        for j in range(N):
            var exp = _host_matmul_nn[.float16, .float16](
                a_host.unsafe_ptr(), b_host.unsafe_ptr(), M, N, K, i, j
            )
            var got = Float32(d_host[i * N + j])
            if not _within_tol[.float16](got, exp):
                print("FAIL:", i, j, "got", got, "expected", exp)
                pass_ = False
    if not pass_:
        raise Error("FAILED (see FAIL lines above)")
    print("PASS")


def test_kernel_64x130x64_nn_fp16_fp16_oddn_bias_epilogue(
    ctx: DeviceContext,
) raises:
    """Lambda epilogue, non-mult-of-4 N (130): exercises the unaligned store
    stride through the user-lambda path.
    """
    print("== test_kernel_64x130x64_nn_fp16_fp16_oddn_bias_epilogue")
    comptime M = 64
    comptime N = 130
    comptime K = 64

    var a_host = ctx.enqueue_create_host_buffer[.float16](M * K)
    var b_host = ctx.enqueue_create_host_buffer[.float16](K * N)
    var bias_host = ctx.enqueue_create_host_buffer[.float16](N)
    for i in range(M * K):
        a_host[i] = Float16(random_si64(Int64(-2), Int64(2)).cast[.float16]())
    for i in range(K * N):
        b_host[i] = Float16(random_si64(Int64(-2), Int64(2)).cast[.float16]())
    for j in range(N):
        bias_host[j] = Float16(
            random_si64(Int64(-2), Int64(2)).cast[.float16]()
        )

    var a_dev = ctx.enqueue_create_buffer[.float16](M * K)
    var b_dev = ctx.enqueue_create_buffer[.float16](K * N)
    var bias_dev = ctx.enqueue_create_buffer[.float16](N)
    var d_dev = ctx.enqueue_create_buffer[.float16](M * N)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)
    ctx.enqueue_copy(bias_dev, bias_host)

    var a_tt = TileTensor(a_dev.unsafe_ptr(), row_major(M, K))
    var b_tt = TileTensor(b_dev.unsafe_ptr(), row_major(K, N))
    var d_tt = TileTensor(d_dev.unsafe_ptr(), row_major(M, N))

    var d_ptr = d_dev.unsafe_ptr()
    var bias_ptr = bias_dev.unsafe_ptr()

    @__parameter
    @inline(.always)
    @__copy_capture(d_ptr, bias_ptr)
    def bias_epilogue[
        dt: DType, w: SIMDLength, *, alignment: Int = 1
    ](coords: IndexList[2], val: SIMD[dt, w]) capturing -> None:
        var b = (bias_ptr + coords[1]).load[width=w]()
        var v_c = rebind[SIMD[.float16, w]](val)
        (d_ptr + coords[0] * N + coords[1]).store[alignment=alignment](v_c + b)

    enqueue_apple_matmul[
        in_type=.float16,
        c_type=.float16,
        transpose_b=False,
        elementwise_lambda_fn=Optional[elementwise_epilogue_type](
            bias_epilogue
        ),
    ](d_tt, a_tt, b_tt, ctx)

    var d_host = ctx.enqueue_create_host_buffer[.float16](M * N)
    ctx.enqueue_copy(d_host, d_dev)
    ctx.synchronize()

    # DRIV-199 workaround: keep device buffers alive past `synchronize`, else
    # ASAP destruction frees them mid-kernel and the suite flakes.
    _ = a_dev^
    _ = b_dev^
    _ = bias_dev^
    _ = d_dev^

    var pass_ = True
    for i in range(M):
        for j in range(N):
            var acc = _host_matmul_nn[.float16, .float16](
                a_host.unsafe_ptr(), b_host.unsafe_ptr(), M, N, K, i, j
            )
            var exp = acc + Float32(bias_host[j])
            var got = Float32(d_host[i * N + j])
            if not _within_tol[.float16](got, exp):
                print("FAIL:", i, j, "got", got, "expected", exp)
                pass_ = False
    if not pass_:
        raise Error("FAILED (see FAIL lines above)")
    print("PASS")


# ===----------------------------------------------------------------------=== #
# Small-batch GEMV (`apple/gemv.mojo`, the M = 2..8 dispatch path; Apple M5)
# ===----------------------------------------------------------------------=== #

comptime _GEMV_SENTINEL = -12345.0


def _gemv_fill_random[
    dtype: DType
](buf: HostBuffer[dtype], count: Int, scale: Float64 = 1.0):
    for i in range(count):
        buf[i] = Scalar[dtype]((random_float64() * 2.0 - 1.0) * scale)


def _gemv_fill_hash[dtype: DType](buf: HostBuffer[dtype], count: Int):
    """Fills `buf` with values in [-1, 1) that differ from element to element.

    Cheaper than `_gemv_fill_random` at billions of elements, and a read from
    the wrong offset still lands on a different value.
    """
    for i in range(count):
        var h = (UInt64(i) * UInt64(0x9E3779B97F4A7C15)) >> 48
        buf[i] = Scalar[dtype](Float64(Int(h)) / 32768.0 - 1.0)


def _gemv_check[
    in_type: DType, c_type: DType
](
    got: HostBuffer[c_type],
    act: HostBuffer[in_type],
    weight: HostBuffer[in_type],
    m: Int,
    rows_alloc: Int,
    n: Int,
    k: Int,
    name: String,
    col_step: Int = 1,
    tail_cols: Int = 0,
) raises:
    """Checks rows `[0, m)` against the host sum and rows past `m` untouched.

    Only columns `j` with `j % col_step == 0` or `j >= n - tail_cols` are
    checked, which keeps the host sum affordable at the largest N.
    """
    comptime rel = Float32(2.0**-8) if c_type != DType.float32 else Float32(
        1e-6
    )
    var bad = 0
    for i in range(m):
        for j in range(n):
            if j % col_step != 0 and j < n - tail_cols:
                continue
            var acc = Float32(0)
            var mag = Float32(0)
            for kk in range(k):
                var p = (
                    act[i * k + kk].cast[.float32]()
                    * weight[j * k + kk].cast[.float32]()
                )
                acc += p
                mag += abs(p)
            var y = got[i * n + j].cast[.float32]()
            # Negated so a NaN output, which fails every comparison, fails.
            if not (abs(y - acc) <= rel * abs(acc) + Float32(1e-5) * mag):
                if bad < 5:
                    print("  FAIL", name, "m", i, "n", j, "got", y, "exp", acc)
                bad += 1
    for i in range(m * n, rows_alloc * n):
        if got[i] != Scalar[c_type](_GEMV_SENTINEL):
            if bad < 5:
                print("  FAIL", name, "wrote past M at flat index", i)
            bad += 1
    if bad:
        raise Error(String(name, ": ", bad, " mismatches"))


def _run_gemv_case[
    in_type: DType,
    c_type: DType,
    N: Int,
    K: Int,
    *,
    tile_m: Int,
    rows_per_warp: Int,
    tile_k: Int = 8,
    unroll: Int = 2,
](ctx: DeviceContext, m: Int, name: String) raises:
    """Runs one launch configuration at `[m, N, K]` and checks it."""
    print("==", name, "m", m, "n", N, "k", K)
    var act_h = ctx.enqueue_create_host_buffer[in_type](m * K)
    var w_h = ctx.enqueue_create_host_buffer[in_type](N * K)
    var c_h = ctx.enqueue_create_host_buffer[c_type](tile_m * N)
    _gemv_fill_random(act_h, m * K)
    _gemv_fill_random(w_h, N * K)
    for i in range(tile_m * N):
        c_h[i] = Scalar[c_type](_GEMV_SENTINEL)

    var act_d = ctx.enqueue_create_buffer[in_type](m * K)
    var w_d = ctx.enqueue_create_buffer[in_type](N * K)
    var c_d = ctx.enqueue_create_buffer[c_type](tile_m * N)
    ctx.enqueue_copy(act_d, act_h)
    ctx.enqueue_copy(w_d, w_h)
    ctx.enqueue_copy(c_d, c_h)

    var a_tt = TileTensor(act_d, row_major(Coord(m, Idx[K]))).as_imm()
    var w_tt = TileTensor(w_d, row_major(Coord(Idx[N], Idx[K]))).as_imm()
    var c_tt = TileTensor(c_d, row_major(Coord(m, Idx[N])))
    enqueue_apple_gemv_config[
        tile_m=tile_m,
        rows_per_warp=rows_per_warp,
        tile_k=tile_k,
        unroll=unroll,
    ](c_tt, a_tt, w_tt, ctx)
    ctx.enqueue_copy(c_h, c_d)
    ctx.synchronize()
    _ = act_d^
    _ = w_d^
    _ = c_d^
    _gemv_check[in_type, c_type](c_h, act_h, w_h, m, tile_m, N, K, name)
    print("  PASS")


def test_gemv_configs(ctx: DeviceContext) raises:
    """Each launch parameter on its own, at small N and K."""
    seed(0)
    _run_gemv_case[.bfloat16, .float32, 256, 512, tile_m=1, rows_per_warp=1](
        ctx, 1, "r1"
    )
    _run_gemv_case[.bfloat16, .float32, 256, 512, tile_m=1, rows_per_warp=2](
        ctx, 1, "r2"
    )
    _run_gemv_case[.bfloat16, .float32, 256, 512, tile_m=1, rows_per_warp=4](
        ctx, 1, "r4"
    )
    _run_gemv_case[
        .bfloat16, .float32, 256, 512, tile_m=1, rows_per_warp=2, unroll=4
    ](ctx, 1, "r2 u4")
    _run_gemv_case[
        .bfloat16, .float32, 256, 512, tile_m=1, rows_per_warp=1, tile_k=16
    ](ctx, 1, "r1 k16")
    _run_gemv_case[.bfloat16, .bfloat16, 256, 512, tile_m=1, rows_per_warp=2](
        ctx, 1, "bf16 out"
    )
    _run_gemv_case[.float16, .float16, 256, 512, tile_m=1, rows_per_warp=2](
        ctx, 1, "fp16"
    )


def test_gemv_ragged(ctx: DeviceContext) raises:
    """Ragged N, K below one warp stride, and partial unrolls."""
    seed(1)
    _run_gemv_case[.bfloat16, .float32, 203, 512, tile_m=1, rows_per_warp=4](
        ctx, 1, "ragged n"
    )
    _run_gemv_case[.bfloat16, .float32, 64, 520, tile_m=1, rows_per_warp=2](
        ctx, 1, "k 520"
    )
    _run_gemv_case[.bfloat16, .float32, 64, 40, tile_m=1, rows_per_warp=2](
        ctx, 1, "k < stride"
    )
    _run_gemv_case[.bfloat16, .float32, 128, 128, tile_m=1, rows_per_warp=1](
        ctx, 1, "k 128"
    )
    # K chunk counts that leave 0, 1 and unroll - 1 chunks after the
    # unrolled loop.
    _run_gemv_case[
        .bfloat16, .float32, 32, 256 * 5, tile_m=1, rows_per_warp=2, unroll=4
    ](ctx, 1, "u4 rem1")
    _run_gemv_case[
        .bfloat16, .float32, 32, 256 * 7, tile_m=1, rows_per_warp=2, unroll=4
    ](ctx, 1, "u4 rem3")


def test_gemv_small_batch(ctx: DeviceContext) raises:
    """M in 2..4 on the multi-row launches, including M below `tile_m`."""
    seed(2)
    _run_gemv_case[.bfloat16, .float32, 256, 512, tile_m=2, rows_per_warp=1](
        ctx, 2, "m2"
    )
    _run_gemv_case[.bfloat16, .float32, 256, 512, tile_m=4, rows_per_warp=2](
        ctx, 4, "m4"
    )
    _run_gemv_case[.bfloat16, .float32, 203, 520, tile_m=4, rows_per_warp=2](
        ctx, 3, "m3 of 4"
    )
    _run_gemv_case[.bfloat16, .bfloat16, 256, 512, tile_m=4, rows_per_warp=1](
        ctx, 4, "m4 bf16 out"
    )


def test_gemv_large_shapes(ctx: DeviceContext) raises:
    """Narrow N, short K, long K, and K below the dispatch floor."""
    seed(3)
    _run_gemv_case[.bfloat16, .bfloat16, 576, 7168, tile_m=1, rows_per_warp=2](
        ctx, 1, "narrow n"
    )
    _run_gemv_case[
        .bfloat16, .bfloat16, 18432, 1536, tile_m=1, rows_per_warp=2
    ](ctx, 1, "short k")
    _run_gemv_case[
        .bfloat16, .bfloat16, 7168, 12288, tile_m=1, rows_per_warp=2
    ](ctx, 1, "long k")
    _run_gemv_case[.bfloat16, .bfloat16, 12288, 128, tile_m=1, rows_per_warp=2](
        ctx, 1, "k 128"
    )


def _run_gemv_dispatch_case[
    N: Int,
    K: Int,
    *,
    in_type: DType = .bfloat16,
    c_type: DType = .float32,
    bias: Bool = False,
    sampled: Bool = False,
](ctx: DeviceContext, m: Int, name: String) raises:
    """Runs `_matmul_gpu[transpose_b=True]` at `[m, N, K]` and checks it.

    With `bias`, the output is written by an elementwise epilogue that adds a
    per-column bias, which checks the `(row, col)` coordinates the kernel
    hands the epilogue. With `sampled`, the weight gets the hash fill and the
    host reference checks every 1021st column plus the last 256.
    """
    print("== dispatch", name, "m", m, "n", N, "k", K, "bias", bias)
    var act_h = ctx.enqueue_create_host_buffer[in_type](m * K)
    var w_h = ctx.enqueue_create_host_buffer[in_type](N * K)
    var bias_h = ctx.enqueue_create_host_buffer[.float32](N)
    var c_h = ctx.enqueue_create_host_buffer[c_type](m * N)
    _gemv_fill_random(act_h, m * K)
    comptime if sampled:
        _gemv_fill_hash(w_h, N * K)
    else:
        _gemv_fill_random(w_h, N * K)
    _gemv_fill_random(bias_h, N, 4.0)
    for i in range(m * N):
        c_h[i] = Scalar[c_type](_GEMV_SENTINEL)

    var act_d = ctx.enqueue_create_buffer[in_type](m * K)
    var w_d = ctx.enqueue_create_buffer[in_type](N * K)
    var bias_d = ctx.enqueue_create_buffer[.float32](N)
    var c_d = ctx.enqueue_create_buffer[c_type](m * N)
    ctx.enqueue_copy(act_d, act_h)
    ctx.enqueue_copy(w_d, w_h)
    ctx.enqueue_copy(bias_d, bias_h)
    ctx.enqueue_copy(c_d, c_h)

    var a_tt = TileTensor(act_d, row_major(Coord(m, Idx[K]))).as_imm()
    var w_tt = TileTensor(w_d, row_major(Coord(Idx[N], Idx[K]))).as_imm()
    var c_tt = TileTensor(c_d, row_major(Coord(m, Idx[N])))

    var bias_tt = TileTensor(bias_d, row_major(Idx[N])).as_imm()

    @inline(.always)
    @__copy_capture(c_tt, bias_tt)
    def bias_epilogue[
        dt: DType, w: SIMDLength, *, alignment: Int = 1
    ](coords: IndexList[2], val: SIMD[dt, w]) capturing -> None:
        var b = bias_tt.load[width=w, alignment=alignment](Coord(coords[1]))
        c_tt.store[alignment=alignment](
            Coord(coords[0], coords[1]),
            (val.cast[.float32]() + b).cast[c_type](),
        )

    comptime if bias:
        _matmul_gpu[
            transpose_b=True,
            elementwise_lambda_fn=Optional[elementwise_epilogue_type](
                bias_epilogue
            ),
        ](c_tt, a_tt, w_tt, ctx)
    else:
        _matmul_gpu[transpose_b=True](c_tt, a_tt, w_tt, ctx)
    ctx.enqueue_copy(c_h, c_d)
    ctx.synchronize()
    _ = act_d^
    _ = w_d^
    _ = bias_d^
    _ = c_d^

    comptime if bias:
        for i in range(m):
            for j in range(N):
                c_h[i * N + j] -= bias_h[j].cast[c_type]()
    comptime if sampled:
        _gemv_check[in_type, c_type](
            c_h, act_h, w_h, m, m, N, K, name, col_step=1021, tail_cols=256
        )
    else:
        _gemv_check[in_type, c_type](c_h, act_h, w_h, m, m, N, K, name)
    print("  PASS")


def test_gemv_dispatch(ctx: DeviceContext) raises:
    """`_matmul_gpu` at every M the GEMV serves, and at its M boundaries.

    M = 1 and M = 9 take the split-K GEMV and the tiled matmul; they are here
    so a boundary change that breaks either neighbour fails this suite too.
    """
    seed(4)
    for m in [1, 2, 3, 4, 5, 8, 9]:
        _run_gemv_dispatch_case[256, 1024](ctx, m, "small")
    _run_gemv_dispatch_case[256, 1024, bias=True](ctx, 1, "bias m1")
    _run_gemv_dispatch_case[256, 1024, bias=True](ctx, 3, "bias m3")
    _run_gemv_dispatch_case[256, 1024, bias=True](ctx, 8, "bias m8")
    # K below the GEMV's K floor stays on the tiled matmul. At K = 512 the
    # floor depends on M: M = 4 takes the GEMV and M = 8 the tiled matmul.
    _run_gemv_dispatch_case[256, 128](ctx, 4, "k128")
    _run_gemv_dispatch_case[256, 512](ctx, 4, "k512 m4")
    _run_gemv_dispatch_case[256, 512](ctx, 8, "k512 m8")
    _run_gemv_dispatch_case[576, 7168](ctx, 2, "narrow n")
    _run_gemv_dispatch_case[3584, 7168](ctx, 4, "k 2n")
    _run_gemv_dispatch_case[7168, 3584](ctx, 8, "n 2k")
    _run_gemv_dispatch_case[18432, 1536](ctx, 4, "short k")
    _run_gemv_dispatch_case[256, 1024, c_type=.bfloat16](ctx, 4, "bf16 out")
    # fp16 takes the same route as bf16.
    for m in [1, 2, 5, 8, 9]:
        _run_gemv_dispatch_case[256, 1024, in_type=.float16, c_type=.float16](
            ctx, m, "fp16"
        )
    _run_gemv_dispatch_case[3584, 7168, in_type=.float16](ctx, 4, "fp16 k 2n")
    _run_gemv_dispatch_case[256, 1024, in_type=.float16, bias=True](
        ctx, 3, "fp16 bias m3"
    )
    # lm_head, the largest weight: the last rows sit past 2^30 elements.
    for m in [2, 5, 8]:
        _run_gemv_dispatch_case[163840, 7168, sampled=True](ctx, m, "lm_head")


def main() raises:
    test_morton_decode_2d()
    test_morton_decode_2d_rect()
    comptime if "metal" not in _accelerator_arch():
        print("SKIP: apple_gpu_matmul tests require Apple GPU")
        return
    var ctx = DeviceContext()

    # 8x8 simdgroup-matrix path (`gemm_kernel_apple_8x8`, the M1-M4 dispatch
    # path). Valid on every Apple GPU, so these run regardless of compute
    # capability.
    _run_8x8_case[.bfloat16, .bfloat16, True](
        ctx, 64, 64, 16, "8x8 bf16 nt min"
    )
    _run_8x8_case[.bfloat16, .bfloat16, True](
        ctx, 512, 1024, 256, "8x8 bf16 nt large"
    )
    _run_8x8_case[.bfloat16, .bfloat16, False](ctx, 128, 256, 64, "8x8 bf16 nn")
    _run_8x8_case[.float16, .float16, True](ctx, 256, 256, 128, "8x8 fp16 nt")
    _run_8x8_case[.bfloat16, .float32, True](
        ctx, 128, 128, 64, "8x8 bf16 in f32 out"
    )
    # Ragged M (real prefill: seq_len not a multiple of 64) and ragged N.
    _run_8x8_case[.bfloat16, .bfloat16, True](
        ctx, 100, 128, 64, "8x8 bf16 nt ragged-m"
    )
    _run_8x8_case[.bfloat16, .bfloat16, False](
        ctx, 100, 200, 64, "8x8 bf16 nn ragged-mn"
    )
    # Odd M/N: edge subtiles where the bound splits a lane's 2-wide fragment.
    # NN + odd N hits the single-slot B-load path (`gj + 1 >= n > gj`); even
    # dims never trigger it since every lane's column index is even.
    _run_8x8_case[.bfloat16, .bfloat16, False](
        ctx, 64, 129, 64, "8x8 bf16 nn odd-n"
    )
    _run_8x8_case[.bfloat16, .bfloat16, True](
        ctx, 65, 129, 64, "8x8 bf16 nt odd-mn"
    )

    # Bias-add `elementwise_lambda_fn` epilogue (clean interior, odd NT edges,
    # and the NN odd-N load edge combined with the epilogue store).
    _run_8x8_bias_case[.bfloat16, .bfloat16, True](
        ctx, 128, 128, 64, "8x8 bias nt"
    )
    _run_8x8_bias_case[.bfloat16, .bfloat16, True](
        ctx, 65, 129, 64, "8x8 bias nt odd"
    )
    _run_8x8_bias_case[.bfloat16, .bfloat16, False](
        ctx, 64, 129, 64, "8x8 bias nn odd-n"
    )

    # f32 in/out: the 8x8 unit is full-precision for f32 (no fp19 truncation),
    # so these check against the tight f32 tolerance. Covers NT/NN, an odd edge,
    # and the epilogue store.
    _run_8x8_case[.float32, .float32, True](ctx, 256, 256, 128, "8x8 f32 nt")
    _run_8x8_case[.float32, .float32, False](ctx, 128, 256, 64, "8x8 f32 nn")
    _run_8x8_case[.float32, .float32, False](
        ctx, 64, 129, 64, "8x8 f32 nn odd-n"
    )
    _run_8x8_bias_case[.float32, .float32, True](
        ctx, 128, 128, 64, "8x8 f32 bias nt"
    )

    # M5 hardware-MMA path (`AppleM5MatMul` / `enqueue_apple_matmul`): Apple M5.
    if ctx.compute_capability() != 5:
        print(
            "SKIP: M5 hardware-MMA matmul tests require Apple M5"
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
    test_partial_m_decode_nt_bf16(ctx)
    test_kernel_128x128x32_nt_fp16(ctx)
    test_kernel_128x128x32_nn_bf16(ctx)
    test_kernel_ragged_100x200x64_nn_bf16_clamp_chain(ctx)
    test_kernel_128x128x32_nn_fp32(ctx)
    test_enqueue_helper_fp16(ctx)
    test_kernel_128_nn_fp16_fp16_no_lambda(ctx)
    test_kernel_128_nn_fp16_bf16_no_lambda(ctx)
    test_kernel_128_nt_fp16_fp16_bias_epilogue(ctx)
    test_kernel_128_nn_fp16_fp16_bias_epilogue(ctx)
    test_kernel_128_nn_fp16_bf16_bias_epilogue(ctx)
    test_kernel_128_nn_fp16_fp32_bias_epilogue(ctx)
    test_kernel_128_nt_fp16_bf16_bias_epilogue(ctx)
    test_kernel_128_nt_fp16_fp32_bias_epilogue(ctx)
    test_kernel_128_nn_bf16_fp16_bias_epilogue(ctx)
    test_kernel_128_nn_bf16_bf16_bias_epilogue(ctx)
    test_kernel_128_nn_bf16_fp32_bias_epilogue(ctx)
    test_kernel_128_nt_bf16_fp16_bias_epilogue(ctx)
    test_kernel_128_nt_bf16_bf16_bias_epilogue(ctx)
    test_kernel_128_nt_bf16_fp32_bias_epilogue(ctx)
    test_kernel_128_nn_fp32_fp16_bias_epilogue(ctx)
    test_kernel_128_nn_fp32_bf16_bias_epilogue(ctx)
    test_kernel_128_nn_fp32_fp32_bias_epilogue(ctx)
    test_kernel_128_nt_fp32_fp16_bias_epilogue(ctx)
    test_kernel_128_nt_fp32_bf16_bias_epilogue(ctx)
    test_kernel_128_nt_fp32_fp32_bias_epilogue(ctx)
    test_kernel_128_nt_fp16_fp16_relu_compose_epilogue(ctx)
    test_kernel_128_nt_fp16_fp16_bias_relu_compose_epilogue(ctx)
    test_kernel_ragged_100x100x97_nt_fp16_fp16_bias_epilogue(ctx)
    test_kernel_64x130x64_nn_fp16_fp16_oddn(ctx)
    test_kernel_64x130x64_nn_fp16_fp16_oddn_bias_epilogue(ctx)

    # Split-K path (folded in from the former test_apple_split_k.mojo).
    # Explicit split counts via enqueue_apple_matmul_split_k:
    _run_split_k_case[.float16, .float32, False](
        ctx, 64, 64, 4096, "splitk nn k4096", splits=4
    )
    _run_split_k_case[.float16, .float32, False](
        ctx, 64, 64, 4096, "splitk nn s8", splits=8
    )
    # K not BK-aligned (last split carries a tail).
    _run_split_k_case[.float16, .float32, False](
        ctx, 64, 64, 4097, "splitk nn k4097 tail", splits=4
    )
    # split hint > num_strips: must cap (no empty splits / OOB).
    _run_split_k_case[.float16, .float32, False](
        ctx, 64, 64, 64, "splitk nn s16cap", splits=16
    )
    # Ragged M/N + multi-tile.
    _run_split_k_case[.float16, .float32, False](
        ctx, 100, 200, 2048, "splitk nn ragged", splits=4
    )
    # NT.
    _run_split_k_case[.float16, .float32, True](
        ctx, 96, 96, 3072, "splitk nt k3072", splits=4
    )
    # bf16 in, fp16 out (exercises the reduce cast).
    _run_split_k_case[.bfloat16, .float16, False](
        ctx, 64, 128, 2048, "splitk bf16->fp16 reduce", splits=4
    )
    # force_split_k=True via enqueue_apple_matmul on balanced shapes that would
    # NOT auto-route: the forced split-K result must still match the reference.
    _run_split_k_case[.float16, .float32, False](
        ctx, 128, 128, 256, "force nn", force_split_k=True
    )
    _run_split_k_case[.float16, .bfloat16, True](
        ctx, 96, 160, 2048, "force nt large-k", force_split_k=True
    )

    test_gemv_configs(ctx)
    test_gemv_ragged(ctx)
    test_gemv_small_batch(ctx)
    test_gemv_large_shapes(ctx)
    test_gemv_dispatch(ctx)
