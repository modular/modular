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
"""Smoke test for MXFP4 dequant-then-FP8 matmul.

Tests the full MXFP4 matmul pipeline:
  1. Dequant packed uint8 FP4 weights + E8M0 scales → FP8
  2. Cast BF16 activations → FP8
  3. FP8 GEMM (transpose_b=True)

Validates against a vendor BLAS reference with dequantized data.
Works on AMD (CDNA) GPUs.
"""

from std.math import ceildiv
from std.memory import bitcast
from std.random import random_float64, random_ui64
from std.sys.info import _accelerator_arch
from max.gpu.host import DeviceContext
import linalg.matmul.vendor.blas as vendor_blas
from layout import (
    ComptimeInt,
    Idx,
    RowMajorLayout,
    TileTensor,
    row_major,
)
from linalg.fp4_utils import E2M1_TO_FLOAT32
from linalg.mxfp4_dequant import dequant_mxfp4
from linalg.mxfp4_dequant import _cast_bf16_to_fp8


def _e8m0_to_float32(bits: UInt8) -> Float32:
    if bits == UInt8(0):
        return Float32(0.0)
    var f32_bits = UInt32(bits) << UInt32(23)
    return bitcast[.float32](f32_bits)


def test_mxfp4_matmul[
    M: Int, N: Int, K: Int
](ctx: DeviceContext,) raises:
    """Tests the MXFP4 matmul pipeline against vendor BLAS on shared FP8 data.

    Dequants and casts once, then feeds the same FP8 buffers to both our
    kernel (_matmul_gpu) and vendor BLAS. This isolates the GEMM comparison
    from any dequant non-determinism.
    """
    comptime packed_K = K // 2
    comptime scale_K = ceildiv(K, 32)
    comptime fp8_type = DType.float8_e4m3fn

    print(t"  M= {M}  N= {N}  K= {K}")

    # Device buffers for MXFP4 inputs
    var a_device = ctx.enqueue_create_buffer[.bfloat16](M * K)
    var b_packed_device = ctx.enqueue_create_buffer[.uint8](N * packed_K)
    var b_scales_device = ctx.enqueue_create_buffer[.float8_e8m0fnu](
        N * scale_K
    )
    # Fill A with random BF16 on host, upload
    with a_device.map_to_host() as ha:
        for i in range(M * K):
            ha[i] = random_float64(-0.5, 0.5).cast[.bfloat16]()

    # Fill B_packed with random uint8, upload
    with b_packed_device.map_to_host() as hbp:
        for i in range(N * packed_K):
            hbp[i] = UInt8(random_ui64(0, 255))

    # Fill scales with random exponents [125..129] -> scales [0.25..4.0]
    var bs_hbuf = ctx.enqueue_create_host_buffer[.float8_e8m0fnu](N * scale_K)
    for i in range(N * scale_K):
        bs_hbuf[i] = bitcast[.float8_e8m0fnu](UInt8(random_ui64(125, 129)))
    ctx.enqueue_copy(b_scales_device, bs_hbuf)
    ctx.synchronize()

    # Step 1: Dequant and cast ONCE to shared FP8 buffers
    # Use M (dynamic) to match mxfp4_dequant_matmul_amd's internal layout.
    var a_tt = TileTensor(a_device, row_major((M, Idx[K])))
    var b_packed_tt = TileTensor(b_packed_device, row_major[N, packed_K]())
    var b_scales_tt = TileTensor(b_scales_device, row_major[N, scale_K]())

    var b_fp8_device = ctx.enqueue_create_buffer[fp8_type](N * K)
    var a_fp8_device = ctx.enqueue_create_buffer[fp8_type](M * K)
    var b_fp8_tt = TileTensor(b_fp8_device, row_major((Idx[N], Idx[K])))
    var a_fp8_tt = TileTensor(a_fp8_device, row_major((M, Idx[K])))

    dequant_mxfp4(
        ctx,
        b_fp8_tt,
        b_packed_tt,
        b_scales_tt,
        num_rows=N,
        num_cols=K,
    )
    _cast_bf16_to_fp8(ctx, a_fp8_tt, a_tt, M, K)
    ctx.synchronize()

    # Step 2: Run our kernel on the shared FP8 data
    var c_device = ctx.enqueue_create_buffer[.bfloat16](M * N)
    var c_tt = TileTensor(c_device, row_major((M, Idx[N])))

    from linalg.matmul.gpu import _matmul_gpu

    _matmul_gpu[transpose_b=True](c_tt, a_fp8_tt, b_fp8_tt, ctx)
    ctx.synchronize()

    # Step 3: Run vendor BLAS on the same shared FP8 data
    var c_ref_device = ctx.enqueue_create_buffer[.bfloat16](M * N)
    var c_ref_tt = TileTensor(c_ref_device, row_major((M, Idx[N])))

    vendor_blas.matmul(
        ctx,
        c_ref_tt,
        a_fp8_tt,
        b_fp8_tt,
        c_row_major=True,
        transpose_b=True,
    )
    ctx.synchronize()

    # Step 4: Compare
    var c_host = ctx.enqueue_create_host_buffer[.bfloat16](M * N)
    var c_ref_host = ctx.enqueue_create_host_buffer[.bfloat16](M * N)
    ctx.enqueue_copy(c_host, c_device)
    ctx.enqueue_copy(c_ref_host, c_ref_device)
    ctx.synchronize()

    # Both paths use identical FP8 data from a single dequant/cast.
    # The FP8 baseline test shows <0.8% max_rel_err between kernel and
    # vendor BLAS. Use 2% threshold with margin.
    var num_mismatches = 0
    var max_rel_err = Float32(0.0)

    for i in range(M * N):
        var got = c_host[i].cast[.float32]()
        var expected = c_ref_host[i].cast[.float32]()
        var magnitude = max(abs(got), abs(expected))
        if magnitude < Float32(1.0):
            continue
        var rel_err = abs(got - expected) / magnitude
        max_rel_err = max(max_rel_err, rel_err)
        if rel_err > Float32(0.02):
            if num_mismatches < 5:
                var row, col = divmod(i, N)
                print(
                    t"    MISMATCH [ {row} , {col} ]: got= {got}  expected="
                    t" {expected}  rel_err= {rel_err}"
                )
            num_mismatches += 1

    if num_mismatches > 0:
        print(
            "    FAIL:",
            num_mismatches,
            "mismatches, max_rel_err=",
            max_rel_err,
        )
        raise Error("MXFP4 matmul test failed")

    print("    PASS max_rel_err=", max_rel_err)


def test_mxfp4_matmul_e2e[
    M: Int, N: Int, K: Int
](ctx: DeviceContext,) raises:
    """End-to-end test of mxfp4_dequant_matmul_amd against vendor BLAS reference.

    Dequants once into shared FP8 buffers, then runs vendor BLAS as reference.
    Separately calls mxfp4_dequant_matmul_amd on the original MXFP4 inputs,
    which runs the native block-scaled MMA on the still-packed weights.
    The dequant is deterministic so both paths operate on identical FP8 data.
    """
    comptime packed_K = K // 2
    comptime scale_K = ceildiv(K, 32)
    comptime fp8_type = DType.float8_e4m3fn

    print(t"  E2E M= {M}  N= {N}  K= {K}")

    var a_device = ctx.enqueue_create_buffer[.bfloat16](M * K)
    var b_packed_device = ctx.enqueue_create_buffer[.uint8](N * packed_K)
    var b_scales_device = ctx.enqueue_create_buffer[.float8_e8m0fnu](
        N * scale_K
    )

    with a_device.map_to_host() as ha:
        for i in range(M * K):
            ha[i] = random_float64(-0.5, 0.5).cast[.bfloat16]()
    with b_packed_device.map_to_host() as hbp:
        for i in range(N * packed_K):
            hbp[i] = UInt8(random_ui64(0, 255))
    var bs_hbuf = ctx.enqueue_create_host_buffer[.float8_e8m0fnu](N * scale_K)
    # The weight metadata the dispatcher gates on: the exponent range of the
    # block scales, computed here because this is where the scales are made.
    var min_scale_exp = 255
    var max_scale_exp = 0
    for i in range(N * scale_K):
        var scale_exp = UInt8(random_ui64(125, 129))
        bs_hbuf[i] = bitcast[.float8_e8m0fnu](scale_exp)
        min_scale_exp = min(min_scale_exp, Int(scale_exp))
        max_scale_exp = max(max_scale_exp, Int(scale_exp))

    ctx.enqueue_copy(b_scales_device, bs_hbuf)
    ctx.synchronize()

    # Use dynamic M (M) matching mxfp4_dequant_matmul_amd's internal layout.
    var a_tt = TileTensor(a_device, row_major((M, Idx[K])))
    var b_packed_tt = TileTensor(b_packed_device, row_major[N, packed_K]())
    var b_scales_tt = TileTensor(b_scales_device, row_major[N, scale_K]())

    # Reference: dequant + cast to FP8, then _matmul_gpu with same
    # dynamic-M layouts as mxfp4_dequant_matmul_amd uses internally.
    var b_fp8_ref = ctx.enqueue_create_buffer[fp8_type](N * K)
    var a_fp8_ref = ctx.enqueue_create_buffer[fp8_type](M * K)
    var b_fp8_tt = TileTensor(b_fp8_ref, row_major((Idx[N], Idx[K])))
    var a_fp8_tt = TileTensor(a_fp8_ref, row_major((M, Idx[K])))

    dequant_mxfp4(
        ctx,
        b_fp8_tt,
        b_packed_tt,
        b_scales_tt,
        num_rows=N,
        num_cols=K,
    )
    _cast_bf16_to_fp8(ctx, a_fp8_tt, a_tt, M, K)
    ctx.synchronize()

    var c_ref_device = ctx.enqueue_create_buffer[.bfloat16](M * N)
    var c_ref_tt = TileTensor(c_ref_device, row_major((M, Idx[N])))

    vendor_blas.matmul(
        ctx,
        c_ref_tt,
        a_fp8_tt,
        b_fp8_tt,
        c_row_major=True,
        transpose_b=True,
    )
    ctx.synchronize()

    # Kernel under test: mxfp4_dequant_matmul_amd (native block-scaled MMA)
    var c_device = ctx.enqueue_create_buffer[.bfloat16](M * N)
    var c_tt = TileTensor(c_device, row_major((M, Idx[N])))

    comptime if "gfx" in _accelerator_arch():
        from linalg.matmul.gpu.amd.mxfp4_dequant_matmul_amd import (
            mxfp4_dequant_matmul_amd,
        )

        mxfp4_dequant_matmul_amd(
            c_tt,
            a_tt,
            b_packed_tt,
            b_scales_tt,
            ctx,
            min_block_scale_exp=min_scale_exp,
            max_block_scale_exp=max_scale_exp,
        )
    elif "sm_90" in _accelerator_arch():
        from linalg.mxfp4_matmul_sm90 import mxfp4_matmul_sm90

        mxfp4_matmul_sm90(c_tt, a_tt, b_packed_tt, b_scales_tt, ctx)
    else:
        print("  SKIP: unsupported GPU architecture")
        return
    ctx.synchronize()

    # Compare mxfp4_dequant_matmul_amd vs vendor BLAS reference
    var c_host = ctx.enqueue_create_host_buffer[.bfloat16](M * N)
    var c_ref_host = ctx.enqueue_create_host_buffer[.bfloat16](M * N)
    ctx.enqueue_copy(c_host, c_device)
    ctx.enqueue_copy(c_ref_host, c_ref_device)
    ctx.synchronize()

    var max_rel_err = Float32(0.0)
    var num_mismatches = 0

    for i in range(M * N):
        var got = c_host[i].cast[.float32]()
        var expected = c_ref_host[i].cast[.float32]()

        var magnitude = max(abs(got), abs(expected))
        if magnitude < Float32(1.0):
            continue
        var rel_err = abs(got - expected) / magnitude
        max_rel_err = max(max_rel_err, rel_err)
        if rel_err > Float32(0.02):
            if num_mismatches < 5:
                var row, col = divmod(i, N)
                print(
                    t"    E2E MISMATCH [ {row} , {col} ]: got= {got}  expected="
                    t" {expected}  rel_err= {rel_err}"
                )
            num_mismatches += 1

    if num_mismatches > 0:
        print(
            "    FAIL:",
            num_mismatches,
            "mismatches, max_rel_err=",
            max_rel_err,
        )
        raise Error("MXFP4 E2E matmul test failed")

    print("    PASS max_rel_err=", max_rel_err)


def test_dequant_only[N: Int, K: Int](ctx: DeviceContext) raises:
    """Isolate the dequant step: verify GPU dequant matches CPU reference."""
    comptime packed_K = K // 2
    comptime scale_K = ceildiv(K, 32)

    print("  Dequant-only: N=", N, " K=", K)

    var b_packed_host = ctx.enqueue_create_host_buffer[.uint8](N * packed_K)
    var b_scales_host = ctx.enqueue_create_host_buffer[.uint8](N * scale_K)

    for i in range(N * packed_K):
        b_packed_host[i] = UInt8(random_ui64(0, 255))
    for i in range(N * scale_K):
        b_scales_host[i] = UInt8(random_ui64(125, 129))

    # Upload to GPU
    var bp_device = ctx.enqueue_create_buffer[.uint8](N * packed_K)
    var bs_device = ctx.enqueue_create_buffer[.float8_e8m0fnu](N * scale_K)
    var out_device = ctx.enqueue_create_buffer[.bfloat16](N * K)

    var bp_hbuf = ctx.enqueue_create_host_buffer[.uint8](N * packed_K)
    var bs_hbuf = ctx.enqueue_create_host_buffer[.float8_e8m0fnu](N * scale_K)
    for i in range(N * packed_K):
        bp_hbuf[i] = b_packed_host[i]
    for i in range(N * scale_K):
        bs_hbuf[i] = bitcast[.float8_e8m0fnu](b_scales_host[i])
    ctx.enqueue_copy(bp_device, bp_hbuf)
    ctx.enqueue_copy(bs_device, bs_hbuf)
    ctx.synchronize()

    var bp_tt = TileTensor(bp_device, row_major[N, packed_K]())
    var bs_tt = TileTensor(bs_device, row_major[N, scale_K]())
    var out_tt = TileTensor(out_device, row_major[N, K]())

    from linalg.mxfp4_dequant import dequant_mxfp4

    dequant_mxfp4(ctx, out_tt, bp_tt, bs_tt, num_rows=N, num_cols=K)
    ctx.synchronize()

    var out_hbuf = ctx.enqueue_create_host_buffer[.bfloat16](N * K)
    ctx.enqueue_copy(out_hbuf, out_device)
    ctx.synchronize()

    var mismatches = 0
    for row in range(N):
        for col in range(K):
            var got = out_hbuf[row * K + col].cast[.float32]()
            var packed_col = col // 2
            var packed_byte = b_packed_host[row * packed_K + packed_col]
            var nibble_shift = UInt8((col % 2) * 4)
            var fp4_bits = Int((packed_byte >> nibble_shift) & UInt8(0x0F))
            var fp4_val = E2M1_TO_FLOAT32[fp4_bits]
            var scale_col = col // 32
            var scale_byte = b_scales_host[row * scale_K + scale_col]
            var scale_f32 = _e8m0_to_float32(scale_byte)
            var expected = fp4_val * scale_f32
            if abs(got - expected) > 0.01:
                if mismatches < 5:
                    print(
                        t"    DEQUANT MISMATCH [ {row} , {col} ]: got= {got} "
                        t" expected= {expected}"
                    )
                mismatches += 1

    if mismatches > 0:
        print("    DEQUANT FAIL:", mismatches, "mismatches")
        raise Error("Dequant test failed")
    print("    DEQUANT PASS")


def test_fp8_kernel_vs_blas[
    M: Int, N: Int, K: Int
](ctx: DeviceContext,) raises:
    """Compare FP8 kernel vs vendor BLAS using MXFP4-dequanted B weights."""
    comptime fp8_type = DType.float8_e4m3fn
    comptime packed_K = K // 2
    comptime scale_K = ceildiv(K, 32)
    print(t"  FP8 kernel-vs-BLAS: M= {M}  N= {N}  K= {K}")

    # A: random BF16 activations cast to FP8
    var a_bf16 = ctx.enqueue_create_buffer[.bfloat16](M * K)
    with a_bf16.map_to_host() as ha:
        for i in range(M * K):
            ha[i] = random_float64(-0.5, 0.5).cast[.bfloat16]()

    var a_fp8 = ctx.enqueue_create_buffer[fp8_type](M * K)
    var a_bf16_tt = TileTensor(a_bf16, row_major((M, Idx[K])))
    var a_fp8_tt = TileTensor(a_fp8, row_major((M, Idx[K])))
    _cast_bf16_to_fp8(ctx, a_fp8_tt, a_bf16_tt, M, K)

    # B: MXFP4 packed weights dequanted to FP8
    var b_packed = ctx.enqueue_create_buffer[.uint8](N * packed_K)
    var b_scales = ctx.enqueue_create_buffer[.float8_e8m0fnu](N * scale_K)
    with b_packed.map_to_host() as hbp:
        for i in range(N * packed_K):
            hbp[i] = UInt8(random_ui64(0, 255))
    var bs_hbuf = ctx.enqueue_create_host_buffer[.float8_e8m0fnu](N * scale_K)
    for i in range(N * scale_K):
        bs_hbuf[i] = bitcast[.float8_e8m0fnu](UInt8(random_ui64(125, 129)))
    ctx.enqueue_copy(b_scales, bs_hbuf)
    ctx.synchronize()

    var b_fp8 = ctx.enqueue_create_buffer[fp8_type](N * K)
    var b_packed_tt = TileTensor(b_packed, row_major[N, packed_K]())
    var b_scales_tt = TileTensor(b_scales, row_major[N, scale_K]())
    var b_fp8_tt = TileTensor(b_fp8, row_major((Idx[N], Idx[K])))
    dequant_mxfp4(
        ctx,
        b_fp8_tt,
        b_packed_tt,
        b_scales_tt,
        num_rows=N,
        num_cols=K,
    )
    ctx.synchronize()

    # Path 1: our kernel
    var c_kernel = ctx.enqueue_create_buffer[.bfloat16](M * N)
    var c_kernel_tt = TileTensor(c_kernel, row_major((M, Idx[N])))
    from linalg.matmul.gpu import _matmul_gpu

    _matmul_gpu[transpose_b=True](c_kernel_tt, a_fp8_tt, b_fp8_tt, ctx)
    ctx.synchronize()

    # Path 2: vendor BLAS
    var c_blas = ctx.enqueue_create_buffer[.bfloat16](M * N)
    var c_blas_tt = TileTensor(c_blas, row_major((M, Idx[N])))
    vendor_blas.matmul(
        ctx,
        c_blas_tt,
        a_fp8_tt,
        b_fp8_tt,
        c_row_major=True,
        transpose_b=True,
    )
    ctx.synchronize()

    var c_k_host = ctx.enqueue_create_host_buffer[.bfloat16](M * N)
    var c_b_host = ctx.enqueue_create_host_buffer[.bfloat16](M * N)
    ctx.enqueue_copy(c_k_host, c_kernel)
    ctx.enqueue_copy(c_b_host, c_blas)
    ctx.synchronize()

    var max_rel_err = Float32(0.0)
    var num_mismatches = 0

    for i in range(M * N):
        var got = c_k_host[i].cast[.float32]()
        var expected = c_b_host[i].cast[.float32]()

        var magnitude = max(abs(got), abs(expected))
        if magnitude < Float32(1.0):
            continue
        var rel_err = abs(got - expected) / magnitude
        max_rel_err = max(max_rel_err, rel_err)
        if rel_err > Float32(0.02):
            if num_mismatches < 5:
                var row, col = divmod(i, N)
                print(
                    t"    [ {row} , {col} ]: kernel= {got}  blas= {expected} "
                    t" rel_err= {rel_err}"
                )
            num_mismatches += 1

    if num_mismatches > 0:
        print(
            "    FAIL:",
            num_mismatches,
            "mismatches, max_rel_err=",
            max_rel_err,
        )
        raise Error("FP8 kernel vs BLAS test failed")

    print("    PASS max_rel_err=", max_rel_err)


def test_n_alignment_gates_route[
    M: Int, N: Int, K: Int
](ctx: DeviceContext) raises:
    """Proves the native route refuses an N that leaves a partial last block.

    The dense block-scaled kernel writes N in 128-wide blocks and its non-fused
    store clamps only at the buffer end, so a partial last block would spill
    into the following row. A gate that tested only `N % 4 == 0` admitted those
    shapes and silently corrupted the output; this pins the routing predicate
    on both sides of the boundary.

    Parameters:
        M: Output rows; must be inside the native route's decode range.
        N: Output columns under test; the caller picks one aligned and one not.
        K: Reduction extent, a multiple of 128.
    """
    from linalg.matmul.gpu.amd.mxfp4_dequant_matmul_amd import (
        mxfp4_native_route_enabled,
    )

    comptime in_range_exp = 127  # 2^0, so only the shape can refuse the route
    print(t"  N-alignment gate: N=", N, " K=", K, " M=", M)

    var routed = mxfp4_native_route_enabled(in_range_exp, in_range_exp, M, N, K)
    comptime whole_blocks = N % 128 == 0
    if routed != whole_blocks:
        print("    FAIL: N=", N, " expected whole_blocks=", whole_blocks)
        raise Error("the native route gate does not require whole N blocks")
    print("    PASS: N=", N, " native=", routed)


def test_runtime_weight_shape_takes_dequant_route[
    M: Int, N: Int, K: Int
](ctx: DeviceContext) raises:
    """Proves a weight view with a runtime dimension takes the dequant route.

    The native branch reads `static_shape` off the weight view and the
    block-scaled kernel states it as a `comptime` assert, so a weight whose row
    count is only known at runtime cannot instantiate it at all: before the
    gate, this call failed to compile instead of falling back. Every runtime
    conjunct of the route predicate passes here, so only the view shape can
    divert the call, and the output must be the dequant route's exact value.

    Parameters:
        M: Output rows; inside the native route's decode range.
        N: Output columns, a multiple of 128.
        K: Reduction extent, a multiple of 128.
    """
    from linalg.matmul.gpu.amd.mxfp4_dequant_matmul_amd import (
        mxfp4_dequant_matmul_amd,
        mxfp4_native_route_enabled,
        mxfp4_native_views_supported,
    )

    comptime packed_K = K // 2
    comptime scale_K = K // 32
    comptime IN_RANGE_EXP = 127  # 2^0, so only the view can refuse the route
    comptime EXPECTED = Float32(K) * Float32(0.5)

    print(t"  Runtime-weight-shape gate: N=", N, " K=", K, " M=", M)

    var a_device = ctx.enqueue_create_buffer[.bfloat16](M * K)
    var b_packed_device = ctx.enqueue_create_buffer[.uint8](N * packed_K)
    var b_scales_device = ctx.enqueue_create_buffer[.float8_e8m0fnu](
        N * scale_K
    )
    with a_device.map_to_host() as ha:
        for i in range(M * K):
            ha[i] = 1.0
    with b_packed_device.map_to_host() as hbp:
        # 0x1 is E2M1's 0.5 in both nibbles.
        for i in range(N * packed_K):
            hbp[i] = UInt8(0x11)
    var bs_hbuf = ctx.enqueue_create_host_buffer[.float8_e8m0fnu](N * scale_K)
    for i in range(N * scale_K):
        bs_hbuf[i] = bitcast[.float8_e8m0fnu](UInt8(IN_RANGE_EXP))
    ctx.enqueue_copy(b_scales_device, bs_hbuf)
    ctx.synchronize()

    var a_tt = TileTensor(a_device, row_major((M, Idx[K])))
    var b_scales_tt = TileTensor(b_scales_device, row_major[N, scale_K]())
    # Same buffer and element type, but the row extent is a runtime value.
    var n_rows = Int(N)
    var b_packed_runtime = TileTensor(
        b_packed_device, row_major((n_rows, Idx[packed_K]))
    )

    if not mxfp4_native_views_supported[
        RowMajorLayout[ComptimeInt[N], ComptimeInt[packed_K]],
        type_of(b_scales_tt).LayoutType,
        scale_K,
    ]():
        print("    FAIL: a static weight and dense scales must be supported")
        raise Error("the view gate refuses the native route's own views")
    if mxfp4_native_views_supported[
        type_of(b_packed_runtime).LayoutType,
        type_of(b_scales_tt).LayoutType,
        scale_K,
    ]():
        print("    FAIL: a runtime weight dimension must not be supported")
        raise Error("the view gate admits a runtime-shaped weight view")
    if not mxfp4_native_route_enabled(IN_RANGE_EXP, IN_RANGE_EXP, M, N, K):
        print("    FAIL: the runtime conjuncts should admit the native route")
        raise Error("the route predicate refuses an in-range decode shape")

    var c_device = ctx.enqueue_create_buffer[.bfloat16](M * N)
    var c_tt = TileTensor(c_device, row_major((M, Idx[N])))
    mxfp4_dequant_matmul_amd(
        c_tt,
        a_tt,
        b_packed_runtime,
        b_scales_tt,
        ctx,
        min_block_scale_exp=IN_RANGE_EXP,
        max_block_scale_exp=IN_RANGE_EXP,
    )
    ctx.synchronize()

    var c_host = ctx.enqueue_create_host_buffer[.bfloat16](M * N)
    ctx.enqueue_copy(c_host, c_device)
    ctx.synchronize()

    var max_err = Float32(0.0)
    for i in range(M * N):
        max_err = max(max_err, abs(c_host[i].cast[.float32]() - EXPECTED))
    if max_err > Float32(1e-4):
        print(
            "    FAIL: expected",
            EXPECTED,
            "from the dequant route, max_err=",
            max_err,
        )
        raise Error("a runtime-shaped weight view did not take dequant route")
    print("    PASS: dequant route, max_err=", max_err)


def test_strided_scale_view_takes_dequant_route[
    M: Int, N: Int, K: Int
](ctx: DeviceContext) raises:
    """Proves a strided scale view takes the dequant route.

    The native scale loader reshapes `b_scales` into a flat row-major buffer,
    so it would read a transposed view as if it were dense and apply another
    block's scale to each column. The gate must refuse that view, and the
    dequant route, which indexes the view through its strides, must reproduce
    the value the scales actually encode.

    The first half of the K blocks carries 2^0 and the second half 2^-1, so a
    contiguous misread changes the row sum from `16 * half + 8 * (K//32 - half)`
    to `16 * K//32`.

    Parameters:
        M: Output rows; inside the native route's decode range.
        N: Output columns, a multiple of 128.
        K: Reduction extent, a multiple of 128 with an even block count.
    """
    from linalg.matmul.gpu.amd.mxfp4_dequant_matmul_amd import (
        mxfp4_dequant_matmul_amd,
        mxfp4_native_route_enabled,
        mxfp4_native_views_supported,
    )

    comptime packed_K = K // 2
    comptime scale_K = K // 32
    comptime half_blocks = scale_K // 2
    comptime UNIT_EXP = 127  # 2^0
    comptime HALF_EXP = 126  # 2^-1
    # Weights are 0.5, so a 2^0 block adds 32 * 0.5 and a 2^-1 block 32 * 0.25.
    comptime EXPECTED = (
        Float32(32.0 * 0.5) * Float32(half_blocks)
        + Float32(32.0 * 0.25) * Float32(scale_K - half_blocks)
    )

    print(t"  Strided-scale gate: N=", N, " K=", K, " M=", M)

    var a_device = ctx.enqueue_create_buffer[.bfloat16](M * K)
    var b_packed_device = ctx.enqueue_create_buffer[.uint8](N * packed_K)
    # Scales stored column-major, i.e. as a [scale_K, N] row-major buffer, so
    # the [N, scale_K] view of it is transposed: stride (1, N).
    var b_scales_device = ctx.enqueue_create_buffer[.float8_e8m0fnu](
        scale_K * N
    )
    with a_device.map_to_host() as ha:
        for i in range(M * K):
            ha[i] = 1.0
    with b_packed_device.map_to_host() as hbp:
        for i in range(N * packed_K):
            hbp[i] = UInt8(0x11)

    var bs_hbuf = ctx.enqueue_create_host_buffer[.float8_e8m0fnu](scale_K * N)
    for k in range(scale_K):
        var exp_byte = UNIT_EXP
        if k >= half_blocks:
            exp_byte = HALF_EXP
        for n in range(N):
            bs_hbuf[k * N + n] = bitcast[.float8_e8m0fnu](UInt8(exp_byte))
    ctx.enqueue_copy(b_scales_device, bs_hbuf)
    ctx.synchronize()

    var a_tt = TileTensor(a_device, row_major((M, Idx[K])))
    var b_packed_tt = TileTensor(b_packed_device, row_major[N, packed_K]())
    var b_scales_col_major = TileTensor(
        b_scales_device, row_major[scale_K, N]()
    )
    var b_scales_strided = b_scales_col_major.transpose()

    if not mxfp4_native_views_supported[
        type_of(b_packed_tt).LayoutType,
        RowMajorLayout[ComptimeInt[N], ComptimeInt[scale_K]],
        scale_K,
    ]():
        print("    FAIL: a dense scale view must be supported")
        raise Error("the view gate refuses the native route's own scale view")
    if mxfp4_native_views_supported[
        type_of(b_packed_tt).LayoutType,
        type_of(b_scales_strided).LayoutType,
        scale_K,
    ]():
        print("    FAIL: a transposed scale view must not be supported")
        raise Error("the view gate admits a strided scale view")
    if not mxfp4_native_route_enabled(UNIT_EXP, UNIT_EXP, M, N, K):
        print("    FAIL: the runtime conjuncts should admit the native route")
        raise Error("the route predicate refuses an in-range decode shape")

    var c_device = ctx.enqueue_create_buffer[.bfloat16](M * N)
    var c_tt = TileTensor(c_device, row_major((M, Idx[N])))
    mxfp4_dequant_matmul_amd(
        c_tt,
        a_tt,
        b_packed_tt,
        b_scales_strided,
        ctx,
        min_block_scale_exp=UNIT_EXP,
        max_block_scale_exp=UNIT_EXP,
    )
    ctx.synchronize()

    var c_host = ctx.enqueue_create_host_buffer[.bfloat16](M * N)
    ctx.enqueue_copy(c_host, c_device)
    ctx.synchronize()

    var max_err = Float32(0.0)
    for i in range(M * N):
        max_err = max(max_err, abs(c_host[i].cast[.float32]() - EXPECTED))
    if max_err > Float32(1e-4):
        print(
            "    FAIL: expected",
            EXPECTED,
            "from the strided scales, max_err=",
            max_err,
        )
        raise Error("a strided scale view did not take the dequant route")
    print("    PASS: dequant route, max_err=", max_err)


def test_scale_range_gates_route[
    M: Int, N: Int, K: Int
](ctx: DeviceContext) raises:
    """Proves the scale bound gates the native route instead of being advised.

    Weights are 0.5 (E2M1) and activations are 1.0, so the output of a
    K-wide dot product is exactly `K * 0.5 * 2^(exp - 127)` whatever route
    runs, and the two routes are told apart by that value alone:

      * block scale 2^-10 puts each product at 2^-11, below E4M3's smallest
        subnormal 2^-9. The dequant route rounds it to zero; the MMA would
        form it. The dispatcher must refuse the native route here, and the
        output must be 0.0.
      * block scale 2^0 puts each product at 0.5, inside E4M3. The
        dispatcher must take the native route, and the output must be
        `K * 0.5`.

    Both the routing predicate and the output are asserted, so neither a
    loosened gate nor a changed rounding can pass silently.

    Parameters:
        M: Output rows; must be inside the native route's decode range.
        N: Output columns.
        K: Reduction extent in logical elements.
    """
    from linalg.matmul.gpu import _matmul_gpu
    from linalg.matmul.gpu.amd.mxfp4_dequant_matmul_amd import (
        mxfp4_dequant_matmul_amd,
        mxfp4_native_route_enabled,
    )

    comptime scale_K = ceildiv(K, 32)
    comptime fp8_type = DType.float8_e4m3fn
    comptime IN_RANGE_EXPECTED = Float32(K) * Float32(0.5)
    comptime OUT_OF_RANGE_EXPECTED = Float32(0.0)
    comptime OUT_OF_RANGE_EXP = 117  # 2^-10
    comptime IN_RANGE_EXP = 127  # 2^0

    print(t"  Scale-range gate: N=", N, " K=", K, " M=", M)

    var a_device = ctx.enqueue_create_buffer[.bfloat16](M * K)
    var b_packed_device = ctx.enqueue_create_buffer[.uint8](N * (K // 2))
    with a_device.map_to_host() as ha:
        for i in range(M * K):
            ha[i] = 1.0
    with b_packed_device.map_to_host() as hbp:
        # 0x1 is E2M1's 0.5 in both nibbles.
        for i in range(N * (K // 2)):
            hbp[i] = UInt8(0x11)

    var a_tt = TileTensor(a_device, row_major((M, Idx[K])))
    var b_packed_tt = TileTensor(b_packed_device, row_major[N, K // 2]())

    # The gate itself, for both scale ranges, before any data is launched.
    if mxfp4_native_route_enabled(OUT_OF_RANGE_EXP, OUT_OF_RANGE_EXP, M, N, K):
        print("    FAIL: out-of-range scales must not take the native route")
        raise Error("scale-range gate admitted an out-of-range weight")
    if not mxfp4_native_route_enabled(IN_RANGE_EXP, IN_RANGE_EXP, M, N, K):
        print("    FAIL: in-range scales should take the native route")
        raise Error("scale-range gate refused an in-range weight")

    # Same weights and activations through both dispatcher states.
    for which in range(2):
        var scale_exp = IN_RANGE_EXP
        if which == 0:
            scale_exp = OUT_OF_RANGE_EXP
        var b_scales_device = ctx.enqueue_create_buffer[.float8_e8m0fnu](
            N * scale_K
        )
        var bs_hbuf = ctx.enqueue_create_host_buffer[.float8_e8m0fnu](
            N * scale_K
        )
        for i in range(N * scale_K):
            bs_hbuf[i] = bitcast[.float8_e8m0fnu](UInt8(scale_exp))
        ctx.enqueue_copy(b_scales_device, bs_hbuf)
        ctx.synchronize()
        var b_scales_tt = TileTensor(b_scales_device, row_major[N, scale_K]())

        var c_device = ctx.enqueue_create_buffer[.bfloat16](M * N)
        var c_tt = TileTensor(c_device, row_major((M, Idx[N])))
        mxfp4_dequant_matmul_amd(
            c_tt,
            a_tt,
            b_packed_tt,
            b_scales_tt,
            ctx,
            min_block_scale_exp=scale_exp,
            max_block_scale_exp=scale_exp,
        )

        # The dequant route to compare against, built by hand.
        var b_fp8 = ctx.enqueue_create_buffer[fp8_type](N * K)
        var a_fp8 = ctx.enqueue_create_buffer[fp8_type](M * K)
        var b_fp8_tt = TileTensor(b_fp8, row_major((Idx[N], Idx[K])))
        var a_fp8_tt = TileTensor(a_fp8, row_major((M, Idx[K])))
        dequant_mxfp4(
            ctx,
            b_fp8_tt,
            b_packed_tt,
            b_scales_tt,
            num_rows=N,
            num_cols=K,
        )
        _cast_bf16_to_fp8(ctx, a_fp8_tt, a_tt, M, K)
        var c_ref_device = ctx.enqueue_create_buffer[.bfloat16](M * N)
        var c_ref_tt = TileTensor(c_ref_device, row_major((M, Idx[N])))
        _matmul_gpu[use_tensor_core=True, transpose_b=True](
            c_ref_tt, a_fp8_tt, b_fp8_tt, ctx
        )
        ctx.synchronize()

        var c_host = ctx.enqueue_create_host_buffer[.bfloat16](M * N)
        var c_ref_host = ctx.enqueue_create_host_buffer[.bfloat16](M * N)
        ctx.enqueue_copy(c_host, c_device)
        ctx.enqueue_copy(c_ref_host, c_ref_device)
        ctx.synchronize()

        var expected = OUT_OF_RANGE_EXPECTED
        if scale_exp == IN_RANGE_EXP:
            expected = IN_RANGE_EXPECTED
        var max_err = Float32(0.0)
        var max_ref_err = Float32(0.0)
        for i in range(M * N):
            max_err = max(max_err, abs(c_host[i].cast[.float32]() - expected))
            max_ref_err = max(
                max_ref_err, abs(c_ref_host[i].cast[.float32]() - expected)
            )

        if max_err > Float32(1e-4) or max_ref_err > Float32(1e-4):
            print(
                "    FAIL: scale_exp=",
                scale_exp,
                "expected",
                expected,
                "got max_err=",
                max_err,
                "dequant max_err=",
                max_ref_err,
            )
            raise Error("routed value disagrees with its expected route")

        var route = String("dequant, E4M3 flush")
        if scale_exp == IN_RANGE_EXP:
            route = String("native")
        print(
            "    PASS scale_exp=",
            scale_exp,
            "routed=",
            expected,
            " route=",
            route,
            " dequant_ref_err=",
            max_ref_err,
        )


def main() raises:
    with DeviceContext() as ctx:
        print("MXFP4 Matmul Smoke Tests")
        print("arch:", _accelerator_arch())
        print("========================")

        print("-- Dequant isolation --")
        test_dequant_only[256, 256](ctx)
        test_dequant_only[4096, 4096](ctx)

        # Baseline: FP8 kernel vs vendor BLAS at same K values
        # This shows the inherent error of the FP8 GEMM kernel itself
        print("-- FP8 kernel vs vendor BLAS (baseline) --")
        test_fp8_kernel_vs_blas[256, 256, 256](ctx)
        test_fp8_kernel_vs_blas[512, 256, 512](ctx)
        test_fp8_kernel_vs_blas[256, 2048, 2048](ctx)

        # Shared-buffer MXFP4 matmul (isolates GEMM from dequant)
        print("-- MXFP4 matmul (shared FP8 buffers) --")
        test_mxfp4_matmul[256, 256, 256](ctx)
        test_mxfp4_matmul[512, 256, 512](ctx)
        test_mxfp4_matmul[256, 2048, 2048](ctx)

        # End-to-end: calls mxfp4_dequant_matmul_amd (native block-scaled MMA)
        print("-- MXFP4 matmul E2E (full pipeline) --")
        test_mxfp4_matmul_e2e[256, 256, 256](ctx)
        test_mxfp4_matmul_e2e[256, 2048, 2048](ctx)
        # M-band coverage: the dispatcher picks a different tile per M regime,
        # so each band needs an end-to-end check, not just the dense one.
        test_mxfp4_matmul_e2e[1, 2048, 2048](ctx)  # decode, split-K
        test_mxfp4_matmul_e2e[64, 2048, 2048](ctx)  # mid-M, split-K band
        test_mxfp4_matmul_e2e[512, 2304, 4096](ctx)  # wide-N split-K band

        # gfx950 only: the native route exists on no other part, so the
        # divergence it pins cannot be observed elsewhere.
        comptime if "gfx950" in _accelerator_arch():
            print("-- Native route precondition (out-of-range scale) --")
            test_scale_range_gates_route[128, 128, 128](ctx)
            # 192 = 128 + 64: a multiple of 4 but not of the 128-wide store
            # block, the shape class the gate used to admit.
            test_n_alignment_gates_route[128, 192, 256](ctx)
            test_n_alignment_gates_route[128, 256, 256](ctx)

            # The view half of the gate: neither case can instantiate the
            # native branch, so both must fall back to the dequant route.
            print("-- Native route view gate --")
            test_runtime_weight_shape_takes_dequant_route[128, 128, 256](ctx)
            test_strided_scale_view_takes_dequant_route[128, 128, 256](ctx)

        print("========================")
        print("ALL TESTS PASSED")
