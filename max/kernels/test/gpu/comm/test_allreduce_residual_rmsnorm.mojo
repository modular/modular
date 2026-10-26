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

"""Tests for the fully-fused allreduce + RMSNorm (+ optional FP8) kernel."""

from std.math import ceildiv, rsqrt
from std.random import random_si64
from std.sys import is_amd_gpu, size_of

from std.memory import bitcast

from comm import Signal, group_start, group_end
from comm.allreduce_lamport_rmsnorm import (
    _lamport_allreduce_rmsnorm_launch,
    lamport_allreduce_rmsnorm_fits,
)
from comm.allreduce_residual_rmsnorm import (
    allreduce_residual_rmsnorm,
    allreduce_rmsnorm,
)
from comm.sync import enable_p2p, init_signal_buffer, is_p2p_enabled
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from layout import (
    Coord,
    Idx,
    TileTensor,
    row_major,
)
from nn.normalization import rms_norm_fused_fp8

from std.testing import assert_true
from std.utils.index import Index
from std.utils.numerics import max_finite

from internal_utils._testing import test_value_for_gpu_element

# Platform-agnostic FP8 output type.
comptime out_fp8_dtype = DType.float8_e4m3fnuz if is_amd_gpu() else DType.float8_e4m3fn


# --- Shared verification helpers ---


def _assert_fp8_close[
    out_dtype: DType,
](
    ref_host: Pointer[Scalar[out_dtype], _],
    fused_host: Pointer[Scalar[out_dtype], _],
    length: Int,
    *,
    max_error_rate: Float32 = 0.05,
) raises:
    """Assert two FP8 host buffers match within ±1 ULP and an error rate cap.

    Every mismatch must be within ±1 FP8 ULP (bit-level). The total
    mismatch rate must not exceed `max_error_rate` (default 5%).
    """
    var num_errors = 0
    var num_ulp_errors = 0
    for i in range(length):
        var ref_val = ref_host[i].cast[.float32]()
        var fused_val = fused_host[i].cast[.float32]()
        if ref_val != fused_val:
            num_errors += 1
            var ref_bits = Int(bitcast[.uint8](ref_host[i]))
            var fused_bits = Int(bitcast[.uint8](fused_host[i]))
            if abs(ref_bits - fused_bits) > 1:
                num_ulp_errors += 1

    if num_ulp_errors > 0:
        raise Error(t"FP8 mismatches exceed 1 ULP: {num_ulp_errors} / {length}")

    var error_rate = Float32(num_errors) / Float32(length)
    if error_rate > max_error_rate:
        var printed = 0
        for i in range(length):
            if printed >= 5:
                break
            var ref_val = ref_host[i].cast[.float32]()
            var fused_val = fused_host[i].cast[.float32]()
            if ref_val != fused_val:
                print(
                    "  FP8 mismatch at",
                    i,
                    ": ref=",
                    ref_val,
                    ", fused=",
                    fused_val,
                )
                printed += 1
        raise Error(
            t"Too many FP8 mismatches: {num_errors} /"
            t" {length} ({error_rate * 100.0}%)"
        )


def _assert_scales_close[
    scales_dtype: DType,
](
    ref_host: Pointer[Scalar[scales_dtype], _],
    fused_host: Pointer[Scalar[scales_dtype], _],
    rows: Int,
    *,
    max_rel_diff: Float32 = 0.005,
) raises:
    """Assert per-row FP8 scale factors match within a relative tolerance.

    Default tolerance is 0.5% relative difference, which accommodates
    the minor divergence from bf16 vs f32 intermediate precision.
    """
    var scale_errors = 0
    for i in range(rows):
        var ref_s = ref_host[i].cast[.float32]()
        var fused_s = fused_host[i].cast[.float32]()
        var denom = max(abs(ref_s), Float32(1e-12))
        var rel_diff = abs(ref_s - fused_s) / denom
        if rel_diff > max_rel_diff:
            scale_errors += 1
            if scale_errors <= 5:
                print(
                    "  Scale mismatch at row",
                    i,
                    ": ref=",
                    ref_s,
                    ", fused=",
                    fused_s,
                    ", rel_diff=",
                    rel_diff,
                )

    if scale_errors > 0:
        raise Error(t"Scale factor mismatches: {scale_errors} / {rows}")


def _assert_bf16_close[
    dtype: DType,
](
    ref_host: Pointer[Scalar[dtype], _],
    fused_host: Pointer[Scalar[dtype], _],
    length: Int,
    *,
    max_ulp: Int = 2,
    max_error_rate: Float32 = 0.05,
) raises:
    """Assert two `dtype` host buffers match within `max_ulp` and a rate cap.

    Used for the no-quant path where the output is written in the input dtype
    (e.g. bf16) rather than quantized to FP8. Both the kernel and the host
    reference normalize in float32 and cast once, so the only expected
    divergence is reduction-order rounding (a small number of ±ULP mismatches).
    """
    var num_errors = 0
    var num_ulp_errors = 0
    for i in range(length):
        var ref_val = ref_host[i].cast[.float32]()
        var fused_val = fused_host[i].cast[.float32]()
        if ref_val != fused_val:
            num_errors += 1
            var ref_bits = Int(bitcast[.uint16](ref_host[i]))
            var fused_bits = Int(bitcast[.uint16](fused_host[i]))
            if abs(ref_bits - fused_bits) > max_ulp:
                num_ulp_errors += 1
                if num_ulp_errors <= 5:
                    print(
                        "  bf16 mismatch at",
                        i,
                        ": ref=",
                        ref_val,
                        ", fused=",
                        fused_val,
                    )

    if num_ulp_errors > 0:
        raise Error(
            t"bf16 mismatches exceed {max_ulp} ULP: {num_ulp_errors} / {length}"
        )

    var error_rate = Float32(num_errors) / Float32(length)
    if error_rate > max_error_rate:
        raise Error(
            t"Too many bf16 mismatches: {num_errors} /"
            t" {length} ({error_rate * 100.0}%)"
        )


def _assert_cast_order[
    dtype: DType, multiply_before_cast: Bool
](
    ref_same: Pointer[Scalar[dtype], _],
    ref_other: Pointer[Scalar[dtype], _],
    got: Pointer[Scalar[dtype], _],
    length: Int,
) raises:
    """Assert `got` used `multiply_before_cast`'s cast order, not the other.

    The two orders differ by about an ulp per element, so a per-element
    tolerance passes either. What separates them is how often they disagree:
    the output must match its own reference on all but 1% of elements and
    disagree with the other order's on more than 1%. Two ulps are allowed on
    the stragglers because the cast-first order rounds twice, and gamma scales
    a flipped intermediate rounding.
    """
    _assert_bf16_close(ref_same, got, length, max_ulp=2, max_error_rate=0.01)
    var other_mismatches = 0
    for i in range(length):
        if got[i] != ref_other[i]:
            other_mismatches += 1
    if Float32(other_mismatches) / Float32(length) <= 0.01:
        comptime other = not multiply_before_cast
        raise Error(
            t"output also matches multiply_before_cast={other}"
            t" ({other_mismatches} / {length} differ)"
        )


def _rms_norm_ref[
    dtype: DType, multiply_before_cast: Bool
](sum: Float32, norm: Float32, gamma: Scalar[dtype]) -> Scalar[dtype]:
    """Normalizes one element the way `ops.rms_norm` does for the cast order."""
    comptime if multiply_before_cast:
        return (sum * norm * gamma.cast[.float32]()).cast[dtype]()
    else:
        return (sum * norm).cast[dtype]() * gamma


# --- Test: fully fused allreduce + RMSNorm + FP8 ---


def test_fused_allreduce_rmsnorm_fp8[
    ngpus: Int,
    in_dtype: DType,
    out_dtype: DType,
    rows: Int,
    cols: Int,
    scales_dtype: DType = .float32,
](list_of_ctx: List[DeviceContext]) raises:
    """Verify fused kernel against separate allreduce → fused RMSNorm+FP8."""
    comptime length = rows * cols

    print(
        "  test_fused_allreduce_rmsnorm_fp8[",
        ngpus,
        ",",
        in_dtype,
        "->",
        out_dtype,
        ",",
        rows,
        "x",
        cols,
        ", scales=",
        scales_dtype,
        "]",
    )

    # --- Setup: per-GPU input buffers ---
    var in_dev = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    var host_bufs = List[HostBuffer[in_dtype]](capacity=ngpus)
    var signal_buffers = List[DeviceBuffer[.uint8]](capacity=ngpus)
    var temp_bytes = ngpus * size_of[in_dtype]() * length

    for i in range(ngpus):
        in_dev.append(list_of_ctx[i].enqueue_create_buffer[in_dtype](length))
        var h = list_of_ctx[i].enqueue_create_host_buffer[in_dtype](length)
        for j in range(length):
            h[j] = test_value_for_gpu_element[in_dtype](i, j)
        list_of_ctx[i].enqueue_copy(in_dev[i], h)
        host_bufs.append(h^)

        signal_buffers.append(
            list_of_ctx[i].create_buffer_sync[.uint8](
                size_of[Signal]() + temp_bytes
            )
        )
        init_signal_buffer(signal_buffers[i], list_of_ctx[i])

    var rank_sigs = Array[_, ngpus](
        fill_with=lambda (i: Int) {ref} -> MutPointer[
            Signal, MutAnyOrigin
        ]: Signal.unsafe_ptr_from(signal_buffers[i])
    )

    comptime in_layout = row_major(Idx[rows], Idx[cols])
    comptime InputTileType = type_of(
        TileTensor(in_dev[0], in_layout).as_unsafe_any_origin().as_imm()
    )
    var in_tiles = Array[_, ngpus](
        fill_with=lambda (i: Int) -> InputTileType: TileTensor(
            in_dev[i], in_layout
        )
        .as_unsafe_any_origin()
        .as_imm()
    )
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    # --- Shared params ---
    var ctx = list_of_ctx[0]
    var gamma_host = ctx.enqueue_create_host_buffer[in_dtype](cols)
    for i in range(cols):
        gamma_host[i] = (Float64(i + cols) / Float64(cols)).cast[in_dtype]()
    var gamma_dev = ctx.enqueue_create_buffer[in_dtype](cols)
    ctx.enqueue_copy(gamma_dev, gamma_host)

    comptime shape = (rows, cols)
    var gamma_tensor = TileTensor(gamma_dev, row_major(cols))
    var epsilon = Float32(1e-5)
    var weight_offset = Scalar[in_dtype](0.0)
    var scale_ub = max_finite[out_dtype]().cast[.float32]()

    # --- Reference path: host-side float32 sum → fused RMSNorm+FP8 ---
    # The fused kernel accumulates the P2P loads in float32, then casts to
    # bf16 internally. We replicate that here to avoid the bf16 rounding
    # that a separate allreduce would introduce.
    var ref_sum_host = ctx.enqueue_create_host_buffer[in_dtype](length)
    for i in range(length):
        var sum_f32 = Float32(0)
        for g in range(ngpus):
            sum_f32 += host_bufs[g][i].cast[.float32]()
        ref_sum_host[i] = sum_f32.cast[in_dtype]()

    # Upload sum to device for the reference fused RMSNorm+FP8 input_fn.
    var ref_sum_dev = ctx.enqueue_create_buffer[in_dtype](length)
    ctx.enqueue_copy(ref_sum_dev, ref_sum_host)
    ctx.synchronize()

    var ref_fp8_dev = ctx.enqueue_create_buffer[out_dtype](length)
    var ref_scales_dev = ctx.enqueue_create_buffer[scales_dtype](rows)

    var ref_sum_ptr = ref_sum_dev.unsafe_ptr()

    @inline(.always)
    def ref_input_fn[
        width: Int
    ](coords: Coord) {var ref_sum_ptr} -> SIMD[in_dtype, width]:
        var linear_idx = Int(coords[0]) * cols + Int(coords[1])
        return ref_sum_ptr.unsafe_load[width=width, alignment=width](linear_idx)

    var ref_fp8_tile = TileTensor(
        ref_fp8_dev,
        row_major(Idx[rows], Idx[cols]),
    )
    var ref_scales_tile = TileTensor(
        ref_scales_dev,
        row_major(Idx[rows], Idx[1]),
    )

    rms_norm_fused_fp8[
        in_dtype,
        out_dtype,
        scales_dtype,
        2,
    ](
        ref_input_fn,
        shape,
        ref_fp8_tile,
        gamma_tensor,
        epsilon,
        weight_offset,
        ctx,
        scale_ub,
        ref_scales_tile,
    )

    ctx.synchronize()

    # --- Fused kernel path ---
    # Reset signal buffers for the fused kernel run.
    for i in range(ngpus):
        init_signal_buffer(signal_buffers[i], list_of_ctx[i])
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    var fused_fp8_dev = ctx.enqueue_create_buffer[out_dtype](length)
    var fused_scales_dev = ctx.enqueue_create_buffer[scales_dtype](rows)

    var fused_fp8_tile = TileTensor(
        fused_fp8_dev,
        row_major(Idx[rows], Idx[cols]),
    )
    var fused_scales_tile = TileTensor(
        fused_scales_dev,
        row_major(Idx[rows], Idx[1]),
    )

    group_start()

    comptime for i in range(ngpus):
        allreduce_rmsnorm(
            in_tiles,
            fused_fp8_tile,
            gamma_tensor,
            epsilon,
            weight_offset,
            scale_ub,
            fused_scales_tile,
            rank_sigs,
            list_of_ctx[i],
        )
    group_end()

    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    # --- Compare FP8 output: reference vs fused kernel ---
    var ref_fp8_host = ctx.enqueue_create_host_buffer[out_dtype](length)
    var fused_fp8_host = ctx.enqueue_create_host_buffer[out_dtype](length)
    ctx.enqueue_copy(ref_fp8_host, ref_fp8_dev)
    ctx.enqueue_copy(fused_fp8_host, fused_fp8_dev)
    ctx.synchronize()

    _assert_fp8_close(
        ref_fp8_host.unsafe_ptr(), fused_fp8_host.unsafe_ptr(), length
    )

    # --- Compare per-row scale factors ---
    var ref_scales_host = ctx.enqueue_create_host_buffer[scales_dtype](rows)
    var fused_scales_host = ctx.enqueue_create_host_buffer[scales_dtype](rows)
    ctx.enqueue_copy(ref_scales_host, ref_scales_dev)
    ctx.enqueue_copy(fused_scales_host, fused_scales_dev)
    ctx.synchronize()

    _assert_scales_close(
        ref_scales_host.unsafe_ptr(), fused_scales_host.unsafe_ptr(), rows
    )

    # Cleanup.
    _ = host_bufs^
    print("    PASS")


# --- Test: fully fused allreduce + RMSNorm (no quant) ---


def test_fused_allreduce_rmsnorm_noquant[
    ngpus: Int,
    dtype: DType,
    rows: Int,
    cols: Int,
](list_of_ctx: List[DeviceContext]) raises:
    """Verify the no-quant path (out_dtype == in_dtype) of the non-residual
    fused kernel.

    Exercises allreduce + RMSNorm with a `dtype` output (no FP8 quantization,
    no scale output) against a host-side float32 reference.
    """
    comptime length = rows * cols

    print(
        "  test_fused_allreduce_rmsnorm_noquant[",
        ngpus,
        ",",
        dtype,
        ",",
        rows,
        "x",
        cols,
        "]",
    )

    # --- Setup: per-GPU input buffers ---
    var in_dev = List[DeviceBuffer[dtype]](capacity=ngpus)
    var host_bufs = List[HostBuffer[dtype]](capacity=ngpus)
    var signal_buffers = List[DeviceBuffer[.uint8]](capacity=ngpus)
    var temp_bytes = ngpus * size_of[dtype]() * length

    for i in range(ngpus):
        in_dev.append(list_of_ctx[i].enqueue_create_buffer[dtype](length))
        var h = list_of_ctx[i].enqueue_create_host_buffer[dtype](length)
        for j in range(length):
            h[j] = test_value_for_gpu_element[dtype](i, j)
        list_of_ctx[i].enqueue_copy(in_dev[i], h)
        host_bufs.append(h^)

        signal_buffers.append(
            list_of_ctx[i].create_buffer_sync[.uint8](
                size_of[Signal]() + temp_bytes
            )
        )
        init_signal_buffer(signal_buffers[i], list_of_ctx[i])

    var rank_sigs = Array[_, ngpus](
        fill_with=lambda (i: Int) {ref} -> MutPointer[
            Signal, MutAnyOrigin
        ]: Signal.unsafe_ptr_from(signal_buffers[i])
    )

    comptime in_layout = row_major(Idx[rows], Idx[cols])
    comptime InputTileType = type_of(
        TileTensor(in_dev[0], in_layout).as_unsafe_any_origin().as_imm()
    )
    var in_tiles = Array[_, ngpus](
        fill_with=lambda (i: Int) -> InputTileType: TileTensor(
            in_dev[i], in_layout
        )
        .as_unsafe_any_origin()
        .as_imm()
    )
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    # --- Shared params ---
    var ctx = list_of_ctx[0]
    var gamma_host = ctx.enqueue_create_host_buffer[dtype](cols)
    for i in range(cols):
        gamma_host[i] = (Float64(i + cols) / Float64(cols)).cast[dtype]()
    var gamma_dev = ctx.enqueue_create_buffer[dtype](cols)
    ctx.enqueue_copy(gamma_dev, gamma_host)

    var gamma_tensor = TileTensor(gamma_dev, row_major(cols))
    var epsilon = Float32(1e-5)
    var weight_offset = Scalar[dtype](0.0)

    # --- Host reference: allreduce in f32, then RMSNorm in f32 ---
    var ref_sum_f32 = ctx.enqueue_create_host_buffer[.float32](length)
    for i in range(length):
        var s = Float32(0)
        for g in range(ngpus):
            s += host_bufs[g][i].cast[.float32]()
        ref_sum_f32[i] = s

    var ref_out_host = ctx.enqueue_create_host_buffer[dtype](length)
    var eps_f32 = epsilon
    var woff_f32 = weight_offset.cast[.float32]()
    for r in range(rows):
        var m2 = Float32(0)
        for c in range(cols):
            var s = ref_sum_f32[r * cols + c]
            m2 += s * s
        var norm = rsqrt(m2 / Float32(cols) + eps_f32)
        for c in range(cols):
            var s = ref_sum_f32[r * cols + c]
            var g_f32 = gamma_host[c].cast[.float32]() + woff_f32
            ref_out_host[r * cols + c] = ((s * norm) * g_f32).cast[dtype]()

    # --- Fused kernel path (out_dtype == in_dtype → no quantization) ---
    for i in range(ngpus):
        init_signal_buffer(signal_buffers[i], list_of_ctx[i])
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    var fused_out_dev = ctx.enqueue_create_buffer[dtype](length)
    # Scale buffer + scale_ub are ignored on the no-quant path; pass a dummy.
    var dummy_scales_dev = ctx.enqueue_create_buffer[.float32](rows)

    var fused_out_tile = TileTensor(
        fused_out_dev, row_major(Idx[rows], Idx[cols])
    )
    var dummy_scales_tile = TileTensor(
        dummy_scales_dev, row_major(Idx[rows], Idx[1])
    )
    var scale_ub = Float32(1.0)

    group_start()

    comptime for i in range(ngpus):
        allreduce_rmsnorm(
            in_tiles,
            fused_out_tile,
            gamma_tensor,
            epsilon,
            weight_offset,
            scale_ub,
            dummy_scales_tile,
            rank_sigs,
            list_of_ctx[i],
        )
    group_end()

    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    # --- Verify normalized output (no quant, written in `dtype`) ---
    var fused_out_host = ctx.enqueue_create_host_buffer[dtype](length)
    ctx.enqueue_copy(fused_out_host, fused_out_dev)
    ctx.synchronize()

    _assert_bf16_close(
        ref_out_host.unsafe_ptr(), fused_out_host.unsafe_ptr(), length
    )

    # Cleanup.
    _ = host_bufs^
    _ = signal_buffers^
    _ = in_dev^
    _ = gamma_dev^
    _ = fused_out_dev^
    _ = dummy_scales_dev^

    print("    PASS")


# --- Test: fully fused allreduce + residual add + RMSNorm + FP8 ---


def test_fused_allreduce_residual_rmsnorm_fp8[
    ngpus: Int,
    in_dtype: DType,
    out_dtype: DType,
    rows: Int,
    cols: Int,
    *,
    no_scratch: Bool = False,
](list_of_ctx: List[DeviceContext]) raises:
    """Verify fused kernel with residual add against separate paths.

    With `no_scratch` the signal buffers end at the `Signal` header, so
    payloads the dispatch would stage through scratch must take 1-stage.
    """
    comptime length = rows * cols

    print(
        "  test_fused_allreduce_residual_rmsnorm_fp8[",
        ngpus,
        ",",
        in_dtype,
        "->",
        out_dtype,
        ",",
        rows,
        "x",
        cols,
        ", no_scratch=",
        no_scratch,
        "]",
    )

    # --- Setup: per-GPU input buffers ---
    var in_dev = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    var host_bufs = List[HostBuffer[in_dtype]](capacity=ngpus)
    var signal_buffers = List[DeviceBuffer[.uint8]](capacity=ngpus)
    var temp_bytes = 0 if no_scratch else ngpus * size_of[in_dtype]() * length
    var scratch_bytes: Optional[Int] = None
    comptime if no_scratch:
        scratch_bytes = 0

    for i in range(ngpus):
        in_dev.append(list_of_ctx[i].enqueue_create_buffer[in_dtype](length))
        var h = list_of_ctx[i].enqueue_create_host_buffer[in_dtype](length)
        for j in range(length):
            h[j] = test_value_for_gpu_element[in_dtype](i, j)
        list_of_ctx[i].enqueue_copy(in_dev[i], h)
        host_bufs.append(h^)

        signal_buffers.append(
            list_of_ctx[i].create_buffer_sync[.uint8](
                size_of[Signal]() + temp_bytes
            )
        )
        init_signal_buffer(signal_buffers[i], list_of_ctx[i])

    var rank_sigs = Array[_, ngpus](
        fill_with=lambda (i: Int) {ref} -> MutPointer[
            Signal, MutAnyOrigin
        ]: Signal.unsafe_ptr_from(signal_buffers[i])
    )

    comptime in_layout = row_major(Idx[rows], Idx[cols])
    comptime InputTileType = type_of(
        TileTensor(in_dev[0], in_layout).as_unsafe_any_origin().as_imm()
    )
    var in_tiles = Array[_, ngpus](
        fill_with=lambda (i: Int) -> InputTileType: TileTensor(
            in_dev[i], in_layout
        )
        .as_unsafe_any_origin()
        .as_imm()
    )
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    # --- Shared params ---
    var ctx = list_of_ctx[0]
    var gamma_dev = ctx.enqueue_create_buffer[in_dtype](cols)
    var gamma_host = ctx.enqueue_create_host_buffer[in_dtype](cols)
    for i in range(cols):
        gamma_host[i] = (Float64(i + cols) / Float64(cols)).cast[in_dtype]()
    ctx.enqueue_copy(gamma_dev, gamma_host)

    var gamma_tensor = TileTensor(gamma_dev, row_major(cols))
    var epsilon = Float32(1e-5)
    var weight_offset = Scalar[in_dtype](0.0)
    var scale_ub = max_finite[out_dtype]().cast[.float32]()

    # --- Residual buffer: deterministic values ---
    var residual_dev = ctx.enqueue_create_buffer[in_dtype](length)
    var residual_host = ctx.enqueue_create_host_buffer[in_dtype](length)
    for i in range(length):
        residual_host[i] = (Float64(i % 127 + 1) / Float64(127)).cast[
            in_dtype
        ]()
    ctx.enqueue_copy(residual_dev, residual_host)

    # --- Reference path: compute allreduce + residual on host in float32 ---
    # The fused kernel accumulates the P2P loads in float32, adds the
    # residual in float32, then casts to bf16. We replicate that here
    # to avoid the bf16 rounding that a separate allreduce would introduce.
    var ref_sum_host = ctx.enqueue_create_host_buffer[in_dtype](length)
    for i in range(length):
        var sum_f32 = Float32(0)
        for g in range(ngpus):
            sum_f32 += host_bufs[g][i].cast[.float32]()
        sum_f32 += residual_host[i].cast[.float32]()
        ref_sum_host[i] = sum_f32.cast[in_dtype]()

    # Upload sum to device for the reference fused RMSNorm+FP8 input_fn.
    var ref_sum_dev = ctx.enqueue_create_buffer[in_dtype](length)
    ctx.enqueue_copy(ref_sum_dev, ref_sum_host)
    ctx.synchronize()

    var ref_fp8_dev = ctx.enqueue_create_buffer[out_dtype](length)
    var ref_scales_dev = ctx.enqueue_create_buffer[.float32](rows)

    var ref_sum_ptr = ref_sum_dev.unsafe_ptr()

    @inline(.always)
    def ref_input_fn[
        width: Int
    ](coords: Coord) {var ref_sum_ptr} -> SIMD[in_dtype, width]:
        var linear_idx = Int(coords[0]) * cols + Int(coords[1])
        return ref_sum_ptr.unsafe_load[width=width, alignment=width](linear_idx)

    comptime shape = (rows, cols)
    var ref_fp8_tile = TileTensor(
        ref_fp8_dev,
        row_major(Idx[rows], Idx[cols]),
    )
    var ref_scales_tile = TileTensor(
        ref_scales_dev,
        row_major(Idx[rows], Idx[1]),
    )

    rms_norm_fused_fp8[
        in_dtype,
        out_dtype,
        DType.float32,
        2,
    ](
        ref_input_fn,
        shape,
        ref_fp8_tile,
        gamma_tensor,
        epsilon,
        weight_offset,
        ctx,
        scale_ub,
        ref_scales_tile,
    )

    ctx.synchronize()

    # --- Fused kernel path ---
    # Reset signal buffers for the fused kernel run.
    for i in range(ngpus):
        init_signal_buffer(signal_buffers[i], list_of_ctx[i])
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    var fused_fp8_dev = ctx.enqueue_create_buffer[out_dtype](length)
    var fused_scales_dev = ctx.enqueue_create_buffer[.float32](rows)
    var fused_residual_output_dev = ctx.enqueue_create_buffer[in_dtype](length)

    var fused_fp8_tile = TileTensor(
        fused_fp8_dev,
        row_major(Idx[rows], Idx[cols]),
    )
    var fused_scales_tile = TileTensor(
        fused_scales_dev,
        row_major(Idx[rows], Idx[1]),
    )
    var residual_tile = TileTensor(
        residual_dev,
        row_major(Idx[rows], Idx[cols]),
    )
    var fused_residual_output_tile = TileTensor(
        fused_residual_output_dev,
        row_major(Idx[rows], Idx[cols]),
    )

    group_start()

    comptime for i in range(ngpus):
        allreduce_residual_rmsnorm(
            in_tiles,
            residual_tile.as_imm(),
            fused_fp8_tile,
            fused_residual_output_tile,
            gamma_tensor,
            epsilon,
            weight_offset,
            scale_ub,
            fused_scales_tile,
            rank_sigs,
            list_of_ctx[i],
            scratch_bytes=scratch_bytes,
        )
    group_end()

    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    # --- Verify residual output: fused vs reference sum ---
    var fused_res_out_host = ctx.enqueue_create_host_buffer[in_dtype](length)
    ctx.enqueue_copy(fused_res_out_host, fused_residual_output_dev)
    ctx.synchronize()

    var res_errors = 0
    for i in range(length):
        var ref_val = ref_sum_host[i].cast[.float32]()
        var fused_val = fused_res_out_host[i].cast[.float32]()
        # The fused 1-stage/2-stage kernel accumulates allreduce + residual
        # in f32 before casting to bf16, giving exact match. The split
        # (2-kernel) path has an inherent bf16 round-trip for the allreduce
        # result before adding residual, which can introduce up to 1 bf16
        # ULP difference. Allow 1 ULP tolerance for both paths.
        var max_mag = max(abs(ref_val), abs(fused_val))
        var bf16_ulp = max_mag * Float32(2.0**-7)  # bf16: 7-bit mantissa
        if abs(ref_val - fused_val) > bf16_ulp:
            res_errors += 1
            if res_errors <= 5:
                print(
                    "  Residual mismatch at",
                    i,
                    ": ref=",
                    ref_val,
                    ", fused=",
                    fused_val,
                )

    if res_errors > 0:
        raise Error(t"Residual output mismatches: {res_errors} / {length}")

    # --- Compare FP8 output: fused vs reference ---
    var ref_fp8_host = ctx.enqueue_create_host_buffer[out_dtype](length)
    var fused_fp8_host = ctx.enqueue_create_host_buffer[out_dtype](length)
    ctx.enqueue_copy(ref_fp8_host, ref_fp8_dev)
    ctx.enqueue_copy(fused_fp8_host, fused_fp8_dev)
    ctx.synchronize()

    _assert_fp8_close(
        ref_fp8_host.unsafe_ptr(), fused_fp8_host.unsafe_ptr(), length
    )

    # --- Compare per-row scale factors ---
    var ref_scales_host = ctx.enqueue_create_host_buffer[.float32](rows)
    var fused_scales_host = ctx.enqueue_create_host_buffer[.float32](rows)
    ctx.enqueue_copy(ref_scales_host, ref_scales_dev)
    ctx.enqueue_copy(fused_scales_host, fused_scales_dev)
    ctx.synchronize()

    _assert_scales_close(
        ref_scales_host.unsafe_ptr(), fused_scales_host.unsafe_ptr(), rows
    )

    # Cleanup.
    _ = signal_buffers^
    _ = in_dev^
    _ = ref_fp8_dev^
    _ = ref_scales_dev^
    _ = ref_sum_dev^
    _ = fused_fp8_dev^
    _ = fused_scales_dev^
    _ = fused_residual_output_dev^
    _ = residual_dev^
    _ = gamma_dev^

    print("    PASS")


# --- Test: fully fused allreduce + residual add + RMSNorm (no quant) ---


def test_fused_allreduce_residual_rmsnorm_noquant[
    ngpus: Int,
    dtype: DType,
    rows: Int,
    cols: Int,
    multiply_before_cast: Bool = True,
](list_of_ctx: List[DeviceContext]) raises:
    """Verify the no-quant path (out_dtype == in_dtype) of the fused kernel.

    Exercises allreduce + residual add + RMSNorm with a `dtype` output (no FP8
    quantization, no scale output) against a host-side float32 reference. The
    reference normalizes the exact float32 sum to match the kernel, which keeps
    the accumulation in float32 before the cast to `dtype`, and applies gamma
    in the order `multiply_before_cast` selects.
    """
    comptime length = rows * cols

    print(
        "  test_fused_allreduce_residual_rmsnorm_noquant[",
        ngpus,
        ",",
        dtype,
        ",",
        rows,
        "x",
        cols,
        ", multiply_before_cast=",
        multiply_before_cast,
        "]",
    )

    # --- Setup: per-GPU input buffers ---
    var in_dev = List[DeviceBuffer[dtype]](capacity=ngpus)
    var host_bufs = List[HostBuffer[dtype]](capacity=ngpus)
    var signal_buffers = List[DeviceBuffer[.uint8]](capacity=ngpus)
    var temp_bytes = ngpus * size_of[dtype]() * length

    for i in range(ngpus):
        in_dev.append(list_of_ctx[i].enqueue_create_buffer[dtype](length))
        var h = list_of_ctx[i].enqueue_create_host_buffer[dtype](length)
        for j in range(length):
            h[j] = test_value_for_gpu_element[dtype](i, j)
        list_of_ctx[i].enqueue_copy(in_dev[i], h)
        host_bufs.append(h^)

        signal_buffers.append(
            list_of_ctx[i].create_buffer_sync[.uint8](
                size_of[Signal]() + temp_bytes
            )
        )
        init_signal_buffer(signal_buffers[i], list_of_ctx[i])

    var rank_sigs = Array[_, ngpus](
        fill_with=lambda (i: Int) {ref} -> MutPointer[
            Signal, MutAnyOrigin
        ]: Signal.unsafe_ptr_from(signal_buffers[i])
    )

    comptime in_layout = row_major(Idx[rows], Idx[cols])
    comptime InputTileType = type_of(
        TileTensor(in_dev[0], in_layout).as_unsafe_any_origin().as_imm()
    )
    var in_tiles = Array[_, ngpus](
        fill_with=lambda (i: Int) -> InputTileType: TileTensor(
            in_dev[i], in_layout
        )
        .as_unsafe_any_origin()
        .as_imm()
    )
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    # --- Shared params ---
    var ctx = list_of_ctx[0]
    var gamma_dev = ctx.enqueue_create_buffer[dtype](cols)
    var gamma_host = ctx.enqueue_create_host_buffer[dtype](cols)
    for i in range(cols):
        gamma_host[i] = (Float64(i + cols) / Float64(cols)).cast[dtype]()
    ctx.enqueue_copy(gamma_dev, gamma_host)

    var gamma_tensor = TileTensor(gamma_dev, row_major(cols))
    var epsilon = Float32(1e-5)
    var weight_offset = Scalar[dtype](0.0)

    # --- Residual buffer: deterministic values ---
    var residual_dev = ctx.enqueue_create_buffer[dtype](length)
    var residual_host = ctx.enqueue_create_host_buffer[dtype](length)
    for i in range(length):
        residual_host[i] = (Float64(i % 127 + 1) / Float64(127)).cast[dtype]()
    ctx.enqueue_copy(residual_dev, residual_host)
    ctx.synchronize()

    # --- Host reference: allreduce + residual in f32, then RMSNorm in f32 ---
    var ref_sum_f32 = ctx.enqueue_create_host_buffer[.float32](length)
    var ref_sum_host = ctx.enqueue_create_host_buffer[dtype](length)
    for i in range(length):
        var s = Float32(0)
        for g in range(ngpus):
            s += host_bufs[g][i].cast[.float32]()
        s += residual_host[i].cast[.float32]()
        ref_sum_f32[i] = s
        # bf16-rounded pre-norm sum (matches kernel's residual_output write).
        ref_sum_host[i] = s.cast[dtype]()

    var ref_out_host = ctx.enqueue_create_host_buffer[dtype](length)
    var ref_other_host = ctx.enqueue_create_host_buffer[dtype](length)
    var eps_f32 = epsilon
    for r in range(rows):
        var m2 = Float32(0)
        for c in range(cols):
            var s = ref_sum_f32[r * cols + c]
            m2 += s * s
        var norm = rsqrt(m2 / Float32(cols) + eps_f32)
        for c in range(cols):
            var s = ref_sum_f32[r * cols + c]
            var g = gamma_host[c] + weight_offset
            ref_out_host[r * cols + c] = _rms_norm_ref[
                dtype, multiply_before_cast
            ](s, norm, g)
            ref_other_host[r * cols + c] = _rms_norm_ref[
                dtype, not multiply_before_cast
            ](s, norm, g)

    # --- Fused kernel path (out_dtype == in_dtype → no quantization) ---
    for i in range(ngpus):
        init_signal_buffer(signal_buffers[i], list_of_ctx[i])
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    var fused_out_dev = ctx.enqueue_create_buffer[dtype](length)
    var fused_res_out_dev = ctx.enqueue_create_buffer[dtype](length)
    # Scale buffer + scale_ub are ignored on the no-quant path; pass a dummy.
    var dummy_scales_dev = ctx.enqueue_create_buffer[.float32](rows)

    var fused_out_tile = TileTensor(
        fused_out_dev, row_major(Idx[rows], Idx[cols])
    )
    var fused_res_out_tile = TileTensor(
        fused_res_out_dev, row_major(Idx[rows], Idx[cols])
    )
    var residual_tile = TileTensor(
        residual_dev, row_major(Idx[rows], Idx[cols])
    )
    var dummy_scales_tile = TileTensor(
        dummy_scales_dev, row_major(Idx[rows], Idx[1])
    )
    var scale_ub = Float32(1.0)

    group_start()

    comptime for i in range(ngpus):
        allreduce_residual_rmsnorm[multiply_before_cast=multiply_before_cast](
            in_tiles,
            residual_tile.as_imm(),
            fused_out_tile,
            fused_res_out_tile,
            gamma_tensor,
            epsilon,
            weight_offset,
            scale_ub,
            dummy_scales_tile,
            rank_sigs,
            list_of_ctx[i],
        )
    group_end()

    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    # --- Verify residual output (pre-norm sum) ---
    var fused_res_out_host = ctx.enqueue_create_host_buffer[dtype](length)
    ctx.enqueue_copy(fused_res_out_host, fused_res_out_dev)
    var fused_out_host = ctx.enqueue_create_host_buffer[dtype](length)
    ctx.enqueue_copy(fused_out_host, fused_out_dev)
    ctx.synchronize()

    var res_errors = 0
    for i in range(length):
        var ref_val = ref_sum_host[i].cast[.float32]()
        var fused_val = fused_res_out_host[i].cast[.float32]()
        var max_mag = max(abs(ref_val), abs(fused_val))
        var bf16_ulp = max_mag * Float32(2.0**-7)
        if abs(ref_val - fused_val) > bf16_ulp:
            res_errors += 1
            if res_errors <= 5:
                print(
                    "  Residual mismatch at",
                    i,
                    ": ref=",
                    ref_val,
                    ", fused=",
                    fused_val,
                )
    if res_errors > 0:
        raise Error(t"Residual output mismatches: {res_errors} / {length}")

    # --- Verify normalized output (no quant, written in `dtype`) ---
    comptime if multiply_before_cast:
        _assert_bf16_close(
            ref_out_host.unsafe_ptr(), fused_out_host.unsafe_ptr(), length
        )
    else:
        _assert_cast_order[dtype, multiply_before_cast](
            ref_out_host.unsafe_ptr(),
            ref_other_host.unsafe_ptr(),
            fused_out_host.unsafe_ptr(),
            length,
        )

    # Cleanup.
    _ = host_bufs^
    _ = signal_buffers^
    _ = in_dev^
    _ = residual_dev^
    _ = gamma_dev^
    _ = fused_out_dev^
    _ = fused_res_out_dev^
    _ = dummy_scales_dev^

    print("    PASS")


# --- Test: Lamport path of the fused allreduce + residual + RMSNorm ---


def _residual_rmsnorm_call[
    ngpus: Int, cols: Int, *, force_lamport: Bool, multiply_before_cast: Bool
](
    list_of_ctx: List[DeviceContext],
    rank_sigs: Array[MutPointer[Signal, MutAnyOrigin], ngpus],
    rows: Int,
    call_idx: Int,
    scratch_bytes: Optional[Int] = None,
) raises:
    """Runs one fused allreduce + residual add + RMSNorm and checks every rank.

    Inputs and the residual are small random integers, so the float32 sum is
    exact in bfloat16 and the residual output is compared bitwise. Each rank
    owns its input, residual, gamma, and outputs, as in a serving graph.
    `force_lamport` launches the Lamport kernel directly; otherwise the call
    goes through the public dispatch, which picks the path itself.
    """
    comptime dtype = DType.bfloat16
    var epsilon = Float32(1e-6)
    var weight_offset = Scalar[dtype](1.0)
    var length = rows * cols

    var residual_vals = List[Scalar[dtype]](capacity=length)
    for _ in range(length):
        residual_vals.append(Scalar[dtype](Int(random_si64(-16, 16))))
    var gamma_vals = List[Scalar[dtype]](capacity=cols)
    for _ in range(cols):
        gamma_vals.append((Float64(random_si64(-8, 8)) / 16.0).cast[dtype]())

    var in_host = List[HostBuffer[dtype]](capacity=ngpus)
    var in_dev = List[DeviceBuffer[dtype]](capacity=ngpus)
    var residual_dev = List[DeviceBuffer[dtype]](capacity=ngpus)
    var gamma_dev = List[DeviceBuffer[dtype]](capacity=ngpus)
    var out_dev = List[DeviceBuffer[dtype]](capacity=ngpus)
    var res_out_dev = List[DeviceBuffer[dtype]](capacity=ngpus)
    var scales_dev = List[DeviceBuffer[.float32]](capacity=ngpus)
    for i in range(ngpus):
        var ctx = list_of_ctx[i]
        var h = ctx.enqueue_create_host_buffer[dtype](length)
        var residual_h = ctx.enqueue_create_host_buffer[dtype](length)
        var gamma_h = ctx.enqueue_create_host_buffer[dtype](cols)
        for j in range(length):
            h[j] = Scalar[dtype](Int(random_si64(-16, 16)))
            residual_h[j] = residual_vals[j]
        for c in range(cols):
            gamma_h[c] = gamma_vals[c]
        in_dev.append(ctx.enqueue_create_buffer[dtype](length))
        residual_dev.append(ctx.enqueue_create_buffer[dtype](length))
        gamma_dev.append(ctx.enqueue_create_buffer[dtype](cols))
        out_dev.append(ctx.enqueue_create_buffer[dtype](length))
        res_out_dev.append(ctx.enqueue_create_buffer[dtype](length))
        scales_dev.append(ctx.enqueue_create_buffer[.float32](rows))
        ctx.enqueue_copy(in_dev[i], h)
        ctx.enqueue_copy(residual_dev[i], residual_h)
        ctx.enqueue_copy(gamma_dev[i], gamma_h)
        ctx.synchronize()
        in_host.append(h^)

    var ref_sum = List[Float32](capacity=length)
    for j in range(length):
        var s = residual_vals[j].cast[.float32]()
        for i in range(ngpus):
            s += in_host[i][j].cast[.float32]()
        ref_sum.append(s)

    comptime InTile = TileTensor[
        dtype, type_of(row_major(rows, cols)), ImmutAnyOrigin
    ]
    var in_tiles = Array[_, ngpus](
        fill_with=lambda (i: Int) -> InTile: TileTensor(
            in_dev[i], row_major(rows, cols)
        ).as_imm()
    )

    comptime for i in range(ngpus):
        var residual_tile = TileTensor(
            residual_dev[i], row_major(rows, cols)
        ).as_imm()
        var res_out_tile = TileTensor(res_out_dev[i], row_major(rows, cols))
        comptime if force_lamport:
            _lamport_allreduce_rmsnorm_launch[
                dtype,
                ngpus,
                has_residual=True,
                multiply_before_cast=multiply_before_cast,
                pdl=False,
            ](
                i,
                rebind[ImmPointer[Scalar[dtype], ImmutAnyOrigin]](
                    in_dev[i].unsafe_ptr()
                ),
                rebind[MutPointer[Scalar[dtype], MutAnyOrigin]](
                    out_dev[i].unsafe_ptr()
                ),
                rebind[ImmPointer[Scalar[dtype], ImmutAnyOrigin]](
                    gamma_dev[i].unsafe_ptr()
                ),
                rank_sigs,
                rows,
                cols,
                Scalar[dtype](epsilon),
                weight_offset,
                list_of_ctx[i],
                residual_tile,
                res_out_tile,
            )
        else:
            allreduce_residual_rmsnorm[
                multiply_before_cast=multiply_before_cast
            ](
                in_tiles,
                residual_tile,
                TileTensor(out_dev[i], row_major(rows, cols)),
                res_out_tile,
                TileTensor(gamma_dev[i], row_major(cols)),
                epsilon,
                weight_offset,
                Float32(1.0),
                TileTensor(scales_dev[i], row_major(rows)),
                rank_sigs,
                list_of_ctx[i],
                my_rank=i,
                scratch_bytes=scratch_bytes,
            )
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    for i in range(ngpus):
        var ctx = list_of_ctx[i]
        var out_h = ctx.enqueue_create_host_buffer[dtype](length)
        var res_out_h = ctx.enqueue_create_host_buffer[dtype](length)
        ctx.enqueue_copy(out_h, out_dev[i])
        ctx.enqueue_copy(res_out_h, res_out_dev[i])
        ctx.synchronize()

        var ref_same = List[Scalar[dtype]](capacity=length)
        var ref_other = List[Scalar[dtype]](capacity=length)
        for r in range(rows):
            var sumsq = Float32(0)
            for c in range(cols):
                var s = ref_sum[r * cols + c]
                sumsq += s * s
            var norm = rsqrt(sumsq / Float32(cols) + epsilon)
            for c in range(cols):
                var j = r * cols + c
                if res_out_h[j].cast[.float32]() != ref_sum[j]:
                    raise Error(
                        t"call {call_idx} GPU {i} residual[{r}, {c}]:"
                        t" got {res_out_h[j]}, expected {ref_sum[j]}"
                    )
                var g = gamma_vals[c] + weight_offset
                ref_same.append(
                    _rms_norm_ref[dtype, multiply_before_cast](
                        ref_sum[j], norm, g
                    )
                )
                ref_other.append(
                    _rms_norm_ref[dtype, not multiply_before_cast](
                        ref_sum[j], norm, g
                    )
                )
        _assert_cast_order[dtype, multiply_before_cast](
            ref_same.unsafe_ptr(),
            ref_other.unsafe_ptr(),
            out_h.unsafe_ptr(),
            length,
        )


def test_lamport_residual_rmsnorm_sequence[
    ngpus: Int, cols: Int, multiply_before_cast: Bool = True
](list_of_ctx: List[DeviceContext]) raises:
    """Chains fused allreduce + residual + RMSNorm calls on one signal buffer
    set, alternating the forced Lamport launch with the public dispatch.

    On B200 the dispatch also takes the Lamport kernel for these sizes, so the
    chain advances one shared generation counter. Row counts shrink and grow so
    calls clear generations written by calls of a different size, and the
    largest count at `cols=8192` fills the whole Lamport workspace. On 8 GPUs
    the sizes straddle the switch between cluster-split and one-block rows, so
    both layouts share the generations. Shapes the kernel cannot take skip the
    forced launch; their dispatch call still runs, on the barrier kernels.
    """
    print(
        "  test_lamport_residual_rmsnorm_sequence[",
        ngpus,
        ", cols=",
        cols,
        ", multiply_before_cast=",
        multiply_before_cast,
        "]",
    )
    var row_counts: List[Int] = [1, 64, 7, 32, 1, 17, 64, 3]
    # Shapes the Lamport kernel cannot take run the 2-stage kernel, which
    # stages through scratch after the Signal header.
    var temp_bytes = ngpus * size_of[DType.bfloat16]() * 64 * cols
    var signal_buffers = List[DeviceBuffer[.uint8]](capacity=ngpus)
    for i in range(ngpus):
        signal_buffers.append(
            list_of_ctx[i].create_buffer_sync[.uint8](
                size_of[Signal]() + temp_bytes
            )
        )
        init_signal_buffer(signal_buffers[i], list_of_ctx[i])
    for i in range(ngpus):
        list_of_ctx[i].synchronize()
    var rank_sigs = Array[_, ngpus](
        fill_with=lambda (i: Int) {ref} -> MutPointer[
            Signal, MutAnyOrigin
        ]: Signal.unsafe_ptr_from(signal_buffers[i])
    )

    for k in range(len(row_counts)):
        if lamport_allreduce_rmsnorm_fits[DType.bfloat16, ngpus](
            row_counts[k], cols
        ):
            _residual_rmsnorm_call[
                ngpus,
                cols,
                force_lamport=True,
                multiply_before_cast=multiply_before_cast,
            ](list_of_ctx, rank_sigs, row_counts[k], 2 * k)
        _residual_rmsnorm_call[
            ngpus,
            cols,
            force_lamport=False,
            multiply_before_cast=multiply_before_cast,
        ](list_of_ctx, rank_sigs, row_counts[k], 2 * k + 1)

    _ = signal_buffers^
    print("    PASS")


# --- Main ---

comptime test_gpu_counts = (2, 4, 8)


def _signal_buffers[
    ngpus: Int
](list_of_ctx: List[DeviceContext], scratch_bytes: Int) raises -> List[
    DeviceBuffer[.uint8]
]:
    var signal_buffers = List[DeviceBuffer[.uint8]](capacity=ngpus)
    for i in range(ngpus):
        signal_buffers.append(
            list_of_ctx[i].create_buffer_sync[.uint8](
                size_of[Signal]() + scratch_bytes
            )
        )
        init_signal_buffer(signal_buffers[i], list_of_ctx[i])
    for i in range(ngpus):
        list_of_ctx[i].synchronize()
    return signal_buffers^


def test_residual_rmsnorm_rank_not_ordinal[
    ngpus: Int
](list_of_ctx: List[DeviceContext]) raises:
    """Runs on GPUs whose ordinals differ from their positions in the group,
    as with `--devices gpu:4,5,6,7`. A rank taken from `ctx.id()` would index
    past the group's buffers. The row counts cover the Lamport kernel and the
    barrier kernels."""
    print(
        "  test_residual_rmsnorm_rank_not_ordinal[",
        ngpus,
        ", first ordinal=",
        list_of_ctx[0].id(),
        "]",
    )
    comptime cols = 4096
    var row_counts: List[Int] = [1, 64, 2048]
    var signal_buffers = _signal_buffers[ngpus](
        list_of_ctx,
        ceildiv(2048, ngpus) * cols * 2 * size_of[DType.bfloat16](),
    )
    var rank_sigs = Array[_, ngpus](
        fill_with=lambda (i: Int) {ref} -> MutPointer[
            Signal, MutAnyOrigin
        ]: Signal.unsafe_ptr_from(signal_buffers[i])
    )
    for k in range(len(row_counts)):
        _residual_rmsnorm_call[
            ngpus, cols, force_lamport=False, multiply_before_cast=False
        ](list_of_ctx, rank_sigs, row_counts[k], k)
    _ = signal_buffers^
    print("    PASS")


def test_residual_rmsnorm_no_scratch[
    ngpus: Int
](list_of_ctx: List[DeviceContext]) raises:
    """Gives the fused op no scratch behind the `Signal` headers. Payloads the
    dispatch would stage through scratch must take the 1-stage kernel."""
    print("  test_residual_rmsnorm_no_scratch[", ngpus, "]")
    var signal_buffers = _signal_buffers[ngpus](list_of_ctx, 0)
    var rank_sigs = Array[_, ngpus](
        fill_with=lambda (i: Int) {ref} -> MutPointer[
            Signal, MutAnyOrigin
        ]: Signal.unsafe_ptr_from(signal_buffers[i])
    )
    _residual_rmsnorm_call[
        ngpus, 4096, force_lamport=False, multiply_before_cast=True
    ](list_of_ctx, rank_sigs, 2048, 0, scratch_bytes=0)
    _ = signal_buffers^
    print("    PASS")


def test_residual_rmsnorm_without_p2p[
    ngpus: Int
](list_of_ctx: List[DeviceContext]) raises:
    """Runs before peer access is enabled, as on a host without P2P. The
    fused op must reduce through the plain allreduce's staged copies and
    normalize separately instead of raising."""
    print("  test_residual_rmsnorm_without_p2p[", ngpus, "]")
    var signal_buffers = _signal_buffers[ngpus](list_of_ctx, 0)
    var rank_sigs = Array[_, ngpus](
        fill_with=lambda (i: Int) {ref} -> MutPointer[
            Signal, MutAnyOrigin
        ]: Signal.unsafe_ptr_from(signal_buffers[i])
    )
    var row_counts: List[Int] = [1, 300]
    for k in range(len(row_counts)):
        _residual_rmsnorm_call[
            ngpus, 4096, force_lamport=False, multiply_before_cast=True
        ](list_of_ctx, rank_sigs, row_counts[k], k)
        # 4100 columns are not a multiple of the norm's SIMD width.
        _residual_rmsnorm_call[
            ngpus, 4100, force_lamport=False, multiply_before_cast=False
        ](list_of_ctx, rank_sigs, row_counts[k], k)
    _ = signal_buffers^
    print("    PASS")


def test_fused_allreduce_rmsnorm_zero_rows[
    ngpus: Int, dtype: DType, cols: Int
](list_of_ctx: List[DeviceContext]) raises:
    """Verify both entry points return without launching on a zero-row batch.

    A step that skips speculative drafting runs the drafter's collectives on
    zero rows on every rank. Buffers are sized for one row so the device
    allocations are real. The views are zero-row, and the output must keep
    its sentinel since nothing may be written.
    """
    print("  test_fused_allreduce_rmsnorm_zero_rows[", ngpus, ",", cols, "]")

    var in_dev = List[DeviceBuffer[dtype]](capacity=ngpus)
    var signal_buffers = List[DeviceBuffer[.uint8]](capacity=ngpus)
    for i in range(ngpus):
        in_dev.append(list_of_ctx[i].enqueue_create_buffer[dtype](cols))
        signal_buffers.append(
            list_of_ctx[i].create_buffer_sync[.uint8](size_of[Signal]())
        )
        init_signal_buffer(signal_buffers[i], list_of_ctx[i])

    var rank_sigs = Array[_, ngpus](
        fill_with=lambda (i: Int) {ref} -> MutPointer[
            Signal, MutAnyOrigin
        ]: Signal.unsafe_ptr_from(signal_buffers[i])
    )

    comptime zero_layout = row_major(Coord(Idx[0], Idx[cols]))
    comptime InputTileType = TileTensor[
        dtype, type_of(zero_layout), ImmutAnyOrigin
    ]
    var in_tiles = Array[_, ngpus](
        fill_with=lambda (i: Int) -> InputTileType: TileTensor(
            in_dev[i], zero_layout
        ).as_imm()
    )

    var ctx = list_of_ctx[0]
    var gamma_dev = ctx.enqueue_create_buffer[dtype](cols)
    var gamma_tensor = TileTensor(gamma_dev, row_major(cols))
    var sentinel = Scalar[dtype](7.0)
    var out_dev = ctx.enqueue_create_buffer[dtype](cols)
    var res_out_dev = ctx.enqueue_create_buffer[dtype](cols)
    var residual_dev = ctx.enqueue_create_buffer[dtype](cols)
    var scales_dev = ctx.enqueue_create_buffer[.float32](1)
    out_dev.enqueue_fill(sentinel)
    res_out_dev.enqueue_fill(sentinel)
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    var out_tile = TileTensor(out_dev, zero_layout)
    var res_out_tile = TileTensor(res_out_dev, zero_layout)
    var residual_tile = TileTensor(residual_dev, zero_layout)
    var scales_tile = TileTensor(scales_dev, row_major(Coord(Idx[0], Idx[1])))

    group_start()
    comptime for i in range(ngpus):
        allreduce_rmsnorm(
            in_tiles,
            out_tile,
            gamma_tensor,
            Float32(1e-5),
            Scalar[dtype](0.0),
            Float32(1.0),
            scales_tile,
            rank_sigs,
            list_of_ctx[i],
        )
        allreduce_residual_rmsnorm(
            in_tiles,
            residual_tile.as_imm(),
            out_tile,
            res_out_tile,
            gamma_tensor,
            Float32(1e-5),
            Scalar[dtype](0.0),
            Float32(1.0),
            scales_tile,
            rank_sigs,
            list_of_ctx[i],
        )
    group_end()
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    var out_host = ctx.enqueue_create_host_buffer[dtype](cols)
    var res_out_host = ctx.enqueue_create_host_buffer[dtype](cols)
    ctx.enqueue_copy(out_host, out_dev)
    ctx.enqueue_copy(res_out_host, res_out_dev)
    ctx.synchronize()
    for i in range(cols):
        assert_true(
            out_host[i] == sentinel and res_out_host[i] == sentinel,
            "a zero-row call wrote its output",
        )


def main() raises:
    var num_devices = DeviceContext.number_of_devices()
    assert_true(num_devices >= 2, "need at least 2 GPUs")

    print("\n=== fused_allreduce_residual_rmsnorm without P2P ===")
    if is_p2p_enabled():
        print("  skipped: peer access is already enabled")
    else:
        var no_p2p_ctx = List[DeviceContext]()
        for i in range(2):
            no_p2p_ctx.append(DeviceContext(device_id=i))
        test_residual_rmsnorm_without_p2p[2](no_p2p_ctx)

    assert_true(enable_p2p(), "failed to enable P2P access between GPUs")

    print("FP8 output dtype:", out_fp8_dtype)

    comptime for gpu_idx in range(len(test_gpu_counts)):
        comptime num_gpus = test_gpu_counts[gpu_idx]
        if num_devices < num_gpus:
            continue

        var list_of_ctx = List[DeviceContext]()
        for i in range(num_gpus):
            list_of_ctx.append(DeviceContext(device_id=i))

        print(
            "\n=== fused_allreduce_rmsnorm_fp8 (",
            num_gpus,
            "GPUs) ===",
        )
        test_fused_allreduce_rmsnorm_fp8[
            num_gpus, DType.bfloat16, out_fp8_dtype, 1, 4096
        ](list_of_ctx)
        test_fused_allreduce_rmsnorm_fp8[
            num_gpus, DType.bfloat16, out_fp8_dtype, 8, 4096
        ](list_of_ctx)
        test_fused_allreduce_rmsnorm_fp8[
            num_gpus, DType.bfloat16, out_fp8_dtype, 1, 8192
        ](list_of_ctx)
        test_fused_allreduce_rmsnorm_fp8[
            num_gpus, DType.bfloat16, out_fp8_dtype, 8, 8192
        ](list_of_ctx)
        test_fused_allreduce_rmsnorm_fp8[
            num_gpus, DType.bfloat16, out_fp8_dtype, 1, 16384
        ](list_of_ctx)

        # --- rows > 512: exercise persistent row loop ---
        test_fused_allreduce_rmsnorm_fp8[
            num_gpus, DType.bfloat16, out_fp8_dtype, 1024, 4096
        ](list_of_ctx)
        test_fused_allreduce_rmsnorm_fp8[
            num_gpus, DType.bfloat16, out_fp8_dtype, 4096, 8192
        ](list_of_ctx)
        test_fused_allreduce_rmsnorm_fp8[
            num_gpus, DType.bfloat16, out_fp8_dtype, 8192, 8192
        ](list_of_ctx)

        # --- 2-stage path with misaligned rows_per_rank (regression for
        #     CUDA_ERROR_MISALIGNED_ADDRESS in scratch_residual pointer).
        #     rows=17 → rows_per_rank=ceildiv(17,ngpus) which is not a
        #     multiple of (simd_width/sizeof(scales_dtype))=4 for ngpus≤8,
        #     triggering the padded-scale-section code path.
        #     cols=16384 matches LLaMA-405B hidden dimension. ---
        test_fused_allreduce_rmsnorm_fp8[
            num_gpus, DType.bfloat16, out_fp8_dtype, 17, 16384
        ](list_of_ctx)
        test_fused_allreduce_rmsnorm_fp8[
            num_gpus, DType.bfloat16, out_fp8_dtype, 20, 16384
        ](list_of_ctx)

        # --- bf16 scale buffer. Checkpoints that store `weight_scale` in
        #     bf16 (compressed-tensors FP8-dynamic) drive the activation
        #     scales dtype to bf16, so both kernels must instantiate for a
        #     scales dtype other than float32. rows=1 takes the 1-stage
        #     kernel, rows=8192 the 2-stage one. ---
        test_fused_allreduce_rmsnorm_fp8[
            num_gpus, DType.bfloat16, out_fp8_dtype, 1, 16384, DType.bfloat16
        ](list_of_ctx)
        test_fused_allreduce_rmsnorm_fp8[
            num_gpus, DType.bfloat16, out_fp8_dtype, 8192, 8192, DType.bfloat16
        ](list_of_ctx)

        print(
            "\n=== fused_allreduce_rmsnorm (no-quant bf16, ",
            num_gpus,
            "GPUs) ===",
        )
        # 1-stage path (small per-rank payloads).
        test_fused_allreduce_rmsnorm_noquant[num_gpus, DType.bfloat16, 1, 4096](
            list_of_ctx
        )
        test_fused_allreduce_rmsnorm_noquant[num_gpus, DType.bfloat16, 8, 8192](
            list_of_ctx
        )
        # 2-stage path (large per-rank payloads) + misaligned rows_per_rank.
        test_fused_allreduce_rmsnorm_noquant[
            num_gpus, DType.bfloat16, 8192, 8192
        ](list_of_ctx)
        test_fused_allreduce_rmsnorm_noquant[
            num_gpus, DType.bfloat16, 17, 16384
        ](list_of_ctx)

        print(
            "\n=== fused_allreduce_residual_rmsnorm_fp8 (",
            num_gpus,
            "GPUs) ===",
        )
        test_fused_allreduce_residual_rmsnorm_fp8[
            num_gpus, DType.bfloat16, out_fp8_dtype, 1, 4096
        ](list_of_ctx)
        test_fused_allreduce_residual_rmsnorm_fp8[
            num_gpus, DType.bfloat16, out_fp8_dtype, 8, 4096
        ](list_of_ctx)
        test_fused_allreduce_residual_rmsnorm_fp8[
            num_gpus, DType.bfloat16, out_fp8_dtype, 1, 8192
        ](list_of_ctx)
        test_fused_allreduce_residual_rmsnorm_fp8[
            num_gpus, DType.bfloat16, out_fp8_dtype, 8, 8192
        ](list_of_ctx)
        test_fused_allreduce_residual_rmsnorm_fp8[
            num_gpus, DType.bfloat16, out_fp8_dtype, 1, 16384
        ](list_of_ctx)

        # --- rows > 512: exercise persistent row loop ---
        test_fused_allreduce_residual_rmsnorm_fp8[
            num_gpus, DType.bfloat16, out_fp8_dtype, 1024, 4096
        ](list_of_ctx)
        test_fused_allreduce_residual_rmsnorm_fp8[
            num_gpus, DType.bfloat16, out_fp8_dtype, 4096, 8192
        ](list_of_ctx)
        test_fused_allreduce_residual_rmsnorm_fp8[
            num_gpus, DType.bfloat16, out_fp8_dtype, 8192, 8192
        ](list_of_ctx)

        # --- 2-stage path with misaligned rows_per_rank (regression for
        #     CUDA_ERROR_MISALIGNED_ADDRESS in scratch_residual pointer).
        #     rows=17 → rows_per_rank=ceildiv(17,ngpus); for ngpus=4:
        #     rows_per_rank=5, scale section = 5*4 = 20 bytes, not a
        #     multiple of simd_width=16, misaligning scratch_residual. ---
        test_fused_allreduce_residual_rmsnorm_fp8[
            num_gpus, DType.bfloat16, out_fp8_dtype, 17, 16384
        ](list_of_ctx)
        test_fused_allreduce_residual_rmsnorm_fp8[
            num_gpus, DType.bfloat16, out_fp8_dtype, 20, 16384
        ](list_of_ctx)

        print(
            "\n=== fused_allreduce_residual_rmsnorm (no-quant bf16, ",
            num_gpus,
            "GPUs) ===",
        )
        # 1-stage path (small per-rank payloads).
        test_fused_allreduce_residual_rmsnorm_noquant[
            num_gpus, DType.bfloat16, 1, 4096
        ](list_of_ctx)
        test_fused_allreduce_residual_rmsnorm_noquant[
            num_gpus, DType.bfloat16, 8, 8192
        ](list_of_ctx)
        # Column-aware-threshold band: on B200 8-GPU the wide-column
        # (cols >= 6144) residual crossover is ~100 KB per-rank, so this
        # shape (rows=48, cols=8192 -> ceildiv(48,8)*8192*2 = 96 KB) now
        # routes to the 1-stage kernel where a flat 80 KB threshold would
        # have used 2-stage. Locks in correctness of the 1-stage path at the
        # newly-selected band.
        test_fused_allreduce_residual_rmsnorm_noquant[
            num_gpus, DType.bfloat16, 48, 8192
        ](list_of_ctx)
        # 2-stage path (large per-rank payloads), including the persistent
        # row loop.
        test_fused_allreduce_residual_rmsnorm_noquant[
            num_gpus, DType.bfloat16, 8192, 8192
        ](list_of_ctx)
        # 2-stage with misaligned rows_per_rank (residual scratch follows the
        # output scratch directly when not quantizing — no scale section).
        test_fused_allreduce_residual_rmsnorm_noquant[
            num_gpus, DType.bfloat16, 17, 16384
        ](list_of_ctx)

        test_fused_allreduce_rmsnorm_zero_rows[num_gpus, DType.bfloat16, 4096](
            list_of_ctx
        )

        print(
            "\n=== fused_allreduce_residual_rmsnorm without scratch (",
            num_gpus,
            "GPUs) ===",
        )
        test_residual_rmsnorm_no_scratch[num_gpus](list_of_ctx)
        test_fused_allreduce_residual_rmsnorm_fp8[
            num_gpus,
            DType.bfloat16,
            out_fp8_dtype,
            2048,
            4096,
            no_scratch=True,
        ](list_of_ctx)

        print(
            "\n=== fused_allreduce_residual_rmsnorm Lamport sequence (",
            num_gpus,
            "GPUs) ===",
        )
        # On 8 GPUs these cover cluster sizes 1, 2, 4 and 8.
        test_lamport_residual_rmsnorm_sequence[num_gpus, 2048](list_of_ctx)
        test_lamport_residual_rmsnorm_sequence[num_gpus, 4096](list_of_ctx)
        test_lamport_residual_rmsnorm_sequence[num_gpus, 7168](list_of_ctx)
        test_lamport_residual_rmsnorm_sequence[num_gpus, 8192](list_of_ctx)
        test_lamport_residual_rmsnorm_sequence[num_gpus, 16384](list_of_ctx)
        test_lamport_residual_rmsnorm_sequence[
            num_gpus, 7168, multiply_before_cast=False
        ](list_of_ctx)

        print(
            "\n=== fused_allreduce_residual_rmsnorm cast-first norm (",
            num_gpus,
            "GPUs) ===",
        )
        # Lamport on B200, 1-stage elsewhere.
        test_fused_allreduce_residual_rmsnorm_noquant[
            num_gpus, DType.bfloat16, 8, 7168, multiply_before_cast=False
        ](list_of_ctx)
        # 1-stage (cols too wide for the Lamport kernel).
        test_fused_allreduce_residual_rmsnorm_noquant[
            num_gpus, DType.bfloat16, 17, 16384, multiply_before_cast=False
        ](list_of_ctx)
        # 2-stage beyond two GPUs.
        test_fused_allreduce_residual_rmsnorm_noquant[
            num_gpus, DType.bfloat16, 8192, 8192, multiply_before_cast=False
        ](list_of_ctx)

    print("\n=== fused_allreduce_residual_rmsnorm rank != ordinal ===")
    comptime for gpu_idx in range(2):
        comptime num_gpus = test_gpu_counts[gpu_idx]
        if num_devices >= 2 * num_gpus:
            var offset_ctx = List[DeviceContext]()
            for i in range(num_gpus):
                offset_ctx.append(DeviceContext(device_id=num_gpus + i))
            test_residual_rmsnorm_rank_not_ordinal[num_gpus](offset_ctx)

    print("\nAll tests passed!")
