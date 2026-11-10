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

"""Numerical E2E test for MLA_SM100_Decode_Sparse_QKV_FP8.

Native FP8-Q+KV sparse decode: Q is float8_e4m3fn, eliminating the
FP8-to-BF16 convert warpgroup present in the BF16-Q path. Output is always
bfloat16 — the kernel dequantizes internally.

Layout tested here per KV row:
    [nope: 512 FP8 bytes] [rope: 64 FP8 bytes]  = 576 bytes total

Q generation: random BF16 values are cast to FP8 before being stored and
uploaded to the device, so the reference sees exactly the same quantized Q
values as the kernel.

This test covers the full feature matrix supported by the kernel:
  * base (NullMask, no extra features)
  * CausalMask
  * multi-token Q (q_max_seq_len > 1)
  * multi-batch (batch_size > 1)
  * variable per-batch topk (has_variable_topk)
  * attention sinks (has_attn_sink)
  * extra KV (has_extra_kv)
  * topk clamping (small cache with topk > actual_tokens)
  * read-once shared-index fold (shared_index=True)
"""

from std.math import ceildiv, exp, sqrt
from std.memory import alloc, bitcast
from std.random import randn, randn_float64, seed
from std.sys import get_defined_int, size_of
from std.time import global_perf_counter_ns

from max.gpu import *
from max.gpu.host import DeviceBuffer, DeviceContext, FuncAttribute, HostBuffer
from max.gpu.host.info import _is_sm10x_gpu
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from max.gpu.primitives.grid_controls import launch_dependent_grids
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import Idx, TileTensor, row_major
from nn.attention.mha_mask import CausalMask, NullMask
from nn.attention.mha_operand import KVCacheMHAOperand
from nn.attention.mha_utils import MHAConfig, NonNullPointer
from nn.attention.gpu.mla import SM100MLA, flare_mla_decoding
from nn.attention.gpu.nvidia.sm100.mla_decode_dispatch import (
    MLADispatchScalarArgs,
    compute_mla_dispatch_scalars,
)
from nn.attention.gpu.nvidia.sm100.mla_decode_utils import (
    _MLA_SPARSE_CG2_ROUTE,
)
from std.utils.index import IndexList
from std.utils.numerics import isnan, min_or_neg_inf


# ===-----------------------------------------------------------------------===#
# Test constants
# ===-----------------------------------------------------------------------===#

# MLA dimensions (matching DeepSeek V3 production config).
comptime Q_DEPTH = 576  # Full Q depth: 512 nope + 64 rope
comptime V_DEPTH = 512  # Output depth (nope only)
comptime ROPE_DEPTH = 64  # Rope dimension
comptime PAGE_SIZE = 128  # Standard page size
comptime NUM_LAYERS = 1
comptime KV_NUM_HEADS = 1  # MLA has 1 KV head

# ALL-FP8 KV layout: 512 (FP8 nope) + 64 (FP8 rope) = 576 FP8 bytes.
comptime KV_HEAD_SIZE = V_DEPTH + ROPE_DEPTH  # 576

# Set by the half-row softmax negative controls (`MLA_HALF_ROWS_BREAK`). Kind
# 3 breaks every row: each cell with at most 16 softmax rows must fail and
# every other cell must pass. Kind 1 (no pair max) leaves the row's max one
# thread's partial, which shows only when the partials differ by a lot: the
# cells marked `hr1_fail` must fail, other cells with at most 16 rows may pass
# or fail, and cells with more rows must pass. Kind 5 breaks only the logical-position ballot
# mask (every causal cell): a padded cell must fail, other causal cells with at
# most 16 rows may fail (one or two masked keys can stay under tolerance), and
# the rest must pass.
comptime HALF_ROWS_BREAK_KIND = get_defined_int["MLA_HALF_ROWS_BREAK", 0]()
# Negative controls for the default h=128 pair (`MLA_CG2_SLICE_BREAK`, see
# mla_decode_utils.mojo): only the pair cells run, and each must fail; one
# unpaired cell must pass. The arms also set `MLA_SPARSE_CG2_ROUTE=0`, or the
# short cells would route to the unpaired kernel and pass.
comptime CG2_SLICE_BREAK = get_defined_int["MLA_CG2_SLICE_BREAK", 0]() != 0
# A slice-break arm that keeps the default routing pins the route's boundary
# instead (`_run_pair_route_negative_control`). The kernels' own value, so the
# test follows that knob's default.
comptime CG2_ROUTED = _MLA_SPARSE_CG2_ROUTE != 0
comptime HALF_ROWS_BREAK = HALF_ROWS_BREAK_KIND != 0
# Kinds 6 (every split reads split 0's keys) and 7 (every split one page
# short) break the 32-bit split-K prologue of every Layout G kernel: the
# forced multi-split cells must fail and the rest may pass or fail.
comptime SPLIT_BREAK = HALF_ROWS_BREAK_KIND == 6 or HALF_ROWS_BREAK_KIND == 7
# Set by the direct-gather negative controls (`MLA_P6_BREAK`, see
# mla_decode_utils.mojo: 5 and 8). A cell names the kinds that must fail it in
# its `p6_fail` bit mask (bit k for kind k); every other cell may pass or
# fail.
comptime P6_BREAK_KIND = get_defined_int["MLA_P6_BREAK", 0]()
comptime P6_BREAK = P6_BREAK_KIND != 0
# PDL and ragged-exit controls. `MLA_DECODE_PDL_WAIT=0` drops the split-K
# decode's grid wait: the late-producer cells (`pdl_mode` 1 and 2) must fail.
# `MLA_COMBINE_TRIGGER_BREAK=1` makes the combine trigger its successor at
# entry; with the wait in place every cell must still pass, and without it
# the reuse cells report whether the race was reached.
# `MLA_COMBINE_RAGGED_EXIT_BREAK` 1 (combine) and 2 (non-split decode) drop a
# ragged exit: the `short_batches` cells at forced np > 1 on the early-load
# combine, or at forced np 1, must fail.
comptime PDL_WAIT = get_defined_int["MLA_DECODE_PDL_WAIT", 1]() != 0
comptime COMBINE_TRIGGER_BREAK = (
    get_defined_int["MLA_COMBINE_TRIGGER_BREAK", 0]() != 0
)
comptime RAGGED_EXIT_BREAK_KIND = get_defined_int[
    "MLA_COMBINE_RAGGED_EXIT_BREAK", 0
]()
comptime PDL_CONTROL = not PDL_WAIT or RAGGED_EXIT_BREAK_KIND != 0
comptime PDL_NONE = 0
comptime PDL_LATE_INDICES = 1
comptime PDL_LATE_Q = 2
comptime PDL_REUSE = 3
comptime PDL_REUSE_CALLS = 16
# A bf16 NaN payload no kernel writes; rows past the packed output keep it.
comptime OUT_SENTINEL_BITS = UInt16(0x7FC1)
comptime BREAK_MUST_PASS = 0
comptime BREAK_MUST_FAIL = 1
# randn and heavy-tailed stimuli raise on pipeline 1 by chance (a causal
# seq6 cell with h24 fails under the knob), so cells default to this.
comptime BREAK_EITHER = 2

# Stimuli for `run_test_sparse_qkv_fp8`'s `stim`. The default is randn(0.5)
# Q and K, whose log2 scores have a standard deviation of about 0.36: a broken
# lazy rescale is almost never exercised. The spike stimuli add a score of
# about 13 (log2) per step through one rotary dimension per query token,
# which reaches the scores but not V: odd tiles only (every raise lands on
# odd-numbered tiles), even tiles only, every tile, and a falling staircase (no
# raise after tile 0). Query token s sees the staircase shifted by s tiles,
# so tokens sharing a warp raise on different tiles. The heavy-tailed ones
# draw Q and K from a Student-t.
comptime STIM_RANDN = 0
comptime STIM_SPIKE_ODD = 1
comptime STIM_SPIKE_EVEN = 2
comptime STIM_SPIKE_EVERY = 3
comptime STIM_SPIKE_FALLING = 4
comptime STIM_T3 = 5
comptime STIM_T7 = 6
# Query token s scores through rope dims A = V_DEPTH + 2s and B = A + 1: even
# heads read A, odd heads B. In every tile the key in column 0 (warp 0, half
# 0) scores 36 on A and 18 on B, the key in column BN_QK / 8 (warp 0, half 1)
# the reverse, and every other key 18 on both, so each head's top key sits
# 26 log2 units above the rest, in the opposite half from the other parity's.
# Q is zero elsewhere, so every top key scores exactly the same and its P is 1.
comptime STIM_HALF_SPLIT = 7


# ===-----------------------------------------------------------------------===#
# Helpers
# ===-----------------------------------------------------------------------===#


def _student_t(nu: Int) -> Float64:
    var chi2 = Float64(0)
    for _ in range(nu):
        var z = randn_float64()
        chi2 += z * z
    return randn_float64() / sqrt(chi2 / Float64(nu))


def _heavy_tailed(stim: Int) -> BFloat16:
    """Returns a scaled, clipped Student-t sample (nu = 3 or 7).

    The scale gives log2 scores a standard deviation of about 2; the clip
    keeps V within a few bf16 ULPs of the 5e-2 gate.
    """
    var nu = 3 if stim == STIM_T3 else 7
    var scale = 0.4 if stim == STIM_T3 else 0.5
    var x = min(max(_student_t(nu) * scale, -4.0), 4.0)
    return x.cast[.bfloat16]()


def _spike_level(stim: Int, tile: Int) -> Int:
    if stim == STIM_SPIKE_ODD:
        return (tile + 1) // 2
    if stim == STIM_SPIKE_EVEN:
        return tile // 2
    if stim == STIM_SPIKE_EVERY:
        return tile
    return -tile


def _gcd(a: Int, b: Int) -> Int:
    var x = a
    var y = b
    while y != 0:
        var t = y
        y = x % y
        x = t
    return x


def _coprime_multiplier(n: Int) -> Int:
    if n <= 1:
        return 1
    if _gcd(3, n) == 1:
        return 3
    if _gcd(5, n) == 1:
        return 5
    if _gcd(7, n) == 1:
        return 7
    if _gcd(11, n) == 1:
        return 11
    return 13


def _late_writer[
    dtype: DType
](
    dst: MutPointer[Scalar[dtype], MutAnyOrigin],
    src: MutPointer[Scalar[dtype], MutAnyOrigin],
    n: Int64,
    delay_ns: UInt64,
):
    """Triggers its PDL successor at once, then overwrites `dst` late."""
    launch_dependent_grids()
    var t0 = global_perf_counter_ns()
    while global_perf_counter_ns() - t0 < delay_ns:
        pass
    for i in range(Int(thread_idx.x), Int(n), Int(block_dim.x)):
        dst[unsafe_offset=i] = src[unsafe_offset=i]


# ===-----------------------------------------------------------------------===#
# Host-side reference: FP8 Q (576) x BF16 K^T (576) -> P -> BF16 O
# ===-----------------------------------------------------------------------===#


def host_reference_with_attn_sink(
    q_ptr: Pointer[Float8_e4m3fn, _],
    k_bf16_ptr: Pointer[BFloat16, _],
    output_ptr: MutPointer[BFloat16, _],
    attn_sink_host: Pointer[Float32, _],
    batch_size: Int,
    num_heads: Int,
    num_keys: Int,
    depth: Int,
    v_depth: Int,
    scale: Float32,
):
    """Reference MLA with attn_sink correction (natural log domain).

    attn_sink is shape [num_heads_q]. The softmax denominator is adjusted:
      sum_exp += exp(attn_sink[h] - max_s)
    which captures the aggregate contribution of non-selected tokens.
    """
    for b in range(batch_size):
        for h in range(num_heads):
            var q_base = b * num_heads * depth + h * depth

            var max_s = Float64(min_or_neg_inf[.float32]())
            var s_buf = List(length=num_keys, fill=Float64(0))
            for k in range(num_keys):
                var k_base = b * num_keys * depth + k * depth
                var dot = Float64(0)
                for d in range(depth):
                    dot += (
                        q_ptr[q_base + d].cast[.float64]()
                        * k_bf16_ptr[k_base + d].cast[.float64]()
                    )
                s_buf[k] = dot * Float64(scale)
                if s_buf[k] > max_s:
                    max_s = s_buf[k]

            var attn_sink_val = Float64(attn_sink_host[h])
            if attn_sink_val > max_s:
                max_s = attn_sink_val

            var sum_exp = Float64(0)
            for k in range(num_keys):
                s_buf[k] = exp(s_buf[k] - max_s)
                sum_exp += s_buf[k]
            sum_exp += exp(attn_sink_val - max_s)

            for k in range(num_keys):
                s_buf[k] = s_buf[k] / sum_exp

            var o_base = b * num_heads * v_depth + h * v_depth
            for d in range(v_depth):
                var acc = Float64(0)
                for k in range(num_keys):
                    var k_base = b * num_keys * depth + k * depth
                    acc += s_buf[k] * k_bf16_ptr[k_base + d].cast[.float64]()
                output_ptr[o_base + d] = acc.cast[.bfloat16]()
            _ = s_buf^


def host_reference_varkeys(
    q_ptr: Pointer[Float8_e4m3fn, _],
    k_bf16_ptr: Pointer[BFloat16, _],
    output_ptr: MutPointer[BFloat16, _],
    batch_size: Int,
    num_heads: Int,
    num_keys_per_batch: List[Int],
    depth: Int,
    v_depth: Int,
    scale: Float32,
):
    """Reference MLA for variable-length per-batch sparse attention.

    K buffer is packed contiguously per batch: k_bf16_ptr contains
    sum_b num_keys_per_batch[b] * depth elements.
    """
    var k_offsets = List(length=batch_size, fill=Int(0))
    var running = 0
    for b in range(batch_size):
        k_offsets[b] = running
        running += num_keys_per_batch[b] * depth

    for b in range(batch_size):
        var num_keys = num_keys_per_batch[b]
        var k_base_offset = k_offsets[b]
        for h in range(num_heads):
            var q_base = b * num_heads * depth + h * depth

            var max_s = Float64(min_or_neg_inf[.float32]())
            var s_buf = List(length=num_keys, fill=Float64(0))
            for k in range(num_keys):
                var k_base = k_base_offset + k * depth
                var dot = Float64(0)
                for d in range(depth):
                    dot += (
                        q_ptr[q_base + d].cast[.float64]()
                        * k_bf16_ptr[k_base + d].cast[.float64]()
                    )
                s_buf[k] = dot * Float64(scale)
                if s_buf[k] > max_s:
                    max_s = s_buf[k]

            var sum_exp = Float64(0)
            for k in range(num_keys):
                s_buf[k] = exp(s_buf[k] - max_s)
                sum_exp += s_buf[k]
            for k in range(num_keys):
                s_buf[k] = s_buf[k] / sum_exp

            var o_base = b * num_heads * v_depth + h * v_depth
            for d in range(v_depth):
                var acc = Float64(0)
                for k in range(num_keys):
                    var k_base = k_base_offset + k * depth
                    acc += s_buf[k] * k_bf16_ptr[k_base + d].cast[.float64]()
                output_ptr[o_base + d] = acc.cast[.bfloat16]()
            _ = s_buf^
    _ = k_offsets^


def _is_pad_slot(slot: Int, logical_pad: Int, logical_pad_half: Int) -> Bool:
    """Whether a logical cell's slot carries the -1 padding index."""
    return (
        logical_pad != 0
        and slot % logical_pad == logical_pad - 1
        and (logical_pad_half < 0 or (slot % 128) // 64 == logical_pad_half)
    )


def _sparse_reference(
    q_ptr: Pointer[Float8_e4m3fn, _],
    k_sparse_ref: Pointer[BFloat16, _],
    ref_ptr: MutPointer[BFloat16, _],
    selected_tokens: List[Int],
    batch_size: Int,
    q_max_seq_len: Int,
    num_heads: Int,
    topk: Int,
    row_depth: Int,
    cache_len: Int,
    use_causal: Bool,
    logical_pad: Int,
    logical_pad_half: Int,
    scale: Float32,
):
    """Reference for `_run_test_sparse_qkv_fp8`, in its padded row order.

    Query token s of every batch sees keys below `cache_len + s + 1` under
    the causal mask, whatever the batch's own length (`OffsetPosition`
    takes the cache length from the KV cache, not from the batch's tokens).
    """
    for b in range(batch_size):
        for s in range(q_max_seq_len):
            var g = b * q_max_seq_len + s
            var causal_limit = cache_len + s + 1
            for h in range(num_heads):
                var q_base = (
                    b * q_max_seq_len * num_heads * row_depth
                    + s * num_heads * row_depth
                    + h * row_depth
                )
                var max_s = Float64(min_or_neg_inf[.float32]())
                var s_buf = List(length=topk, fill=Float64(0))
                var valid = List(length=topk, fill=True)
                for i in range(topk):
                    if use_causal:
                        var tok = selected_tokens[g * topk + i]
                        if tok >= causal_limit or _is_pad_slot(
                            i, logical_pad, logical_pad_half
                        ):
                            valid[i] = False
                            s_buf[i] = Float64(min_or_neg_inf[.float32]())
                            continue
                    var k_base = g * topk * row_depth + i * row_depth
                    var dot = Float64(0)
                    for d in range(row_depth):
                        dot += (
                            q_ptr[q_base + d].cast[.float64]()
                            * k_sparse_ref[k_base + d].cast[.float64]()
                        )
                    s_buf[i] = dot * Float64(scale)
                    if s_buf[i] > max_s:
                        max_s = s_buf[i]

                var sum_exp = Float64(0)
                for i in range(topk):
                    if not valid[i]:
                        s_buf[i] = Float64(0)
                        continue
                    s_buf[i] = exp(s_buf[i] - max_s)
                    sum_exp += s_buf[i]
                if sum_exp > Float64(0):
                    for i in range(topk):
                        if valid[i]:
                            s_buf[i] = s_buf[i] / sum_exp

                var o_base = (
                    b * q_max_seq_len * num_heads * V_DEPTH
                    + s * num_heads * V_DEPTH
                    + h * V_DEPTH
                )
                for d in range(V_DEPTH):
                    var acc = Float64(0)
                    for i in range(topk):
                        if not valid[i]:
                            continue
                        var k_base = g * topk * row_depth + i * row_depth
                        acc += (
                            s_buf[i] * k_sparse_ref[k_base + d].cast[.float64]()
                        )
                    ref_ptr[o_base + d] = acc.cast[.bfloat16]()
                _ = valid^
                _ = s_buf^


# ===-----------------------------------------------------------------------===#
# Aggregate output verification (mirrors test_mla_prefill_sparse_qkv_fp8.mojo).
#
# A single element outside tolerance used to abort the whole suite, hiding the
# shape of a failure. This computes the same aggregate metrics the prefill
# harness reports -- cosine similarity, mean/max abs error, error-tail fraction,
# non-finite counts, non-zero fraction -- and PRINTS them for every case before
# gating, so a regression shows its full character (systemic vs a few outliers)
# in one run instead of dying on element 0.
# ===-----------------------------------------------------------------------===#


def verify_mla_output(
    name: StringLiteral,
    ref_ptr: Pointer[BFloat16, _],
    out_ptr: Pointer[BFloat16, _],
    num_elems: Int,
    atol: Float64 = 5e-2,
    mean_err_max: Float64 = 0.01,
    # Fraction of finite elements allowed to exceed |err| > 1e-2. Default matches
    # the prefill harness (0.15%); few-key cases (topk <= ~8, cache_len <= ~6)
    # pass a looser bound at their call site -- with so few softmax terms, FP8-P
    # quantization pushes a percent-ish of dims just past 1e-2 while the vector
    # stays cosine > 0.999 correct.
    tail_max: Float64 = 0.0015,
    break_expect: Int = BREAK_EITHER,
) raises:
    _ = _verify_mla_output(
        name,
        ref_ptr,
        out_ptr,
        num_elems,
        atol=atol,
        mean_err_max=mean_err_max,
        tail_max=tail_max,
        break_expect=break_expect,
    )


def _verify_mla_output(
    name: StringLiteral,
    ref_ptr: Pointer[BFloat16, _],
    out_ptr: Pointer[BFloat16, _],
    num_elems: Int,
    atol: Float64 = 5e-2,
    mean_err_max: Float64 = 0.01,
    tail_max: Float64 = 0.0015,
    break_expect: Int = BREAK_EITHER,
    # A failure the caller found outside the compared elements (a canary).
    extra_failure: String = String(),
) raises -> Bool:
    """Checks `out_ptr` against `ref_ptr`; returns whether it failed.

    Raises on a failure unless a negative control expects or ignores it,
    and on a pass the control expected to fail.
    """
    comptime COS_MIN = Float64(0.99)

    var max_err = Float64(0)
    var max_actual = Float64(0)
    var num_nonzero = 0
    var nan_actual = 0
    var inf_actual = 0
    var nan_ref = 0
    var inf_ref = 0
    var sum_abs_err = Float64(0)
    var dot_ar = Float64(0)
    var norm_a = Float64(0)
    var norm_r = Float64(0)
    var n_finite = 0
    var n_err_gt_1em2 = 0

    for i in range(num_elems):
        var ref_val = ref_ptr[i].cast[.float64]()
        var actual_val = out_ptr[i].cast[.float64]()
        var err = abs(actual_val - ref_val)
        if err > max_err:
            max_err = err
        if abs(actual_val) > max_actual:
            max_actual = abs(actual_val)
        if abs(actual_val) > 1e-6:
            num_nonzero += 1
        if isnan(actual_val):
            nan_actual += 1
        elif abs(actual_val) > 1.0e300:
            inf_actual += 1
        elif not isnan(ref_val) and abs(ref_val) <= 1.0e300:
            sum_abs_err += err
            if err > 1.0e-2:
                n_err_gt_1em2 += 1
            dot_ar += actual_val * ref_val
            norm_a += actual_val * actual_val
            norm_r += ref_val * ref_val
            n_finite += 1
        if isnan(ref_val):
            nan_ref += 1
        elif abs(ref_val) > 1.0e300:
            inf_ref += 1

    var mean_abs_err = sum_abs_err / Float64(max(n_finite, 1))
    var cos_denom = sqrt(norm_a) * sqrt(norm_r)
    var cosine: Float64
    if norm_a < 1e-12 and norm_r < 1e-12:
        cosine = 1.0
    elif cos_denom < 1e-12:
        cosine = 0.0
    else:
        cosine = dot_ar / cos_denom
    var tail_frac = Float64(n_err_gt_1em2) / Float64(max(n_finite, 1))

    print(
        t"  DIAG: {name}  cosine= {cosine}  max_err= {max_err}  mean_abs_err="
        t" {mean_abs_err}  |err|>1e-2= {n_err_gt_1em2}  tail= {tail_frac} "
        t" nan_actual= {nan_actual}  inf_actual= {inf_actual}  nonzero="
        t" {num_nonzero} / {num_elems}  max_abs_actual= {max_actual}"
    )

    # Gates apply AFTER the DIAG print so metrics are always visible on failure.
    var failure = String()
    if nan_actual != 0 or inf_actual != 0 or nan_ref != 0 or inf_ref != 0:
        failure = (
            String("non-finite values in ")
            + String(name)
            + ": nan_actual="
            + String(nan_actual)
            + " inf_actual="
            + String(inf_actual)
            + " nan_ref="
            + String(nan_ref)
            + " inf_ref="
            + String(inf_ref)
        )
    elif num_nonzero == 0 and num_elems > 0:
        failure = String("output is all-zero in ") + String(name)
    elif mean_abs_err > mean_err_max:
        failure = (
            String("mean_abs_err exceeded tolerance in ")
            + String(name)
            + ": "
            + String(mean_abs_err)
            + " > "
            + String(mean_err_max)
        )
    elif cosine < COS_MIN:
        failure = (
            String("cosine below tolerance in ")
            + String(name)
            + ": "
            + String(cosine)
            + " < "
            + String(COS_MIN)
        )
    elif tail_frac >= tail_max:
        failure = (
            String("error-tail exceeded bound in ")
            + String(name)
            + ": "
            + String(n_err_gt_1em2)
            + " elems (|err| > 1e-2) = "
            + String(tail_frac * 100)
            + "% >= "
            + String(tail_max * 100)
            + "%"
        )
    elif max_err > atol:
        failure = (
            String("max_err exceeded tolerance in ")
            + String(name)
            + ": "
            + String(max_err)
            + " > atol "
            + String(atol)
        )
    elif extra_failure.byte_length() > 0:
        failure = extra_failure

    comptime if HALF_ROWS_BREAK or P6_BREAK or PDL_CONTROL or CG2_SLICE_BREAK:
        if break_expect == BREAK_MUST_FAIL:
            if failure.byte_length() == 0:
                raise Error(
                    String("negative control passed in ")
                    + String(name)
                    + ": the broken kernel must fail it"
                )
            print("  EXPECTED_FAIL:", name, failure)
            return True
        if break_expect == BREAK_EITHER and failure.byte_length() > 0:
            print("  IGNORED_FAIL:", name, failure)
            return True
    if failure.byte_length() > 0:
        raise Error(failure)

    print(
        t"  PASSED: {name}  cosine= {cosine}  max_err= {max_err}  checked="
        t" {num_elems}  elements"
    )
    return False


def _pack_rows[
    dtype: DType
](
    ctx: DeviceContext,
    src: HostBuffer[dtype],
    row_g: List[Int],
    row_elems: Int,
    rows: Int,
) raises -> HostBuffer[dtype]:
    """Copies padded row `row_g[r]` of `src` to row r, and row 0 to the rows
    past `len(row_g)`."""
    var dst = ctx.enqueue_create_host_buffer[dtype](rows * row_elems)
    for r in range(rows):
        var g = row_g[r] if r < len(row_g) else row_g[0]
        for e in range(row_elems):
            dst[r * row_elems + e] = src[g * row_elems + e]
    return dst


def _decoy_indices(
    ctx: DeviceContext,
    selected_tokens: List[Int],
    lookup_table_host: HostBuffer[DType.uint32],
    batch_size: Int,
    q_max_seq_len: Int,
    topk: Int,
    num_keys: Int,
    max_pages_per_batch: Int,
) raises -> HostBuffer[DType.int32]:
    """Physical indices of `topk` tokens per query row, none of them in that
    row's selection, so a decode that reads them computes a wrong output."""
    var h_decoy = ctx.enqueue_create_host_buffer[.int32](
        batch_size * q_max_seq_len * topk
    )
    for bi in range(batch_size):
        for s in range(q_max_seq_len):
            var g = bi * q_max_seq_len + s
            var taken = List(length=num_keys, fill=False)
            for i in range(topk):
                taken[selected_tokens[g * topk + i]] = True
            var t = (num_keys // 2 + s) % num_keys
            for i in range(topk):
                while taken[t]:
                    t = (t + 1) % num_keys
                taken[t] = True
                var page_idx, tok_in_page = divmod(t, PAGE_SIZE)
                var block_id = Int(
                    lookup_table_host[bi * max_pages_per_batch + page_idx]
                )
                h_decoy[g * topk + i] = Int32(
                    block_id * PAGE_SIZE + tok_in_page
                )
            _ = taken^
    return h_decoy


def _check_output(
    name: StringLiteral,
    ctx: DeviceContext,
    out_dev: DeviceBuffer[DType.bfloat16],
    ref_ptr: Pointer[BFloat16, _],
    row_g: List[Int],
    verify_rows: Int,
    canary_lo: Int,
    out_rows: Int,
    row_elems: Int,
    tail_max: Float64,
    break_expect: Int,
    extra_failure: String = String(),
) raises -> Bool:
    """Verifies the first `verify_rows` packed rows against the padded
    reference and rows [canary_lo, out_rows) against the sentinel; returns
    whether the output failed."""
    var out_host = ctx.enqueue_create_host_buffer[.bfloat16](
        out_rows * row_elems
    )
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()
    var ref_packed = ctx.enqueue_create_host_buffer[.bfloat16](
        max(verify_rows, 1) * row_elems
    )
    for r in range(verify_rows):
        for e in range(row_elems):
            ref_packed[r * row_elems + e] = ref_ptr[row_g[r] * row_elems + e]
    var overwritten = 0
    for i in range(canary_lo * row_elems, out_rows * row_elems):
        if bitcast[DType.uint16](out_host[i]) != OUT_SENTINEL_BITS:
            overwritten += 1
    var canary = extra_failure
    if overwritten > 0:
        canary = (
            String("rows past the packed output overwritten in ")
            + String(name)
            + ": "
            + String(overwritten)
            + " elements"
        )
    return _verify_mla_output(
        name,
        ref_packed.unsafe_ptr(),
        out_host.unsafe_ptr(),
        verify_rows * row_elems,
        tail_max=tail_max,
        break_expect=break_expect,
        extra_failure=canary,
    )


# ===-----------------------------------------------------------------------===#
# Shared KV fixtures: one paged latent cache and its host reference per
# (batch, cache length, q length, top-k, row width, page layout, stimulus).
# Heads and query features vary inside a fixture; the kernel only reads the
# cache, so every cell built from it can run against the same device buffers.
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct _SparseKVFixture[kv_type: DType](ImplicitlyCopyable, Movable):
    var batch_size: Int
    var cache_len: Int
    var q_max_seq_len: Int
    var topk: Int
    var row_depth: Int
    var shared_index: Bool
    var order_mode: Int
    var stim: Int
    var blocks_host: HostBuffer[Self.kv_type]
    var blocks_device: DeviceBuffer[Self.kv_type]
    var lookup_table_host: HostBuffer[DType.uint32]
    var lookup_table_device: DeviceBuffer[DType.uint32]
    var cache_lengths_host: HostBuffer[DType.uint32]
    var cache_lengths_device: DeviceBuffer[DType.uint32]
    var k_sparse_ref: HostBuffer[DType.bfloat16]
    var indices_host: HostBuffer[DType.int32]
    var logical_indices_host: HostBuffer[DType.int32]
    var num_keys: Int

    def matches(
        self,
        batch_size: Int,
        cache_len: Int,
        q_max_seq_len: Int,
        topk: Int,
        row_depth: Int,
        shared_index: Bool,
        order_mode: Int,
        stim: Int,
    ) -> Bool:
        return (
            self.batch_size == batch_size
            and self.cache_len == cache_len
            and self.q_max_seq_len == q_max_seq_len
            and self.topk == topk
            and self.row_depth == row_depth
            and self.shared_index == shared_index
            and self.order_mode == order_mode
            and self.stim == stim
        )


def _build_sparse_kv_fixture[
    kv_type: DType,
    row_depth: Int,
](
    ctx: DeviceContext,
    batch_size: Int,
    cache_len: Int,
    q_max_seq_len: Int,
    topk: Int,
    shared_index: Bool,
    order_mode: Int,
    stim: Int,
) raises -> _SparseKVFixture[kv_type]:
    """Builds the paged cache and the reference K rows for one fixture."""
    comptime kv_params = KVCacheStaticParams(
        num_heads=KV_NUM_HEADS, head_size=row_depth, is_mla=True
    )
    comptime kv_dim2 = 1

    var num_keys = cache_len + q_max_seq_len
    var total_q_tokens = batch_size * q_max_seq_len
    var max_pages_per_batch = ceildiv(num_keys, PAGE_SIZE)
    var total_pages = batch_size * max_pages_per_batch
    var block_elems = (
        total_pages
        * kv_dim2
        * NUM_LAYERS
        * PAGE_SIZE
        * kv_params.num_heads
        * kv_params.head_size
    )

    var selected_tokens = List(length=total_q_tokens * topk, fill=Int(0))
    var sel_mult = _coprime_multiplier(num_keys)
    var perm_mult = _coprime_multiplier(topk)
    for bi in range(batch_size):
        for s in range(q_max_seq_len):
            var g = bi * q_max_seq_len + s
            for i in range(topk):
                if shared_index:
                    if order_mode == 1:
                        selected_tokens[g * topk + i] = topk - 1 - i
                    elif order_mode == 2:
                        selected_tokens[g * topk + i] = (i * perm_mult) % topk
                    else:
                        selected_tokens[g * topk + i] = i
                else:
                    selected_tokens[g * topk + i] = (
                        i * sel_mult + 1 + s
                    ) % num_keys

    var lut_size = batch_size * max_pages_per_batch
    var lookup_table_host = ctx.enqueue_create_host_buffer[.uint32](lut_size)
    for i in range(lut_size):
        lookup_table_host[i] = UInt32(0)
    var page_offset = 0
    for bi in range(batch_size):
        var np = ceildiv(num_keys, PAGE_SIZE)
        var mult = _coprime_multiplier(np)
        for p in range(np):
            var shuffled_p = (p * mult + 1) % np
            lookup_table_host[bi * max_pages_per_batch + p] = UInt32(
                page_offset + shuffled_p
            )
        page_offset += np

    var cache_lengths_host = ctx.enqueue_create_host_buffer[.uint32](batch_size)
    for i in range(batch_size):
        cache_lengths_host[i] = UInt32(cache_len)

    var blocks_host = ctx.enqueue_create_host_buffer[kv_type](block_elems)
    for i in range(block_elems):
        blocks_host[i] = Scalar[kv_type](0)

    var k_sparse_ref = ctx.enqueue_create_host_buffer[.bfloat16](
        total_q_tokens * topk * row_depth
    )

    if stim == STIM_RANDN:
        # The kernel only gathers the selected tokens, so randomize those rows
        # once and leave the rest of the cache zero. The paged layout, not the
        # token count, decides where each selected row lands.
        var token_row = List[Int](length=batch_size * num_keys, fill=-1)
        var unique_tokens = List[Int]()
        var batch_base = List[Int](length=batch_size + 1, fill=0)
        for bi in range(batch_size):
            batch_base[bi] = len(unique_tokens)
            for s in range(q_max_seq_len):
                var g = bi * q_max_seq_len + s
                for i in range(topk):
                    var t = selected_tokens[g * topk + i]
                    if token_row[bi * num_keys + t] < 0:
                        token_row[bi * num_keys + t] = len(unique_tokens)
                        unique_tokens.append(t)
        batch_base[batch_size] = len(unique_tokens)
        var rand_host = ctx.enqueue_create_host_buffer[.bfloat16](
            len(unique_tokens) * row_depth
        )
        randn(rand_host.as_span(), mean=0.0, standard_deviation=0.5)
        var compact = ctx.enqueue_create_host_buffer[kv_type](
            len(unique_tokens) * row_depth
        )
        for r in range(len(unique_tokens)):
            for d in range(row_depth):
                compact[r * row_depth + d] = rand_host[r * row_depth + d].cast[
                    kv_type
                ]()
        var tok_stride = kv_params.head_size
        var page_stride_elems = (
            kv_dim2
            * NUM_LAYERS
            * PAGE_SIZE
            * kv_params.num_heads
            * kv_params.head_size
        )
        for bi in range(batch_size):
            for u in range(batch_base[bi], batch_base[bi + 1]):
                var t = unique_tokens[u]
                var page_idx, tok_in_page = divmod(t, PAGE_SIZE)
                var block_id = Int(
                    lookup_table_host[bi * max_pages_per_batch + page_idx]
                )
                var base = (
                    block_id * page_stride_elems + tok_in_page * tok_stride
                )
                for d in range(row_depth):
                    blocks_host[base + d] = compact[u * row_depth + d]
        for bi in range(batch_size):
            for s in range(q_max_seq_len):
                var g = bi * q_max_seq_len + s
                for i in range(topk):
                    var t = selected_tokens[g * topk + i]
                    var u = token_row[bi * num_keys + t]
                    var dst_base = g * topk * row_depth + i * row_depth
                    for d in range(row_depth):
                        k_sparse_ref[dst_base + d] = compact[
                            u * row_depth + d
                        ].cast[DType.bfloat16]()
        _ = token_row^
        _ = unique_tokens^
        _ = batch_base^
        _ = rand_host
        _ = compact
    else:
        var k_bf16_total = batch_size * num_keys * row_depth
        var k_bf16_host = ctx.enqueue_create_host_buffer[.bfloat16](
            k_bf16_total
        )
        randn(k_bf16_host.as_span(), mean=0.0, standard_deviation=0.5)
        if stim == STIM_T3 or stim == STIM_T7:
            for i in range(k_bf16_total):
                k_bf16_host[i] = _heavy_tailed(stim)
        elif stim == STIM_HALF_SPLIT:
            if row_depth - V_DEPTH < 2 * q_max_seq_len:
                raise Error(
                    "STIM_HALF_SPLIT needs two rotary dims per query token"
                )
            for bi in range(batch_size):
                for s in range(q_max_seq_len):
                    var g = bi * q_max_seq_len + s
                    for i in range(topk):
                        var t = selected_tokens[g * topk + i]
                        var col = i % 64
                        var a_val = 18
                        var b_val = 18
                        if col == 0:
                            a_val = 36
                        elif col == 64 // 8:
                            b_val = 36
                        var a_dim = (
                            bi * num_keys * row_depth
                            + t * row_depth
                            + V_DEPTH
                            + 2 * s
                        )
                        k_bf16_host[a_dim] = BFloat16(a_val)
                        k_bf16_host[a_dim + 1] = BFloat16(b_val)
        else:
            # Rotary dimension V_DEPTH + s carries query token s's spike, so
            # it moves the scores and leaves V alone.
            if row_depth - V_DEPTH < q_max_seq_len:
                raise Error("spike stimuli need a rotary dim per query token")
            for bi in range(batch_size):
                for s in range(q_max_seq_len):
                    var g = bi * q_max_seq_len + s
                    for i in range(topk):
                        var t = selected_tokens[g * topk + i]
                        k_bf16_host[
                            bi * num_keys * row_depth
                            + t * row_depth
                            + V_DEPTH
                            + s
                        ] = Float64(9 * _spike_level(stim, i // 64 + s)).cast[
                            .bfloat16
                        ]()

        var tok_stride = kv_params.head_size
        var page_stride_elems = (
            kv_dim2
            * NUM_LAYERS
            * PAGE_SIZE
            * kv_params.num_heads
            * kv_params.head_size
        )
        for bi in range(batch_size):
            for t in range(num_keys):
                var page_idx, tok_in_page = divmod(t, PAGE_SIZE)
                var block_id = Int(
                    lookup_table_host[bi * max_pages_per_batch + page_idx]
                )
                var base = (
                    block_id * page_stride_elems + tok_in_page * tok_stride
                )
                var k_base = bi * num_keys * row_depth + t * row_depth
                for d in range(row_depth):
                    blocks_host[base + d] = k_bf16_host[k_base + d].cast[
                        kv_type
                    ]()
        var k_ref_host = ctx.enqueue_create_host_buffer[.bfloat16](k_bf16_total)
        for bi in range(batch_size):
            for t in range(num_keys):
                var page_idx, tok_in_page = divmod(t, PAGE_SIZE)
                var block_id = Int(
                    lookup_table_host[bi * max_pages_per_batch + page_idx]
                )
                var base = (
                    block_id * page_stride_elems + tok_in_page * tok_stride
                )
                var k_base = bi * num_keys * row_depth + t * row_depth
                for d in range(row_depth):
                    k_ref_host[k_base + d] = blocks_host[base + d].cast[
                        DType.bfloat16
                    ]()
        for bi in range(batch_size):
            for s in range(q_max_seq_len):
                var g = bi * q_max_seq_len + s
                for i in range(topk):
                    var t = selected_tokens[g * topk + i]
                    var src_base = bi * num_keys * row_depth + t * row_depth
                    var dst_base = g * topk * row_depth + i * row_depth
                    for d in range(row_depth):
                        k_sparse_ref[dst_base + d] = k_ref_host[src_base + d]
        _ = k_bf16_host
        _ = k_ref_host

    var total_indices = total_q_tokens * topk
    var indices_host = ctx.enqueue_create_host_buffer[.int32](total_indices)
    var logical_indices_host = ctx.enqueue_create_host_buffer[.int32](
        total_indices
    )
    for bi in range(batch_size):
        for s in range(q_max_seq_len):
            var g = bi * q_max_seq_len + s
            for i in range(topk):
                var t = selected_tokens[g * topk + i]
                var page_idx, tok_in_page = divmod(t, PAGE_SIZE)
                var block_id = Int(
                    lookup_table_host[bi * max_pages_per_batch + page_idx]
                )
                indices_host[g * topk + i] = Int32(
                    block_id * PAGE_SIZE + tok_in_page
                )
                logical_indices_host[g * topk + i] = Int32(t)

    var blocks_device = ctx.enqueue_create_buffer[kv_type](block_elems)
    ctx.enqueue_copy(blocks_device, blocks_host)
    var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    ctx.enqueue_copy(cache_lengths_device, cache_lengths_host)
    var lookup_table_device = ctx.enqueue_create_buffer[.uint32](lut_size)
    ctx.enqueue_copy(lookup_table_device, lookup_table_host)
    ctx.synchronize()

    _ = selected_tokens^
    return _SparseKVFixture[kv_type](
        batch_size=batch_size,
        cache_len=cache_len,
        q_max_seq_len=q_max_seq_len,
        topk=topk,
        row_depth=row_depth,
        shared_index=shared_index,
        order_mode=order_mode,
        stim=stim,
        blocks_host=blocks_host,
        blocks_device=blocks_device,
        lookup_table_host=lookup_table_host,
        lookup_table_device=lookup_table_device,
        cache_lengths_host=cache_lengths_host,
        cache_lengths_device=cache_lengths_device,
        k_sparse_ref=k_sparse_ref,
        indices_host=indices_host,
        logical_indices_host=logical_indices_host,
        num_keys=num_keys,
    )


def _sparse_kv_fixture_index[
    kv_type: DType,
    row_depth: Int,
](
    mut fixtures: List[_SparseKVFixture[kv_type]],
    ctx: DeviceContext,
    batch_size: Int,
    cache_len: Int,
    q_max_seq_len: Int,
    topk: Int,
    shared_index: Bool,
    order_mode: Int,
    stim: Int,
) raises -> Int:
    """Returns the fixture for this shape, building it on first use."""
    for i in range(len(fixtures)):
        if fixtures[i].matches(
            batch_size,
            cache_len,
            q_max_seq_len,
            topk,
            row_depth,
            shared_index,
            order_mode,
            stim,
        ):
            return i
    fixtures.append(
        _build_sparse_kv_fixture[kv_type, row_depth](
            ctx,
            batch_size,
            cache_len,
            q_max_seq_len,
            topk,
            shared_index,
            order_mode,
            stim,
        )
    )
    return len(fixtures) - 1


# ===-----------------------------------------------------------------------===#
# Core test function: native FP8-Q + all-FP8-KV sparse decode.
# Parametrized over mask type and shared-index fold via comptime flags.
# ===-----------------------------------------------------------------------===#


comptime LATENT_DEPTH = 512


def run_nope_native_512_fp8[
    num_heads: Int,
](
    ctx: DeviceContext,
    batch_size: Int,
    cache_len: Int,
    *,
    topk: Int = -1,
    q_max_seq_len: Int = 1,
    name: StringLiteral = "",
) raises:
    _run_nope_native_512_fp8[num_heads=num_heads,](
        ctx=ctx,
        batch_size=batch_size,
        cache_len=cache_len,
        topk=topk,
        q_max_seq_len=q_max_seq_len,
        name=name,
    )


def _run_nope_native_512_fp8[
    num_heads: Int,
](
    ctx: DeviceContext,
    batch_size: Int,
    cache_len: Int,
    *,
    topk: Int = -1,
    q_max_seq_len: Int = 1,
    name: StringLiteral = "",
) raises:
    """The native-FP8 kernel takes a 512-wide NoPE row and answers identically.

    Same property as the BF16-KV oracle
    (`test_mla_decode_sparse_kv_bf16.run_nope_native_512`), on the backend a
    NoPE model actually runs: one problem, run once with both operands padded
    to 576 with a zero tail and once at their native 512, required
    **bit-identical**.

    This path is worth pinning separately because it reaches the tail through
    different machinery -- an INT64-packed gather into a linear staging buffer,
    then a re-swizzle warpgroup that permutes it into the SW64 MMA layout. The
    narrow row has to shorten the gather, the staging row stride and the
    re-swizzle's column count together; any one of them left wide would corrupt
    the operand rather than merely waste bandwidth.
    """
    comptime q_type = DType.float8_e4m3fn
    var num_keys = cache_len + q_max_seq_len
    var eff_topk = num_keys if topk < 0 else topk
    comptime scale = Float32(0.125)

    print(
        t"NoPE native 512 (QKV FP8):  {name}  batch_size= {batch_size} "
        t" cache_len= {cache_len}  num_heads= {num_heads}"
    )
    seed(0x5EED18)

    comptime kv_params_576 = KVCacheStaticParams(
        num_heads=KV_NUM_HEADS, head_size=Q_DEPTH, is_mla=True
    )
    comptime kv_params_512 = KVCacheStaticParams(
        num_heads=KV_NUM_HEADS, head_size=LATENT_DEPTH, is_mla=True
    )
    comptime kv_dim2 = 1

    var pages_per_batch = ceildiv(num_keys, PAGE_SIZE)
    var total_pages = batch_size * pages_per_batch
    var rows_meta = (
        total_pages * kv_dim2 * NUM_LAYERS * PAGE_SIZE * KV_NUM_HEADS
    )

    var block_shape_576 = (
        total_pages,
        kv_dim2,
        NUM_LAYERS,
        PAGE_SIZE,
        KV_NUM_HEADS,
        Q_DEPTH,
    )
    var block_shape_512 = (
        total_pages,
        kv_dim2,
        NUM_LAYERS,
        PAGE_SIZE,
        KV_NUM_HEADS,
        LATENT_DEPTH,
    )

    # One latent cache, quantized once, laid out at both widths so the two runs
    # see exactly the same FP8 bytes over [0, 512).
    var src = ctx.enqueue_create_host_buffer[.bfloat16](
        rows_meta * LATENT_DEPTH
    )
    randn(src.as_span(), mean=0.0, standard_deviation=0.5)
    var blocks_512_host = ctx.enqueue_create_host_buffer[q_type](
        rows_meta * LATENT_DEPTH
    )
    var blocks_576_host = ctx.enqueue_create_host_buffer[q_type](
        rows_meta * Q_DEPTH
    )
    for r in range(rows_meta):
        for d in range(LATENT_DEPTH):
            var v = src[r * LATENT_DEPTH + d].cast[q_type]()
            blocks_512_host[r * LATENT_DEPTH + d] = v
            blocks_576_host[r * Q_DEPTH + d] = v
        for d in range(LATENT_DEPTH, Q_DEPTH):
            blocks_576_host[r * Q_DEPTH + d] = Scalar[q_type](0)

    var lut_size = batch_size * pages_per_batch
    var lookup_table_host = ctx.enqueue_create_host_buffer[.uint32](lut_size)
    for bi in range(batch_size):
        for p in range(pages_per_batch):
            lookup_table_host[bi * pages_per_batch + p] = UInt32(
                bi * pages_per_batch + p
            )
    var cache_lengths_host = ctx.enqueue_create_host_buffer[.uint32](batch_size)
    for i in range(batch_size):
        cache_lengths_host[i] = UInt32(cache_len)

    var total_q_tokens = batch_size * q_max_seq_len
    var q_rows = total_q_tokens * num_heads
    var q_src = ctx.enqueue_create_host_buffer[.bfloat16](q_rows * LATENT_DEPTH)
    randn(q_src.as_span(), mean=0.0, standard_deviation=0.5)
    var q_512_host = ctx.enqueue_create_host_buffer[q_type](
        q_rows * LATENT_DEPTH
    )
    var q_576_host = ctx.enqueue_create_host_buffer[q_type](q_rows * Q_DEPTH)
    for row in range(q_rows):
        for d in range(LATENT_DEPTH):
            var v = q_src[row * LATENT_DEPTH + d].cast[q_type]()
            q_512_host[row * LATENT_DEPTH + d] = v
            q_576_host[row * Q_DEPTH + d] = v
        for d in range(LATENT_DEPTH, Q_DEPTH):
            q_576_host[row * Q_DEPTH + d] = Scalar[q_type](0)

    # Indices are per QUERY TOKEN, not per batch: the kernel walks
    # `d_indices` with a stride of `topk` for each of the `total_q_tokens`
    # rows. These coincide only when `q_max_seq_len == 1`.
    var total_indices = total_q_tokens * eff_topk
    var h_indices = ctx.enqueue_create_host_buffer[.int32](total_indices)
    for bi in range(batch_size):
        for si in range(q_max_seq_len):
            var g = bi * q_max_seq_len + si
            for i in range(eff_topk):
                var page_idx, tok_in_page = divmod(i, PAGE_SIZE)
                var block_id = Int(
                    lookup_table_host[bi * pages_per_batch + page_idx]
                )
                h_indices[g * eff_topk + i] = Int32(
                    block_id * PAGE_SIZE + tok_in_page
                )

    var out_size = total_q_tokens * num_heads * V_DEPTH

    var blocks_576_dev = ctx.enqueue_create_buffer[q_type](rows_meta * Q_DEPTH)
    ctx.enqueue_copy(blocks_576_dev, blocks_576_host)
    var blocks_512_dev = ctx.enqueue_create_buffer[q_type](
        rows_meta * LATENT_DEPTH
    )
    ctx.enqueue_copy(blocks_512_dev, blocks_512_host)
    var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    ctx.enqueue_copy(cache_lengths_device, cache_lengths_host)
    var lookup_table_device = ctx.enqueue_create_buffer[.uint32](lut_size)
    ctx.enqueue_copy(lookup_table_device, lookup_table_host)
    var q_576_dev = ctx.enqueue_create_buffer[q_type](q_rows * Q_DEPTH)
    ctx.enqueue_copy(q_576_dev, q_576_host)
    var q_512_dev = ctx.enqueue_create_buffer[q_type](q_rows * LATENT_DEPTH)
    ctx.enqueue_copy(q_512_dev, q_512_host)
    var d_indices_device = ctx.enqueue_create_buffer[.int32](total_indices)
    ctx.enqueue_copy(d_indices_device, h_indices)
    var out_576 = ctx.enqueue_create_buffer[.bfloat16](out_size)
    var out_512 = ctx.enqueue_create_buffer[.bfloat16](out_size)

    var row_offsets_host = ctx.enqueue_create_host_buffer[.uint32](
        batch_size + 1
    )
    for i in range(batch_size + 1):
        row_offsets_host[i] = UInt32(i * q_max_seq_len)
    var row_offsets_device = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    ctx.enqueue_copy(row_offsets_device, row_offsets_host)
    ctx.synchronize()

    var cache_lengths_tt = (
        TileTensor(cache_lengths_device, row_major(Int64(batch_size)))
        .as_imm()
        .as_unsafe_any_origin()
    )
    var lookup_table_tt = (
        TileTensor(
            lookup_table_device,
            row_major(Int64(batch_size), Int64(pages_per_batch)),
        )
        .as_imm()
        .as_unsafe_any_origin()
    )
    comptime Collection576 = PagedKVCacheCollection[
        q_type,
        kv_params_576,
        PAGE_SIZE,
        MutAnyOrigin,
        ImmutAnyOrigin,
        ImmutAnyOrigin,
        MutAnyOrigin,
    ]
    var kv_576 = Collection576(
        rebind[Collection576.blocks_tt_type](
            TileTensor(
                blocks_576_dev,
                row_major(
                    Int64(block_shape_576[0]),
                    Idx[1],
                    Int64(block_shape_576[2]),
                    Idx[PAGE_SIZE],
                    Idx[kv_params_576.num_heads],
                    Idx[kv_params_576.head_size],
                ),
            ).as_unsafe_any_origin()
        ),
        cache_lengths_tt,
        lookup_table_tt,
        UInt32(q_max_seq_len),
        UInt32(cache_len),
    )
    comptime Collection512 = PagedKVCacheCollection[
        q_type,
        kv_params_512,
        PAGE_SIZE,
        MutAnyOrigin,
        ImmutAnyOrigin,
        ImmutAnyOrigin,
        MutAnyOrigin,
    ]
    var kv_512 = Collection512(
        rebind[Collection512.blocks_tt_type](
            TileTensor(
                blocks_512_dev,
                row_major(
                    Int64(block_shape_512[0]),
                    Idx[1],
                    Int64(block_shape_512[2]),
                    Idx[PAGE_SIZE],
                    Idx[kv_params_512.num_heads],
                    Idx[kv_params_512.head_size],
                ),
            ).as_unsafe_any_origin()
        ),
        cache_lengths_tt,
        lookup_table_tt,
        UInt32(q_max_seq_len),
        UInt32(cache_len),
    )

    var row_offsets_tt = TileTensor(
        row_offsets_device, row_major(batch_size + 1)
    )
    var args_576 = MLADispatchScalarArgs[num_heads=num_heads, is_fp8_kv=True](
        batch_size, cache_len, q_max_seq_len, ctx
    )
    var args_512 = MLADispatchScalarArgs[num_heads=num_heads, is_fp8_kv=True](
        batch_size, cache_len, q_max_seq_len, ctx
    )

    flare_mla_decoding[
        rank=3,
        config=MHAConfig[q_type](num_heads, Q_DEPTH),
        ragged=True,
        sparse=True,
        kernels=SM100MLA,
    ](
        TileTensor(
            out_576,
            row_major(total_q_tokens, Idx[num_heads], Idx[V_DEPTH]),
        ),
        TileTensor(
            q_576_dev,
            row_major(total_q_tokens, Idx[num_heads], Idx[Q_DEPTH]),
        ),
        kv_576.get_key_cache(0),
        NullMask(),
        row_offsets_tt,
        scale,
        ctx,
        args_576.gpu_tile_tensor(),
        d_indices=rebind[MutPointer[Int32, MutAnyOrigin]](
            d_indices_device.unsafe_ptr()
        ),
        indices_stride=eff_topk,
    )
    ctx.synchronize()

    flare_mla_decoding[
        rank=3,
        config=MHAConfig[q_type](num_heads, LATENT_DEPTH),
        ragged=True,
        sparse=True,
        kernels=SM100MLA,
    ](
        TileTensor(
            out_512,
            row_major(total_q_tokens, Idx[num_heads], Idx[V_DEPTH]),
        ),
        TileTensor(
            q_512_dev,
            row_major(total_q_tokens, Idx[num_heads], Idx[LATENT_DEPTH]),
        ),
        kv_512.get_key_cache(0),
        NullMask(),
        row_offsets_tt,
        scale,
        ctx,
        args_512.gpu_tile_tensor(),
        d_indices=rebind[MutPointer[Int32, MutAnyOrigin]](
            d_indices_device.unsafe_ptr()
        ),
        indices_stride=eff_topk,
    )
    ctx.synchronize()

    var h576 = ctx.enqueue_create_host_buffer[.bfloat16](out_size)
    var h512 = ctx.enqueue_create_host_buffer[.bfloat16](out_size)
    ctx.enqueue_copy(h576, out_576)
    ctx.enqueue_copy(h512, out_512)
    ctx.synchronize()

    var mismatches = 0
    for i in range(out_size):
        if isnan(h512[i].cast[.float64]()):
            raise Error("NaN in native-512 QKV-FP8 decode output")
        if h576[i] != h512[i]:
            mismatches += 1
            if mismatches <= 5:
                print(
                    t"  mismatch idx= {i}  padded576= {h576[i]}  native512="
                    t" {h512[i]}"
                )
    if mismatches > 0:
        print(t"  FAILED:  {mismatches} of {out_size} outputs differ")
        raise Error("native-512 QKV-FP8 decode does not match padded 576")
    print("  PASSED: native 512 is bit-identical to zero-padded 576")
    _ = args_576
    _ = args_512
    _ = blocks_576_dev
    _ = blocks_512_dev
    _ = cache_lengths_device
    _ = lookup_table_device
    _ = q_576_dev
    _ = q_512_dev
    _ = d_indices_device
    _ = out_576
    _ = out_512
    _ = row_offsets_device


def run_test_sparse_qkv_fp8[
    kv_type: DType,  # float8_e4m3fn
    num_heads: Int,
    use_causal: Bool = False,
    # Read-once shared-index fold (KERN-3141).
    shared_index: Bool = False,
    # Physical gather order of the ONE shared set (shared_index only).
    # 0=identity, 1=reversed, 2=coprime permutation.
    order_mode: Int = 0,
    # Phase 5 seq_len=0: make the LAST batch a 0-length (empty) ragged sequence.
    empty_last_batch: Bool = False,
    # Width of the KV row and the absorbed Q as the model stores them. 576
    # carries a rotary tail after the 512 latent; a NoPE model stores the
    # latent alone and the kernel takes that row directly.
    row_depth: Int = Q_DEPTH,
    # Pass token positions and let the kernel decode them through the page
    # table (the graph-op path) instead of precomputed physical rows.
    logical: Bool = False,
](
    name: StringLiteral,
    batch_size: Int,
    cache_len: Int,
    ctx: DeviceContext,
    mut fixtures: List[_SparseKVFixture[kv_type]],
    topk: Int,
    q_max_seq_len: Int = 1,
    forced_np: Int = 0,
    tail_max: Float64 = 0.0015,
    stim: Int = STIM_RANDN,
    break_expect: Int = BREAK_EITHER,
    # Logical cells only: every `logical_pad`-th slot's logical index is -1
    # (padding), which the kernel must mask.
    logical_pad: Int = 0,
    # Restricts that padding to one half of every 128-slot pair tile: 0 the
    # leader's keys (slots [0, 64)), 1 the follower's; -1 pads both.
    logical_pad_half: Int = -1,
    # `MLA_P6_BREAK` kinds that must fail this cell (bit k for kind k).
    p6_fail: Int = 0,
    # The split-K controls (`SPLIT_BREAK`) must fail this cell.
    split_fail: Bool = False,
    # The no-pair-max control (`MLA_HALF_ROWS_BREAK=1`) must fail this cell.
    hr1_fail: Bool = False,
    # Odd batches (and the last) keep fewer than q_max_seq_len query tokens,
    # packed back to back; rows past the packed output must stay untouched.
    short_batches: Bool = False,
    # PDL_LATE_INDICES / PDL_LATE_Q: a kernel that triggers at entry writes
    # the true indices or Q 200 us late, right before the decode launch.
    # PDL_REUSE: PDL_REUSE_CALLS decodes back to back with no host sync,
    # alternating Q and -Q, into distinct outputs.
    pdl_mode: Int = PDL_NONE,
) raises:
    _run_test_sparse_qkv_fp8[
        kv_type=kv_type,
        num_heads=num_heads,
        use_causal=use_causal,
        shared_index=shared_index,
        order_mode=order_mode,
        empty_last_batch=empty_last_batch,
        row_depth=row_depth,
        logical=logical,
    ](
        name=name,
        batch_size=batch_size,
        cache_len=cache_len,
        ctx=ctx,
        fixtures=fixtures,
        topk=topk,
        q_max_seq_len=q_max_seq_len,
        forced_np=forced_np,
        tail_max=tail_max,
        stim=stim,
        break_expect=break_expect,
        logical_pad=logical_pad,
        logical_pad_half=logical_pad_half,
        p6_fail=p6_fail,
        split_fail=split_fail,
        hr1_fail=hr1_fail,
        short_batches=short_batches,
        pdl_mode=pdl_mode,
    )


def _run_test_sparse_qkv_fp8[
    kv_type: DType,  # float8_e4m3fn
    num_heads: Int,
    use_causal: Bool = False,
    # Read-once shared-index fold (KERN-3141).
    shared_index: Bool = False,
    # Physical gather order of the ONE shared set (shared_index only).
    # 0=identity, 1=reversed, 2=coprime permutation.
    order_mode: Int = 0,
    # Phase 5 seq_len=0: make the LAST batch a 0-length (empty) ragged sequence.
    empty_last_batch: Bool = False,
    # Width of the KV row and the absorbed Q as the model stores them. 576
    # carries a rotary tail after the 512 latent; a NoPE model stores the
    # latent alone and the kernel takes that row directly.
    row_depth: Int = Q_DEPTH,
    # Pass token positions and let the kernel decode them through the page
    # table (the graph-op path) instead of precomputed physical rows.
    logical: Bool = False,
](
    name: StringLiteral,
    batch_size: Int,
    cache_len: Int,
    ctx: DeviceContext,
    mut fixtures: List[_SparseKVFixture[kv_type]],
    topk: Int,
    q_max_seq_len: Int = 1,
    forced_np: Int = 0,
    tail_max: Float64 = 0.0015,
    stim: Int = STIM_RANDN,
    break_expect: Int = BREAK_EITHER,
    # Logical cells only: every `logical_pad`-th slot's logical index is -1
    # (padding), which the kernel must mask.
    logical_pad: Int = 0,
    # Restricts that padding to one half of every 128-slot pair tile: 0 the
    # leader's keys (slots [0, 64)), 1 the follower's; -1 pads both.
    logical_pad_half: Int = -1,
    # `MLA_P6_BREAK` kinds that must fail this cell (bit k for kind k).
    p6_fail: Int = 0,
    # The split-K controls (`SPLIT_BREAK`) must fail this cell.
    split_fail: Bool = False,
    # The no-pair-max control (`MLA_HALF_ROWS_BREAK=1`) must fail this cell.
    hr1_fail: Bool = False,
    # Odd batches (and the last) keep fewer than q_max_seq_len query tokens,
    # packed back to back; rows past the packed output must stay untouched.
    short_batches: Bool = False,
    # PDL_LATE_INDICES / PDL_LATE_Q: a kernel that triggers at entry writes
    # the true indices or Q 200 us late, right before the decode launch.
    # PDL_REUSE: PDL_REUSE_CALLS decodes back to back with no host sync,
    # alternating Q and -Q, into distinct outputs.
    pdl_mode: Int = PDL_NONE,
) raises:
    print(
        t"test: {name}  batch_size: {batch_size}  cache_len: {cache_len} "
        t" num_heads: {num_heads}  topk: {topk}  q_max_seq_len:"
        t" {q_max_seq_len}  causal: {use_causal}"
    )

    var num_keys = cache_len + q_max_seq_len
    var total_q_tokens = batch_size * q_max_seq_len
    comptime scale = Float32(0.125)
    if logical_pad != 0 and not (logical and use_causal):
        raise Error("logical_pad needs a causal logical cell")

    comptime kv_params = KVCacheStaticParams(
        num_heads=KV_NUM_HEADS, head_size=row_depth, is_mla=True
    )
    comptime kv_dim2 = 1

    var total_pages = batch_size * ceildiv(num_keys, PAGE_SIZE)
    var max_pages_per_batch = ceildiv(num_keys, PAGE_SIZE)

    var block_shape = (
        total_pages,
        kv_dim2,
        NUM_LAYERS,
        PAGE_SIZE,
        kv_params.num_heads,
        kv_params.head_size,
    )
    # Select topk unique tokens PER QUERY TOKEN via deterministic permutation.
    var selected_tokens = List(length=total_q_tokens * topk, fill=Int(0))
    var sel_mult = _coprime_multiplier(num_keys)
    var perm_mult = _coprime_multiplier(topk)
    for bi in range(batch_size):
        for s in range(q_max_seq_len):
            var g = bi * q_max_seq_len + s
            for i in range(topk):
                comptime if shared_index:
                    if order_mode == 1:
                        selected_tokens[g * topk + i] = topk - 1 - i
                    elif order_mode == 2:
                        selected_tokens[g * topk + i] = (i * perm_mult) % topk
                    else:
                        selected_tokens[g * topk + i] = i
                else:
                    selected_tokens[g * topk + i] = (
                        i * sel_mult + 1 + s
                    ) % num_keys

    var fixture_idx = _sparse_kv_fixture_index[kv_type, row_depth](
        fixtures,
        ctx,
        batch_size,
        cache_len,
        q_max_seq_len,
        topk,
        shared_index,
        order_mode,
        stim,
    )
    var fixture = fixtures[fixture_idx]

    # Q tensor: generate BF16 randn, cast to FP8 for the kernel.
    var q_size = batch_size * q_max_seq_len * num_heads * row_depth
    var q_bf16_scratch = ctx.enqueue_create_host_buffer[.bfloat16](q_size)
    randn(q_bf16_scratch.as_span(), mean=0.0, standard_deviation=0.5)
    if stim == STIM_T3 or stim == STIM_T7:
        for i in range(q_size):
            q_bf16_scratch[i] = _heavy_tailed(stim)
    elif stim == STIM_HALF_SPLIT:
        for bi in range(batch_size):
            for s in range(q_max_seq_len):
                for h in range(num_heads):
                    var q_base = (
                        (bi * q_max_seq_len + s) * num_heads + h
                    ) * row_depth
                    # The nope dims are zeroed too: their randn scores would
                    # make the tiles' top keys differ by about 1 log2 unit,
                    # and the FP8 P of the runner-up then misses the float64
                    # reference by up to 6% on every form, half-row or not.
                    for d in range(row_depth):
                        q_bf16_scratch[q_base + d] = BFloat16(0)
                    q_bf16_scratch[q_base + V_DEPTH + 2 * s + h % 2] = BFloat16(
                        8
                    )
    elif stim != STIM_RANDN:
        # 8 * 9 * scale * log2(e) = 13 log2 units per spike step. Zeroing
        # the other tokens' spike dimensions keeps their levels out of this
        # token's scores.
        for bi in range(batch_size):
            for s in range(q_max_seq_len):
                for h in range(num_heads):
                    var q_base = (
                        (bi * q_max_seq_len + s) * num_heads + h
                    ) * row_depth
                    for s2 in range(q_max_seq_len):
                        q_bf16_scratch[q_base + V_DEPTH + s2] = BFloat16(
                            8 if s2 == s else 0
                        )
    var q_host = ctx.enqueue_create_host_buffer[.float8_e4m3fn](q_size)
    for i in range(q_size):
        q_host[i] = q_bf16_scratch[i].cast[.float8_e4m3fn]()
    var out_size = batch_size * q_max_seq_len * num_heads * V_DEPTH
    var ref_host = ctx.enqueue_create_host_buffer[.bfloat16](out_size)

    _sparse_reference(
        q_host.unsafe_ptr(),
        fixture.k_sparse_ref.unsafe_ptr(),
        ref_host.unsafe_ptr(),
        selected_tokens,
        batch_size,
        q_max_seq_len,
        num_heads,
        topk,
        row_depth,
        cache_len,
        use_causal,
        logical_pad,
        logical_pad_half,
        scale,
    )
    # -Q: the late-Q poison and the reuse cells' second input set.
    var q_neg_host = ctx.enqueue_create_host_buffer[.float8_e4m3fn](q_size)
    var ref_neg_host = ctx.enqueue_create_host_buffer[.bfloat16](out_size)
    if pdl_mode == PDL_LATE_Q or pdl_mode == PDL_REUSE:
        for i in range(q_size):
            q_neg_host[i] = (-q_host[i].cast[.float32]()).cast[.float8_e4m3fn]()
        if pdl_mode == PDL_REUSE:
            _sparse_reference(
                q_neg_host.unsafe_ptr(),
                fixture.k_sparse_ref.unsafe_ptr(),
                ref_neg_host.unsafe_ptr(),
                selected_tokens,
                batch_size,
                q_max_seq_len,
                num_heads,
                topk,
                row_depth,
                cache_len,
                use_causal,
                logical_pad,
                logical_pad_half,
                scale,
            )

    # Device rows are packed: row r holds padded query row row_g[r]. Without
    # short batches the map is the identity. A short batch b keeps
    # seq_lens[b] tokens; the q_max_seq_len index and output rows past the
    # packed ones stay in range for a kernel that misses a ragged exit, and
    # the output ones must keep the sentinel.
    var seq_lens = List(length=batch_size, fill=q_max_seq_len)
    if short_batches:
        for b in range(batch_size):
            if b % 2 == 1:
                seq_lens[b] = max(1, q_max_seq_len - 1 - b % 3)
        seq_lens[batch_size - 1] = max(1, q_max_seq_len - 2)
    var row_g = List[Int]()
    for b in range(batch_size):
        for s in range(seq_lens[b]):
            row_g.append(b * q_max_seq_len + s)
    var q_rows = len(row_g)
    var out_rows = q_rows + (q_max_seq_len if short_batches else 0)
    var q_row_elems = num_heads * row_depth
    var out_row_elems = num_heads * V_DEPTH

    # -----------------------------------------------------------------------
    # Copy to device
    # -----------------------------------------------------------------------
    var q_dev_host = _pack_rows(ctx, q_host, row_g, q_row_elems, q_rows)
    var q_device = ctx.enqueue_create_buffer[.float8_e4m3fn](
        q_rows * q_row_elems
    )
    ctx.enqueue_copy(q_device, q_dev_host)
    var q_neg_dev_host = _pack_rows(ctx, q_neg_host, row_g, q_row_elems, q_rows)
    var q_neg_device = ctx.enqueue_create_buffer[.float8_e4m3fn](
        q_rows * q_row_elems
    )
    ctx.enqueue_copy(q_neg_device, q_neg_dev_host)

    var out_init_host = ctx.enqueue_create_host_buffer[.bfloat16](
        out_rows * out_row_elems
    )
    for i in range(out_rows * out_row_elems):
        out_init_host[i] = bitcast[DType.bfloat16](OUT_SENTINEL_BITS)
    var out_device = ctx.enqueue_create_buffer[.bfloat16](
        out_rows * out_row_elems
    )
    ctx.enqueue_copy(out_device, out_init_host)

    ctx.synchronize()

    # -----------------------------------------------------------------------
    # Build PagedKVCacheCollection on device
    # -----------------------------------------------------------------------
    var blocks_tt = TileTensor(
        fixture.blocks_device,
        row_major(
            Int64(block_shape[0]),
            Idx[1],
            Int64(block_shape[2]),
            Idx[PAGE_SIZE],
            Idx[kv_params.num_heads],
            Idx[kv_params.head_size],
        ),
    )

    var cache_lengths_tt = TileTensor(
        fixture.cache_lengths_device, row_major(Int64(batch_size))
    )

    var lookup_table_tt = TileTensor(
        fixture.lookup_table_device,
        row_major(Int64(batch_size), Int64(max_pages_per_batch)),
    )

    comptime Collection = PagedKVCacheCollection[
        kv_type,
        kv_params,
        PAGE_SIZE,
        MutAnyOrigin,
        ImmutAnyOrigin,
        ImmutAnyOrigin,
        MutAnyOrigin,
    ]
    var kv_collection = Collection(
        rebind[Collection.blocks_tt_type](blocks_tt.as_unsafe_any_origin()),
        cache_lengths_tt.as_imm().as_unsafe_any_origin(),
        lookup_table_tt.as_imm().as_unsafe_any_origin(),
        UInt32(q_max_seq_len),
        UInt32(cache_len),
    )

    var kv_cache = kv_collection.get_key_cache(0)
    var kv_lut = KVCacheMHAOperand(kv_cache)

    # -----------------------------------------------------------------------
    # Build gather4 indices for the selected topk tokens.
    # -----------------------------------------------------------------------
    var total_indices = total_q_tokens * topk
    var h_indices_dev: HostBuffer[DType.int32]
    comptime if logical:
        h_indices_dev = _pack_rows(
            ctx, fixture.logical_indices_host, row_g, topk, out_rows
        )
    else:
        h_indices_dev = _pack_rows(
            ctx, fixture.indices_host, row_g, topk, out_rows
        )
    var d_indices_device = ctx.enqueue_create_buffer[.int32](out_rows * topk)
    ctx.enqueue_copy(d_indices_device, h_indices_dev)
    ctx.synchronize()

    # Logical sparse indices, matching production: `selected_tokens` is a
    # score-sorted, non-monotonic permutation of logical positions, so this
    # feeds the causal-by-logical-position mask its expected input. Only
    # meaningful when use_causal.
    var h_logical_indices = ctx.enqueue_create_host_buffer[.int32](
        total_indices
    )
    for idx in range(total_indices):
        h_logical_indices[idx] = Int32(selected_tokens[idx])
        if _is_pad_slot(idx % topk, logical_pad, logical_pad_half):
            h_logical_indices[idx] = -1
    var h_logical_dev = _pack_rows(
        ctx, h_logical_indices, row_g, topk, out_rows
    )
    var logical_indices_device = ctx.enqueue_create_buffer[.int32](
        out_rows * topk
    )
    ctx.enqueue_copy(logical_indices_device, h_logical_dev)
    ctx.synchronize()

    # -----------------------------------------------------------------------
    # Build TileTensors and call flare_mla_decoding.
    # -----------------------------------------------------------------------
    var row_offsets_host = ctx.enqueue_create_host_buffer[.uint32](
        batch_size + 1
    )
    row_offsets_host[0] = UInt32(0)
    for i in range(batch_size):
        row_offsets_host[i + 1] = row_offsets_host[i] + UInt32(seq_lens[i])
    if empty_last_batch and batch_size > 1:
        row_offsets_host[batch_size] = row_offsets_host[batch_size - 1]
    var row_offsets_device = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    ctx.enqueue_copy(row_offsets_device, row_offsets_host)
    ctx.synchronize()

    var row_offsets_tt = TileTensor(
        row_offsets_device, row_major(batch_size + 1)
    )

    var mla_args = MLADispatchScalarArgs[
        num_heads=num_heads,
        is_fp8_kv=True,
        fold_shared_index=shared_index,
    ](batch_size, cache_len, q_max_seq_len, ctx)
    var scalar_args_buf_tt = mla_args.gpu_tile_tensor()

    comptime sm_count = ctx.default_device_info.sm_count
    var dispatch_scalars = compute_mla_dispatch_scalars[
        num_heads=num_heads,
        is_fp8_kv=True,
        half_sms=sm_count // 2,
        fold_shared_index=shared_index,
    ](batch_size, cache_len, q_max_seq_len, sm_count)
    var num_partitions = dispatch_scalars[2]
    print(
        t"  num_partitions= {num_partitions}  (split-K"
        t""" {"ACTIVE" if num_partitions > 1 else "OFF"} )"""
    )

    var indices_stride = topk

    var _np_ovr = Optional[Int](forced_np) if forced_np > 0 else Optional[Int](
        None
    )

    print(
        t"  Launching MLA sparse QKV_FP8 decode kernel...  topk= {topk} "
        t" num_keys= {num_keys}"
    )

    def _decode(
        q_buf: DeviceBuffer[.float8_e4m3fn],
        mut out_buf: DeviceBuffer[.bfloat16],
    ) raises {imm}:
        var q_tt = TileTensor(
            q_buf, row_major(q_rows, Idx[num_heads], Idx[row_depth])
        )
        var out_tt = TileTensor(
            out_buf, row_major(out_rows, Idx[num_heads], Idx[V_DEPTH])
        )
        comptime if use_causal:
            flare_mla_decoding[
                rank=3,
                config=MHAConfig[.float8_e4m3fn](num_heads, row_depth),
                ragged=True,
                sparse=True,
                fold_shared_index=shared_index,
                physical_indices=not logical,
                kernels=SM100MLA,
            ](
                out_tt,
                q_tt,
                kv_cache,
                CausalMask(),
                row_offsets_tt,
                scale,
                ctx,
                scalar_args_buf_tt,
                d_indices=rebind[MutPointer[Int32, MutAnyOrigin]](
                    d_indices_device.unsafe_ptr()
                ),
                indices_stride=indices_stride,
                num_partitions_in=_np_ovr,
                logical_indices=rebind[MutPointer[Int32, MutAnyOrigin]](
                    logical_indices_device.unsafe_ptr()
                ),
            )
        else:
            flare_mla_decoding[
                rank=3,
                config=MHAConfig[.float8_e4m3fn](num_heads, row_depth),
                ragged=True,
                sparse=True,
                fold_shared_index=shared_index,
                physical_indices=not logical,
                kernels=SM100MLA,
            ](
                out_tt,
                q_tt,
                kv_cache,
                NullMask(),
                row_offsets_tt,
                scale,
                ctx,
                scalar_args_buf_tt,
                d_indices=rebind[MutPointer[Int32, MutAnyOrigin]](
                    d_indices_device.unsafe_ptr()
                ),
                indices_stride=indices_stride,
                num_partitions_in=_np_ovr,
            )

    # The late writer and the decode sit back to back in the stream, so the
    # PDL-launched split decode may start as soon as the writer triggers. A
    # warm-up decode first loads the kernels and grows the allocator, since a
    # module load or an allocation on the first call blocks the host until
    # the writer is done and the race is never reached.
    comptime late_delay_ns = UInt64(1_000_000)
    var reuse_outs = List[DeviceBuffer[.bfloat16]]()
    if pdl_mode == PDL_LATE_INDICES:
        if logical or 2 * topk > num_keys:
            raise Error(
                "PDL_LATE_INDICES needs physical indices, 2*topk <= keys"
            )
        var h_decoy = _decoy_indices(
            ctx,
            selected_tokens,
            fixture.lookup_table_host,
            batch_size,
            q_max_seq_len,
            topk,
            num_keys,
            max_pages_per_batch,
        )
        var h_decoy_dev = _pack_rows(ctx, h_decoy, row_g, topk, out_rows)
        _decode(q_device, out_device)
        ctx.enqueue_copy(out_device, out_init_host)
        var d_true = ctx.enqueue_create_buffer[.int32](out_rows * topk)
        ctx.enqueue_copy(d_true, h_indices_dev)
        ctx.enqueue_copy(d_indices_device, h_decoy_dev)
        ctx.synchronize()
        ctx.enqueue_function[_late_writer[DType.int32]](
            rebind[MutPointer[Int32, MutAnyOrigin]](
                d_indices_device.unsafe_ptr()
            ),
            rebind[MutPointer[Int32, MutAnyOrigin]](d_true.unsafe_ptr()),
            Int64(out_rows * topk),
            late_delay_ns,
            grid_dim=1,
            block_dim=256,
        )
        _decode(q_device, out_device)
        ctx.synchronize()
        _ = d_true
    elif pdl_mode == PDL_LATE_Q:
        _decode(q_device, out_device)
        ctx.enqueue_copy(out_device, out_init_host)
        var q_true = ctx.enqueue_create_buffer[.float8_e4m3fn](
            q_rows * q_row_elems
        )
        ctx.enqueue_copy(q_true, q_dev_host)
        ctx.enqueue_copy(q_device, q_neg_dev_host)
        ctx.synchronize()
        ctx.enqueue_function[_late_writer[DType.float8_e4m3fn]](
            rebind[MutPointer[Float8_e4m3fn, MutAnyOrigin]](
                q_device.unsafe_ptr()
            ),
            rebind[MutPointer[Float8_e4m3fn, MutAnyOrigin]](
                q_true.unsafe_ptr()
            ),
            Int64(q_rows * q_row_elems),
            late_delay_ns,
            grid_dim=1,
            block_dim=256,
        )
        _decode(q_device, out_device)
        ctx.synchronize()
        _ = q_true
    elif pdl_mode == PDL_REUSE:
        for _ in range(PDL_REUSE_CALLS):
            var o = ctx.enqueue_create_buffer[.bfloat16](
                out_rows * out_row_elems
            )
            ctx.enqueue_copy(o, out_init_host)
            reuse_outs.append(o)
        ctx.synchronize()
        for c in range(PDL_REUSE_CALLS):
            if c % 2 == 0:
                _decode(q_device, reuse_outs[c])
            else:
                _decode(q_neg_device, reuse_outs[c])
        ctx.synchronize()
    else:
        _decode(q_device, out_device)
        ctx.synchronize()

    # -----------------------------------------------------------------------
    # Verify output.
    # -----------------------------------------------------------------------
    # `empty_last_batch` zero-fills the final ragged sequence; skip it by
    # verifying only the contiguous valid prefix (output is batch-major).
    var valid_batches = batch_size - 1 if empty_last_batch else batch_size
    var verify_rows = q_rows if short_batches else valid_batches * q_max_seq_len
    var expect = break_expect
    comptime if SPLIT_BREAK:
        expect = BREAK_MUST_FAIL if split_fail else BREAK_EITHER
    elif HALF_ROWS_BREAK:
        var rows = num_heads * (q_max_seq_len if shared_index else 1)
        comptime if HALF_ROWS_BREAK_KIND == 5:
            # Every causal sparse cell passes `logical_indices`, so the
            # ballot decides its mask whether or not the gather is logical.
            if not (use_causal and not shared_index and rows <= 16):
                expect = BREAK_MUST_PASS
            elif logical_pad != 0:
                expect = BREAK_MUST_FAIL
            else:
                expect = BREAK_EITHER
        elif HALF_ROWS_BREAK_KIND == 1:
            # Both threads of a row still reduce to one stored max, so only
            # a max below the row's true one, which inflates P, can show.
            if rows > 16:
                expect = BREAK_MUST_PASS
            elif hr1_fail:
                expect = BREAK_MUST_FAIL
            else:
                expect = BREAK_EITHER
        else:
            expect = BREAK_MUST_FAIL if rows <= 16 else BREAK_MUST_PASS
    comptime if P6_BREAK:
        expect = (
            BREAK_MUST_FAIL if (p6_fail >> P6_BREAK_KIND) & 1 else BREAK_EITHER
        )
    comptime if PDL_CONTROL:
        var late = pdl_mode == PDL_LATE_INDICES or pdl_mode == PDL_LATE_Q
        var ragged_cell = short_batches and not shared_index
        if not PDL_WAIT and late:
            expect = BREAK_MUST_FAIL
        elif (
            RAGGED_EXIT_BREAK_KIND == 1
            and ragged_cell
            and forced_np > 1
            and forced_np <= 8
        ):
            expect = BREAK_MUST_FAIL
        elif RAGGED_EXIT_BREAK_KIND == 2 and ragged_cell and forced_np == 1:
            expect = BREAK_MUST_FAIL
        else:
            expect = BREAK_EITHER
    var mismatch = String()
    if pdl_mode == PDL_REUSE:
        var failed = 0
        for c in range(PDL_REUSE_CALLS):
            var bad = _check_output(
                name,
                ctx,
                reuse_outs[c],
                ref_host.unsafe_ptr() if c % 2
                == 0 else ref_neg_host.unsafe_ptr(),
                row_g,
                verify_rows,
                q_rows,
                out_rows,
                out_row_elems,
                tail_max,
                expect,
            )
            if bad:
                failed += 1
        print("  REUSE:", name, failed, "of", PDL_REUSE_CALLS, "outputs failed")
    else:
        _ = _check_output(
            name,
            ctx,
            out_device,
            ref_host.unsafe_ptr(),
            row_g,
            verify_rows,
            q_rows if short_batches else out_rows,
            out_rows,
            out_row_elems,
            tail_max,
            expect,
            mismatch,
        )
    _ = mla_args
    _ = q_neg_device
    _ = logical_indices_device
    _ = fixture.blocks_device
    _ = fixture.cache_lengths_device
    _ = fixture.lookup_table_device
    _ = q_device
    _ = out_device
    _ = d_indices_device
    _ = row_offsets_device
    _ = selected_tokens^


# ===-----------------------------------------------------------------------===#
# Variable-topk sparse test (has_variable_topk=True)
# ===-----------------------------------------------------------------------===#


def run_test_sparse_qkv_fp8_variable_topk[
    kv_type: DType,
    num_heads: Int,
    # Width of the KV row and the absorbed Q as the model stores them. 576
    # carries a rotary tail after the 512 latent; a NoPE model stores the
    # latent alone and the kernel takes that row directly.
    row_depth: Int = Q_DEPTH,
](
    name: StringLiteral,
    cache_lengths: List[Int],
    topk_per_batch: List[Int],
    ctx: DeviceContext,
) raises:
    _run_test_sparse_qkv_fp8_variable_topk[
        kv_type=kv_type,
        num_heads=num_heads,
        row_depth=row_depth,
    ](
        name=name,
        cache_lengths=cache_lengths,
        topk_per_batch=topk_per_batch,
        ctx=ctx,
    )


def _run_test_sparse_qkv_fp8_variable_topk[
    kv_type: DType,
    num_heads: Int,
    # Width of the KV row and the absorbed Q as the model stores them. 576
    # carries a rotary tail after the 512 latent; a NoPE model stores the
    # latent alone and the kernel takes that row directly.
    row_depth: Int = Q_DEPTH,
](
    name: StringLiteral,
    cache_lengths: List[Int],
    topk_per_batch: List[Int],
    ctx: DeviceContext,
) raises:
    """Variable-topk per-batch with native FP8-Q + all-FP8-KV (576-byte row).

    Per-batch cache lengths and topk counts. indices_stride = max(topk).
    """
    var batch_size = len(cache_lengths)
    comptime q_max_seq_len = 1
    comptime scale = Float32(0.125)

    var max_topk = 0
    for bi in range(batch_size):
        if topk_per_batch[bi] > max_topk:
            max_topk = topk_per_batch[bi]

    print(t"test: {name}  batch_size: {batch_size}  num_heads: {num_heads}")
    for i in range(batch_size):
        print(
            t"  batch {i} : cache_len= {cache_lengths[i]}  topk="
            t" {topk_per_batch[i]}"
        )
    print("  indices_stride (max topk)=", max_topk)

    var max_cache_len = 0
    var total_pages = 0
    var num_keys_list = List[Int]()
    for i in range(batch_size):
        var cl = cache_lengths[i]
        if cl > max_cache_len:
            max_cache_len = cl
        var nk = cl + q_max_seq_len
        num_keys_list.append(nk)
        total_pages += ceildiv(nk, PAGE_SIZE)

    var max_num_keys = max_cache_len + q_max_seq_len
    var max_pages_per_batch = ceildiv(max_num_keys, PAGE_SIZE)

    comptime kv_params = KVCacheStaticParams(
        num_heads=KV_NUM_HEADS, head_size=row_depth, is_mla=True
    )
    comptime kv_dim2 = 1

    var block_shape = (
        total_pages,
        kv_dim2,
        NUM_LAYERS,
        PAGE_SIZE,
        kv_params.num_heads,
        kv_params.head_size,
    )
    var block_elems = (
        total_pages
        * kv_dim2
        * NUM_LAYERS
        * PAGE_SIZE
        * kv_params.num_heads
        * kv_params.head_size
    )
    var tok_stride = kv_params.head_size

    var blocks_host = ctx.enqueue_create_host_buffer[kv_type](block_elems)
    for i in range(block_elems):
        blocks_host[i] = Scalar[kv_type](0)
    var lut_size = batch_size * max_pages_per_batch
    var lookup_table_host = ctx.enqueue_create_host_buffer[.uint32](lut_size)
    for i in range(lut_size):
        lookup_table_host[i] = UInt32(0)
    var page_offset = 0
    for bi in range(batch_size):
        var np_bi = ceildiv(num_keys_list[bi], PAGE_SIZE)
        var mult_bi = _coprime_multiplier(np_bi)
        for p in range(np_bi):
            var shuffled_p = (p * mult_bi + 1) % np_bi
            lookup_table_host[bi * max_pages_per_batch + p] = UInt32(
                page_offset + shuffled_p
            )
        page_offset += np_bi

    var cache_lengths_host = ctx.enqueue_create_host_buffer[.uint32](batch_size)
    for i in range(batch_size):
        cache_lengths_host[i] = UInt32(cache_lengths[i])

    var page_stride_elems = (
        kv_dim2
        * NUM_LAYERS
        * PAGE_SIZE
        * kv_params.num_heads
        * kv_params.head_size
    )

    var total_k_elems = 0
    for bi in range(batch_size):
        total_k_elems += num_keys_list[bi] * row_depth
    var k_bf16_host = ctx.enqueue_create_host_buffer[.bfloat16](total_k_elems)
    randn(k_bf16_host.as_span(), mean=0.0, standard_deviation=0.5)

    var k_ref_host = ctx.enqueue_create_host_buffer[.bfloat16](total_k_elems)

    var k_offset = 0
    for bi in range(batch_size):
        var nk = num_keys_list[bi]
        for t in range(nk):
            var page_idx, tok_in_page = divmod(t, PAGE_SIZE)
            var physical_page = Int(
                lookup_table_host[bi * max_pages_per_batch + page_idx]
            )
            var base = (
                physical_page * page_stride_elems + tok_in_page * tok_stride
            )
            var k_base = k_offset + t * row_depth
            for d in range(row_depth):
                var fp8_val = k_bf16_host[k_base + d].cast[kv_type]()
                blocks_host[base + d] = fp8_val
                k_ref_host[k_base + d] = fp8_val.cast[.bfloat16]()
        k_offset += nk * row_depth

    # Q: generate BF16 randn, cast to FP8.
    var q_size = batch_size * num_heads * row_depth
    var q_bf16_scratch = ctx.enqueue_create_host_buffer[.bfloat16](q_size)
    randn(q_bf16_scratch.as_span(), mean=0.0, standard_deviation=0.5)
    var q_host = ctx.enqueue_create_host_buffer[.float8_e4m3fn](q_size)
    for i in range(q_size):
        q_host[i] = q_bf16_scratch[i].cast[.float8_e4m3fn]()

    var total_indices = batch_size * max_topk
    var h_indices = ctx.enqueue_create_host_buffer[.int32](total_indices)
    for i in range(total_indices):
        h_indices[i] = Int32(0)
    var total_sparse_ref_elems = 0
    for bi in range(batch_size):
        total_sparse_ref_elems += topk_per_batch[bi] * row_depth
    var k_sparse_ref = ctx.enqueue_create_host_buffer[.bfloat16](
        total_sparse_ref_elems
    )

    var sparse_ref_offset = 0
    var k_offset_src = 0
    for bi in range(batch_size):
        var nk = num_keys_list[bi]
        var topk_bi = topk_per_batch[bi]
        var mult = _coprime_multiplier(nk)
        for i in range(topk_bi):
            var t = (i * mult + 1) % nk
            var page_idx, tok_in_page = divmod(t, PAGE_SIZE)
            var physical_page = Int(
                lookup_table_host[bi * max_pages_per_batch + page_idx]
            )
            h_indices[bi * max_topk + i] = Int32(
                physical_page * PAGE_SIZE + tok_in_page
            )
            var src_base = k_offset_src + t * row_depth
            var dst_base = sparse_ref_offset + i * row_depth
            for d in range(row_depth):
                k_sparse_ref[dst_base + d] = k_ref_host[src_base + d]
        sparse_ref_offset += topk_bi * row_depth
        k_offset_src += nk * row_depth

    var sparse_num_keys_list = List[Int]()
    for bi in range(batch_size):
        sparse_num_keys_list.append(topk_per_batch[bi])

    var out_size = batch_size * num_heads * V_DEPTH
    var ref_host = ctx.enqueue_create_host_buffer[.bfloat16](out_size)
    host_reference_varkeys(
        q_host.unsafe_ptr(),
        k_sparse_ref.unsafe_ptr(),
        ref_host.unsafe_ptr(),
        batch_size,
        num_heads,
        sparse_num_keys_list,
        row_depth,
        V_DEPTH,
        scale,
    )

    var blocks_device = ctx.enqueue_create_buffer[kv_type](block_elems)
    ctx.enqueue_copy(blocks_device, blocks_host)
    var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    ctx.enqueue_copy(cache_lengths_device, cache_lengths_host)
    var lookup_table_device = ctx.enqueue_create_buffer[.uint32](lut_size)
    ctx.enqueue_copy(lookup_table_device, lookup_table_host)
    var q_device = ctx.enqueue_create_buffer[.float8_e4m3fn](q_size)
    ctx.enqueue_copy(q_device, q_host)
    var out_device = ctx.enqueue_create_buffer[.bfloat16](out_size)
    var d_indices_device = ctx.enqueue_create_buffer[.int32](total_indices)
    ctx.enqueue_copy(d_indices_device, h_indices)

    var topk_lengths_host = ctx.enqueue_create_host_buffer[.int32](batch_size)
    for bi in range(batch_size):
        topk_lengths_host[bi] = Int32(topk_per_batch[bi])
    var topk_lengths_device = ctx.enqueue_create_buffer[.int32](batch_size)
    ctx.enqueue_copy(topk_lengths_device, topk_lengths_host)

    ctx.synchronize()

    var blocks_tt = TileTensor(
        blocks_device,
        row_major(
            Int64(block_shape[0]),
            Idx[1],
            Int64(block_shape[2]),
            Idx[PAGE_SIZE],
            Idx[kv_params.num_heads],
            Idx[kv_params.head_size],
        ),
    )
    var cache_lengths_tt = TileTensor(
        cache_lengths_device, row_major(Int64(batch_size))
    )
    var lookup_table_tt = TileTensor(
        lookup_table_device,
        row_major(Int64(batch_size), Int64(max_pages_per_batch)),
    )
    comptime Collection = PagedKVCacheCollection[
        kv_type,
        kv_params,
        PAGE_SIZE,
        MutAnyOrigin,
        ImmutAnyOrigin,
        ImmutAnyOrigin,
        MutAnyOrigin,
    ]
    var kv_collection = Collection(
        rebind[Collection.blocks_tt_type](blocks_tt.as_unsafe_any_origin()),
        cache_lengths_tt.as_imm().as_unsafe_any_origin(),
        lookup_table_tt.as_imm().as_unsafe_any_origin(),
        UInt32(q_max_seq_len),
        UInt32(max_cache_len),
    )
    var kv_cache = kv_collection.get_key_cache(0)

    var q_tt = TileTensor(
        q_device, row_major(batch_size, Idx[num_heads], Idx[row_depth])
    )
    var out_tt = TileTensor(
        out_device, row_major(batch_size, Idx[num_heads], Idx[V_DEPTH])
    )

    var row_offsets_host = ctx.enqueue_create_host_buffer[.uint32](
        batch_size + 1
    )
    for i in range(batch_size + 1):
        row_offsets_host[i] = UInt32(i)
    var row_offsets_device = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    ctx.enqueue_copy(row_offsets_device, row_offsets_host)
    ctx.synchronize()

    var row_offsets_tt = TileTensor(
        row_offsets_device, row_major(batch_size + 1)
    )

    var mla_args = MLADispatchScalarArgs[
        num_heads=num_heads,
        is_fp8_kv=True,
    ](batch_size, max_cache_len, q_max_seq_len, ctx)
    var scalar_args_buf_tt = mla_args.gpu_tile_tensor()

    comptime sm_count = ctx.default_device_info.sm_count
    var dispatch_scalars = compute_mla_dispatch_scalars[
        num_heads=num_heads, is_fp8_kv=True, half_sms=sm_count // 2
    ](batch_size, max_cache_len, q_max_seq_len, sm_count)
    var num_partitions = dispatch_scalars[2]
    print(
        t"  num_partitions= {num_partitions}  (split-K"
        t""" {"ACTIVE" if num_partitions > 1 else "OFF"} )"""
    )

    print("  Launching MLA sparse QKV_FP8 (variable topk)...")

    flare_mla_decoding[
        rank=3,
        config=MHAConfig[.float8_e4m3fn](num_heads, row_depth),
        ragged=True,
        sparse=True,
        kernels=SM100MLA,
    ](
        out_tt,
        q_tt,
        kv_cache,
        NullMask(),
        row_offsets_tt,
        scale,
        ctx,
        scalar_args_buf_tt,
        d_indices=rebind[MutPointer[Int32, MutAnyOrigin]](
            d_indices_device.unsafe_ptr()
        ),
        indices_stride=max_topk,
        topk_lengths=NonNullPointer[DType.int32](topk_lengths_device),
    )
    ctx.synchronize()

    var out_host = ctx.enqueue_create_host_buffer[.bfloat16](out_size)
    ctx.enqueue_copy(out_host, out_device)
    ctx.synchronize()

    verify_mla_output(
        name,
        ref_host.unsafe_ptr(),
        out_host.unsafe_ptr(),
        batch_size * num_heads * V_DEPTH,
    )

    _ = mla_args
    _ = blocks_device
    _ = cache_lengths_device
    _ = lookup_table_device
    _ = q_device
    _ = out_device
    _ = d_indices_device
    _ = topk_lengths_device
    _ = row_offsets_device


# ===-----------------------------------------------------------------------===#
# Attention sink sparse test (has_attn_sink=True)
# ===-----------------------------------------------------------------------===#


def run_test_sparse_qkv_fp8_attn_sink[
    kv_type: DType,
    num_heads: Int,
    # Width of the KV row and the absorbed Q as the model stores them. 576
    # carries a rotary tail after the 512 latent; a NoPE model stores the
    # latent alone and the kernel takes that row directly.
    row_depth: Int = Q_DEPTH,
](
    name: StringLiteral,
    batch_size: Int,
    cache_len: Int,
    ctx: DeviceContext,
    topk: Int,
    forced_np: Int = 0,
    neg_inf_sinks: Bool = False,
) raises:
    _run_test_sparse_qkv_fp8_attn_sink[
        kv_type=kv_type,
        num_heads=num_heads,
        row_depth=row_depth,
    ](
        name=name,
        batch_size=batch_size,
        cache_len=cache_len,
        ctx=ctx,
        topk=topk,
        forced_np=forced_np,
        neg_inf_sinks=neg_inf_sinks,
    )


def _run_test_sparse_qkv_fp8_attn_sink[
    kv_type: DType,
    num_heads: Int,
    # Width of the KV row and the absorbed Q as the model stores them. 576
    # carries a rotary tail after the 512 latent; a NoPE model stores the
    # latent alone and the kernel takes that row directly.
    row_depth: Int = Q_DEPTH,
](
    name: StringLiteral,
    batch_size: Int,
    cache_len: Int,
    ctx: DeviceContext,
    topk: Int,
    forced_np: Int = 0,
    # Every fourth head gets a -inf sink, the sink-free arm of the
    # correction.
    neg_inf_sinks: Bool = False,
) raises:
    """Native FP8-Q sparse MLA decode with attn_sink correction.

    attn_sink shape is [num_heads_q] per the MOGG fix.
    """
    print(
        t"test: {name}  batch_size: {batch_size}  cache_len: {cache_len} "
        t" num_heads: {num_heads}  topk: {topk}"
    )

    comptime q_max_seq_len = 1
    var num_keys = cache_len + q_max_seq_len
    comptime scale = Float32(0.125)

    comptime kv_params = KVCacheStaticParams(
        num_heads=KV_NUM_HEADS, head_size=row_depth, is_mla=True
    )
    comptime kv_dim2 = 1

    var total_pages = batch_size * ceildiv(num_keys, PAGE_SIZE)
    var max_pages_per_batch = ceildiv(num_keys, PAGE_SIZE)

    var block_shape = (
        total_pages,
        kv_dim2,
        NUM_LAYERS,
        PAGE_SIZE,
        kv_params.num_heads,
        kv_params.head_size,
    )
    var block_elems = (
        total_pages
        * kv_dim2
        * NUM_LAYERS
        * PAGE_SIZE
        * kv_params.num_heads
        * kv_params.head_size
    )

    var blocks_host = ctx.enqueue_create_host_buffer[kv_type](block_elems)
    for i in range(block_elems):
        blocks_host[i] = Scalar[kv_type](0)
    var k_bf16_total = batch_size * num_keys * row_depth
    var k_bf16_host = ctx.enqueue_create_host_buffer[.bfloat16](k_bf16_total)
    randn(k_bf16_host.as_span(), mean=0.0, standard_deviation=0.5)

    var tok_stride = kv_params.head_size
    var lut_size = batch_size * max_pages_per_batch
    var lookup_table_host = ctx.enqueue_create_host_buffer[.uint32](lut_size)
    for i in range(lut_size):
        lookup_table_host[i] = UInt32(0)
    var page_offset = 0
    for bi in range(batch_size):
        var np = ceildiv(num_keys, PAGE_SIZE)
        var mult = _coprime_multiplier(np)
        for p in range(np):
            var shuffled_p = (p * mult + 1) % np
            lookup_table_host[bi * max_pages_per_batch + p] = UInt32(
                page_offset + shuffled_p
            )
        page_offset += np

    var page_stride_elems = (
        kv_dim2
        * NUM_LAYERS
        * PAGE_SIZE
        * kv_params.num_heads
        * kv_params.head_size
    )
    for bi in range(batch_size):
        for t in range(num_keys):
            var page_idx, tok_in_page = divmod(t, PAGE_SIZE)
            var block_id = Int(
                lookup_table_host[bi * max_pages_per_batch + page_idx]
            )
            var base = block_id * page_stride_elems + tok_in_page * tok_stride
            var k_base = bi * num_keys * row_depth + t * row_depth
            for d in range(row_depth):
                blocks_host[base + d] = k_bf16_host[k_base + d].cast[kv_type]()

    var k_ref_host = ctx.enqueue_create_host_buffer[.bfloat16](k_bf16_total)
    for bi in range(batch_size):
        for t in range(num_keys):
            var page_idx, tok_in_page = divmod(t, PAGE_SIZE)
            var block_id = Int(
                lookup_table_host[bi * max_pages_per_batch + page_idx]
            )
            var base = block_id * page_stride_elems + tok_in_page * tok_stride
            var k_base = bi * num_keys * row_depth + t * row_depth
            for d in range(row_depth):
                k_ref_host[k_base + d] = blocks_host[base + d].cast[
                    DType.bfloat16
                ]()

    var selected_tokens = List(length=batch_size * topk, fill=Int(0))
    for bi in range(batch_size):
        var mult = _coprime_multiplier(num_keys)
        for i in range(topk):
            selected_tokens[bi * topk + i] = (i * mult + 1) % num_keys

    var k_sparse_ref = ctx.enqueue_create_host_buffer[.bfloat16](
        batch_size * topk * row_depth
    )
    for bi in range(batch_size):
        for i in range(topk):
            var t = selected_tokens[bi * topk + i]
            var src = bi * num_keys * row_depth + t * row_depth
            var dst = bi * topk * row_depth + i * row_depth
            for d in range(row_depth):
                k_sparse_ref[dst + d] = k_ref_host[src + d]

    var attn_sink_host = ctx.enqueue_create_host_buffer[.float32](num_heads)
    for h in range(num_heads):
        attn_sink_host[h] = Float32(
            -1.0 + 3.0 * Float32(h) / Float32(num_heads)
        )
        if neg_inf_sinks and h % 4 == 3:
            attn_sink_host[h] = min_or_neg_inf[.float32]()

    # Q: generate BF16 randn, cast to FP8.
    var q_size = batch_size * num_heads * row_depth
    var q_bf16_scratch = ctx.enqueue_create_host_buffer[.bfloat16](q_size)
    randn(q_bf16_scratch.as_span(), mean=0.0, standard_deviation=0.3)
    var q_host = ctx.enqueue_create_host_buffer[.float8_e4m3fn](q_size)
    for i in range(q_size):
        q_host[i] = q_bf16_scratch[i].cast[.float8_e4m3fn]()

    var out_size = batch_size * num_heads * V_DEPTH
    var ref_host = ctx.enqueue_create_host_buffer[.bfloat16](out_size)
    host_reference_with_attn_sink(
        q_host.unsafe_ptr(),
        k_sparse_ref.unsafe_ptr(),
        ref_host.unsafe_ptr(),
        attn_sink_host.unsafe_ptr(),
        batch_size,
        num_heads,
        topk,
        row_depth,
        V_DEPTH,
        scale,
    )

    var blocks_device = ctx.enqueue_create_buffer[kv_type](block_elems)
    ctx.enqueue_copy(blocks_device, blocks_host)
    var cache_lengths_host = ctx.enqueue_create_host_buffer[.uint32](batch_size)
    for i in range(batch_size):
        cache_lengths_host[i] = UInt32(cache_len)
    var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    ctx.enqueue_copy(cache_lengths_device, cache_lengths_host)
    var lookup_table_device = ctx.enqueue_create_buffer[.uint32](lut_size)
    ctx.enqueue_copy(lookup_table_device, lookup_table_host)
    var q_device = ctx.enqueue_create_buffer[.float8_e4m3fn](q_size)
    ctx.enqueue_copy(q_device, q_host)
    var out_device = ctx.enqueue_create_buffer[.bfloat16](out_size)
    var attn_sink_device = ctx.enqueue_create_buffer[.float32](num_heads)
    ctx.enqueue_copy(attn_sink_device, attn_sink_host)

    ctx.synchronize()

    var blocks_tt = TileTensor(
        blocks_device,
        row_major(
            Int64(block_shape[0]),
            Idx[1],
            Int64(block_shape[2]),
            Idx[PAGE_SIZE],
            Idx[kv_params.num_heads],
            Idx[kv_params.head_size],
        ),
    )
    var cache_lengths_tt = TileTensor(
        cache_lengths_device, row_major(Int64(batch_size))
    )
    var lookup_table_tt = TileTensor(
        lookup_table_device,
        row_major(Int64(batch_size), Int64(max_pages_per_batch)),
    )
    comptime Collection = PagedKVCacheCollection[
        kv_type,
        kv_params,
        PAGE_SIZE,
        MutAnyOrigin,
        ImmutAnyOrigin,
        ImmutAnyOrigin,
        MutAnyOrigin,
    ]
    var kv_collection = Collection(
        rebind[Collection.blocks_tt_type](blocks_tt.as_unsafe_any_origin()),
        cache_lengths_tt.as_imm().as_unsafe_any_origin(),
        lookup_table_tt.as_imm().as_unsafe_any_origin(),
        UInt32(q_max_seq_len),
        UInt32(cache_len),
    )
    var kv_cache = kv_collection.get_key_cache(0)

    var total_indices = batch_size * topk
    var h_indices = ctx.enqueue_create_host_buffer[.int32](total_indices)
    for bi in range(batch_size):
        for i in range(topk):
            var t = selected_tokens[bi * topk + i]
            var page_idx, tok_in_page = divmod(t, PAGE_SIZE)
            var block_id = Int(
                lookup_table_host[bi * max_pages_per_batch + page_idx]
            )
            h_indices[bi * topk + i] = Int32(block_id * PAGE_SIZE + tok_in_page)

    var d_indices_device = ctx.enqueue_create_buffer[.int32](total_indices)
    ctx.enqueue_copy(d_indices_device, h_indices)
    ctx.synchronize()

    var q_tt = TileTensor(
        q_device, row_major(batch_size, Idx[num_heads], Idx[row_depth])
    )

    var row_offsets_host = ctx.enqueue_create_host_buffer[.uint32](
        batch_size + 1
    )
    for i in range(batch_size + 1):
        row_offsets_host[i] = UInt32(i)
    var row_offsets_device = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    ctx.enqueue_copy(row_offsets_device, row_offsets_host)
    ctx.synchronize()

    var row_offsets_tt = TileTensor(
        row_offsets_device, row_major(batch_size + 1)
    )

    var mla_args = MLADispatchScalarArgs[
        num_heads=num_heads,
        is_fp8_kv=True,
    ](batch_size, cache_len, q_max_seq_len, ctx)
    var scalar_args_buf_tt = mla_args.gpu_tile_tensor()

    comptime sm_count = ctx.default_device_info.sm_count
    var dispatch_scalars = compute_mla_dispatch_scalars[
        num_heads=num_heads, is_fp8_kv=True, half_sms=sm_count // 2
    ](batch_size, cache_len, q_max_seq_len, sm_count)
    var num_partitions = dispatch_scalars[2]
    print(
        t"  num_partitions= {num_partitions}  (split-K"
        t""" {"ACTIVE" if num_partitions > 1 else "OFF"} )"""
    )

    print("  Launching MLA sparse QKV_FP8 (attn_sink)...")
    var _np_ovr = Optional[Int](forced_np) if forced_np > 0 else Optional[Int](
        None
    )

    def _decode(mut out_buf: DeviceBuffer[.bfloat16]) raises {imm}:
        var out_tt = TileTensor(
            out_buf, row_major(batch_size, Idx[num_heads], Idx[V_DEPTH])
        )
        flare_mla_decoding[
            rank=3,
            config=MHAConfig[.float8_e4m3fn](num_heads, row_depth),
            ragged=True,
            sparse=True,
            kernels=SM100MLA,
        ](
            out_tt,
            q_tt,
            kv_cache,
            NullMask(),
            row_offsets_tt,
            scale,
            ctx,
            scalar_args_buf_tt,
            d_indices=rebind[MutPointer[Int32, MutAnyOrigin]](
                d_indices_device.unsafe_ptr()
            ),
            indices_stride=topk,
            num_partitions_in=_np_ovr,
            attn_sink_ptr=NonNullPointer[DType.float32](attn_sink_device),
        )

    _decode(out_device)
    ctx.synchronize()

    var out_host = ctx.enqueue_create_host_buffer[.bfloat16](out_size)
    ctx.enqueue_copy(out_host, out_device)
    ctx.synchronize()

    var mismatch = String()
    var expect = BREAK_EITHER
    _ = _verify_mla_output(
        name,
        ref_host.unsafe_ptr(),
        out_host.unsafe_ptr(),
        batch_size * num_heads * V_DEPTH,
        break_expect=expect,
        extra_failure=mismatch,
    )

    _ = mla_args
    _ = blocks_device
    _ = cache_lengths_device
    _ = lookup_table_device
    _ = q_device
    _ = out_device
    _ = d_indices_device
    _ = row_offsets_device
    _ = attn_sink_device
    _ = selected_tokens^


# ===-----------------------------------------------------------------------===#
# Extra KV sparse test (has_extra_kv=True)
# Both caches use the same 576-byte all-FP8 layout.
# ===-----------------------------------------------------------------------===#


def run_test_sparse_qkv_fp8_extra_kv[
    kv_type: DType,
    num_heads: Int,
    # Width of the KV row and the absorbed Q as the model stores them. 576
    # carries a rotary tail after the 512 latent; a NoPE model stores the
    # latent alone and the kernel takes that row directly.
    row_depth: Int = Q_DEPTH,
](
    name: StringLiteral,
    cache_lengths: List[Int],
    topk_per_batch: List[Int],
    extra_cache_lengths: List[Int],
    extra_topk_per_batch: List[Int],
    ctx: DeviceContext,
) raises:
    _run_test_sparse_qkv_fp8_extra_kv[
        kv_type=kv_type,
        num_heads=num_heads,
        row_depth=row_depth,
    ](
        name=name,
        cache_lengths=cache_lengths,
        topk_per_batch=topk_per_batch,
        extra_cache_lengths=extra_cache_lengths,
        extra_topk_per_batch=extra_topk_per_batch,
        ctx=ctx,
    )


def _run_test_sparse_qkv_fp8_extra_kv[
    kv_type: DType,
    num_heads: Int,
    # Width of the KV row and the absorbed Q as the model stores them. 576
    # carries a rotary tail after the 512 latent; a NoPE model stores the
    # latent alone and the kernel takes that row directly.
    row_depth: Int = Q_DEPTH,
](
    name: StringLiteral,
    cache_lengths: List[Int],
    topk_per_batch: List[Int],
    extra_cache_lengths: List[Int],
    extra_topk_per_batch: List[Int],
    ctx: DeviceContext,
) raises:
    """Two paged KV caches (main + always-attend). Both all-FP8, native FP8-Q.
    """
    var batch_size = len(cache_lengths)
    comptime q_max_seq_len = 1
    comptime scale = Float32(0.125)

    var max_topk = 0
    var max_extra_topk = 0
    for bi in range(batch_size):
        if topk_per_batch[bi] > max_topk:
            max_topk = topk_per_batch[bi]
        if extra_topk_per_batch[bi] > max_extra_topk:
            max_extra_topk = extra_topk_per_batch[bi]

    print(t"test: {name}  batch_size: {batch_size}  num_heads: {num_heads}")
    for i in range(batch_size):
        print(
            t"  batch {i} : cache_len= {cache_lengths[i]}  topk="
            t" {topk_per_batch[i]}  extra_cache_len= {extra_cache_lengths[i]} "
            t" extra_topk= {extra_topk_per_batch[i]}"
        )

    comptime kv_params = KVCacheStaticParams(
        num_heads=KV_NUM_HEADS, head_size=row_depth, is_mla=True
    )
    comptime kv_dim2 = 1
    var tok_stride = kv_params.head_size
    var page_stride_elems = (
        kv_dim2
        * NUM_LAYERS
        * PAGE_SIZE
        * kv_params.num_heads
        * kv_params.head_size
    )

    # --- ORIGINAL cache ---
    var max_cache_len = 0
    var total_pages = 0
    var num_keys_list = List[Int]()
    for i in range(batch_size):
        var cl = cache_lengths[i]
        if cl > max_cache_len:
            max_cache_len = cl
        var nk = cl + q_max_seq_len
        num_keys_list.append(nk)
        total_pages += ceildiv(nk, PAGE_SIZE)
    var max_num_keys = max_cache_len + q_max_seq_len
    var max_pages_per_batch = ceildiv(max_num_keys, PAGE_SIZE)

    var block_shape = (
        total_pages,
        kv_dim2,
        NUM_LAYERS,
        PAGE_SIZE,
        kv_params.num_heads,
        kv_params.head_size,
    )
    var block_elems = (
        total_pages
        * kv_dim2
        * NUM_LAYERS
        * PAGE_SIZE
        * kv_params.num_heads
        * kv_params.head_size
    )

    var blocks_host = ctx.enqueue_create_host_buffer[kv_type](block_elems)
    for i in range(block_elems):
        blocks_host[i] = Scalar[kv_type](0)
    var lut_size = batch_size * max_pages_per_batch
    var lookup_table_host = ctx.enqueue_create_host_buffer[.uint32](lut_size)
    for i in range(lut_size):
        lookup_table_host[i] = UInt32(0)
    var page_offset = 0
    for bi in range(batch_size):
        var np_bi = ceildiv(num_keys_list[bi], PAGE_SIZE)
        var mult_bi = _coprime_multiplier(np_bi)
        for p in range(np_bi):
            var shuffled_p = (p * mult_bi + 1) % np_bi
            lookup_table_host[bi * max_pages_per_batch + p] = UInt32(
                page_offset + shuffled_p
            )
        page_offset += np_bi

    var cache_lengths_host = ctx.enqueue_create_host_buffer[.uint32](batch_size)
    for i in range(batch_size):
        cache_lengths_host[i] = UInt32(cache_lengths[i])

    var total_k_elems = 0
    for bi in range(batch_size):
        total_k_elems += num_keys_list[bi] * row_depth
    var k_bf16_host = ctx.enqueue_create_host_buffer[.bfloat16](total_k_elems)
    randn(k_bf16_host.as_span(), mean=0.0, standard_deviation=0.5)
    var k_ref_host = ctx.enqueue_create_host_buffer[.bfloat16](total_k_elems)

    var k_offset = 0
    for bi in range(batch_size):
        var nk = num_keys_list[bi]
        for t in range(nk):
            var page_idx, tok_in_page = divmod(t, PAGE_SIZE)
            var physical_page = Int(
                lookup_table_host[bi * max_pages_per_batch + page_idx]
            )
            var base = (
                physical_page * page_stride_elems + tok_in_page * tok_stride
            )
            var k_base = k_offset + t * row_depth
            for d in range(row_depth):
                var fp8_val = k_bf16_host[k_base + d].cast[kv_type]()
                blocks_host[base + d] = fp8_val
                k_ref_host[k_base + d] = fp8_val.cast[.bfloat16]()
        k_offset += nk * row_depth

    # --- EXTRA cache ---
    var max_extra_cache_len = 0
    var extra_total_pages = 0
    var extra_num_keys_list = List[Int]()
    for i in range(batch_size):
        var ecl = extra_cache_lengths[i]
        if ecl > max_extra_cache_len:
            max_extra_cache_len = ecl
        var enk = ecl + q_max_seq_len
        extra_num_keys_list.append(enk)
        extra_total_pages += ceildiv(enk, PAGE_SIZE)
    var max_extra_num_keys = max_extra_cache_len + q_max_seq_len
    var max_extra_pages_per_batch = ceildiv(max_extra_num_keys, PAGE_SIZE)

    var extra_block_shape = (
        extra_total_pages,
        kv_dim2,
        NUM_LAYERS,
        PAGE_SIZE,
        kv_params.num_heads,
        kv_params.head_size,
    )
    var extra_block_elems = (
        extra_total_pages
        * kv_dim2
        * NUM_LAYERS
        * PAGE_SIZE
        * kv_params.num_heads
        * kv_params.head_size
    )

    var extra_blocks_host = ctx.enqueue_create_host_buffer[kv_type](
        extra_block_elems
    )
    for i in range(extra_block_elems):
        extra_blocks_host[i] = Scalar[kv_type](0)
    var extra_lut_size = batch_size * max_extra_pages_per_batch
    var extra_lookup_table_host = ctx.enqueue_create_host_buffer[.uint32](
        extra_lut_size
    )
    for i in range(extra_lut_size):
        extra_lookup_table_host[i] = UInt32(0)
    var extra_page_offset = 0
    for bi in range(batch_size):
        var np_bi = ceildiv(extra_num_keys_list[bi], PAGE_SIZE)
        var mult_bi = _coprime_multiplier(np_bi)
        for p in range(np_bi):
            var shuffled_p = (p * mult_bi + 1) % np_bi
            extra_lookup_table_host[
                bi * max_extra_pages_per_batch + p
            ] = UInt32(extra_page_offset + shuffled_p)
        extra_page_offset += np_bi

    var extra_cache_lengths_host = ctx.enqueue_create_host_buffer[.uint32](
        batch_size
    )
    for i in range(batch_size):
        extra_cache_lengths_host[i] = UInt32(extra_cache_lengths[i])

    var extra_total_k_elems = 0
    for bi in range(batch_size):
        extra_total_k_elems += extra_num_keys_list[bi] * row_depth
    var extra_k_bf16_host = ctx.enqueue_create_host_buffer[.bfloat16](
        extra_total_k_elems
    )
    randn(extra_k_bf16_host.as_span(), mean=0.0, standard_deviation=0.5)
    var extra_k_ref_host = ctx.enqueue_create_host_buffer[.bfloat16](
        extra_total_k_elems
    )

    var ek_offset = 0
    for bi in range(batch_size):
        var enk = extra_num_keys_list[bi]
        for t in range(enk):
            var page_idx, tok_in_page = divmod(t, PAGE_SIZE)
            var physical_page = Int(
                extra_lookup_table_host[
                    bi * max_extra_pages_per_batch + page_idx
                ]
            )
            var base = (
                physical_page * page_stride_elems + tok_in_page * tok_stride
            )
            var k_base = ek_offset + t * row_depth
            for d in range(row_depth):
                var fp8_val = extra_k_bf16_host[k_base + d].cast[kv_type]()
                extra_blocks_host[base + d] = fp8_val
                extra_k_ref_host[k_base + d] = fp8_val.cast[.bfloat16]()
        ek_offset += enk * row_depth

    # Q: generate BF16 randn, cast to FP8.
    var q_size = batch_size * num_heads * row_depth
    var q_bf16_scratch = ctx.enqueue_create_host_buffer[.bfloat16](q_size)
    randn(q_bf16_scratch.as_span(), mean=0.0, standard_deviation=0.5)
    var q_host = ctx.enqueue_create_host_buffer[.float8_e4m3fn](q_size)
    for i in range(q_size):
        q_host[i] = q_bf16_scratch[i].cast[.float8_e4m3fn]()

    # Build combined reference (main topk + extra topk per batch).
    var total_indices = batch_size * max_topk
    var h_indices = ctx.enqueue_create_host_buffer[.int32](total_indices)
    for i in range(total_indices):
        h_indices[i] = Int32(0)
    var extra_total_indices = batch_size * max_extra_topk
    var extra_h_indices = ctx.enqueue_create_host_buffer[.int32](
        extra_total_indices
    )
    for i in range(extra_total_indices):
        extra_h_indices[i] = Int32(0)
    var total_combined_ref_elems = 0
    for bi in range(batch_size):
        total_combined_ref_elems += (
            topk_per_batch[bi] + extra_topk_per_batch[bi]
        ) * row_depth
    var k_combined_ref = ctx.enqueue_create_host_buffer[.bfloat16](
        total_combined_ref_elems
    )

    var combined_ref_offset = 0
    var k_offset_src = 0
    var ek_offset_src = 0
    for bi in range(batch_size):
        var nk = num_keys_list[bi]
        var topk_bi = topk_per_batch[bi]
        var mult = _coprime_multiplier(nk)
        for i in range(topk_bi):
            var t = (i * mult + 1) % nk
            var page_idx, tok_in_page = divmod(t, PAGE_SIZE)
            var physical_page = Int(
                lookup_table_host[bi * max_pages_per_batch + page_idx]
            )
            h_indices[bi * max_topk + i] = Int32(
                physical_page * PAGE_SIZE + tok_in_page
            )
            var src_base = k_offset_src + t * row_depth
            var dst_base = combined_ref_offset + i * row_depth
            for d in range(row_depth):
                k_combined_ref[dst_base + d] = k_ref_host[src_base + d]
        combined_ref_offset += topk_bi * row_depth

        var enk = extra_num_keys_list[bi]
        var extra_topk_bi = extra_topk_per_batch[bi]
        var extra_mult = _coprime_multiplier(enk)
        for i in range(extra_topk_bi):
            var t = (i * extra_mult + 1) % enk
            var page_idx, tok_in_page = divmod(t, PAGE_SIZE)
            var physical_page = Int(
                extra_lookup_table_host[
                    bi * max_extra_pages_per_batch + page_idx
                ]
            )
            extra_h_indices[bi * max_extra_topk + i] = Int32(
                physical_page * PAGE_SIZE + tok_in_page
            )
            var src_base = ek_offset_src + t * row_depth
            var dst_base = combined_ref_offset + i * row_depth
            for d in range(row_depth):
                k_combined_ref[dst_base + d] = extra_k_ref_host[src_base + d]
        combined_ref_offset += extra_topk_bi * row_depth
        k_offset_src += nk * row_depth
        ek_offset_src += enk * row_depth

    var combined_num_keys_list = List[Int]()
    for bi in range(batch_size):
        combined_num_keys_list.append(
            topk_per_batch[bi] + extra_topk_per_batch[bi]
        )

    var out_size = batch_size * num_heads * V_DEPTH
    var ref_host = ctx.enqueue_create_host_buffer[.bfloat16](out_size)
    host_reference_varkeys(
        q_host.unsafe_ptr(),
        k_combined_ref.unsafe_ptr(),
        ref_host.unsafe_ptr(),
        batch_size,
        num_heads,
        combined_num_keys_list,
        row_depth,
        V_DEPTH,
        scale,
    )

    var blocks_device = ctx.enqueue_create_buffer[kv_type](block_elems)
    ctx.enqueue_copy(blocks_device, blocks_host)
    var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    ctx.enqueue_copy(cache_lengths_device, cache_lengths_host)
    var lookup_table_device = ctx.enqueue_create_buffer[.uint32](lut_size)
    ctx.enqueue_copy(lookup_table_device, lookup_table_host)

    var extra_blocks_device = ctx.enqueue_create_buffer[kv_type](
        extra_block_elems
    )
    ctx.enqueue_copy(extra_blocks_device, extra_blocks_host)
    var extra_cache_lengths_device = ctx.enqueue_create_buffer[.uint32](
        batch_size
    )
    ctx.enqueue_copy(extra_cache_lengths_device, extra_cache_lengths_host)
    var extra_lookup_table_device = ctx.enqueue_create_buffer[.uint32](
        extra_lut_size
    )
    ctx.enqueue_copy(extra_lookup_table_device, extra_lookup_table_host)

    var q_device = ctx.enqueue_create_buffer[.float8_e4m3fn](q_size)
    ctx.enqueue_copy(q_device, q_host)
    var out_device = ctx.enqueue_create_buffer[.bfloat16](out_size)

    var d_indices_device = ctx.enqueue_create_buffer[.int32](total_indices)
    ctx.enqueue_copy(d_indices_device, h_indices)
    var extra_d_indices_device = ctx.enqueue_create_buffer[.int32](
        extra_total_indices
    )
    ctx.enqueue_copy(extra_d_indices_device, extra_h_indices)

    var topk_lengths_host = ctx.enqueue_create_host_buffer[.int32](batch_size)
    for bi in range(batch_size):
        topk_lengths_host[bi] = Int32(topk_per_batch[bi])
    var topk_lengths_device = ctx.enqueue_create_buffer[.int32](batch_size)
    ctx.enqueue_copy(topk_lengths_device, topk_lengths_host)

    var extra_topk_lengths_host = ctx.enqueue_create_host_buffer[.int32](
        batch_size
    )
    for bi in range(batch_size):
        extra_topk_lengths_host[bi] = Int32(extra_topk_per_batch[bi])
    var extra_topk_lengths_device = ctx.enqueue_create_buffer[.int32](
        batch_size
    )
    ctx.enqueue_copy(extra_topk_lengths_device, extra_topk_lengths_host)

    ctx.synchronize()

    var blocks_tt = TileTensor(
        blocks_device,
        row_major(
            Int64(block_shape[0]),
            Idx[1],
            Int64(block_shape[2]),
            Idx[PAGE_SIZE],
            Idx[kv_params.num_heads],
            Idx[kv_params.head_size],
        ),
    )
    var cache_lengths_tt = TileTensor(
        cache_lengths_device, row_major(Int64(batch_size))
    )
    var lookup_table_tt = TileTensor(
        lookup_table_device,
        row_major(Int64(batch_size), Int64(max_pages_per_batch)),
    )
    comptime Collection = PagedKVCacheCollection[
        kv_type,
        kv_params,
        PAGE_SIZE,
        MutAnyOrigin,
        ImmutAnyOrigin,
        ImmutAnyOrigin,
        MutAnyOrigin,
    ]
    var kv_collection = Collection(
        rebind[Collection.blocks_tt_type](blocks_tt.as_unsafe_any_origin()),
        cache_lengths_tt.as_imm().as_unsafe_any_origin(),
        lookup_table_tt.as_imm().as_unsafe_any_origin(),
        UInt32(q_max_seq_len),
        UInt32(max_cache_len),
    )
    var kv_cache = kv_collection.get_key_cache(0)

    var extra_blocks_tt = TileTensor(
        extra_blocks_device,
        row_major(
            Int64(extra_block_shape[0]),
            Idx[1],
            Int64(extra_block_shape[2]),
            Idx[PAGE_SIZE],
            Idx[kv_params.num_heads],
            Idx[kv_params.head_size],
        ),
    )
    var extra_cache_lengths_tt = TileTensor(
        extra_cache_lengths_device, row_major(Int64(batch_size))
    )
    var extra_lookup_table_tt = TileTensor(
        extra_lookup_table_device,
        row_major(Int64(batch_size), Int64(max_extra_pages_per_batch)),
    )
    var extra_kv_collection = Collection(
        rebind[Collection.blocks_tt_type](
            extra_blocks_tt.as_unsafe_any_origin()
        ),
        extra_cache_lengths_tt.as_imm().as_unsafe_any_origin(),
        extra_lookup_table_tt.as_imm().as_unsafe_any_origin(),
        UInt32(q_max_seq_len),
        UInt32(max_extra_cache_len),
    )
    var extra_kv_cache = extra_kv_collection.get_key_cache(0)

    var q_tt = TileTensor(
        q_device, row_major(batch_size, Idx[num_heads], Idx[row_depth])
    )
    var out_tt = TileTensor(
        out_device, row_major(batch_size, Idx[num_heads], Idx[V_DEPTH])
    )

    var row_offsets_host = ctx.enqueue_create_host_buffer[.uint32](
        batch_size + 1
    )
    for i in range(batch_size + 1):
        row_offsets_host[i] = UInt32(i)
    var row_offsets_device = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    ctx.enqueue_copy(row_offsets_device, row_offsets_host)
    ctx.synchronize()

    var row_offsets_tt = TileTensor(
        row_offsets_device, row_major(batch_size + 1)
    )

    var mla_args = MLADispatchScalarArgs[
        num_heads=num_heads,
        is_fp8_kv=True,
    ](batch_size, max_cache_len, q_max_seq_len, ctx)
    var scalar_args_buf_tt = mla_args.gpu_tile_tensor()

    comptime sm_count = ctx.default_device_info.sm_count
    # The sparse dispatch splits the gathered keys, max top-k plus max
    # extra top-k, not the cache length.
    var dispatch_scalars = compute_mla_dispatch_scalars[
        num_heads=num_heads,
        _is_cache_length_accurate=True,
        is_fp8_kv=True,
        half_sms=sm_count // 2,
    ](batch_size, max_topk + max_extra_topk, q_max_seq_len, sm_count)
    var num_partitions = dispatch_scalars[2]
    print(
        t"  num_partitions= {num_partitions}  (split-K"
        t""" {"ACTIVE" if num_partitions > 1 else "OFF"} )"""
    )

    print("  Launching MLA sparse QKV_FP8 (extra KV)...")

    flare_mla_decoding[
        rank=3,
        config=MHAConfig[.float8_e4m3fn](num_heads, row_depth),
        ragged=True,
        sparse=True,
        has_extra_k=True,
        kernels=SM100MLA,
    ](
        out_tt,
        q_tt,
        kv_cache,
        NullMask(),
        row_offsets_tt,
        scale,
        ctx,
        scalar_args_buf_tt,
        d_indices=rebind[MutPointer[Int32, MutAnyOrigin]](
            d_indices_device.unsafe_ptr()
        ),
        indices_stride=max_topk,
        topk_lengths=NonNullPointer[DType.int32](topk_lengths_device),
        extra_k=extra_kv_cache,
        extra_d_indices=rebind[MutPointer[Int32, MutAnyOrigin]](
            extra_d_indices_device.unsafe_ptr()
        ),
        extra_indices_stride=max_extra_topk,
        extra_topk_lengths=NonNullPointer[DType.int32](
            extra_topk_lengths_device
        ),
    )
    ctx.synchronize()

    var out_host = ctx.enqueue_create_host_buffer[.bfloat16](out_size)
    ctx.enqueue_copy(out_host, out_device)
    ctx.synchronize()

    verify_mla_output(
        name,
        ref_host.unsafe_ptr(),
        out_host.unsafe_ptr(),
        batch_size * num_heads * V_DEPTH,
    )

    _ = mla_args
    _ = blocks_device
    _ = cache_lengths_device
    _ = lookup_table_device
    _ = extra_blocks_device
    _ = extra_cache_lengths_device
    _ = extra_lookup_table_device
    _ = q_device
    _ = out_device
    _ = d_indices_device
    _ = extra_d_indices_device
    _ = topk_lengths_device
    _ = extra_topk_lengths_device
    _ = row_offsets_device


# ===-----------------------------------------------------------------------===#
# Topk-clamping sparse test (small caches with topk > actual_tokens)
# ===-----------------------------------------------------------------------===#


def run_test_sparse_qkv_fp8_topk_clamping[
    kv_type: DType,
    num_heads: Int,
    # Width of the KV row and the absorbed Q as the model stores them. 576
    # carries a rotary tail after the 512 latent; a NoPE model stores the
    # latent alone and the kernel takes that row directly.
    row_depth: Int = Q_DEPTH,
](
    name: StringLiteral,
    cache_lengths: List[Int],
    topk_per_batch: List[Int],
    ctx: DeviceContext,
    tail_max: Float64 = 0.0015,
) raises:
    _run_test_sparse_qkv_fp8_topk_clamping[
        kv_type=kv_type,
        num_heads=num_heads,
        row_depth=row_depth,
    ](
        name=name,
        cache_lengths=cache_lengths,
        topk_per_batch=topk_per_batch,
        ctx=ctx,
        tail_max=tail_max,
    )


def _run_test_sparse_qkv_fp8_topk_clamping[
    kv_type: DType,
    num_heads: Int,
    # Width of the KV row and the absorbed Q as the model stores them. 576
    # carries a rotary tail after the 512 latent; a NoPE model stores the
    # latent alone and the kernel takes that row directly.
    row_depth: Int = Q_DEPTH,
](
    name: StringLiteral,
    cache_lengths: List[Int],
    topk_per_batch: List[Int],
    ctx: DeviceContext,
    tail_max: Float64 = 0.0015,
) raises:
    """Topk clamping with native FP8-Q + all-FP8-KV layout.

    When topk_per_batch[b] > actual_tokens[b], the kernel clamps to actual
    tokens. Indices beyond effective_topk are filled with -1 to catch OOB reads.
    """
    var batch_size = len(cache_lengths)
    comptime q_max_seq_len = 1
    comptime scale = Float32(0.125)

    var max_topk = 0
    var effective_topk_list = List[Int]()
    for bi in range(batch_size):
        if topk_per_batch[bi] > max_topk:
            max_topk = topk_per_batch[bi]
        var actual_tokens = cache_lengths[bi] + q_max_seq_len
        var eff_topk = topk_per_batch[bi]
        if eff_topk > actual_tokens:
            eff_topk = actual_tokens
        effective_topk_list.append(eff_topk)

    print(t"test: {name}  batch_size: {batch_size}  num_heads: {num_heads}")
    for i in range(batch_size):
        print(
            t"  batch {i} : cache_len= {cache_lengths[i]}  topk="
            t" {topk_per_batch[i]}  actual_tokens="
            t" {cache_lengths[i] + q_max_seq_len}  effective_topk="
            t" {effective_topk_list[i]}"
        )

    var max_cache_len = 0
    var total_pages = 0
    var num_keys_list = List[Int]()
    for i in range(batch_size):
        var cl = cache_lengths[i]
        if cl > max_cache_len:
            max_cache_len = cl
        var nk = cl + q_max_seq_len
        num_keys_list.append(nk)
        total_pages += ceildiv(nk, PAGE_SIZE)

    var max_num_keys = max_cache_len + q_max_seq_len
    var max_pages_per_batch = ceildiv(max_num_keys, PAGE_SIZE)

    comptime kv_params = KVCacheStaticParams(
        num_heads=KV_NUM_HEADS, head_size=row_depth, is_mla=True
    )
    comptime kv_dim2 = 1

    var block_shape = (
        total_pages,
        kv_dim2,
        NUM_LAYERS,
        PAGE_SIZE,
        kv_params.num_heads,
        kv_params.head_size,
    )
    var block_elems = (
        total_pages
        * kv_dim2
        * NUM_LAYERS
        * PAGE_SIZE
        * kv_params.num_heads
        * kv_params.head_size
    )
    var tok_stride = kv_params.head_size

    var blocks_host = ctx.enqueue_create_host_buffer[kv_type](block_elems)
    for i in range(block_elems):
        blocks_host[i] = Scalar[kv_type](0)
    var lut_size = batch_size * max_pages_per_batch
    var lookup_table_host = ctx.enqueue_create_host_buffer[.uint32](lut_size)
    for i in range(lut_size):
        lookup_table_host[i] = UInt32(0)
    var page_offset = 0
    for bi in range(batch_size):
        var np_bi = ceildiv(num_keys_list[bi], PAGE_SIZE)
        var mult_bi = _coprime_multiplier(np_bi)
        for p in range(np_bi):
            var shuffled_p = (p * mult_bi + 1) % np_bi
            lookup_table_host[bi * max_pages_per_batch + p] = UInt32(
                page_offset + shuffled_p
            )
        page_offset += np_bi

    var cache_lengths_host = ctx.enqueue_create_host_buffer[.uint32](batch_size)
    for i in range(batch_size):
        cache_lengths_host[i] = UInt32(cache_lengths[i])

    var page_stride_elems = (
        kv_dim2
        * NUM_LAYERS
        * PAGE_SIZE
        * kv_params.num_heads
        * kv_params.head_size
    )

    var total_k_elems = 0
    for bi in range(batch_size):
        total_k_elems += num_keys_list[bi] * row_depth
    var k_bf16_host = ctx.enqueue_create_host_buffer[.bfloat16](total_k_elems)
    randn(k_bf16_host.as_span(), mean=0.0, standard_deviation=0.5)
    var k_ref_host = ctx.enqueue_create_host_buffer[.bfloat16](total_k_elems)

    var k_offset = 0
    for bi in range(batch_size):
        var nk = num_keys_list[bi]
        for t in range(nk):
            var page_idx, tok_in_page = divmod(t, PAGE_SIZE)
            var physical_page = Int(
                lookup_table_host[bi * max_pages_per_batch + page_idx]
            )
            var base = (
                physical_page * page_stride_elems + tok_in_page * tok_stride
            )
            var k_base = k_offset + t * row_depth
            for d in range(row_depth):
                var fp8_val = k_bf16_host[k_base + d].cast[kv_type]()
                blocks_host[base + d] = fp8_val
                k_ref_host[k_base + d] = fp8_val.cast[.bfloat16]()
        k_offset += nk * row_depth

    # Q: generate BF16 randn, cast to FP8.
    var q_size = batch_size * num_heads * row_depth
    var q_bf16_scratch = ctx.enqueue_create_host_buffer[.bfloat16](q_size)
    randn(q_bf16_scratch.as_span(), mean=0.0, standard_deviation=0.5)
    var q_host = ctx.enqueue_create_host_buffer[.float8_e4m3fn](q_size)
    for i in range(q_size):
        q_host[i] = q_bf16_scratch[i].cast[.float8_e4m3fn]()

    # d_indices: first effective_topk entries valid; rest are -1 sentinel.
    var total_indices = batch_size * max_topk
    var h_indices = ctx.enqueue_create_host_buffer[.int32](total_indices)
    for i in range(total_indices):
        h_indices[i] = Int32(-1)

    var total_sparse_ref_elems = 0
    for bi in range(batch_size):
        total_sparse_ref_elems += effective_topk_list[bi] * row_depth
    var k_sparse_ref = ctx.enqueue_create_host_buffer[.bfloat16](
        total_sparse_ref_elems
    )

    var sparse_ref_offset = 0
    var k_offset_src = 0
    for bi in range(batch_size):
        var nk = num_keys_list[bi]
        var eff_topk = effective_topk_list[bi]
        var mult = _coprime_multiplier(nk) if nk > 1 else 1
        for i in range(eff_topk):
            var t: Int
            if nk == 1:
                t = 0
            else:
                t = (i * mult + 1) % nk
            var page_idx, tok_in_page = divmod(t, PAGE_SIZE)
            var physical_page = Int(
                lookup_table_host[bi * max_pages_per_batch + page_idx]
            )
            h_indices[bi * max_topk + i] = Int32(
                physical_page * PAGE_SIZE + tok_in_page
            )
            var src_base = k_offset_src + t * row_depth
            var dst_base = sparse_ref_offset + i * row_depth
            for d in range(row_depth):
                k_sparse_ref[dst_base + d] = k_ref_host[src_base + d]
        sparse_ref_offset += eff_topk * row_depth
        k_offset_src += nk * row_depth

    var sparse_num_keys_list = List[Int]()
    for bi in range(batch_size):
        sparse_num_keys_list.append(effective_topk_list[bi])

    var out_size = batch_size * num_heads * V_DEPTH
    var ref_host = ctx.enqueue_create_host_buffer[.bfloat16](out_size)
    host_reference_varkeys(
        q_host.unsafe_ptr(),
        k_sparse_ref.unsafe_ptr(),
        ref_host.unsafe_ptr(),
        batch_size,
        num_heads,
        sparse_num_keys_list,
        row_depth,
        V_DEPTH,
        scale,
    )

    var blocks_device = ctx.enqueue_create_buffer[kv_type](block_elems)
    ctx.enqueue_copy(blocks_device, blocks_host)
    var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    ctx.enqueue_copy(cache_lengths_device, cache_lengths_host)
    var lookup_table_device = ctx.enqueue_create_buffer[.uint32](lut_size)
    ctx.enqueue_copy(lookup_table_device, lookup_table_host)
    var q_device = ctx.enqueue_create_buffer[.float8_e4m3fn](q_size)
    ctx.enqueue_copy(q_device, q_host)
    var out_device = ctx.enqueue_create_buffer[.bfloat16](out_size)
    var d_indices_device = ctx.enqueue_create_buffer[.int32](total_indices)
    ctx.enqueue_copy(d_indices_device, h_indices)

    var topk_lengths_host = ctx.enqueue_create_host_buffer[.int32](batch_size)
    for bi in range(batch_size):
        topk_lengths_host[bi] = Int32(topk_per_batch[bi])  # UNCLAMPED
    var topk_lengths_device = ctx.enqueue_create_buffer[.int32](batch_size)
    ctx.enqueue_copy(topk_lengths_device, topk_lengths_host)

    ctx.synchronize()

    var blocks_tt = TileTensor(
        blocks_device,
        row_major(
            Int64(block_shape[0]),
            Idx[1],
            Int64(block_shape[2]),
            Idx[PAGE_SIZE],
            Idx[kv_params.num_heads],
            Idx[kv_params.head_size],
        ),
    )
    var cache_lengths_tt = TileTensor(
        cache_lengths_device, row_major(Int64(batch_size))
    )
    var lookup_table_tt = TileTensor(
        lookup_table_device,
        row_major(Int64(batch_size), Int64(max_pages_per_batch)),
    )
    comptime Collection = PagedKVCacheCollection[
        kv_type,
        kv_params,
        PAGE_SIZE,
        MutAnyOrigin,
        ImmutAnyOrigin,
        ImmutAnyOrigin,
        MutAnyOrigin,
    ]
    var kv_collection = Collection(
        rebind[Collection.blocks_tt_type](blocks_tt.as_unsafe_any_origin()),
        cache_lengths_tt.as_imm().as_unsafe_any_origin(),
        lookup_table_tt.as_imm().as_unsafe_any_origin(),
        UInt32(q_max_seq_len),
        UInt32(max_cache_len),
    )
    var kv_cache = kv_collection.get_key_cache(0)

    var q_tt = TileTensor(
        q_device, row_major(batch_size, Idx[num_heads], Idx[row_depth])
    )
    var out_tt = TileTensor(
        out_device, row_major(batch_size, Idx[num_heads], Idx[V_DEPTH])
    )

    var row_offsets_host = ctx.enqueue_create_host_buffer[.uint32](
        batch_size + 1
    )
    for i in range(batch_size + 1):
        row_offsets_host[i] = UInt32(i)
    var row_offsets_device = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    ctx.enqueue_copy(row_offsets_device, row_offsets_host)
    ctx.synchronize()

    var row_offsets_tt = TileTensor(
        row_offsets_device, row_major(batch_size + 1)
    )

    var mla_args = MLADispatchScalarArgs[
        num_heads=num_heads,
        is_fp8_kv=True,
    ](batch_size, max_cache_len, q_max_seq_len, ctx)
    var scalar_args_buf_tt = mla_args.gpu_tile_tensor()

    comptime sm_count = ctx.default_device_info.sm_count
    var dispatch_scalars = compute_mla_dispatch_scalars[
        num_heads=num_heads, is_fp8_kv=True, half_sms=sm_count // 2
    ](batch_size, max_cache_len, q_max_seq_len, sm_count)
    var num_partitions = dispatch_scalars[2]
    print(
        t"  num_partitions= {num_partitions}  (split-K"
        t""" {"ACTIVE" if num_partitions > 1 else "OFF"} )"""
    )

    print("  Launching MLA sparse QKV_FP8 (topk clamping)...")

    flare_mla_decoding[
        rank=3,
        config=MHAConfig[.float8_e4m3fn](num_heads, row_depth),
        ragged=True,
        sparse=True,
        kernels=SM100MLA,
    ](
        out_tt,
        q_tt,
        kv_cache,
        NullMask(),
        row_offsets_tt,
        scale,
        ctx,
        scalar_args_buf_tt,
        d_indices=rebind[MutPointer[Int32, MutAnyOrigin]](
            d_indices_device.unsafe_ptr()
        ),
        indices_stride=max_topk,
        topk_lengths=NonNullPointer[DType.int32](topk_lengths_device),
    )
    ctx.synchronize()

    var out_host = ctx.enqueue_create_host_buffer[.bfloat16](out_size)
    ctx.enqueue_copy(out_host, out_device)
    ctx.synchronize()

    verify_mla_output(
        name,
        ref_host.unsafe_ptr(),
        out_host.unsafe_ptr(),
        batch_size * num_heads * V_DEPTH,
        tail_max=tail_max,
    )

    _ = mla_args
    _ = blocks_device
    _ = cache_lengths_device
    _ = lookup_table_device
    _ = q_device
    _ = out_device
    _ = d_indices_device
    _ = topk_lengths_device
    _ = row_offsets_device


def _p6_bits(k0: Int, k1: Int = 0) -> Int:
    var bits = 0
    for k in [k0, k1]:
        if k > 0:
            bits |= 1 << k
    return bits


def _run_p6_cells(
    ctx: DeviceContext,
    mut fixtures: List[_SparseKVFixture[DType.float8_e4m3fn]],
) raises:
    """Cells for the direct gather, at 64- and 128-key tiles.

    Key counts around the 16-, 64- and 128-key boundaries (partial
    sub-slots, a dead second tile half, 1 to 5 tiles), one-tile split-K
    partitions, a cache shorter than one tile, the extra-KV gap inside the
    last main tile, causal horizons inside each 16-key half, full rows at
    h=24 and h=32, rescales on every tile, and the B >= 32, short-cache
    launch that keeps 64-key tiles. `p6_fail` names the negative controls
    each cell must catch.
    """
    run_test_sparse_qkv_fp8[.float8_e4m3fn, 8](
        "p6_h8_cl512_topk300",
        2,
        512,
        ctx,
        fixtures=fixtures,
        topk=300,
        p6_fail=_p6_bits(5, 8),
    )
    run_test_sparse_qkv_fp8[.float8_e4m3fn, 32](
        "p6_h32_cl512_topk300",
        1,
        512,
        ctx,
        fixtures=fixtures,
        topk=300,
        p6_fail=_p6_bits(5, 8),
    )
    run_test_sparse_qkv_fp8[.float8_e4m3fn, 8](
        "p6_h8_cl512_topk200_np1",
        1,
        512,
        ctx,
        fixtures=fixtures,
        topk=200,
        forced_np=1,
    )
    run_test_sparse_qkv_fp8[.float8_e4m3fn, 8](
        "p6_spike_every_h8_cl1024_topk640",
        1,
        1024,
        ctx,
        fixtures=fixtures,
        topk=640,
        forced_np=1,
        stim=STIM_SPIKE_EVERY,
    )
    run_test_sparse_qkv_fp8[.float8_e4m3fn, 8, use_causal=True, logical=True](
        "p6_causal_logical_pad3_h8_cl2048_topk203_seq6",
        2,
        2048,
        ctx,
        fixtures=fixtures,
        topk=203,
        q_max_seq_len=6,
        logical_pad=3,
    )
    for topk in [1, 40, 64, 65, 100, 127, 128, 129, 191, 255]:
        run_test_sparse_qkv_fp8[.float8_e4m3fn, 8](
            "p6_h8_cl512_keys", 1, 512, ctx, fixtures=fixtures, topk=topk
        )
    run_test_sparse_qkv_fp8[.float8_e4m3fn, 32](
        "p6_h32_cl512_topk129", 1, 512, ctx, fixtures=fixtures, topk=129
    )
    run_test_sparse_qkv_fp8[.float8_e4m3fn, 24](
        "p6_h24_cl1024_topk512", 2, 1024, ctx, fixtures=fixtures, topk=512
    )
    run_test_sparse_qkv_fp8[.float8_e4m3fn, 8, use_causal=True](
        "p6_causal_h8_cl2048_topk384_np3_seq6",
        2,
        2048,
        ctx,
        fixtures=fixtures,
        topk=384,
        q_max_seq_len=6,
        forced_np=3,
    )
    run_test_sparse_qkv_fp8[.float8_e4m3fn, 8](
        "p6_h8_cl100_topk100", 2, 100, ctx, fixtures=fixtures, topk=100
    )
    run_test_sparse_qkv_fp8[.float8_e4m3fn, 8, use_causal=True](
        "p6_causal_h8_cl120_topk120_seq6",
        2,
        120,
        ctx,
        fixtures=fixtures,
        topk=120,
        q_max_seq_len=6,
    )
    run_test_sparse_qkv_fp8[.float8_e4m3fn, 16](
        "p6_t3_h16_cl1024_topk640",
        2,
        1024,
        ctx,
        fixtures=fixtures,
        topk=640,
        stim=STIM_T3,
    )
    run_test_sparse_qkv_fp8[.float8_e4m3fn, 8, shared_index=True](
        "p6_fold2_h8_cl1024_topk300_seq2",
        2,
        1024,
        ctx,
        fixtures=fixtures,
        topk=300,
        q_max_seq_len=2,
    )
    run_test_sparse_qkv_fp8_attn_sink[.float8_e4m3fn, 8](
        "p6_sink_h8_cl512_topk300", 1, 512, ctx, topk=300
    )
    run_test_sparse_qkv_fp8_extra_kv[.float8_e4m3fn, 16](
        "p6_extra_kv_h16_topk1900_extra64",
        [2048],
        [1900],
        [256],
        [64],
        ctx,
    )
    # B >= 32 with a short cache launches with 64-key split pages, so the
    # 64-key kernel runs under the 128 knob.
    run_test_sparse_qkv_fp8[.float8_e4m3fn, 8](
        "p6_fallback64_h8_b32_cl256_topk64",
        32,
        256,
        ctx,
        fixtures=fixtures,
        topk=64,
    )
    run_test_sparse_qkv_fp8[.float8_e4m3fn, 8](
        "p6_direct_h8_b2_cl131072_topk2048",
        2,
        131072,
        ctx,
        fixtures=fixtures,
        topk=2048,
    )


def _run_pdl_cells(
    ctx: DeviceContext,
    mut fixtures: List[_SparseKVFixture[DType.float8_e4m3fn]],
) raises:
    """Split-K decodes launched under PDL right behind a kernel that
    triggers at entry and writes their indices or Q 200 us late (each must
    wait for it), and back-to-back calls that reuse the split workspaces."""
    for np in [2, 3]:
        for mode in [PDL_LATE_INDICES, PDL_LATE_Q]:
            run_test_sparse_qkv_fp8[.float8_e4m3fn, 8](
                "pdl_late_h8_cl32768_topk2048_seq6",
                2,
                32768,
                ctx,
                fixtures=fixtures,
                topk=2048,
                q_max_seq_len=6,
                forced_np=np,
                pdl_mode=mode,
            )
            run_test_sparse_qkv_fp8[.float8_e4m3fn, 8](
                "pdl_late_h8_cl32768_topk512_seq6",
                2,
                32768,
                ctx,
                fixtures=fixtures,
                topk=512,
                q_max_seq_len=6,
                forced_np=np,
                pdl_mode=mode,
            )
            run_test_sparse_qkv_fp8[.float8_e4m3fn, 24](
                "pdl_late_h24_cl8192_topk2048_seq2",
                2,
                8192,
                ctx,
                fixtures=fixtures,
                topk=2048,
                q_max_seq_len=2,
                forced_np=np,
                pdl_mode=mode,
            )
            run_test_sparse_qkv_fp8[.float8_e4m3fn, 64](
                "pdl_late_h64_cl8192_topk2048_seq2",
                2,
                8192,
                ctx,
                fixtures=fixtures,
                topk=2048,
                q_max_seq_len=2,
                forced_np=np,
                pdl_mode=mode,
            )
    run_test_sparse_qkv_fp8[.float8_e4m3fn, 8](
        "pdl_reuse_h8_b8_cl4096_topk512_seq6_np3",
        8,
        4096,
        ctx,
        fixtures=fixtures,
        topk=512,
        q_max_seq_len=6,
        forced_np=3,
        pdl_mode=PDL_REUSE,
    )
    run_test_sparse_qkv_fp8[.float8_e4m3fn, 8](
        "pdl_reuse_h8_b8_cl4096_topk512_seq6",
        8,
        4096,
        ctx,
        fixtures=fixtures,
        topk=512,
        q_max_seq_len=6,
        pdl_mode=PDL_REUSE,
    )
    run_test_sparse_qkv_fp8[.float8_e4m3fn, 8](
        "pdl_reuse_h8_b32_cl4096_topk512_seq6",
        32,
        4096,
        ctx,
        fixtures=fixtures,
        topk=512,
        q_max_seq_len=6,
        pdl_mode=PDL_REUSE,
    )


def _run_ragged_cells(
    ctx: DeviceContext,
    mut fixtures: List[_SparseKVFixture[DType.float8_e4m3fn]],
) raises:
    """Short batches in the middle and at the end: the split-K combine's
    deferred ragged exit (np 3, 2), the non-split decode's exit (np 1), the
    fold, and a split with no keys."""
    comptime for causal in [False, True]:
        for np in [3, 1]:
            run_test_sparse_qkv_fp8[.float8_e4m3fn, 8, use_causal=causal](
                "ragged_h8_b4_cl4096_topk512_seq6",
                4,
                4096,
                ctx,
                fixtures=fixtures,
                topk=512,
                q_max_seq_len=6,
                forced_np=np,
                short_batches=True,
            )
    run_test_sparse_qkv_fp8[.float8_e4m3fn, 8, shared_index=True](
        "ragged_fold4_h8_b4_cl2048_topk300_seq4",
        4,
        2048,
        ctx,
        fixtures=fixtures,
        topk=300,
        q_max_seq_len=4,
        short_batches=True,
    )
    run_test_sparse_qkv_fp8[.float8_e4m3fn, 64](
        "ragged_h64_b4_cl2048_topk512_seq2_np2",
        4,
        2048,
        ctx,
        fixtures=fixtures,
        topk=512,
        q_max_seq_len=2,
        forced_np=2,
        short_batches=True,
    )
    run_test_sparse_qkv_fp8[.float8_e4m3fn, 8](
        "ragged_empty_split_h8_b4_cl256_topk128_seq6_np3",
        4,
        256,
        ctx,
        fixtures=fixtures,
        topk=128,
        q_max_seq_len=6,
        forced_np=3,
        short_batches=True,
    )


def _run_half_split_cells(
    ctx: DeviceContext,
    mut fixtures: List[_SparseKVFixture[DType.float8_e4m3fn]],
) raises:
    """STIM_HALF_SPLIT cells: with at most 16 rows the no-pair-max control
    (`MLA_HALF_ROWS_BREAK=1`) must fail every one."""
    run_test_sparse_qkv_fp8[.float8_e4m3fn, 8](
        "hs_h8_cl1024_topk128_np1",
        2,
        1024,
        ctx,
        fixtures=fixtures,
        topk=128,
        forced_np=1,
        stim=STIM_HALF_SPLIT,
        hr1_fail=True,
    )
    run_test_sparse_qkv_fp8[.float8_e4m3fn, 8](
        "hs_h8_cl2048_topk640_np3",
        2,
        2048,
        ctx,
        fixtures=fixtures,
        topk=640,
        forced_np=3,
        stim=STIM_HALF_SPLIT,
        hr1_fail=True,
    )
    run_test_sparse_qkv_fp8[.float8_e4m3fn, 8, use_causal=True](
        "hs_causal_h8_cl4096_topk2048_seq6",
        2,
        4096,
        ctx,
        fixtures=fixtures,
        topk=2048,
        q_max_seq_len=6,
        stim=STIM_HALF_SPLIT,
        hr1_fail=True,
    )
    run_test_sparse_qkv_fp8[.float8_e4m3fn, 16](
        "hs_h16_cl2048_topk512",
        2,
        2048,
        ctx,
        fixtures=fixtures,
        topk=512,
        stim=STIM_HALF_SPLIT,
        hr1_fail=True,
    )
    run_test_sparse_qkv_fp8[.float8_e4m3fn, 8, shared_index=True](
        "hs_fold2_h8_cl1024_topk512_seq2",
        2,
        1024,
        ctx,
        fixtures=fixtures,
        topk=512,
        q_max_seq_len=2,
        stim=STIM_HALF_SPLIT,
        hr1_fail=True,
    )
    run_test_sparse_qkv_fp8[.float8_e4m3fn, 8, use_causal=True, logical=True](
        "hs_causal_logical_h8_cl2048_topk512_seq2",
        2,
        2048,
        ctx,
        fixtures=fixtures,
        topk=512,
        q_max_seq_len=2,
        stim=STIM_HALF_SPLIT,
        hr1_fail=True,
    )


def _run_pair_cells(
    ctx: DeviceContext,
    mut fixtures: List[_SparseKVFixture[DType.float8_e4m3fn]],
) raises:
    """The default h=128 CTA pair's edge cases.

    The pair splits every 128-key tile into the leader's keys (slots
    [0, 64)) and the follower's ([64, 128)), so each family below lands its
    last tile with the follower half partial, empty, or holding a single
    key, alone or under split-K, extra KV, padding and logical indices.
    Under the default routing, cells whose CTAs walk at most three tiles run
    the unpaired kernel; the `_cg2_unrouted` arm (`MLA_SPARSE_CG2_ROUTE=0`)
    runs every cell on the pair.
    """
    comptime fp8 = DType.float8_e4m3fn
    # Ragged tails: follower half partial (200 = 128 + 64 + 8), empty
    # (300 = 2 * 128 + 44, 129 = 128 + 1), one short of full (191), and a
    # lone partial leader half (40).
    run_test_sparse_qkv_fp8[fp8, 128](
        "pair_tail_h128_cl1024_topk200",
        1,
        1024,
        ctx,
        fixtures=fixtures,
        topk=200,
    )
    run_test_sparse_qkv_fp8[fp8, 128](
        "pair_tail_h128_cl1024_topk300",
        1,
        1024,
        ctx,
        fixtures=fixtures,
        topk=300,
    )
    run_test_sparse_qkv_fp8[fp8, 128](
        "pair_tail_h128_cl512_topk129", 1, 512, ctx, fixtures=fixtures, topk=129
    )
    run_test_sparse_qkv_fp8[fp8, 128](
        "pair_tail_b2_h128_cl1024_topk191",
        2,
        1024,
        ctx,
        fixtures=fixtures,
        topk=191,
    )
    run_test_sparse_qkv_fp8[fp8, 128](
        "pair_tail_h128_cl256_topk40", 1, 256, ctx, fixtures=fixtures, topk=40
    )
    run_test_sparse_qkv_fp8[fp8, 128, use_causal=True](
        "pair_causal_b3_h128_cl509_topk73_seq3",
        3,
        509,
        ctx,
        fixtures=fixtures,
        topk=73,
        q_max_seq_len=3,
    )
    # Split-K: whole tiles per split, a ragged last split, and splits whose
    # follower halves are partial or empty.
    run_test_sparse_qkv_fp8[fp8, 128](
        "pair_np2_h128_cl4096_topk2048",
        1,
        4096,
        ctx,
        fixtures=fixtures,
        topk=2048,
        forced_np=2,
    )
    run_test_sparse_qkv_fp8[fp8, 128](
        "pair_np3_b2_h128_cl4096_topk1000",
        2,
        4096,
        ctx,
        fixtures=fixtures,
        topk=1000,
        forced_np=3,
    )
    run_test_sparse_qkv_fp8[fp8, 128](
        "pair_np4_h128_cl2048_topk300",
        1,
        2048,
        ctx,
        fixtures=fixtures,
        topk=300,
        forced_np=4,
    )
    run_test_sparse_qkv_fp8[fp8, 128](
        "pair_np8_h128_cl4096_topk2048",
        1,
        4096,
        ctx,
        fixtures=fixtures,
        topk=2048,
        forced_np=8,
    )
    # -1 logical indices in both halves, the leader's only and the
    # follower's only, then with split-K and two query tokens.
    run_test_sparse_qkv_fp8[fp8, 128, use_causal=True, logical=True](
        "pair_logical_pad3_h128_cl2048_topk203",
        2,
        2048,
        ctx,
        fixtures=fixtures,
        topk=203,
        logical_pad=3,
    )
    run_test_sparse_qkv_fp8[fp8, 128, use_causal=True, logical=True](
        "pair_logical_pad2_leader_h128_cl2048_topk203",
        2,
        2048,
        ctx,
        fixtures=fixtures,
        topk=203,
        logical_pad=2,
        logical_pad_half=0,
    )
    run_test_sparse_qkv_fp8[fp8, 128, use_causal=True, logical=True](
        "pair_logical_pad2_follower_h128_cl2048_topk203",
        2,
        2048,
        ctx,
        fixtures=fixtures,
        topk=203,
        logical_pad=2,
        logical_pad_half=1,
    )
    run_test_sparse_qkv_fp8[fp8, 128, use_causal=True, logical=True](
        "pair_logical_pad3_np3_h128_cl4096_topk1000",
        1,
        4096,
        ctx,
        fixtures=fixtures,
        topk=1000,
        forced_np=3,
        logical_pad=3,
    )
    run_test_sparse_qkv_fp8[fp8, 128, use_causal=True, logical=True](
        "pair_logical_h128_cl2048_topk512_seq2",
        2,
        2048,
        ctx,
        fixtures=fixtures,
        topk=512,
        q_max_seq_len=2,
    )
    run_test_sparse_qkv_fp8[fp8, 128, logical=True](
        "pair_logical_b2_h128_cl1024_topk300",
        2,
        1024,
        ctx,
        fixtures=fixtures,
        topk=300,
    )
    # Extra KV: tile-aligned, unaligned main segments that split, and
    # short lists that cannot.
    var ek_cls_a: List[Int] = [256, 384]
    var ek_topk_a: List[Int] = [64, 64]
    var ek_ecls_a: List[Int] = [64, 128]
    var ek_etopk_a: List[Int] = [64, 64]
    run_test_sparse_qkv_fp8_extra_kv[fp8, 128](
        "pair_extra_kv_b2_h128_variable",
        ek_cls_a,
        ek_topk_a,
        ek_ecls_a,
        ek_etopk_a,
        ctx,
    )
    var ek_cls_b: List[Int] = [512]
    var ek_topk_b: List[Int] = [300]
    var ek_ecls_b: List[Int] = [256]
    var ek_etopk_b: List[Int] = [200]
    run_test_sparse_qkv_fp8_extra_kv[fp8, 128](
        "pair_extra_kv_h128_topk300_extra200",
        ek_cls_b,
        ek_topk_b,
        ek_ecls_b,
        ek_etopk_b,
        ctx,
    )
    var ek_cls_c: List[Int] = [64, 256]
    var ek_topk_c: List[Int] = [37, 45]
    var ek_ecls_c: List[Int] = [64, 128]
    var ek_etopk_c: List[Int] = [20, 24]
    run_test_sparse_qkv_fp8_extra_kv[fp8, 128](
        "pair_extra_kv_b2_h128_topk_unaligned",
        ek_cls_c,
        ek_topk_c,
        ek_ecls_c,
        ek_etopk_c,
        ctx,
    )
    # Per-batch top-k and clamped near-empty caches (a lone partial leader
    # half, or no keys at all past the clamp).
    var vt_cls: List[Int] = [1024, 300, 128, 2048]
    var vt_topk: List[Int] = [200, 70, 32, 1000]
    run_test_sparse_qkv_fp8_variable_topk[fp8, 128](
        "pair_variable_topk_b4_h128", vt_cls, vt_topk, ctx
    )
    var tc_cls_a: List[Int] = [5]
    var tc_topk_a: List[Int] = [64]
    run_test_sparse_qkv_fp8_topk_clamping[fp8, 128](
        "pair_topk_clamp_small_cache_h128",
        tc_cls_a,
        tc_topk_a,
        ctx,
        tail_max=0.05,
    )
    var tc_cls_b: List[Int] = [0, 256]
    var tc_topk_b: List[Int] = [64, 64]
    run_test_sparse_qkv_fp8_topk_clamping[fp8, 128](
        "pair_topk_clamp_mixed_b2_h128", tc_cls_b, tc_topk_b, ctx
    )
    run_test_sparse_qkv_fp8_attn_sink[fp8, 128](
        "pair_attn_sink_b2_h128_cl1024_topk200", 2, 1024, ctx, topk=200
    )
    # The NoPE row on the pair, bit-identical to its zero-tailed 576 run.
    run_nope_native_512_fp8[128](
        ctx, batch_size=2, cache_len=1024, topk=300, name="pair_nope512"
    )
    # Either side of the route: one split of 384 keys runs the unpaired
    # kernel, 385 keys the pair.
    run_test_sparse_qkv_fp8[fp8, 128](
        "pair_route_h128_cl4096_topk384",
        1,
        4096,
        ctx,
        fixtures=fixtures,
        topk=384,
    )
    run_test_sparse_qkv_fp8[fp8, 128](
        "pair_route_h128_cl4096_topk385",
        1,
        4096,
        ctx,
        fixtures=fixtures,
        topk=385,
    )


def _run_pair_negative_control(
    ctx: DeviceContext,
    mut fixtures: List[_SparseKVFixture[DType.float8_e4m3fn]],
) raises:
    """`MLA_CG2_SLICE_BREAK`: the pair cells must fail, an unpaired one pass.

    Every cell's follower half holds keys, so both kinds reach it.
    """
    comptime fp8 = DType.float8_e4m3fn
    run_test_sparse_qkv_fp8[fp8, 128](
        "pair_break_h128_cl2048_topk2048",
        1,
        2048,
        ctx,
        fixtures=fixtures,
        topk=2048,
        break_expect=BREAK_MUST_FAIL,
    )
    run_test_sparse_qkv_fp8[fp8, 128](
        "pair_break_tail_h128_cl1024_topk200",
        1,
        1024,
        ctx,
        fixtures=fixtures,
        topk=200,
        break_expect=BREAK_MUST_FAIL,
    )
    run_test_sparse_qkv_fp8[fp8, 128](
        "pair_break_np3_b2_h128_cl4096_topk1000",
        2,
        4096,
        ctx,
        fixtures=fixtures,
        topk=1000,
        forced_np=3,
        break_expect=BREAK_MUST_FAIL,
    )
    run_test_sparse_qkv_fp8[fp8, 128, use_causal=True, logical=True](
        "pair_break_logical_pad3_h128_cl2048_topk203",
        2,
        2048,
        ctx,
        fixtures=fixtures,
        topk=203,
        logical_pad=3,
        break_expect=BREAK_MUST_FAIL,
    )
    # The route's direct-side boundary cell reaches the pair once the
    # routing is off, so its pass in the routed arm is the route's doing.
    run_test_sparse_qkv_fp8[fp8, 128](
        "pair_break_route_h128_cl4096_topk384",
        1,
        4096,
        ctx,
        fixtures=fixtures,
        topk=384,
        forced_np=1,
        break_expect=BREAK_MUST_FAIL,
    )
    run_test_sparse_qkv_fp8[fp8, 64](
        "pair_break_unpaired_h64_cl2048_topk512",
        1,
        2048,
        ctx,
        fixtures=fixtures,
        topk=512,
        break_expect=BREAK_MUST_PASS,
    )


def _run_pair_route_negative_control(
    ctx: DeviceContext,
    mut fixtures: List[_SparseKVFixture[DType.float8_e4m3fn]],
) raises:
    """`MLA_CG2_SLICE_BREAK` under the default routing: one split of 384 keys
    runs the unpaired kernel and must pass, 385 keys the broken pair and must
    fail. Split, the walk rounds up to whole split pages: two splits of 768
    keys walk 384 each and pass, of 769 keys 512 and fail.

    Each forces its split, since the route reads the critical CTA's walk.
    """
    comptime fp8 = DType.float8_e4m3fn
    run_test_sparse_qkv_fp8[fp8, 128](
        "pair_route_h128_cl4096_topk384",
        1,
        4096,
        ctx,
        fixtures=fixtures,
        topk=384,
        forced_np=1,
        break_expect=BREAK_MUST_PASS,
    )
    run_test_sparse_qkv_fp8[fp8, 128](
        "pair_route_h128_cl4096_topk385",
        1,
        4096,
        ctx,
        fixtures=fixtures,
        topk=385,
        forced_np=1,
        break_expect=BREAK_MUST_FAIL,
    )
    run_test_sparse_qkv_fp8[fp8, 128](
        "pair_route_np2_h128_cl4096_topk768",
        1,
        4096,
        ctx,
        fixtures=fixtures,
        topk=768,
        forced_np=2,
        break_expect=BREAK_MUST_PASS,
    )
    run_test_sparse_qkv_fp8[fp8, 128](
        "pair_route_np2_h128_cl4096_topk769",
        1,
        4096,
        ctx,
        fixtures=fixtures,
        topk=769,
        forced_np=2,
        break_expect=BREAK_MUST_FAIL,
    )


def main() raises:
    with DeviceContext() as ctx:
        comptime if ctx.target.is_nvidia_gpu() and _is_sm10x_gpu(
            ctx.default_device_info
        ):
            seed(42)
            var kv_fixtures = List[_SparseKVFixture[DType.float8_e4m3fn]]()

            comptime if CG2_SLICE_BREAK:
                comptime if CG2_ROUTED:
                    _run_pair_route_negative_control(ctx, kv_fixtures)
                else:
                    _run_pair_negative_control(ctx, kv_fixtures)
                print("negative control: all expectations met")
                return

            comptime if not (HALF_ROWS_BREAK or P6_BREAK):
                _run_pdl_cells(ctx, kv_fixtures)
                _run_ragged_cells(ctx, kv_fixtures)
                run_test_sparse_qkv_fp8[.float8_e4m3fn, 8, use_causal=True](
                    "topk2048_h8_b26_seq6",
                    26,
                    32768,
                    ctx,
                    fixtures=kv_fixtures,
                    topk=2048,
                    q_max_seq_len=6,
                )
                run_test_sparse_qkv_fp8[.float8_e4m3fn, 64, use_causal=True](
                    "topk2048_h64_b8_seq2",
                    8,
                    4096,
                    ctx,
                    fixtures=kv_fixtures,
                    topk=2048,
                    q_max_seq_len=2,
                )
            # `_pdl_nowait` alone runs every cell, so its build is the full
            # kernel set (the wait's codegen-identity gate compares it).
            comptime if (RAGGED_EXIT_BREAK_KIND != 0 or COMBINE_TRIGGER_BREAK):
                print("negative control: all expectations met")
                return

            # ===============================================================
            # Layout G (MMA_M=32) cells. Heads past the existing h=8/16
            # coverage, unfolded MTP verify, folds that fit 32 rows, top-k
            # remainders, and 1-, 2- and 4-tile key ranges (the last two leave
            # the final tile in SW64 stage 0 and 1 of 3).
            # ===============================================================
            run_test_sparse_qkv_fp8[.float8_e4m3fn, 24](
                "g_h24_cl512_topk64", 1, 512, ctx, fixtures=kv_fixtures, topk=64
            )
            run_test_sparse_qkv_fp8[.float8_e4m3fn, 32](
                "g_h32_cl1024_topk256",
                2,
                1024,
                ctx,
                fixtures=kv_fixtures,
                topk=256,
            )
            run_test_sparse_qkv_fp8[.float8_e4m3fn, 24, use_causal=True](
                "g_causal_h24_cl512_topk128_seq6",
                2,
                512,
                ctx,
                fixtures=kv_fixtures,
                topk=128,
                q_max_seq_len=6,
            )
            run_test_sparse_qkv_fp8[.float8_e4m3fn, 32, use_causal=True](
                "g_causal_h32_cl2048_topk512_seq6",
                2,
                2048,
                ctx,
                fixtures=kv_fixtures,
                topk=512,
                q_max_seq_len=6,
            )
            run_test_sparse_qkv_fp8[
                .float8_e4m3fn, 8, use_causal=True, logical=True
            ](
                "g_causal_logical_h8_cl2048_topk512_seq6",
                4,
                2048,
                ctx,
                fixtures=kv_fixtures,
                topk=512,
                q_max_seq_len=6,
            )
            run_test_sparse_qkv_fp8[.float8_e4m3fn, 8, use_causal=True](
                "g_causal_h8_cl4096_topk2048_seq6",
                8,
                4096,
                ctx,
                fixtures=kv_fixtures,
                topk=2048,
                q_max_seq_len=6,
            )
            run_test_sparse_qkv_fp8[.float8_e4m3fn, 8](
                "g_h8_cl256_topk43", 2, 256, ctx, fixtures=kv_fixtures, topk=43
            )
            run_test_sparse_qkv_fp8[.float8_e4m3fn, 32](
                "g_h32_cl512_topk79", 1, 512, ctx, fixtures=kv_fixtures, topk=79
            )
            run_test_sparse_qkv_fp8[.float8_e4m3fn, 8, use_causal=True](
                "g_causal_h8_cl256_topk73_seq6",
                2,
                256,
                ctx,
                fixtures=kv_fixtures,
                topk=73,
                q_max_seq_len=6,
            )
            run_test_sparse_qkv_fp8[.float8_e4m3fn, 8](
                "g_h8_cl40_topk40_1tile",
                1,
                40,
                ctx,
                fixtures=kv_fixtures,
                topk=40,
            )
            run_test_sparse_qkv_fp8[.float8_e4m3fn, 16](
                "g_h16_cl128_topk128_2tile",
                1,
                128,
                ctx,
                fixtures=kv_fixtures,
                topk=128,
            )
            run_test_sparse_qkv_fp8[.float8_e4m3fn, 8](
                "g_h8_cl256_topk256_4tile",
                1,
                256,
                ctx,
                fixtures=kv_fixtures,
                topk=256,
            )
            # Odd tile counts exercise the partial final tile, and a
            # 1-tile partition leaves pipeline 1 empty.
            run_test_sparse_qkv_fp8[.float8_e4m3fn, 8](
                "g_h8_cl192_topk192_3tile",
                1,
                192,
                ctx,
                fixtures=kv_fixtures,
                topk=192,
            )
            run_test_sparse_qkv_fp8[.float8_e4m3fn, 8, use_causal=True](
                "g_causal_h8_cl2048_topk320_np3_seq6",
                2,
                2048,
                ctx,
                fixtures=kv_fixtures,
                topk=320,
                q_max_seq_len=6,
                forced_np=3,
            )
            run_test_sparse_qkv_fp8[.float8_e4m3fn, 8](
                "g_h8_cl512_topk128_np2",
                1,
                512,
                ctx,
                fixtures=kv_fixtures,
                topk=128,
                forced_np=2,
            )
            # Forced splits down to one 128-key page per split, which
            # the split cost model picks less often than the shared one.
            for np in [2, 3, 8]:
                run_test_sparse_qkv_fp8[.float8_e4m3fn, 8](
                    "g_h8_cl2048_topk1024_forced_np",
                    2,
                    2048,
                    ctx,
                    fixtures=kv_fixtures,
                    topk=1024,
                    forced_np=np,
                    split_fail=True,
                )
                run_test_sparse_qkv_fp8[.float8_e4m3fn, 32](
                    "g_h32_cl2048_topk1024_forced_np",
                    2,
                    2048,
                    ctx,
                    fixtures=kv_fixtures,
                    topk=1024,
                    forced_np=np,
                    split_fail=True,
                )
            run_test_sparse_qkv_fp8[.float8_e4m3fn, 8, shared_index=True](
                "g_fold4_h8_cl1024_topk160_seq4",
                2,
                1024,
                ctx,
                fixtures=kv_fixtures,
                topk=160,
                q_max_seq_len=4,
            )
            run_test_sparse_qkv_fp8[.float8_e4m3fn, 16, shared_index=True](
                "g_fold2_h16_cl512_topk96_seq2",
                2,
                512,
                ctx,
                fixtures=kv_fixtures,
                topk=96,
                q_max_seq_len=2,
            )
            run_test_sparse_qkv_fp8_attn_sink[.float8_e4m3fn, 32](
                "g_sink_h32_cl512_topk64", 1, 512, ctx, topk=64
            )
            # Lazy-rescale stimuli (see STIM_*): 2-, 3- and 5-tile ranges
            # and a 3-way split. The odd-tile spike puts every raise on
            # odd tiles, so the online-softmax rescale is exercised
            # it (`break_expect`). Under the every-tile spike the next
            # pipeline-0 raise scales a skipped rescale's stale terms by
            # about 2^-13, so one CTA ending on pipeline 0 passes anyway.
            run_test_sparse_qkv_fp8[.float8_e4m3fn, 8](
                "g_spike_odd_h8_topk128_2tile",
                1,
                128,
                ctx,
                fixtures=kv_fixtures,
                topk=128,
                stim=STIM_SPIKE_ODD,
                break_expect=BREAK_MUST_FAIL,
            )
            run_test_sparse_qkv_fp8[.float8_e4m3fn, 8](
                "g_spike_odd_h8_topk192_3tile",
                1,
                192,
                ctx,
                fixtures=kv_fixtures,
                topk=192,
                stim=STIM_SPIKE_ODD,
                break_expect=BREAK_MUST_FAIL,
            )
            run_test_sparse_qkv_fp8[.float8_e4m3fn, 8](
                "g_spike_odd_h8_topk320_5tile",
                2,
                320,
                ctx,
                fixtures=kv_fixtures,
                topk=320,
                stim=STIM_SPIKE_ODD,
                break_expect=BREAK_MUST_FAIL,
            )
            run_test_sparse_qkv_fp8[.float8_e4m3fn, 32](
                "g_spike_even_h32_topk320_5tile",
                1,
                320,
                ctx,
                fixtures=kv_fixtures,
                topk=320,
                stim=STIM_SPIKE_EVEN,
                break_expect=BREAK_MUST_PASS,
            )
            run_test_sparse_qkv_fp8[.float8_e4m3fn, 8](
                "g_spike_every_h8_topk320_5tile",
                1,
                320,
                ctx,
                fixtures=kv_fixtures,
                topk=320,
                stim=STIM_SPIKE_EVERY,
                break_expect=BREAK_MUST_PASS,
            )
            run_test_sparse_qkv_fp8[.float8_e4m3fn, 8](
                "g_spike_falling_h8_topk320_5tile",
                1,
                320,
                ctx,
                fixtures=kv_fixtures,
                topk=320,
                stim=STIM_SPIKE_FALLING,
                break_expect=BREAK_MUST_PASS,
            )
            run_test_sparse_qkv_fp8[.float8_e4m3fn, 8](
                "g_spike_odd_h8_cl2048_topk320_np3",
                1,
                2048,
                ctx,
                fixtures=kv_fixtures,
                topk=320,
                forced_np=3,
                stim=STIM_SPIKE_ODD,
                break_expect=BREAK_MUST_FAIL,
            )
            run_test_sparse_qkv_fp8[.float8_e4m3fn, 8](
                "g_spike_every_h8_cl2048_topk320_np3",
                1,
                2048,
                ctx,
                fixtures=kv_fixtures,
                topk=320,
                forced_np=3,
                stim=STIM_SPIKE_EVERY,
                break_expect=BREAK_EITHER,
            )
            run_test_sparse_qkv_fp8[.float8_e4m3fn, 8](
                "g_t3_h8_cl2048_topk2048",
                2,
                2048,
                ctx,
                fixtures=kv_fixtures,
                topk=2048,
                stim=STIM_T3,
                break_expect=BREAK_EITHER,
                hr1_fail=True,
            )
            run_test_sparse_qkv_fp8[.float8_e4m3fn, 8](
                "g_t7_h8_cl2048_topk2048",
                2,
                2048,
                ctx,
                fixtures=kv_fixtures,
                topk=2048,
                stim=STIM_T7,
                break_expect=BREAK_EITHER,
            )
            run_test_sparse_qkv_fp8[.float8_e4m3fn, 32](
                "g_t3_h32_cl512_topk320",
                1,
                512,
                ctx,
                fixtures=kv_fixtures,
                topk=320,
                stim=STIM_T3,
                break_expect=BREAK_EITHER,
            )
            run_test_sparse_qkv_fp8[.float8_e4m3fn, 8](
                "g_t3_h8_cl2048_topk320_np3",
                1,
                2048,
                ctx,
                fixtures=kv_fixtures,
                topk=320,
                forced_np=3,
                stim=STIM_T3,
                break_expect=BREAK_EITHER,
            )
            # Spikes under the causal slot limit, the logical-position
            # mask, MTP verify and the shared-index fold. Query token s
            # raises on tiles shifted by s, so tokens sharing a warp
            # disagree on which tiles raise.
            run_test_sparse_qkv_fp8[
                .float8_e4m3fn, 8, use_causal=True, logical=True
            ](
                "g_spike_odd_causal_logical_h8_cl2048_topk512_seq6",
                2,
                2048,
                ctx,
                fixtures=kv_fixtures,
                topk=512,
                q_max_seq_len=6,
                stim=STIM_SPIKE_ODD,
                break_expect=BREAK_MUST_FAIL,
            )
            run_test_sparse_qkv_fp8[.float8_e4m3fn, 8, use_causal=True](
                "g_spike_every_causal_h8_cl2048_topk320_np3_seq6",
                2,
                2048,
                ctx,
                fixtures=kv_fixtures,
                topk=320,
                q_max_seq_len=6,
                forced_np=3,
                stim=STIM_SPIKE_EVERY,
                break_expect=BREAK_EITHER,
            )
            run_test_sparse_qkv_fp8[.float8_e4m3fn, 8, shared_index=True](
                "g_spike_odd_fold4_h8_cl1024_topk320_seq4",
                2,
                1024,
                ctx,
                fixtures=kv_fixtures,
                topk=320,
                q_max_seq_len=4,
                stim=STIM_SPIKE_ODD,
                break_expect=BREAK_MUST_FAIL,
            )
            # Half-row softmax cells: at most 16 rows (h=8, h=16, fold 2
            # at h=8), the attention sink, logical-mask tails and splits.
            run_test_sparse_qkv_fp8[.float8_e4m3fn, 8, shared_index=True](
                "g_fold2_h8_cl512_topk96_seq2",
                2,
                512,
                ctx,
                fixtures=kv_fixtures,
                topk=96,
                q_max_seq_len=2,
            )
            run_test_sparse_qkv_fp8[
                .float8_e4m3fn, 8, use_causal=True, shared_index=True
            ](
                "g_fold2_causal_h8_cl1024_topk160_seq2",
                2,
                1024,
                ctx,
                fixtures=kv_fixtures,
                topk=160,
                q_max_seq_len=2,
            )
            run_test_sparse_qkv_fp8_attn_sink[.float8_e4m3fn, 8](
                "g_sink_h8_cl512_topk64", 1, 512, ctx, topk=64
            )
            run_test_sparse_qkv_fp8_attn_sink[.float8_e4m3fn, 16](
                "g_sink_h16_cl512_topk64", 1, 512, ctx, topk=64
            )
            run_test_sparse_qkv_fp8[
                .float8_e4m3fn, 16, use_causal=True, logical=True
            ](
                "g_causal_logical_h16_cl2048_topk512_seq6",
                2,
                2048,
                ctx,
                fixtures=kv_fixtures,
                topk=512,
                q_max_seq_len=6,
            )
            run_test_sparse_qkv_fp8[
                .float8_e4m3fn, 8, use_causal=True, logical=True
            ](
                "g_causal_logical_h8_cl2048_topk203_np3_seq6",
                2,
                2048,
                ctx,
                fixtures=kv_fixtures,
                topk=203,
                q_max_seq_len=6,
                forced_np=3,
            )
            # Logical positions past the causal horizon (cache_len +
            # s): cl58 masks slots 20 (warp 1, half 0) and 41-42
            # (warp 2, half 1) at s=0; cl250 masks 84, 169 and 170.
            # The pad cell's -1 slots fall in both halves of every
            # warp, so a ballot half swap must fail it.
            run_test_sparse_qkv_fp8[
                .float8_e4m3fn, 8, use_causal=True, logical=True
            ](
                "g_causal_logical_h8_cl58_topk60_seq6",
                2,
                58,
                ctx,
                fixtures=kv_fixtures,
                topk=60,
                q_max_seq_len=6,
            )
            run_test_sparse_qkv_fp8[
                .float8_e4m3fn, 8, use_causal=True, logical=True
            ](
                "g_causal_logical_h8_cl250_topk200_seq6",
                2,
                250,
                ctx,
                fixtures=kv_fixtures,
                topk=200,
                q_max_seq_len=6,
            )
            run_test_sparse_qkv_fp8[
                .float8_e4m3fn, 8, use_causal=True, logical=True
            ](
                "g_causal_logical_pad3_h8_cl2048_topk203_seq6",
                2,
                2048,
                ctx,
                fixtures=kv_fixtures,
                topk=203,
                q_max_seq_len=6,
                logical_pad=3,
            )
            run_test_sparse_qkv_fp8[.float8_e4m3fn, 8](
                "g_h8_cl256_topk41", 2, 256, ctx, fixtures=kv_fixtures, topk=41
            )
            run_test_sparse_qkv_fp8[.float8_e4m3fn, 8](
                "g_h8_cl256_topk47", 2, 256, ctx, fixtures=kv_fixtures, topk=47
            )
            run_test_sparse_qkv_fp8[.float8_e4m3fn, 8](
                "g_h8_cl256_topk56", 2, 256, ctx, fixtures=kv_fixtures, topk=56
            )
            run_test_sparse_qkv_fp8[.float8_e4m3fn, 16](
                "g_t3_h16_cl2048_topk1024",
                2,
                2048,
                ctx,
                fixtures=kv_fixtures,
                topk=1024,
                stim=STIM_T3,
            )
            run_test_sparse_qkv_fp8[.float8_e4m3fn, 8, shared_index=True](
                "g_spike_odd_fold2_h8_cl1024_topk320_seq2",
                2,
                1024,
                ctx,
                fixtures=kv_fixtures,
                topk=320,
                q_max_seq_len=2,
                stim=STIM_SPIKE_ODD,
            )
            _run_p6_cells(ctx, kv_fixtures)
            _run_half_split_cells(ctx, kv_fixtures)
            comptime if HALF_ROWS_BREAK or P6_BREAK:
                # The rest of main does not exercise Layout G.
                print("negative control: all expectations met")
                return

            # ===============================================================
            # NoPE native 512 on the production (native-FP8) decode backend.
            # ---------------------------------------------------------------
            # Breadth comes from differentials against the zero-tailed 576 run
            # rather than a host reference: both arms are GPU runs, so the
            # production shapes are affordable and the comparison is bit-exact.
            # ===============================================================
            run_nope_native_512_fp8[64](ctx, batch_size=1, cache_len=256)
            run_nope_native_512_fp8[64](ctx, batch_size=4, cache_len=2048)

            # Primes and the page-size boundary (PAGE_SIZE=128).
            run_nope_native_512_fp8[64](
                ctx, batch_size=1, cache_len=2, name="prime_cl2"
            )
            run_nope_native_512_fp8[64](
                ctx, batch_size=1, cache_len=7, name="prime_cl7"
            )
            run_nope_native_512_fp8[64](
                ctx, batch_size=3, cache_len=13, name="prime_cl13"
            )
            run_nope_native_512_fp8[64](
                ctx, batch_size=1, cache_len=97, name="prime_cl97"
            )
            run_nope_native_512_fp8[64](
                ctx, batch_size=1, cache_len=127, name="page_cl127"
            )
            run_nope_native_512_fp8[64](
                ctx, batch_size=1, cache_len=128, name="page_cl128"
            )
            run_nope_native_512_fp8[64](
                ctx, batch_size=1, cache_len=129, name="page_cl129"
            )
            run_nope_native_512_fp8[64](
                ctx, batch_size=2, cache_len=257, topk=64, name="prime_cl257"
            )

            # Production selection width, long context, and batching.
            run_nope_native_512_fp8[64](
                ctx,
                batch_size=1,
                cache_len=2048,
                topk=2048,
                name="prod_topk2048_saturating",
            )
            run_nope_native_512_fp8[64](
                ctx,
                batch_size=1,
                cache_len=8192,
                topk=2048,
                name="prod_cl8192_topk2048",
            )
            run_nope_native_512_fp8[64](
                ctx,
                batch_size=4,
                cache_len=4096,
                topk=2048,
                name="prod_b4_cl4096_topk2048",
            )
            # The real GLM decode step: TP=8 -> 8 heads, MTP=5 -> q_len 6.
            run_nope_native_512_fp8[8](
                ctx,
                batch_size=1,
                cache_len=4096,
                topk=2048,
                q_max_seq_len=6,
                name="prod_tp8_mtp6_topk2048",
            )
            run_nope_native_512_fp8[8](
                ctx,
                batch_size=4,
                cache_len=2048,
                topk=2048,
                q_max_seq_len=6,
                name="prod_tp8_mtp6_b4",
            )

            # Production decode batch sweep at the deployed shape. Batch moves
            # the decode grid and the split-K decision, so the narrow row has
            # to hold across it, not only at the one batch the mechanism was
            # first proved on.
            run_nope_native_512_fp8[8](
                ctx,
                batch_size=8,
                cache_len=4096,
                topk=2048,
                q_max_seq_len=6,
                name="prod_tp8_mtp6_b8",
            )
            run_nope_native_512_fp8[8](
                ctx,
                batch_size=16,
                cache_len=4096,
                topk=2048,
                q_max_seq_len=6,
                name="prod_tp8_mtp6_b16",
            )
            run_nope_native_512_fp8[8](
                ctx,
                batch_size=24,
                cache_len=4096,
                topk=2048,
                q_max_seq_len=6,
                name="prod_tp8_mtp6_b24",
            )
            run_nope_native_512_fp8[8](
                ctx,
                batch_size=32,
                cache_len=4096,
                topk=2048,
                q_max_seq_len=6,
                name="prod_tp8_mtp6_b32",
            )
            # The selection width is not a round 2048. Pooling by 4 selects 512
            # pools of four contiguous tokens and always appends a tail of up to
            # three, so the row the kernel sees is 2051 wide; an engine that
            # tiles its index by 128 columns rounds that to 2176. Neither is a
            # multiple of the gather tile, so both run the remainder path at
            # production scale rather than at the toy sizes the prime cases use.
            run_nope_native_512_fp8[8](
                ctx,
                batch_size=8,
                cache_len=8192,
                topk=2051,
                q_max_seq_len=6,
                name="prod_kpool_tail_topk2051",
            )
            run_nope_native_512_fp8[8](
                ctx,
                batch_size=8,
                cache_len=8192,
                topk=2176,
                q_max_seq_len=6,
                name="prod_engine_padded_topk2176",
            )
            # Speculative depth sweep. The deployment recipe is MTP=5 (q_len 6)
            # but the engine varies draft depth at runtime and falls back to
            # q_len 1 with speculation off.
            run_nope_native_512_fp8[8](
                ctx,
                batch_size=8,
                cache_len=4096,
                topk=2048,
                q_max_seq_len=1,
                name="prod_tp8_qlen1_b8",
            )
            run_nope_native_512_fp8[8](
                ctx,
                batch_size=8,
                cache_len=4096,
                topk=2048,
                q_max_seq_len=2,
                name="prod_tp8_qlen2_b8",
            )
            run_nope_native_512_fp8[8](
                ctx,
                batch_size=8,
                cache_len=4096,
                topk=2048,
                q_max_seq_len=4,
                name="prod_tp8_qlen4_b8",
            )
            # Multi-token Q at smaller scale, and prime-on-prime.
            run_nope_native_512_fp8[64](
                ctx,
                batch_size=2,
                cache_len=512,
                topk=64,
                q_max_seq_len=4,
                name="qlen4_b2",
            )
            run_nope_native_512_fp8[64](
                ctx,
                batch_size=2,
                cache_len=1021,
                topk=509,
                name="prime_cl1021_topk509",
            )

            # =====================================================
            # Read-once shared-index MTP fold (KERN-3141), shared_index=True.
            # Exercises the fold branch with native FP8-Q.
            # =====================================================

            # NullMask, single tile.
            run_test_sparse_qkv_fp8[.float8_e4m3fn, 8, shared_index=True](
                "shared_fold_b1_h8_cl256_topk64_seq6",
                1,
                256,
                ctx,
                fixtures=kv_fixtures,
                topk=64,
                q_max_seq_len=6,
            )
            # Multi-tile shared list.
            run_test_sparse_qkv_fp8[.float8_e4m3fn, 8, shared_index=True](
                "shared_fold_b2_h8_cl1024_topk160_seq6",
                2,
                1024,
                ctx,
                fixtures=kv_fixtures,
                topk=160,
                q_max_seq_len=6,
            )
            # CausalMask fold.
            run_test_sparse_qkv_fp8[
                DType.float8_e4m3fn, 8, use_causal=True, shared_index=True
            ](
                "shared_fold_causal_b1_h8_cl64_topk70_seq6",
                1,
                64,
                ctx,
                fixtures=kv_fixtures,
                topk=70,
                q_max_seq_len=6,
            )
            # Order audit: reversed gather order — NullMask output must be
            # invariant to gather order.
            run_test_sparse_qkv_fp8[
                DType.float8_e4m3fn, 8, shared_index=True, order_mode=1
            ](
                "shared_fold_null_reversed_b2_h8_cl256_topk64_seq6",
                2,
                256,
                ctx,
                fixtures=kv_fixtures,
                topk=64,
                q_max_seq_len=6,
            )
            # Coprime-permuted gather order.
            run_test_sparse_qkv_fp8[
                DType.float8_e4m3fn, 8, shared_index=True, order_mode=2
            ](
                "shared_fold_null_permuted_b2_h8_cl512_topk160_seq6",
                2,
                512,
                ctx,
                fixtures=kv_fixtures,
                topk=160,
                q_max_seq_len=6,
            )
            # Negative: per-position distinct sets (fold not selected).
            run_test_sparse_qkv_fp8[.float8_e4m3fn, 8, shared_index=False](
                "neg_distinct_sets_unfolded_b2_h8_cl512_topk64_seq6",
                2,
                512,
                ctx,
                fixtures=kv_fixtures,
                topk=64,
                q_max_seq_len=6,
            )
            # Production shape: bs8, cache=2048, topk=2048.
            run_test_sparse_qkv_fp8[.float8_e4m3fn, 8, shared_index=True](
                "shared_fold_prod_b8_h8_cl2048_topk2048_seq6",
                8,
                2048,
                ctx,
                fixtures=kv_fixtures,
                topk=2048,
                q_max_seq_len=6,
            )
            # seq_len=0: empty last batch under the fold.
            run_test_sparse_qkv_fp8[
                DType.float8_e4m3fn, 8, shared_index=True, empty_last_batch=True
            ](
                "shared_fold_seqlen0_b2_h8_cl256_topk64_seq6",
                2,
                256,
                ctx,
                fixtures=kv_fixtures,
                topk=64,
                q_max_seq_len=6,
            )

            # =====================================================
            # Base variants: NullMask, no feature flags.
            # =====================================================

            # Minimal config: bs=1, h=16, cl=128, topk=8.
            # topk=8 is few enough keys that FP8-P softmax quantization pushes a
            # ~1% error tail past 1e-2 (cosine stays > 0.999); loosen tail bound.
            run_test_sparse_qkv_fp8[.float8_e4m3fn, 16](
                "sparse_qkv_fp8_min_b1_h16_cl128_topk8",
                1,
                128,
                ctx,
                fixtures=kv_fixtures,
                topk=8,
                tail_max=0.03,
            )

            # bs=1, h=16, cl=512, topk=64.
            run_test_sparse_qkv_fp8[.float8_e4m3fn, 16](
                "sparse_qkv_fp8_b1_h16_cl512_topk64",
                1,
                512,
                ctx,
                fixtures=kv_fixtures,
                topk=64,
            )

            # Production-ish: bs=1, h=128, cl=512, topk=64.
            run_test_sparse_qkv_fp8[.float8_e4m3fn, 128](
                "sparse_qkv_fp8_b1_h128_cl512_topk64",
                1,
                512,
                ctx,
                fixtures=kv_fixtures,
                topk=64,
            )

            # Larger cache with split-K: bs=1, h=128, cl=2048, topk=64.
            run_test_sparse_qkv_fp8[.float8_e4m3fn, 128](
                "sparse_qkv_fp8_b1_h128_cl2048_topk64",
                1,
                2048,
                ctx,
                fixtures=kv_fixtures,
                topk=64,
            )
            _run_pair_cells(ctx, kv_fixtures)

            # =====================================================
            # Multi-batch (batch_size > 1).
            # =====================================================

            # bs=2, h=16, cl=128, topk=64.
            run_test_sparse_qkv_fp8[.float8_e4m3fn, 16](
                "sparse_qkv_fp8_b2_h16_cl128_topk64",
                2,
                128,
                ctx,
                fixtures=kv_fixtures,
                topk=64,
            )

            # bs=4, h=16, cl=256, topk=64.
            run_test_sparse_qkv_fp8[.float8_e4m3fn, 16](
                "sparse_qkv_fp8_b4_h16_cl256_topk64",
                4,
                256,
                ctx,
                fixtures=kv_fixtures,
                topk=64,
            )

            # =====================================================
            # Multi-token Q (speculative decode).
            # =====================================================

            # q_max_seq_len=2.
            run_test_sparse_qkv_fp8[.float8_e4m3fn, 16](
                "sparse_qkv_fp8_b1_h16_cl256_topk64_seq2",
                1,
                256,
                ctx,
                fixtures=kv_fixtures,
                topk=64,
                q_max_seq_len=2,
            )

            # q_max_seq_len=4.
            run_test_sparse_qkv_fp8[.float8_e4m3fn, 16](
                "sparse_qkv_fp8_b1_h16_cl256_topk64_seq4",
                1,
                256,
                ctx,
                fixtures=kv_fixtures,
                topk=64,
                q_max_seq_len=4,
            )

            # q_max_seq_len=8, multi-batch.
            run_test_sparse_qkv_fp8[.float8_e4m3fn, 16](
                "sparse_qkv_fp8_b2_h16_cl256_topk64_seq8",
                2,
                256,
                ctx,
                fixtures=kv_fixtures,
                topk=64,
                q_max_seq_len=8,
            )

            # =====================================================
            # CausalMask variant.
            # =====================================================

            run_test_sparse_qkv_fp8[.float8_e4m3fn, 16, use_causal=True](
                "sparse_qkv_fp8_causal_b1_h16_cl256_topk64",
                1,
                256,
                ctx,
                fixtures=kv_fixtures,
                topk=64,
            )

            # Multi-token Q with genuine sparsity (topk << num_keys) so the
            # score-sorted selection is not logical-position-sorted: exercises
            # causal masking by logical position rather than gather-slot order.
            run_test_sparse_qkv_fp8[.float8_e4m3fn, 16, use_causal=True](
                "sparse_qkv_fp8_causal_multitoken_b2_h16_cl256_topk64_seq8",
                2,
                256,
                ctx,
                fixtures=kv_fixtures,
                topk=64,
                q_max_seq_len=8,
            )

            # The logical-position mask reads its slots in batches of 8 and
            # picks up the remainder one at a time. A thread owns 32 slots, so
            # the cases above (topk a multiple of 32) only ever run whole
            # batches, and the topk=70 case above only ever runs the remainder:
            # the tail's shift base is carried out of the batch loop and no
            # shape yet reaches it with a nonzero carry. These land the
            # remainder at 3, 7 and 1 slots after a whole batch. Prime cache
            # lengths and q also keep the paged gather off its tile boundaries.
            run_test_sparse_qkv_fp8[.float8_e4m3fn, 16, use_causal=True](
                "sparse_qkv_fp8_causal_b1_h16_cl251_topk43",
                1,
                251,
                ctx,
                fixtures=kv_fixtures,
                topk=43,
            )
            run_test_sparse_qkv_fp8[.float8_e4m3fn, 16, use_causal=True](
                "sparse_qkv_fp8_causal_multitoken_b2_h16_cl257_topk79_seq5",
                2,
                257,
                ctx,
                fixtures=kv_fixtures,
                topk=79,
                q_max_seq_len=5,
            )
            run_test_sparse_qkv_fp8[.float8_e4m3fn, 16, use_causal=True](
                "sparse_qkv_fp8_causal_multitoken_b3_h16_cl509_topk73_seq3",
                3,
                509,
                ctx,
                fixtures=kv_fixtures,
                topk=73,
                q_max_seq_len=3,
            )

            # =====================================================
            # Logical token positions, decoded through the shuffled per-batch
            # page table inside the kernel (the graph-op path). A wrong batch
            # row or a missed conversion site gathers another batch's pages.
            # =====================================================
            run_test_sparse_qkv_fp8[.float8_e4m3fn, 128, logical=True](
                "logical_b4_h128_cl2048_topk512",
                4,
                2048,
                ctx,
                fixtures=kv_fixtures,
                topk=512,
            )
            run_test_sparse_qkv_fp8[
                .float8_e4m3fn, 16, use_causal=True, logical=True
            ](
                "logical_causal_multitoken_b3_h16_cl509_topk73_seq3",
                3,
                509,
                ctx,
                fixtures=kv_fixtures,
                topk=73,
                q_max_seq_len=3,
            )
            run_test_sparse_qkv_fp8[
                .float8_e4m3fn, 8, shared_index=True, logical=True
            ](
                "logical_shared_fold_b2_h8_cl1024_topk160_seq6",
                2,
                1024,
                ctx,
                fixtures=kv_fixtures,
                topk=160,
                q_max_seq_len=6,
            )

            # =====================================================
            # Variable per-batch topk (has_variable_topk=True).
            # =====================================================

            var vt_cls_2: List[Int] = [256, 128]
            var vt_topk_2: List[Int] = [64, 32]
            run_test_sparse_qkv_fp8_variable_topk[.float8_e4m3fn, 16](
                "sparse_qkv_fp8_variable_topk_b2_h16",
                vt_cls_2,
                vt_topk_2,
                ctx,
            )

            var vt_cls_4: List[Int] = [256, 384, 128, 512]
            var vt_topk_4: List[Int] = [64, 128, 32, 64]
            run_test_sparse_qkv_fp8_variable_topk[.float8_e4m3fn, 16](
                "sparse_qkv_fp8_variable_topk_b4_h16",
                vt_cls_4,
                vt_topk_4,
                ctx,
            )

            # =====================================================
            # Attention sink (has_attn_sink=True).
            # =====================================================

            run_test_sparse_qkv_fp8_attn_sink[.float8_e4m3fn, 16](
                "sparse_qkv_fp8_attn_sink_b1_h16_cl128_topk64",
                1,
                128,
                ctx,
                topk=64,
            )

            run_test_sparse_qkv_fp8_attn_sink[.float8_e4m3fn, 128](
                "sparse_qkv_fp8_attn_sink_b1_h128_cl512_topk64",
                1,
                512,
                ctx,
                topk=64,
            )

            run_test_sparse_qkv_fp8_attn_sink[.float8_e4m3fn, 16](
                "sparse_qkv_fp8_attn_sink_b4_h16_cl256_topk64",
                4,
                256,
                ctx,
                topk=64,
            )

            # =====================================================
            # Extra KV (has_extra_kv=True).
            # =====================================================

            var ek_cls_1: List[Int] = [256]
            var ek_topk_1: List[Int] = [64]
            var ek_ecls_1: List[Int] = [64]
            var ek_etopk_1: List[Int] = [64]
            run_test_sparse_qkv_fp8_extra_kv[.float8_e4m3fn, 16](
                "sparse_qkv_fp8_extra_kv_b1_h16_topk64_extra64",
                ek_cls_1,
                ek_topk_1,
                ek_ecls_1,
                ek_etopk_1,
                ctx,
            )

            var ek_cls_2: List[Int] = [256, 384]
            var ek_topk_2: List[Int] = [64, 64]
            var ek_ecls_2: List[Int] = [64, 128]
            var ek_etopk_2: List[Int] = [64, 64]
            run_test_sparse_qkv_fp8_extra_kv[.float8_e4m3fn, 16](
                "sparse_qkv_fp8_extra_kv_b2_h16_variable",
                ek_cls_2,
                ek_topk_2,
                ek_ecls_2,
                ek_etopk_2,
                ctx,
            )

            # Main top-k values that are not tile multiples pad the main
            # segment. The long lists split; the short ones (45 + 24 keys,
            # one page) cannot. The split-K line printed above each launch
            # says which ran.
            var ek_cls_3: List[Int] = [512]
            var ek_topk_3: List[Int] = [300]
            var ek_ecls_3: List[Int] = [256]
            var ek_etopk_3: List[Int] = [200]
            run_test_sparse_qkv_fp8_extra_kv[.float8_e4m3fn, 16](
                "sparse_qkv_fp8_extra_kv_b1_h16_topk300_extra200",
                ek_cls_3,
                ek_topk_3,
                ek_ecls_3,
                ek_etopk_3,
                ctx,
            )

            var ek_cls_4: List[Int] = [64, 256]
            var ek_topk_4: List[Int] = [37, 45]
            var ek_ecls_4: List[Int] = [64, 128]
            var ek_etopk_4: List[Int] = [20, 24]
            run_test_sparse_qkv_fp8_extra_kv[.float8_e4m3fn, 16](
                "sparse_qkv_fp8_extra_kv_b2_h16_topk_unaligned",
                ek_cls_4,
                ek_topk_4,
                ek_ecls_4,
                ek_etopk_4,
                ctx,
            )

            # =====================================================
            # Topk clamping (small caches).
            # =====================================================

            # First decode: cache_length=0, actual=1, topk=64 -> clamp to 1.
            var tc_cls_1: List[Int] = [0]
            var tc_topk_1: List[Int] = [64]
            run_test_sparse_qkv_fp8_topk_clamping[.float8_e4m3fn, 16](
                "sparse_qkv_fp8_topk_clamp_first_exec_b1_h16",
                tc_cls_1,
                tc_topk_1,
                ctx,
            )

            # Small cache: cache_length=5, topk=64 -> clamp to 6.
            # Only 6 keys after clamping: FP8-P softmax quantization gives a ~3%
            # error tail past 1e-2 (cosine stays > 0.999); loosen tail bound.
            var tc_cls_2: List[Int] = [5]
            var tc_topk_2: List[Int] = [64]
            run_test_sparse_qkv_fp8_topk_clamping[.float8_e4m3fn, 16](
                "sparse_qkv_fp8_topk_clamp_small_cache_b1_h16",
                tc_cls_2,
                tc_topk_2,
                ctx,
                tail_max=0.05,
            )

            # Mixed batch: cl=0 and cl=256.
            var tc_cls_3: List[Int] = [0, 256]
            var tc_topk_3: List[Int] = [64, 64]
            run_test_sparse_qkv_fp8_topk_clamping[.float8_e4m3fn, 16](
                "sparse_qkv_fp8_topk_clamp_mixed_b2_h16",
                tc_cls_3,
                tc_topk_3,
                ctx,
            )

            # =====================================================
            # NoPE native 512, the ragged and edge-case families.
            # ---------------------------------------------------------------
            # The differential above runs the narrow row at production scale,
            # but only on shapes that are uniform across the batch. These
            # re-run the ragged families -- per-batch topk and cache length,
            # attention sink, a second always-attend cache, and clamping on a
            # near-empty cache -- at 512 against the same host reference the
            # 576 row is held to, so the narrow row clears the whole bar and
            # not a friendly subset of it.
            #
            # The extra-KV case is the one that answers a specific question:
            # the tail is zeroed once at kernel entry, so a second cache that
            # wrote the tail afterwards would break that. It does not.
            # =====================================================

            run_test_sparse_qkv_fp8[.float8_e4m3fn, 16, row_depth=512](
                "nope512_qkv_fp8_b4_h16_cl256_topk64",
                4,
                256,
                ctx,
                fixtures=kv_fixtures,
                topk=64,
            )
            run_test_sparse_qkv_fp8[
                .float8_e4m3fn, 16, use_causal=True, row_depth=512
            ](
                "nope512_qkv_fp8_causal_b3_h16_cl509_topk73_seq3",
                3,
                509,
                ctx,
                fixtures=kv_fixtures,
                topk=73,
                q_max_seq_len=3,
            )

            var n512_vt_cls: List[Int] = [256, 384, 128, 512]
            var n512_vt_topk: List[Int] = [64, 128, 32, 64]
            run_test_sparse_qkv_fp8_variable_topk[
                .float8_e4m3fn, 16, row_depth=512
            ](
                "nope512_qkv_fp8_variable_topk_b4_h16",
                n512_vt_cls,
                n512_vt_topk,
                ctx,
            )

            run_test_sparse_qkv_fp8_attn_sink[
                .float8_e4m3fn, 16, row_depth=512
            ](
                "nope512_qkv_fp8_attn_sink_b4_h16_cl256_topk64",
                4,
                256,
                ctx,
                topk=64,
            )

            var n512_ek_cls: List[Int] = [256, 384]
            var n512_ek_topk: List[Int] = [64, 64]
            var n512_ek_ecls: List[Int] = [64, 128]
            var n512_ek_etopk: List[Int] = [64, 64]
            run_test_sparse_qkv_fp8_extra_kv[.float8_e4m3fn, 16, row_depth=512](
                "nope512_qkv_fp8_extra_kv_b2_h16_variable",
                n512_ek_cls,
                n512_ek_topk,
                n512_ek_ecls,
                n512_ek_etopk,
                ctx,
            )

            var n512_tc_cls: List[Int] = [0, 256]
            var n512_tc_topk: List[Int] = [64, 64]
            run_test_sparse_qkv_fp8_topk_clamping[
                .float8_e4m3fn, 16, row_depth=512
            ](
                "nope512_qkv_fp8_topk_clamp_mixed_b2_h16",
                n512_tc_cls,
                n512_tc_topk,
                ctx,
            )
