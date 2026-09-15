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
"""Tests for the KDA decode CPU scalar reference (M0).

Compares the M0 reference (`kda_decode_ref`) against embedded FP64-gold
values derived from `naive_recurrent_kda` (FLA). All three test cases are
fixed-length sequences; the criterion is an RMSE ratio,
RMSE(gold - ref) / RMSE(gold).

The bound is 1e-5, not the 0.005 the GPU tests use: this is deterministic CPU
arithmetic against FP64 goldens, with none of the tensor-core nondeterminism
0.005 absorbs. Observed error is ~1e-7; at 0.005 a 0.1% drift in the oracle
passes unnoticed.

Also includes explicit K_FIRST / V_FIRST layout equivalence tests that verify
both layouts produce identical outputs and final states for the same logical
state matrix.
"""

import std.math
from std.math import isfinite, sqrt
from std.testing import TestSuite, assert_true

from kda.reference import kda_decode_ref
from testdata.kda_goldens import (
    case0_meta,
    case0_q,
    case0_k,
    case0_v,
    case0_raw_gate,
    case0_beta_logits,
    case0_a_log,
    case0_dt_bias,
    case0_h0,
    case0_cu_seqlens,
    case0_gold_o,
    case0_gold_ht,
    case1_meta,
    case1_q,
    case1_k,
    case1_v,
    case1_raw_gate,
    case1_beta_logits,
    case1_a_log,
    case1_dt_bias,
    case1_h0,
    case1_cu_seqlens,
    case1_gold_o,
    case1_gold_ht,
    case2_meta,
    case2_q,
    case2_k,
    case2_v,
    case2_raw_gate,
    case2_beta_logits,
    case2_a_log,
    case2_dt_bias,
    case2_h0,
    case2_cu_seqlens,
    case2_gold_o,
    case2_gold_ht,
)


def _rmse(a: List[Float32], b: List[Float32]) -> Float64:
    """RMSE between two float32 lists."""
    var n = len(a)
    var sum_sq = Float64(0.0)
    for i in range(n):
        var d = Float64(a[i]) - Float64(b[i])
        sum_sq = sum_sq + d * d
    return sqrt(sum_sq / Float64(n))


def _rel_err(gold: List[Float32], cand: List[Float32]) -> Float64:
    """rel_err = RMSE(gold - cand) / RMSE(gold) (FLA get_err_ratio convention).
    """
    var diff_rmse = _rmse(gold, cand)
    # RMSE of gold against zero (denominator of the FLA err-ratio).
    var n = len(gold)
    var g_sq = Float64(0.0)
    for i in range(n):
        var g = Float64(gold[i])
        g_sq = g_sq + g * g
    var gold_rmse = sqrt(g_sq / Float64(n))
    return diff_rmse / (gold_rmse + Float64(1e-8))


def _print_rel_err(
    tag: String, gold: List[Float32], cand: List[Float32]
) -> Float64:
    var r = _rel_err(gold, cand)
    print(tag + " rel_err=" + String(r))
    return r


def _fill_from_list(
    ptr: UnsafePointer[Scalar[DType.float32], MutUntrackedOrigin],
    vals: List[Scalar[DType.float32]],
):
    for i in range(len(vals)):
        ptr[i] = vals[i]


def _fill_from_list_i32(
    ptr: UnsafePointer[Scalar[DType.int32], MutUntrackedOrigin],
    vals: List[Scalar[DType.int32]],
):
    for i in range(len(vals)):
        ptr[i] = vals[i]


def _collect_output(
    ptr: UnsafePointer[Scalar[DType.float32], MutUntrackedOrigin], n: Int
) -> List[Float32]:
    var out = List[Float32](capacity=n)
    for i in range(n):
        out.append(Float32(ptr[i]))
    return out^


# ===----------------------------------------------------------------------=== #
# Generic runner: calls kda_decode_ref and returns (output, final_state).
# ===----------------------------------------------------------------------=== #


def _run_ref[
    gate_mode: StaticString,
    beta_mode: StaticString,
    state_layout: StaticString,
    K: Int,
    V: Int,
](
    N: Int,
    HV: Int,
    H: Int,
    T: Int,
    q_vals: List[Scalar[DType.float32]],
    k_vals: List[Scalar[DType.float32]],
    v_vals: List[Scalar[DType.float32]],
    raw_gate_vals: List[Scalar[DType.float32]],
    beta_logits_vals: List[Scalar[DType.float32]],
    a_log_vals: List[Scalar[DType.float32]],
    dt_bias_vals: List[Scalar[DType.float32]],
    h0_vals: List[Scalar[DType.float32]],
    cu_seqlens_vals: List[Scalar[DType.int32]],
) raises -> Tuple[List[Float32], List[Float32]]:
    """Run reference and return (output_flat, final_state_flat)."""
    var total_T = N * T

    # Allocate flat arrays on the heap.
    var q_buf = alloc[Scalar[DType.float32]](total_T * H * K)
    var k_buf = alloc[Scalar[DType.float32]](total_T * H * K)
    var v_buf = alloc[Scalar[DType.float32]](total_T * HV * V)
    var rg_buf = alloc[Scalar[DType.float32]](total_T * HV * K)
    var bl_buf = alloc[Scalar[DType.float32]](total_T * HV)
    var al_buf = alloc[Scalar[DType.float32]](HV)
    var dt_buf = alloc[Scalar[DType.float32]](HV * K)
    var cu_buf = alloc[Scalar[DType.int32]](N + 1)
    var state_buf = alloc[Scalar[DType.float32]](N * HV * K * V)
    var out_buf = alloc[Scalar[DType.float32]](total_T * HV * V)

    _fill_from_list(q_buf.bitcast[Scalar[DType.float32]](), q_vals)
    _fill_from_list(k_buf.bitcast[Scalar[DType.float32]](), k_vals)
    _fill_from_list(v_buf.bitcast[Scalar[DType.float32]](), v_vals)
    _fill_from_list(rg_buf.bitcast[Scalar[DType.float32]](), raw_gate_vals)
    _fill_from_list(bl_buf.bitcast[Scalar[DType.float32]](), beta_logits_vals)
    _fill_from_list(al_buf.bitcast[Scalar[DType.float32]](), a_log_vals)
    _fill_from_list(dt_buf.bitcast[Scalar[DType.float32]](), dt_bias_vals)
    _fill_from_list_i32(cu_buf.bitcast[Scalar[DType.int32]](), cu_seqlens_vals)
    _fill_from_list(state_buf.bitcast[Scalar[DType.float32]](), h0_vals)

    # State indices: batch item b uses slot b.
    var si_buf = alloc[Scalar[DType.int32]](N)
    for b in range(N):
        si_buf[b] = Int32(b)

    # Zero output buffer.
    for i in range(total_T * HV * V):
        out_buf[i] = Scalar[DType.float32](0.0)

    # State strides (K_FIRST: [N,HV,K,V] or V_FIRST: [N,HV,V,K]).
    var state_dim1_stride: Int
    var state_dim2_stride: Int
    comptime if state_layout == "K_FIRST":
        state_dim1_stride = V  # k outer
        state_dim2_stride = 1  # v inner
    else:
        state_dim1_stride = K  # v outer
        state_dim2_stride = 1  # k inner

    kda_decode_ref[
        DType.float32,
        DType.float32,
        DType.float32,
        DType.float32,
        gate_mode,
        beta_mode,
        state_layout,
    ](
        batch_size=N,
        num_value_heads=HV,
        num_key_heads=H,
        key_head_dim=K,
        value_head_dim=V,
        output_ptr=out_buf.bitcast[Scalar[DType.float32]](),
        q_ptr=q_buf.bitcast[Scalar[DType.float32]](),
        k_ptr=k_buf.bitcast[Scalar[DType.float32]](),
        v_ptr=v_buf.bitcast[Scalar[DType.float32]](),
        raw_gate_ptr=rg_buf.bitcast[Scalar[DType.float32]](),
        beta_logits_ptr=bl_buf.bitcast[Scalar[DType.float32]](),
        a_log_ptr=al_buf.bitcast[Scalar[DType.float32]](),
        dt_bias_ptr=dt_buf.bitcast[Scalar[DType.float32]](),
        cu_seqlens_ptr=cu_buf.bitcast[Scalar[DType.int32]](),
        state_pool_ptr=state_buf.bitcast[Scalar[DType.float32]](),
        state_indices_ptr=si_buf.bitcast[Scalar[DType.int32]](),
        # q strides: [1, total_T, H, K] → seqlen=H*K, head=K, key=1
        q_seqlen_stride=H * K,
        q_head_stride=K,
        q_key_stride=1,
        # k strides
        k_seqlen_stride=H * K,
        k_head_stride=K,
        k_key_stride=1,
        # v strides: [1, total_T, HV, V]
        v_seqlen_stride=HV * V,
        v_head_stride=V,
        v_value_stride=1,
        # raw_gate strides: [1, total_T, HV, K]
        raw_gate_seqlen_stride=HV * K,
        raw_gate_head_stride=K,
        raw_gate_key_stride=1,
        # beta strides: [1, total_T, HV]
        beta_seqlen_stride=HV,
        beta_head_stride=1,
        # dt_bias strides: [HV, K]
        dt_bias_head_stride=K,
        dt_bias_key_stride=1,
        # state strides: [N, HV, dim1, dim2]
        state_slot_stride=HV * K * V,
        state_head_stride=K * V,
        state_dim1_stride=state_dim1_stride,
        state_dim2_stride=state_dim2_stride,
        # output strides: [1, total_T, HV, V]
        out_seqlen_stride=HV * V,
        out_head_stride=V,
        out_value_stride=1,
    )

    var out_list = _collect_output(
        out_buf.bitcast[Scalar[DType.float32]](), total_T * HV * V
    )
    var state_list = _collect_output(
        state_buf.bitcast[Scalar[DType.float32]](), N * HV * K * V
    )

    q_buf.free()
    k_buf.free()
    v_buf.free()
    rg_buf.free()
    bl_buf.free()
    al_buf.free()
    dt_buf.free()
    cu_buf.free()
    state_buf.free()
    out_buf.free()
    si_buf.free()

    return out_list^, state_list^


# ===----------------------------------------------------------------------=== #
# Golden-comparison tests (M0 vs FP64 gold)
# ===----------------------------------------------------------------------=== #


def test_case0_original_logits_k16() raises:
    """Case 0: K=V=16, gate=original, beta=logits. Checks output and state."""
    var _, _, N, HV, H, T, _ = case0_meta()
    var gold_o = case0_gold_o()
    var gold_ht = case0_gold_ht()

    var _kda_res1 = _run_ref["original", "logits", "K_FIRST", 16, 16](
        N,
        HV,
        H,
        T,
        case0_q(),
        case0_k(),
        case0_v(),
        case0_raw_gate(),
        case0_beta_logits(),
        case0_a_log(),
        case0_dt_bias(),
        case0_h0(),
        case0_cu_seqlens(),
    )
    var out = _kda_res1[0].copy()
    var state = _kda_res1[1].copy()

    var err_o = _print_rel_err("case0 output", gold_o, out)
    var err_ht = _print_rel_err("case0 final_state", gold_ht, state)
    assert_true(
        err_o < Float64(1e-5),
        "case0 output rel_err=" + String(err_o) + " >= 1e-5",
    )
    assert_true(
        err_ht < Float64(1e-5),
        "case0 final_state rel_err=" + String(err_ht) + " >= 1e-5",
    )


def test_case1_safe_probability_k16() raises:
    """Case 1: K=V=16, gate=safe, beta=probability. Checks output and state."""
    var _, _, N, HV, H, T, _ = case1_meta()
    var gold_o = case1_gold_o()
    var gold_ht = case1_gold_ht()

    var _kda_res2 = _run_ref["safe", "probability", "K_FIRST", 16, 16](
        N,
        HV,
        H,
        T,
        case1_q(),
        case1_k(),
        case1_v(),
        case1_raw_gate(),
        case1_beta_logits(),
        case1_a_log(),
        case1_dt_bias(),
        case1_h0(),
        case1_cu_seqlens(),
    )
    var out = _kda_res2[0].copy()
    var state = _kda_res2[1].copy()

    var err_o = _print_rel_err("case1 output", gold_o, out)
    var err_ht = _print_rel_err("case1 final_state", gold_ht, state)
    assert_true(
        err_o < Float64(1e-5),
        "case1 output rel_err=" + String(err_o) + " >= 1e-5",
    )
    assert_true(
        err_ht < Float64(1e-5),
        "case1 final_state rel_err=" + String(err_ht) + " >= 1e-5",
    )


def test_case2_original_logits_k32() raises:
    """Case 2: K=V=32, gate=original, beta=logits. Checks output and state."""
    var _, _, N, HV, H, T, _ = case2_meta()
    var gold_o = case2_gold_o()
    var gold_ht = case2_gold_ht()

    var _kda_res3 = _run_ref["original", "logits", "K_FIRST", 32, 32](
        N,
        HV,
        H,
        T,
        case2_q(),
        case2_k(),
        case2_v(),
        case2_raw_gate(),
        case2_beta_logits(),
        case2_a_log(),
        case2_dt_bias(),
        case2_h0(),
        case2_cu_seqlens(),
    )
    var out = _kda_res3[0].copy()
    var state = _kda_res3[1].copy()

    var err_o = _print_rel_err("case2 output", gold_o, out)
    var err_ht = _print_rel_err("case2 final_state", gold_ht, state)
    assert_true(
        err_o < Float64(1e-5),
        "case2 output rel_err=" + String(err_o) + " >= 1e-5",
    )
    assert_true(
        err_ht < Float64(1e-5),
        "case2 final_state rel_err=" + String(err_ht) + " >= 1e-5",
    )


# ===----------------------------------------------------------------------=== #
# Layout equivalence test: K_FIRST vs V_FIRST must give same math
# ===----------------------------------------------------------------------=== #


def _transpose_state[
    K: Int, V: Int
](state_kfirst: List[Float32], N: Int, HV: Int) -> List[Float32]:
    """Transpose state [N,HV,K,V] → [N,HV,V,K]."""
    var out = List[Float32](capacity=N * HV * K * V)
    for _ in range(N * HV * K * V):
        out.append(Float32(0.0))
    for n in range(N):
        for hv in range(HV):
            for kd in range(K):
                for vd in range(V):
                    var src = n * HV * K * V + hv * K * V + kd * V + vd
                    var dst = n * HV * V * K + hv * V * K + vd * K + kd
                    out[dst] = state_kfirst[src]
    return out^


def test_layout_equivalence_k16_short() raises:
    """K=V=16, T=3: K_FIRST and V_FIRST produce identical output/state."""
    comptime K = 16
    comptime V = 16
    var N = 1
    var HV = 1
    var H = 1
    var T = 3

    var q_vals = case0_q()
    var k_vals = case0_k()
    var v_vals = case0_v()
    var rg_vals = case0_raw_gate()
    var bl_vals = case0_beta_logits()
    var al_vals = case0_a_log()
    var dt_vals = case0_dt_bias()
    var h0_vals = case0_h0()
    var cu_vals = case0_cu_seqlens()

    var _kda_res4 = _run_ref["original", "logits", "K_FIRST", K, V](
        N,
        HV,
        H,
        T,
        q_vals,
        k_vals,
        v_vals,
        rg_vals,
        bl_vals,
        al_vals,
        dt_vals,
        h0_vals,
        cu_vals,
    )
    var out_kf = _kda_res4[0].copy()
    var st_kf = _kda_res4[1].copy()

    # Transpose h0 [N,HV,K,V] → [N,HV,V,K] for V_FIRST input.
    var h0_vf = _transpose_state[K, V](h0_vals, N, HV)

    var _kda_res5 = _run_ref["original", "logits", "V_FIRST", K, V](
        N,
        HV,
        H,
        T,
        q_vals,
        k_vals,
        v_vals,
        rg_vals,
        bl_vals,
        al_vals,
        dt_vals,
        h0_vf,
        cu_vals,
    )
    var out_vf = _kda_res5[0].copy()
    var st_vf_raw = _kda_res5[1].copy()

    # Transpose V_FIRST final state back to K_FIRST for comparison.
    var st_vf = _transpose_state[V, K](st_vf_raw, N, HV)

    var err_o = _print_rel_err("layout_equiv output (kf vs vf)", out_kf, out_vf)
    var err_s = _print_rel_err(
        "layout_equiv state (kf vs vf-transposed)", st_kf, st_vf
    )
    assert_true(
        err_o < Float64(1e-6),
        "layout equiv output not identical: rel_err=" + String(err_o),
    )
    assert_true(
        err_s < Float64(1e-6),
        "layout equiv state not identical: rel_err=" + String(err_s),
    )


# ===----------------------------------------------------------------------=== #
# Long T / nonzero initial state regression (T=256, K=16)
# ===----------------------------------------------------------------------=== #


def test_long_sequence_nonzero_state_k16() raises:
    """Long T=256, K=16: M0 reference produces finite output and nonzero state.
    """
    comptime K = 16
    comptime V = 16
    var N = 1
    var HV = 1
    var H = 1
    var T = 256
    var total_T = T

    var q_buf = alloc[Scalar[DType.float32]](total_T * H * K)
    var k_buf = alloc[Scalar[DType.float32]](total_T * H * K)
    var v_buf = alloc[Scalar[DType.float32]](total_T * HV * V)
    var rg_buf = alloc[Scalar[DType.float32]](total_T * HV * K)
    var bl_buf = alloc[Scalar[DType.float32]](total_T * HV)
    var al_buf = alloc[Scalar[DType.float32]](HV)
    var dt_buf = alloc[Scalar[DType.float32]](HV * K)
    var h0_buf = alloc[Scalar[DType.float32]](N * HV * K * V)
    var si_buf = alloc[Scalar[DType.int32]](N)
    var cu_buf = alloc[Scalar[DType.int32]](N + 1)
    var out_buf = alloc[Scalar[DType.float32]](total_T * HV * V)

    # Deterministic analytic fill, so the case needs no RNG and no goldens.
    for i in range(total_T * H * K):
        q_buf[i] = Scalar[DType.float32](
            std.math.sin(Float32(i) * Float32(0.3))
        )
        k_buf[i] = Scalar[DType.float32](
            std.math.cos(Float32(i) * Float32(0.5))
        )
    for i in range(total_T * HV * V):
        v_buf[i] = Scalar[DType.float32](
            std.math.sin(Float32(i) * Float32(0.7))
        )
    for i in range(total_T * HV * K):
        rg_buf[i] = Scalar[DType.float32](
            std.math.sin(Float32(i) * Float32(0.11))
        )
    for i in range(total_T * HV):
        bl_buf[i] = Scalar[DType.float32](Float32(0.5))
    al_buf[0] = Scalar[DType.float32](Float32(0.1))
    for i in range(HV * K):
        dt_buf[i] = Scalar[DType.float32](Float32(0.0))
    # Non-zero initial state.
    for i in range(N * HV * K * V):
        h0_buf[i] = Scalar[DType.float32](
            std.math.sin(Float32(i) * Float32(0.2)) * Float32(0.1)
        )
    si_buf[0] = Int32(0)
    cu_buf[0] = Int32(0)
    cu_buf[1] = Int32(T)
    for i in range(total_T * HV * V):
        out_buf[i] = Scalar[DType.float32](0.0)

    kda_decode_ref[
        DType.float32,
        DType.float32,
        DType.float32,
        DType.float32,
        "original",
        "logits",
        "K_FIRST",
    ](
        batch_size=N,
        num_value_heads=HV,
        num_key_heads=H,
        key_head_dim=K,
        value_head_dim=V,
        output_ptr=out_buf.bitcast[Scalar[DType.float32]](),
        q_ptr=q_buf.bitcast[Scalar[DType.float32]](),
        k_ptr=k_buf.bitcast[Scalar[DType.float32]](),
        v_ptr=v_buf.bitcast[Scalar[DType.float32]](),
        raw_gate_ptr=rg_buf.bitcast[Scalar[DType.float32]](),
        beta_logits_ptr=bl_buf.bitcast[Scalar[DType.float32]](),
        a_log_ptr=al_buf.bitcast[Scalar[DType.float32]](),
        dt_bias_ptr=dt_buf.bitcast[Scalar[DType.float32]](),
        cu_seqlens_ptr=cu_buf.bitcast[Scalar[DType.int32]](),
        state_pool_ptr=h0_buf.bitcast[Scalar[DType.float32]](),
        state_indices_ptr=si_buf.bitcast[Scalar[DType.int32]](),
        q_seqlen_stride=H * K,
        q_head_stride=K,
        q_key_stride=1,
        k_seqlen_stride=H * K,
        k_head_stride=K,
        k_key_stride=1,
        v_seqlen_stride=HV * V,
        v_head_stride=V,
        v_value_stride=1,
        raw_gate_seqlen_stride=HV * K,
        raw_gate_head_stride=K,
        raw_gate_key_stride=1,
        beta_seqlen_stride=HV,
        beta_head_stride=1,
        dt_bias_head_stride=K,
        dt_bias_key_stride=1,
        state_slot_stride=HV * K * V,
        state_head_stride=K * V,
        state_dim1_stride=V,
        state_dim2_stride=1,
        out_seqlen_stride=HV * V,
        out_head_stride=V,
        out_value_stride=1,
    )

    # Check output is finite and nonzero.
    var any_nonzero = False
    for i in range(total_T * HV * V):
        var v_val = Float32(out_buf[i])
        assert_true(isfinite(v_val), "non-finite output at index " + String(i))
        if abs(v_val) > Float32(1e-10):
            any_nonzero = True
    assert_true(any_nonzero, "All output values are zero (regression)")
    print("test_long_sequence_nonzero_state_k16: output finite and nonzero OK")

    q_buf.free()
    k_buf.free()
    v_buf.free()
    rg_buf.free()
    bl_buf.free()
    al_buf.free()
    dt_buf.free()
    h0_buf.free()
    si_buf.free()
    cu_buf.free()
    out_buf.free()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
