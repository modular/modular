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
"""Large-M router-gate GEMM benchmark and accuracy check for SM100 (B200).

Compares the production prefill route (bf16->fp32 cast, then the fp32 matmul
dispatch) with `router_gate_tf32_gemm` (weight rounded to TF32 to nearest;
the CTA-group-1 variant is `cta1`) on `[M, 6144] bf16 x [128, 6144] fp32^T`.
The weight, norm gamma and score-correction bias are synthetic unless
`--real-dir=DIR --layer=N` points at raw fp32 dumps of a real layer (see
`run`). Per `M` it prints the durations (the baseline's cast and matmul separately), each route's largest
error against the IEEE-fp32 `router_gate_mixed_gemv`, and how many rows pick a
different top-4 expert set than that reference and than the baseline.
"""

from std.math import ceildiv, exp
from std.random import randn, random_float64, seed
from std.sys import argv, get_defined_int

from layout import Idx, TileTensor, row_major
from linalg.gemv import router_gate_mixed_gemv
import linalg.matmul.vendor.blas as vendor_blas
from linalg.matmul import matmul
from linalg.router_gate_gemm import router_gate_tf32_gemm
from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from router_gate_prefill_utils.validation import max_abs_diff, read_f32

comptime N = 128
comptime K = 6144
comptime ITERS = get_defined_int["ITERS", 30]()
comptime CTA_GROUP = get_defined_int["CTA_GROUP", 2]()
comptime TOPK = 4


def cast_bf16_to_f32(
    src: MutPointer[Scalar[DType.bfloat16], MutUntrackedOrigin],
    dst: MutPointer[Scalar[DType.float32], MutUntrackedOrigin],
    n: Int32,
):
    var i = (block_idx.x * block_dim.x + thread_idx.x) * 8
    if i < Int(n):
        dst.unsafe_store(i, src.unsafe_load[width=8](i).cast[DType.float32]())


def top4(
    logits: Span[Scalar[DType.float32], _],
    bias: Span[Scalar[DType.float32], _],
    use_bias: Bool,
    row: Int,
) -> Int:
    """Returns a bitmask-free key: the sorted top-4 expert ids packed in an Int.
    """
    var chosen = List[Int](length=TOPK, fill=-1)
    for t in range(TOPK):
        var best = -1
        var best_v = Float32(-1e30)
        for e in range(N):
            var taken = False
            for u in range(t):
                if chosen[u] == e:
                    taken = True
            if taken:
                continue
            var v = logits[row * N + e]
            if use_bias:
                # Routing score as in the MoE gate: sigmoid plus the expert
                # score-correction bias.
                v = 1.0 / (1.0 + exp(-v)) + bias[e]
            if v > best_v:
                best_v = v
                best = e
        chosen[t] = best
    var key = 0
    for i in range(TOPK):
        for j in range(i + 1, TOPK):
            if chosen[j] < chosen[i]:
                var tmp = chosen[i]
                chosen[i] = chosen[j]
                chosen[j] = tmp
    for t in range(TOPK):
        key = key * 131 + chosen[t]
    return key


def top4_rows_differ(
    a: Span[Scalar[DType.float32], _],
    b: Span[Scalar[DType.float32], _],
    bias: Span[Scalar[DType.float32], _],
    m: Int,
    use_bias: Bool,
) -> Int:
    var diff = 0
    for r in range(m):
        if top4(a, bias, use_bias, r) != top4(b, bias, use_bias, r):
            diff += 1
    return diff


def flag_value(name: String) -> String:
    for arg in argv():
        var a = String(arg)
        if a.startswith(name):
            return String(a.removeprefix(name))
    return String()


def run(ctx: DeviceContext, m: Int, layer: Int) raises:
    var a_bf16 = ctx.enqueue_create_buffer[DType.bfloat16](m * K)
    var a_f32 = ctx.enqueue_create_buffer[DType.float32](m * K)
    var w = ctx.enqueue_create_buffer[DType.float32](N * K)
    var c_base = ctx.enqueue_create_buffer[DType.float32](m * N)
    var c_ref = ctx.enqueue_create_buffer[DType.float32](m * N)
    var c_trunc = ctx.enqueue_create_buffer[DType.float32](m * N)
    var c_rn = ctx.enqueue_create_buffer[DType.float32](m * N)
    var c_vendor = ctx.enqueue_create_buffer[DType.float32](m * N)

    seed(7)
    # Gate weight, post-attention norm gamma and score-correction bias.
    # Synthetic by default; `--real-dir=DIR` reads `DIR/l<layer>_gate.f32`,
    # `_gamma.f32` and `_bias.f32` (raw little-endian fp32) instead. Activations
    # are unit-RMS normal rows scaled by gamma, the shape of a post-attention
    # norm output.
    var real_dir = flag_value("--real-dir=")
    var w_vals: List[Float32]
    var gamma: List[Float32]
    var bias_h: List[Float32]
    if real_dir:
        var prefix = String(real_dir, "/l", layer, "_")
        w_vals = read_f32(prefix + "gate.f32", N * K)
        gamma = read_f32(prefix + "gamma.f32", K)
        bias_h = read_f32(prefix + "bias.f32", N)
    else:
        w_vals = List[Float32](length=N * K, fill=0)
        for i in range(N * K):
            w_vals[i] = Float32(random_float64(-0.05, 0.05))
        gamma = List[Float32](length=K, fill=0)
        for i in range(K):
            gamma[i] = Float32(random_float64(-1.0, 0.3))
        bias_h = List[Float32](length=N, fill=0)
        for i in range(N):
            bias_h[i] = 12.0 + Float32(random_float64(0.0, 0.2))
    var a_host = ctx.enqueue_create_host_buffer[DType.bfloat16](m * K)
    var a_tmp = ctx.enqueue_create_host_buffer[DType.float32](m * K)
    randn(a_tmp.unsafe_ptr(), m * K)
    for i in range(m * K):
        a_host[i] = (a_tmp[i] * gamma[i % K]).cast[DType.bfloat16]()
    var w_host = ctx.enqueue_create_host_buffer[DType.float32](N * K)
    for i in range(N * K):
        w_host[i] = w_vals[i]
    ctx.enqueue_copy(a_bf16, a_host)
    ctx.enqueue_copy(w, w_host)
    ctx.synchronize()

    var a_bf16_tt = TileTensor(a_bf16, row_major(m, Idx[K])).as_imm()
    var a_f32_tt = TileTensor(a_f32, row_major(m, Idx[K])).as_imm()
    var w_tt = TileTensor(w, row_major(Idx[N], Idx[K])).as_imm()
    var c_base_tt = TileTensor(c_base, row_major(m, Idx[N]))
    var c_ref_tt = TileTensor(c_ref, row_major(m, Idx[N]))
    var c_trunc_tt = TileTensor(c_trunc, row_major(m, Idx[N]))
    var c_rn_tt = TileTensor(c_rn, row_major(m, Idx[N]))
    var c_vendor_tt = TileTensor(c_vendor, row_major(m, Idx[N]))

    def cast_only() raises {imm}:
        ctx.enqueue_function[cast_bf16_to_f32](
            a_bf16.unsafe_ptr(),
            a_f32.unsafe_ptr(),
            Int32(m * K),
            grid_dim=ceildiv(m * K, 8 * 256),
            block_dim=256,
        )

    def matmul_only() raises {imm}:
        matmul[transpose_b=True](c_base_tt, a_f32_tt, w_tt, ctx)

    def vendor_only() raises {imm}:
        vendor_blas.matmul(
            ctx,
            c_vendor_tt,
            a_f32_tt,
            w_tt,
            c_row_major=True,
            transpose_b=True,
        )

    def baseline() raises {imm}:
        cast_only()
        matmul_only()

    def cand_cta1() raises {imm}:
        router_gate_tf32_gemm[cta_group=1](c_trunc_tt, a_bf16_tt, w_tt, m, ctx)

    def cand_rn() raises {imm}:
        router_gate_tf32_gemm[cta_group=CTA_GROUP](
            c_rn_tt, a_bf16_tt, w_tt, m, ctx
        )

    baseline()
    cast_only()
    vendor_only()
    cand_cta1()
    cand_rn()
    router_gate_mixed_gemv[N](c_ref_tt, a_bf16_tt, w_tt, m, N, K, ctx)
    ctx.synchronize()
    var it = Float64(ITERS)
    var t_cast = Float64(ctx.execution_time(cast_only, ITERS)) / 1e3 / it
    var t_mm = Float64(ctx.execution_time(matmul_only, ITERS)) / 1e3 / it
    var t_vendor = Float64(ctx.execution_time(vendor_only, ITERS)) / 1e3 / it
    var t_base = Float64(ctx.execution_time(baseline, ITERS)) / 1e3 / it
    var t_trunc = Float64(ctx.execution_time(cand_cta1, ITERS)) / 1e3 / it
    var t_rn = Float64(ctx.execution_time(cand_rn, ITERS)) / 1e3 / it

    var base_h = ctx.enqueue_create_host_buffer[DType.float32](m * N)
    var ref_h = ctx.enqueue_create_host_buffer[DType.float32](m * N)
    var trunc_h = ctx.enqueue_create_host_buffer[DType.float32](m * N)
    var rn_h = ctx.enqueue_create_host_buffer[DType.float32](m * N)
    var vendor_h = ctx.enqueue_create_host_buffer[DType.float32](m * N)
    ctx.enqueue_copy(base_h, c_base)
    ctx.enqueue_copy(ref_h, c_ref)
    ctx.enqueue_copy(trunc_h, c_trunc)
    ctx.enqueue_copy(rn_h, c_rn)
    ctx.enqueue_copy(vendor_h, c_vendor)
    ctx.synchronize()
    var n = m * N
    var b = base_h.as_span()
    var r = ref_h.as_span()
    var t = trunc_h.as_span()
    var q = rn_h.as_span()
    var v = vendor_h.as_span()

    print(
        "layer=" + String(layer),
        "M=" + String(m),
        "cast_us=" + String(t_cast),
        "matmul_us=" + String(t_mm),
        "vendor_us=" + String(t_vendor),
        "baseline_us=" + String(t_base),
        "new_cta1_us=" + String(t_trunc),
        "new_rn_us=" + String(t_rn),
    )
    print(
        "  max_abs_vs_ieee: base=" + String(max_abs_diff(b, r, n)),
        "cta1=" + String(max_abs_diff(t, r, n)),
        "rn=" + String(max_abs_diff(q, r, n)),
        " max_abs_new_vs_base: cta1=" + String(max_abs_diff(t, b, n)),
        "rn=" + String(max_abs_diff(q, b, n)),
    )
    print(
        "  vendor(cuBLAS) vs_ieee: max_abs=" + String(max_abs_diff(v, r, n)),
        "top4=" + String(top4_rows_differ(v, r, Span(bias_h), m, False)),
        "top4_bias=" + String(top4_rows_differ(v, r, Span(bias_h), m, True)),
        " vs_matmul_baseline max_abs=" + String(max_abs_diff(v, b, n)),
        " | new_trunc vs vendor max_abs=" + String(max_abs_diff(t, v, n)),
        "top4_bias=" + String(top4_rows_differ(t, v, Span(bias_h), m, True)),
        " new_rn vs vendor max_abs=" + String(max_abs_diff(q, v, n)),
        "top4_bias=" + String(top4_rows_differ(q, v, Span(bias_h), m, True)),
    )
    print(
        "  top4_diff_rows(bias) vs_ieee: base="
        + String(top4_rows_differ(b, r, Span(bias_h), m, False))
        + "("
        + String(top4_rows_differ(b, r, Span(bias_h), m, True))
        + ") cta1="
        + String(top4_rows_differ(t, r, Span(bias_h), m, False))
        + "("
        + String(top4_rows_differ(t, r, Span(bias_h), m, True))
        + ") rn="
        + String(top4_rows_differ(q, r, Span(bias_h), m, False))
        + "("
        + String(top4_rows_differ(q, r, Span(bias_h), m, True))
        + ")  vs_base: cta1="
        + String(top4_rows_differ(t, b, Span(bias_h), m, False))
        + "("
        + String(top4_rows_differ(t, b, Span(bias_h), m, True))
        + ") rn="
        + String(top4_rows_differ(q, b, Span(bias_h), m, False))
        + "("
        + String(top4_rows_differ(q, b, Span(bias_h), m, True))
        + ")"
    )


def main() raises:
    # `--layer=N` picks the layer of a `--real-dir` run.
    var layer_flag = flag_value("--layer=")
    var layer = Int(layer_flag) if layer_flag else 0
    with DeviceContext() as ctx:
        for m in [16384, 15000, 9216, 8704, 8449, 8200, 8193, 8192, 4096]:
            run(ctx, m, layer)
