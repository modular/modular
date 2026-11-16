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

"""Times K3's MLA `w_uv` output projection with and without its output gate.

The latent MLA layer multiplies the attention output by `sigmoid(gate_pre)`
before `o_proj`, which is one elementwise launch per MLA layer. Folding that
multiply into the projection's own store (`_mla_gated_w_uv`) removes it; the
three arms below measure what that buys at K3's shapes:

  * `w_uv_projection` -- the unfused projection alone (one launch),
  * `w_uv_projection_plus_gate` -- the unfused chain (projection, then the
    gated pass), which is what the layer runs today,
  * `w_uv_projection_gated_store` -- the fused projection (one launch).

All three run in one process over the same buffers, so the difference between
arms is the launch (and the extra read/write of the attention output) the fold
removes, not a change of inputs.
"""

from std.sys import get_defined_int

from layout import Coord, Idx, TileTensor, row_major
from layout.layout import Layout
from linalg.bmm import _batched_matmul_gpu
from max.benchmark import bencher_iter_custom
from max.gpu.host import DeviceBuffer, DeviceContext
from nn.attention.gpu.mla_graph import _mla_apply_output_gate, _mla_gated_w_uv
from std.benchmark import Bench, BenchConfig, Bencher, BenchId

comptime dtype = DType.bfloat16
# K3 per rank over TP8: 12 of 96 query heads, latent 512.
comptime num_heads = 12
comptime kv_latent_dim = 512


def _value(seed: Int) -> Float32:
    """Deterministic pseudo-random value in [-1, 1)."""
    var x = UInt32(seed) * UInt32(1664525) + UInt32(1013904223)
    x = x ^ (x >> UInt32(13))
    x = x * UInt32(1274126177)
    x = x ^ (x >> UInt32(16))
    return Float32(Int(x & UInt32(0xFFFF))) * (2.0 / 65536.0) - 1.0


def _fill(ctx: DeviceContext[], buf: DeviceBuffer[dtype], count: Int) raises:
    """Fills a bfloat16 buffer with deterministic values."""
    var host = List[Scalar[dtype]](length=count, fill=Scalar[dtype](0))
    for i in range(count):
        host[i] = Scalar[dtype](_value(i + 1))
    ctx.enqueue_copy(buf, host)


def bench_gated_w_uv[
    v_head_dim: Int, gate_row_stride: Int
](num_rows: Int, ctx: DeviceContext[]) raises:
    """Times the three arms at `num_rows` query tokens.

    Parameters:
        v_head_dim: Output width per head.
        gate_row_stride: Row stride of the gate view, in elements.

    Args:
        num_rows: Query tokens in the step (K3: 32 at verify, 4 at decode).
        ctx: Device context.
    """
    var out_elems = num_heads * num_rows * v_head_dim
    var a_elems = num_heads * num_rows * kv_latent_dim
    var w_elems = num_heads * v_head_dim * kv_latent_dim
    var gate_elems = num_rows * gate_row_stride

    var a_buf = ctx.enqueue_create_buffer[dtype](a_elems)
    var w_buf = ctx.enqueue_create_buffer[dtype](w_elems)
    var c_buf = ctx.enqueue_create_buffer[dtype](out_elems)
    var gate_buf = ctx.enqueue_create_buffer[dtype](gate_elems)

    _fill(ctx, a_buf, a_elems)
    _fill(ctx, w_buf, w_elems)
    _fill(ctx, gate_buf, gate_elems)
    ctx.synchronize()

    var a_rows = TileTensor(
        a_buf, row_major((num_rows, Idx[num_heads], Idx[kv_latent_dim]))
    )
    var a_3d = a_rows.reshape(
        Layout(
            (Idx[num_heads], num_rows, Idx[kv_latent_dim]),
            (Idx[kv_latent_dim], Idx[num_heads * kv_latent_dim], Idx[1]),
        )
    )
    var w_3d = TileTensor(
        w_buf, row_major((Idx[num_heads], Idx[v_head_dim], Idx[kv_latent_dim]))
    )
    var c_rows = TileTensor(
        c_buf, row_major((num_rows, Idx[num_heads], Idx[v_head_dim]))
    )
    var c_3d = c_rows.reshape(
        Layout(
            (Idx[num_heads], num_rows, Idx[v_head_dim]),
            (Idx[v_head_dim], Idx[num_heads * v_head_dim], Idx[1]),
        )
    )
    var gate = TileTensor(
        gate_buf,
        Layout(
            Coord(num_rows, Idx[num_heads * v_head_dim]),
            Coord(gate_row_stride, Idx[1]),
        ),
    )
    var gate_ptr = gate.unsafe_ptr().as_imm()

    var m = Bench(
        BenchConfig(
            min_runtime_secs=0.0,
            max_runtime_secs=600.0,
            max_iters=10_000,
            max_batch_size=100,
        )
    )

    @inline(.always)
    def bench_projection(mut b: Bencher) raises {imm}:
        @inline(.always)
        def launch(ctx_: DeviceContext[]) raises {imm}:
            _batched_matmul_gpu[transpose_b=True](c_3d, a_3d, w_3d, ctx_)

        bencher_iter_custom(b, launch, ctx)

    m.bench_function(
        bench_projection,
        BenchId(String("w_uv_projection_v", v_head_dim, "_rows", num_rows)),
    )

    @inline(.always)
    def bench_chain(mut b: Bencher) raises {imm}:
        @inline(.always)
        def launch(ctx_: DeviceContext[]) raises {imm}:
            _batched_matmul_gpu[transpose_b=True](c_3d, a_3d, w_3d, ctx_)
            _mla_apply_output_gate[v_head_dim](
                c_rows, gate_ptr, gate_row_stride, ctx_
            )

        bencher_iter_custom(b, launch, ctx)

    m.bench_function(
        bench_chain,
        BenchId(
            String("w_uv_projection_plus_gate_v", v_head_dim, "_rows", num_rows)
        ),
    )

    @inline(.always)
    def bench_fused(mut b: Bencher) raises {imm}:
        @inline(.always)
        def launch(ctx_: DeviceContext[]) raises {imm}:
            _mla_gated_w_uv(c_3d, a_3d, w_3d, gate, ctx_)

        bencher_iter_custom(b, launch, ctx)

    m.bench_function(
        bench_fused,
        BenchId(
            String(
                "w_uv_projection_gated_store_v", v_head_dim, "_rows", num_rows
            )
        ),
    )

    m.dump_report()


def main() raises:
    with DeviceContext() as ctx:
        # K3: v_head_dim 128, and the gate is the 1536-wide tail slice of the
        # per-rank fused input projection (q_a 1536 + kv_a 576 + gate 1536).
        # The verify step is 4 requests x 8 verify positions.
        bench_gated_w_uv[128, 3648](get_defined_int["ROWS", 32](), ctx)
        # A plain decode step: 4 requests x 1 token.
        bench_gated_w_uv[128, 3648](4, ctx)
        # A square `w_uv`, four times the output per head.
        bench_gated_w_uv[512, 6144](32, ctx)
