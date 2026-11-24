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

"""Checks the MLA layer's gated `w_uv` projection (`_mla_gated_w_uv`).

K3's latent MLA layer multiplies the attention output by `sigmoid(gate_pre)`
before its `o_proj`. `_mla_gated_w_uv` folds that multiply into the `w_uv`
projection's own store, so the layer runs one kernel fewer per MLA layer.
The fold changes which kernel writes the output, so this test builds both
chains at K3's verify and decode shapes, at a square `w_uv`, and at an odd
shape whose gate row stride is not a vector multiple:

  unfused: `_batched_matmul_gpu` into the output view, then one
           `_mla_apply_output_gate` pass over it
  fused:   `_batched_matmul_gpu` with the gate as its compute function

and requires

  * the two to agree bit for bit (the fold's contract -- a wrong axis, a
    dropped row or a stale row stride shows up here), and
  * the fused result to equal the unfused projection times the host's
    `sigmoid(gate)` within a bfloat16 rounding (a structural check that the
    gate value at `(token, head, v)` is the one applied there).

The output buffer is NaN-poisoned before the fused chain, so a tile its
epilogue failed to store fails the test instead of reading the previous
chain's result.
"""

from std.math import exp, isnan, nan

from layout import Coord, Idx, TileTensor, row_major
from layout.layout import Layout
from linalg.bmm import _batched_matmul_gpu
from max.gpu.host import DeviceBuffer, DeviceContext
from nn.attention.gpu.mla_graph import _mla_apply_output_gate, _mla_gated_w_uv
from std.testing import assert_almost_equal, assert_equal, assert_true

comptime dtype = DType.bfloat16


def _value(seed: Int) -> Float32:
    """Deterministic pseudo-random value in [-1, 1)."""
    var x = UInt32(seed) * UInt32(1664525) + UInt32(1013904223)
    x = x ^ (x >> UInt32(13))
    x = x * UInt32(1274126177)
    x = x ^ (x >> UInt32(16))
    return Float32(Int(x & UInt32(0xFFFF))) * (2.0 / 65536.0) - 1.0


def _fill(
    ctx: DeviceContext, buf: DeviceBuffer[dtype], count: Int, seed: Int
) raises:
    """Fills `count` bfloat16 elements with deterministic values."""
    var host = List[Scalar[dtype]](length=count, fill=Scalar[dtype](0))
    for i in range(count):
        host[i] = Scalar[dtype](_value(seed + i))
    ctx.enqueue_copy(buf, host)


def _read_back(
    ctx: DeviceContext, buf: DeviceBuffer[dtype], count: Int
) raises -> List[Scalar[dtype]]:
    """Reads a bfloat16 buffer back to the host."""
    var host = List[Scalar[dtype]](length=count, fill=Scalar[dtype](0))
    ctx.enqueue_copy(host, buf)
    ctx.synchronize()
    return host^


def test_gated_w_uv[
    num_heads: Int,
    num_rows: Int,
    kv_latent_dim: Int,
    v_head_dim: Int,
    gate_row_stride: Int,
](ctx: DeviceContext) raises:
    """Runs both chains at one shape and compares them.

    Parameters:
        num_heads: Query heads per rank (the batched matmul's batch).
        num_rows: Query tokens (decode: batch x verify width).
        kv_latent_dim: Inner dimension of the `w_uv` projection.
        v_head_dim: Output width per head.
        gate_row_stride: Row stride of the gate view, in elements, which may
            be wider than the gate.
    """
    comptime out_elems = num_heads * num_rows * v_head_dim
    comptime a_elems = num_heads * num_rows * kv_latent_dim
    comptime w_elems = num_heads * v_head_dim * kv_latent_dim
    comptime gate_cols = num_heads * v_head_dim
    comptime gate_elems = num_rows * gate_row_stride
    comptime assert gate_row_stride >= gate_cols

    var a_buf = ctx.enqueue_create_buffer[dtype](a_elems)
    var w_buf = ctx.enqueue_create_buffer[dtype](w_elems)
    var c_buf = ctx.enqueue_create_buffer[dtype](out_elems)
    var gate_buf = ctx.enqueue_create_buffer[dtype](gate_elems)

    _fill(ctx, a_buf, a_elems, 1)
    _fill(ctx, w_buf, w_elems, 7)
    _fill(ctx, gate_buf, gate_elems, 13)
    var gate_host = _read_back(ctx, gate_buf, gate_elems)

    # The graph's views: `q`-shaped buffers are `[rows, heads, dim]` in
    # memory and the projection sees them batched over heads. The gate is a
    # row slice of the fused input projection, so its row stride is the
    # parent's width.
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
            Coord(num_rows, Idx[gate_cols]),
            Coord(gate_row_stride, Idx[1]),
        ),
    )

    # Chain 1: the unfused projection, then one gated pass over its output.
    _batched_matmul_gpu[transpose_b=True](c_3d, a_3d, w_3d, ctx)
    ctx.synchronize()
    var c_unfused = _read_back(ctx, c_buf, out_elems)
    _mla_apply_output_gate[v_head_dim](
        c_rows, gate.unsafe_ptr().as_imm(), gate_row_stride, ctx
    )
    ctx.synchronize()
    var c_unfused_gated = _read_back(ctx, c_buf, out_elems)

    # Chain 2: the folded projection. Its epilogue owns every store, so the
    # poison must be gone afterwards.
    c_buf.enqueue_fill(nan[dtype]())
    _mla_gated_w_uv(c_3d, a_3d, w_3d, gate, ctx)
    ctx.synchronize()
    var c_fused = _read_back(ctx, c_buf, out_elems)

    for i in range(out_elems):
        assert_true(
            not isnan(c_fused[i]),
            msg="the gated store left an output element unwritten",
        )

    # The fold's contract: bit-identical to the separate pass.
    for i in range(out_elems):
        assert_equal(c_fused[i], c_unfused_gated[i])

    # The gate is applied at the right `(token, head, v)`: the fused value is
    # the unfused projection times the host's sigmoid of that gate element.
    for row in range(num_rows):
        for head in range(num_heads):
            for v in range(v_head_dim):
                comptime row_stride = num_heads * v_head_dim
                var idx = row * row_stride + head * v_head_dim + v
                var gate_val = Float32(
                    gate_host[row * gate_row_stride + head * v_head_dim + v]
                )
                var expected = Float32(c_unfused[idx]) / (1.0 + exp(-gate_val))
                assert_almost_equal(
                    Float32(c_fused[idx]),
                    expected,
                    atol=0.02,
                    rtol=0.02,
                    msg="the fused store applied a different gate value",
                )


def main() raises:
    with DeviceContext() as ctx:
        # K3's verify shape: 4 requests x 8 verify positions, 12 heads per
        # rank (96 heads over TP8), latent 512, v_head_dim 128. The gate is
        # the 1536-wide tail slice of the per-rank fused input projection
        # (q_a 1536 + kv_a 576 + gate 1536), so its row stride is 3648.
        test_gated_w_uv[12, 32, 512, 128, 3648](ctx)
        # K3's decode shape: 4 requests x 1 token.
        test_gated_w_uv[12, 4, 512, 128, 3648](ctx)
        # The gate as a dense tensor of its own, row stride equal to its
        # width.
        test_gated_w_uv[12, 32, 512, 128, 1536](ctx)
        # A square `w_uv`, where a projection that dropped `transpose_b`
        # would still be shape-valid, and a wide gate stride.
        test_gated_w_uv[12, 32, 512, 512, 23040](ctx)
        # Odd shape: an unaligned head count and row count, a narrow
        # projection, and a gate stride that is not a vector multiple.
        test_gated_w_uv[3, 5, 128, 64, 200](ctx)
