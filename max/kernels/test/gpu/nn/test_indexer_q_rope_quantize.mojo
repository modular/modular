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
"""The indexer query op without SSMax against the SSMax op at identity SSMax.

`indexer_q_rope_quantize` is `indexer_q_ssmax_rope_quantize` without the SSMax
step, writing each head's scale as float32. A zero SSMax scale and a unit bias
make the SSMax factor exactly one, so both ops must store the same FP8 bytes,
and the float32 scales must be the E8M0 scales widened. Two requests sit at
different cache positions, so each rotates its rows at its own positions, and a
row past the last offset belongs to no request: neither op may write it.

Both ops run one kernel body, so a rope case and the NoPE case also compare the
op with the chain it replaces, `rope_ragged` and then
`quantize_dynamic_scaled_fp8` with an E8M0 scale per head.
"""

from std.memory import bitcast
from std.testing import assert_equal
from layout import Coord, Idx, TileTensor, row_major
from linalg.fp8_quantization import quantize_dynamic_scaled_fp8
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from nn.indexer_qk_quantize import (
    indexer_q_rope_quantize,
    indexer_q_ssmax_rope_quantize,
)
from nn.rope import rope_ragged

comptime _HEAD_DIM = 128
comptime _HEADS = 32
comptime _TABLE_ROPE = 64
"""Width of the rope table; a NoPE launch passes it unread."""
comptime _POSITIONS = 512
comptime _EMPTY = UInt8(0xA5)
comptime _EMPTY_SCALE = Float32(-1.0)


def _q_value(row: Int, head: Int, col: Int) -> BFloat16:
    """Heads of very different magnitudes, signs mixed."""
    var magnitude = Float32(1 << ((row + head) % 9)) / 64
    var level = Float32((row * 31 + head * 7 + col * 13) % 101 - 50) / 50
    return (level * magnitude).cast[.bfloat16]()


def _check_reference[
    live: Int, rope_dim: Int
](
    ctx: DeviceContext,
    d_q: DeviceBuffer[.bfloat16],
    d_freqs: DeviceBuffer[.bfloat16],
    d_offsets: DeviceBuffer[.uint32],
    d_start: DeviceBuffer[.uint32],
    q_plain: HostBuffer[.uint8],
    scale_plain: HostBuffer[.float32],
) raises:
    """Checks the op's live rows against the chain it replaces.

    `rope_ragged` rotates the leading `rope_dim` columns of each head, then
    `quantize_dynamic_scaled_fp8` quantizes each head with an E8M0 scale. Neither
    runs the kernel body the two indexer ops share.
    """
    comptime row_width = _HEADS * _HEAD_DIM
    # The quantize's input: the rotated query, or the query itself.
    var d_x = ctx.enqueue_create_buffer[.bfloat16](len(d_q))
    var d_q_ref = ctx.enqueue_create_buffer[.uint8](live * row_width)
    var d_scale_ref = ctx.enqueue_create_buffer[.uint8](_HEADS * live)
    var rows_shape = row_major(Idx[live], Idx[row_width])
    comptime if rope_dim > 0:
        comptime assert (
            rope_dim == _TABLE_ROPE
        ), "rope_ragged rotates as many columns as its table is wide"
        var head_shape = row_major(Idx[live], Idx[_HEADS], Idx[_HEAD_DIM])
        var roped = TileTensor(d_x, head_shape)

        @inline(.always)
        def output_fn[
            width: SIMDLength, alignment: Int
        ](idx: Coord, val: SIMD[.bfloat16, width]) {var roped} -> None:
            roped.store(idx, val)

        rope_ragged[.bfloat16, .bfloat16, interleaved=True, rope_first=True](
            x=TileTensor(d_q, head_shape).as_unsafe_any_origin(),
            input_row_offsets=TileTensor(
                d_offsets, row_major(Idx[3])
            ).as_unsafe_any_origin(),
            start_pos=TileTensor(
                d_start, row_major(Idx[2])
            ).as_unsafe_any_origin(),
            freqs_cis=TileTensor(
                d_freqs, row_major(Idx[_POSITIONS], Idx[rope_dim])
            ).as_unsafe_any_origin(),
            context=ctx,
            output_fn=output_fn,
        )
    else:
        ctx.enqueue_copy(dst_buf=d_x, src_buf=d_q)
    var x = TileTensor(d_x, rows_shape)

    @inline(.always)
    def input_fn[
        width: Int, alignment: Int
    ](row: Int, col: Int) {var x} -> SIMD[.bfloat16, width]:
        return x.load[width=width, alignment=alignment]((row, col))

    quantize_dynamic_scaled_fp8[
        in_dtype=.bfloat16,
        group_size_or_per_token=_HEAD_DIM,
        num_cols=row_width,
    ](
        input_fn,
        TileTensor(
            rebind[MutPointer[Scalar[.float8_e4m3fn], MutAnyOrigin]](
                d_q_ref.unsafe_ptr()
            ),
            rows_shape,
        ),
        TileTensor(
            rebind[MutPointer[Scalar[.float8_e8m0fnu], MutAnyOrigin]](
                d_scale_ref.unsafe_ptr()
            ),
            row_major(Idx[_HEADS], Idx[live]),
        ),
        1200.0,
        ctx,
        live,
    )
    var q_ref = ctx.enqueue_create_host_buffer[.uint8](live * row_width)
    var scale_ref = ctx.enqueue_create_host_buffer[.uint8](_HEADS * live)
    ctx.enqueue_copy(dst_buf=q_ref, src_buf=d_q_ref)
    ctx.enqueue_copy(dst_buf=scale_ref, src_buf=d_scale_ref)
    ctx.synchronize()
    _ = d_x^

    for i in range(live * row_width):
        assert_equal(q_plain[i], q_ref[i])
    # The chain writes its scales as `[heads, rows]`.
    for row in range(live):
        for head in range(_HEADS):
            var widened = bitcast[.float8_e8m0fnu](
                scale_ref[head * live + row]
            ).cast[.float32]()
            assert_equal(
                bitcast[.uint32](scale_plain[row * _HEADS + head]),
                bitcast[.uint32](widened),
            )


def _case[rows: Int, rope_dim: Int, reference: Bool](ctx: DeviceContext) raises:
    """Both ops at `rows` rows, the last one padding past the offsets.

    With `reference`, the op is also checked against the chain it replaces.
    """
    comptime live = rows - 1
    comptime first = live // 2
    comptime q_size = rows * _HEADS * _HEAD_DIM
    comptime scale_size = rows * _HEADS

    var q = ctx.enqueue_create_host_buffer[.bfloat16](q_size)
    var positions = ctx.enqueue_create_host_buffer[.uint32](rows)
    var zeros = ctx.enqueue_create_host_buffer[.bfloat16](_HEADS)
    var ones = ctx.enqueue_create_host_buffer[.bfloat16](_HEADS)
    var freqs = ctx.enqueue_create_host_buffer[.bfloat16](
        _POSITIONS * _TABLE_ROPE
    )
    var offsets = ctx.enqueue_create_host_buffer[.uint32](3)
    var start_pos = ctx.enqueue_create_host_buffer[.uint32](2)
    for row in range(rows):
        positions[row] = UInt32(5 * row + 3)
        for head in range(_HEADS):
            for col in range(_HEAD_DIM):
                q[(row * _HEADS + head) * _HEAD_DIM + col] = _q_value(
                    row, head, col
                )
    for head in range(_HEADS):
        zeros[head] = BFloat16(0)
        ones[head] = BFloat16(1)
    for pos in range(_POSITIONS):
        for col in range(_TABLE_ROPE):
            var angle = Float32(pos * (col // 2 + 1)) / 37
            var phase = angle - Float32(Int(angle))
            freqs[pos * _TABLE_ROPE + col] = (phase * 2 - 1).cast[.bfloat16]()
    offsets[0] = 0
    offsets[1] = UInt32(first)
    offsets[2] = UInt32(live)
    start_pos[0] = 300
    start_pos[1] = 17

    var d_q = ctx.enqueue_create_buffer[.bfloat16](q_size)
    var d_positions = ctx.enqueue_create_buffer[.uint32](rows)
    var d_zeros = ctx.enqueue_create_buffer[.bfloat16](_HEADS)
    var d_ones = ctx.enqueue_create_buffer[.bfloat16](_HEADS)
    var d_freqs = ctx.enqueue_create_buffer[.bfloat16](_POSITIONS * _TABLE_ROPE)
    var d_offsets = ctx.enqueue_create_buffer[.uint32](3)
    var d_start = ctx.enqueue_create_buffer[.uint32](2)
    var d_q_ssmax = ctx.enqueue_create_buffer[.uint8](q_size)
    var d_q_plain = ctx.enqueue_create_buffer[.uint8](q_size)
    var d_scale_ssmax = ctx.enqueue_create_buffer[.uint8](scale_size)
    var d_scale_plain = ctx.enqueue_create_buffer[.float32](scale_size)
    var empty_scales = ctx.enqueue_create_host_buffer[.float32](scale_size)
    for i in range(scale_size):
        empty_scales[i] = _EMPTY_SCALE
    ctx.enqueue_copy(dst_buf=d_q, src_buf=q)
    ctx.enqueue_copy(dst_buf=d_positions, src_buf=positions)
    ctx.enqueue_copy(dst_buf=d_zeros, src_buf=zeros)
    ctx.enqueue_copy(dst_buf=d_ones, src_buf=ones)
    ctx.enqueue_copy(dst_buf=d_freqs, src_buf=freqs)
    ctx.enqueue_copy(dst_buf=d_offsets, src_buf=offsets)
    ctx.enqueue_copy(dst_buf=d_start, src_buf=start_pos)
    d_q_ssmax.enqueue_fill(_EMPTY)
    d_q_plain.enqueue_fill(_EMPTY)
    d_scale_ssmax.enqueue_fill(_EMPTY)
    ctx.enqueue_copy(dst_buf=d_scale_plain, src_buf=empty_scales)

    var q_shape = row_major(Idx[rows], Idx[_HEADS], Idx[_HEAD_DIM])
    var scale_shape = row_major(Idx[rows], Idx[_HEADS])
    var q_in = TileTensor(d_q, q_shape).as_imm()
    var freqs_in = TileTensor(
        d_freqs, row_major(Idx[_POSITIONS], Idx[_TABLE_ROPE])
    ).as_imm()
    var offsets_in = TileTensor(d_offsets, row_major(Idx[3])).as_imm()
    var start_in = TileTensor(d_start, row_major(Idx[2])).as_imm()

    indexer_q_ssmax_rope_quantize[rope_dim](
        TileTensor(
            rebind[MutPointer[Scalar[.float8_e4m3fn], MutAnyOrigin]](
                d_q_ssmax.unsafe_ptr()
            ),
            q_shape,
        ),
        TileTensor(
            rebind[MutPointer[Scalar[.float8_e8m0fnu], MutAnyOrigin]](
                d_scale_ssmax.unsafe_ptr()
            ),
            scale_shape,
        ),
        q_in,
        TileTensor(d_positions, row_major(Idx[rows])).as_imm(),
        TileTensor(d_zeros, row_major(Idx[_HEADS])).as_imm(),
        TileTensor(d_ones, row_major(Idx[_HEADS])).as_imm(),
        freqs_in,
        offsets_in,
        start_in,
        ctx,
    )
    indexer_q_rope_quantize[rope_dim](
        TileTensor(
            rebind[MutPointer[Scalar[.float8_e4m3fn], MutAnyOrigin]](
                d_q_plain.unsafe_ptr()
            ),
            q_shape,
        ),
        TileTensor(d_scale_plain, scale_shape),
        q_in,
        freqs_in,
        offsets_in,
        start_in,
        ctx,
    )

    var q_ssmax = ctx.enqueue_create_host_buffer[.uint8](q_size)
    var q_plain = ctx.enqueue_create_host_buffer[.uint8](q_size)
    var scale_ssmax = ctx.enqueue_create_host_buffer[.uint8](scale_size)
    var scale_plain = ctx.enqueue_create_host_buffer[.float32](scale_size)
    ctx.enqueue_copy(dst_buf=q_ssmax, src_buf=d_q_ssmax)
    ctx.enqueue_copy(dst_buf=q_plain, src_buf=d_q_plain)
    ctx.enqueue_copy(dst_buf=scale_ssmax, src_buf=d_scale_ssmax)
    ctx.enqueue_copy(dst_buf=scale_plain, src_buf=d_scale_plain)
    ctx.synchronize()

    var written = 0
    for i in range(q_size):
        assert_equal(q_plain[i], q_ssmax[i])
        if i >= live * _HEADS * _HEAD_DIM:
            assert_equal(q_plain[i], _EMPTY)
        elif q_plain[i] != _EMPTY:
            written += 1
    assert_equal(written > 0, True)
    for i in range(scale_size):
        if i >= live * _HEADS:
            assert_equal(scale_ssmax[i], _EMPTY)
            assert_equal(scale_plain[i], _EMPTY_SCALE)
            continue
        var widened = bitcast[.float8_e8m0fnu](scale_ssmax[i]).cast[.float32]()
        assert_equal(
            bitcast[.uint32](scale_plain[i]), bitcast[.uint32](widened)
        )
    comptime if reference:
        _check_reference[live, rope_dim](
            ctx, d_q, d_freqs, d_offsets, d_start, q_plain, scale_plain
        )
    print(t"[PASS] rows={rows} rope_dim={rope_dim} reference={reference}")


def main() raises:
    with DeviceContext() as ctx:
        _case[3, 64, reference=False](ctx)
        _case[9, 64, reference=False](ctx)
        _case[131, 64, reference=True](ctx)
        _case[9, 0, reference=True](ctx)
