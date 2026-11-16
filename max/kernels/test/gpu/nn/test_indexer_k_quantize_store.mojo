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
"""The fused indexer key tail reads its key out of the merged projection.

The op is handed the whole projection -- the latent's columns first, the key's
128 after them -- and a column offset for the key, so it reads its channels
where the GEMM wrote them and nothing copies them out first. A caller with a
packed key passes offset 0. Both shapes must store the same bytes, and the
leading columns of the projection hold a value no key carries, so an op that
read its key at the wrong offset or at a packed row stride stores different
bytes rather than the same ones by luck.
"""

from std.math import ceildiv
from std.memory import bitcast
from std.testing import assert_equal
from kv_cache.types import KVCacheStaticParams
from layout import Idx, TileTensor, row_major
from max.gpu.host import DeviceContext
from nn.indexer_qk_quantize import indexer_k_quantize_store
from nn.kv_cache import generic_get_paged_cache_with_scales

comptime _HEAD_DIM = 128
"""Channels of one index key, which is also its quantization group."""

comptime _ROPE_DIM = 64
"""Leading columns the interleaved rope rotates on a rope layer."""

comptime _ROPE_CACHE_HEAD_DIM = 576
"""Columns before the key on a rope layer: the latent plus the rope slice."""

comptime _NOPE_CACHE_HEAD_DIM = 512
"""Columns before the key on a NoPE layer: the latent alone."""

comptime _PAGE_SIZE = 128
"""Tokens a cache page holds."""

comptime _PREFIX_VALUE = 3.0
"""What the projection's leading columns hold.

No key row carries this value, so an op reading its key at the wrong offset, or
at the packed row stride, lands on it.
"""

comptime _EMPTY = UInt8(0xAB)
"""The byte the cache starts as, so an unwritten page still compares."""


def _key_value(row: Int, col: Int) -> BFloat16:
    """The deterministic key byte pattern every arm reads."""
    return (Float32((row * 17 + col * 13) % 97 - 48) / 16).cast[.bfloat16]()


def _run[rows: Int, offset: Int](ctx: DeviceContext) raises -> List[UInt8]:
    """Runs the op once and returns its key cache plus its scales, as bytes.

    `offset` is the column the key starts at: 0 for a packed key, or the merged
    projection's latent width, in which case the operand is the whole
    `[rows, offset + 128]` tensor. A page pool per run keeps runs independent.
    """
    comptime key_width = offset + _HEAD_DIM
    comptime num_pages = ceildiv(rows, _PAGE_SIZE)
    comptime page_elements = _PAGE_SIZE * _HEAD_DIM

    var keys = ctx.enqueue_create_host_buffer[.bfloat16](rows * key_width)
    var gamma = ctx.enqueue_create_host_buffer[.bfloat16](_HEAD_DIM)
    var beta = ctx.enqueue_create_host_buffer[.bfloat16](_HEAD_DIM)
    var freqs = ctx.enqueue_create_host_buffer[.bfloat16](rows * _ROPE_DIM)
    var offsets = ctx.enqueue_create_host_buffer[.uint32](2)
    var cache_lengths = ctx.enqueue_create_host_buffer[.uint32](1)
    var start_pos = ctx.enqueue_create_host_buffer[.uint32](1)
    var max_prompt = ctx.enqueue_create_host_buffer[.uint32](1)
    var max_cache = ctx.enqueue_create_host_buffer[.uint32](1)
    var lut = ctx.enqueue_create_host_buffer[.uint32](num_pages)
    var page_stride = ctx.enqueue_create_host_buffer[.int64](1)
    var scales_page_stride = ctx.enqueue_create_host_buffer[.int64](1)

    for row in range(rows):
        for col in range(key_width):
            keys[row * key_width + col] = BFloat16(_PREFIX_VALUE)
        for col in range(_HEAD_DIM):
            keys[row * key_width + offset + col] = _key_value(row, col)
    for col in range(_HEAD_DIM):
        gamma[col] = (Float32(1) + Float32((col * 5) % 11 - 5) / 64).cast[
            .bfloat16
        ]()
        beta[col] = (Float32((col * 7) % 13 - 6) / 32).cast[.bfloat16]()
    for pos in range(rows):
        for col in range(_ROPE_DIM):
            var phase = (pos + col // 2) % 4
            var quarter = 1.0 if phase == 0 else (-1.0 if phase == 2 else 0.0)
            freqs[pos * _ROPE_DIM + col] = quarter.cast[.bfloat16]()
    offsets[0] = 0
    offsets[1] = UInt32(rows)
    cache_lengths[0] = 0
    start_pos[0] = 0
    max_prompt[0] = UInt32(rows)
    max_cache[0] = UInt32(rows)
    for page in range(num_pages):
        lut[page] = UInt32(page)
    # -1 is the packed convention: each page is the product of the leaf dims.
    page_stride[0] = -1
    scales_page_stride[0] = -1

    var d_gamma = ctx.enqueue_create_buffer[.bfloat16](_HEAD_DIM)
    var d_beta = ctx.enqueue_create_buffer[.bfloat16](_HEAD_DIM)
    var d_freqs = ctx.enqueue_create_buffer[.bfloat16](rows * _ROPE_DIM)
    var d_offsets = ctx.enqueue_create_buffer[.uint32](2)
    var d_cache_lengths = ctx.enqueue_create_buffer[.uint32](1)
    var d_start = ctx.enqueue_create_buffer[.uint32](1)
    var d_lut = ctx.enqueue_create_buffer[.uint32](num_pages)
    var d_keys = ctx.enqueue_create_buffer[.bfloat16](rows * key_width)
    var d_blocks = ctx.enqueue_create_buffer[.uint8](num_pages * page_elements)
    var d_scales = ctx.enqueue_create_buffer[.float32](num_pages * _PAGE_SIZE)
    var host_blocks = ctx.enqueue_create_host_buffer[.uint8](
        num_pages * page_elements
    )
    var host_scales = ctx.enqueue_create_host_buffer[.float32](
        num_pages * _PAGE_SIZE
    )

    ctx.enqueue_copy(dst_buf=d_gamma, src_buf=gamma)
    ctx.enqueue_copy(dst_buf=d_beta, src_buf=beta)
    ctx.enqueue_copy(dst_buf=d_freqs, src_buf=freqs)
    ctx.enqueue_copy(dst_buf=d_offsets, src_buf=offsets)
    ctx.enqueue_copy(dst_buf=d_cache_lengths, src_buf=cache_lengths)
    ctx.enqueue_copy(dst_buf=d_start, src_buf=start_pos)
    ctx.enqueue_copy(dst_buf=d_lut, src_buf=lut)
    ctx.enqueue_copy(dst_buf=d_keys, src_buf=keys)
    d_blocks.enqueue_fill(_EMPTY)
    for page in range(num_pages * _PAGE_SIZE):
        host_scales[page] = -1.0
    ctx.enqueue_copy(dst_buf=d_scales, src_buf=host_scales)

    var fp8_blocks = rebind[MutPointer[Scalar[.float8_e4m3fn], MutAnyOrigin]](
        d_blocks.unsafe_ptr()
    )
    var collection = generic_get_paged_cache_with_scales[
        .float8_e4m3fn,
        .float32,
        KVCacheStaticParams(num_heads=1, head_size=_HEAD_DIM, is_mla=True),
        _PAGE_SIZE,
        _HEAD_DIM,
    ](
        TileTensor(
            fp8_blocks,
            row_major(
                Idx[num_pages],
                Idx[1],
                Idx[1],
                Idx[_PAGE_SIZE],
                Idx[1],
                Idx[_HEAD_DIM],
            ),
        ),
        # The collection resolves these four on the host when it is built, so
        # they stay host-visible; the kernel reads the lengths and the LUT from
        # the device.
        TileTensor(page_stride, row_major(Idx[1])).as_imm(),
        TileTensor(d_cache_lengths, row_major(Idx[1])).as_imm(),
        TileTensor(d_lut, row_major(Idx[1], Idx[num_pages])).as_imm(),
        TileTensor(max_prompt, row_major(Idx[1])).as_imm(),
        TileTensor(max_cache, row_major(Idx[1])).as_imm(),
        TileTensor(
            d_scales,
            row_major(
                Idx[num_pages],
                Idx[1],
                Idx[1],
                Idx[_PAGE_SIZE],
                Idx[1],
                Idx[1],
            ),
        ),
        TileTensor(scales_page_stride, row_major(Idx[1])).as_imm(),
        TileTensor(d_lut, row_major(Idx[1], Idx[num_pages])).as_imm(),
    )
    var key_cache = collection.get_key_cache(Int(0))

    indexer_k_quantize_store[
        type_of(key_cache), .bfloat16, .bfloat16, _ROPE_DIM, offset
    ](
        key_cache,
        TileTensor(d_keys, row_major(Idx[rows], Idx[key_width])).as_imm(),
        TileTensor(d_gamma, row_major(Idx[_HEAD_DIM])).as_imm(),
        TileTensor(d_beta, row_major(Idx[_HEAD_DIM])).as_imm(),
        Float32(1e-6),
        TileTensor(d_freqs, row_major(Idx[rows], Idx[_ROPE_DIM])).as_imm(),
        TileTensor(d_offsets, row_major(Idx[2])).as_imm(),
        TileTensor(d_start, row_major(Idx[1])).as_imm(),
        ctx,
    )

    ctx.enqueue_copy(dst_buf=host_blocks, src_buf=d_blocks)
    ctx.enqueue_copy(dst_buf=host_scales, src_buf=d_scales)
    ctx.synchronize()

    var result = List[UInt8](
        capacity=num_pages * page_elements + 4 * num_pages * _PAGE_SIZE
    )
    for i in range(num_pages * page_elements):
        result.append(host_blocks[i])
    for i in range(num_pages * _PAGE_SIZE):
        for byte in range(4):
            result.append(
                UInt8(
                    (bitcast[.uint32](host_scales[i]) >> UInt32(byte * 8))
                    & UInt32(0xFF)
                )
            )

    # The buffers outlive the collection's untracked views of them.
    _ = d_keys^
    _ = d_blocks^
    _ = d_scales^
    return result^


def _case[rows: Int, offset: Int](ctx: DeviceContext) raises:
    """Reading the key at `offset` stores what a packed key stores, at `rows`.
    """
    var packed = _run[rows, 0](ctx)
    var projection = _run[rows, offset](ctx)

    # The run stored something: an op that wrote nothing would agree with
    # itself and with an op that read garbage.
    var written = 0
    for i in range(len(packed)):
        if packed[i] != _EMPTY:
            written += 1
    assert_equal(written > 0, True)

    var moved = 0
    for i in range(len(packed)):
        if projection[i] != packed[i]:
            moved += 1
    assert_equal(moved, 0)


def main() raises:
    with DeviceContext() as ctx:
        _case[1, _ROPE_CACHE_HEAD_DIM](ctx)
        _case[32, _ROPE_CACHE_HEAD_DIM](ctx)
        _case[8192, _ROPE_CACHE_HEAD_DIM](ctx)
        # A NoPE layer's projection is narrower: it has no rope slice in front
        # of the key, so its offset is the latent's width alone.
        _case[1, _NOPE_CACHE_HEAD_DIM](ctx)
        _case[8192, _NOPE_CACHE_HEAD_DIM](ctx)
    print("PASS the fused key op reads its columns out of the projection")
