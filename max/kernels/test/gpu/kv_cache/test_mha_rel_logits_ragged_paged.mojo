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
"""Runs flash attention through `RelativeLogitsMask` over a paged
ragged KV cache and checks it against `mha_gpu_naive`, which calls the same
mask functor -- a cross-implementation check of the mask wired into the
production kernel path, not just the mask struct in isolation.

The `token_sink` cases also give the mask per-token sink logits and check both
launches against a host reference that appends the sink as an extra softmax key
with a zero value.
"""

from std.math import align_up, ceildiv, exp, rsqrt
from std.sys import size_of
from std.random import seed

from max.gpu.host import DeviceContext
from max.gpu.host.info import B200
from layout import Idx, row_major
from layout._fillers import random
from layout._host_device_tile_tensor import HostDeviceTileTensor
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from kv_cache_test_utils import CacheLengthsTable, PagedLookupTable
from nn.attention.gpu.mha import flash_attention, mha_gpu_naive
from nn.attention.mha_mask import NoView
from nn.attention.operand_masks import RelativeLogitsMask
from extensibility import Loadable
from layout import Coord, TensorLayout, TileTensor
from std.testing import assert_almost_equal, assert_true


@fieldwise_init
struct TileView[dtype_: DType, LT: TensorLayout, origin: ImmOrigin](
    Loadable, TrivialRegisterPassable
):
    """Reads a device tile through `load`, as a mask reads a graph operand."""

    comptime dtype: DType = Self.dtype_
    comptime rank: Int = Self.LT.rank
    comptime __shape_types = Self.LT.__shape_types
    comptime _shape_types = Self.LT._shape_types

    var tile: TileTensor[Self.dtype_, Self.LT, Self.origin]

    def shape(self) -> Coord[*Self._shape_types]:
        return self.tile.layout.shape_coord()

    def load[
        width: Int, element_alignment: Int = 1
    ](self, idx: Coord) -> SIMD[Self.dtype, width]:
        return self.tile.load[
            width, alignment=element_alignment * size_of[Self.dtype]()
        ](idx)


def execute_rel_logits_flash_attention_test[
    window: Int,
    num_q_heads: Int,
    kv_params: KVCacheStaticParams,
    page_size: Int,
    extent: Int,
    token_sink: Bool = False,
](
    valid_lengths: List[Int],
    cache_lengths: List[Int],
    label: String,
    ctx: DeviceContext,
    num_partitions: Optional[Int] = None,
) raises:
    comptime dtype = DType.bfloat16
    comptime group = num_q_heads // kv_params.num_heads
    comptime head_size = kv_params.head_size
    comptime num_layers = 1
    comptime layer_idx = 0

    var batch_size = len(valid_lengths)
    var cache_table = CacheLengthsTable.build(valid_lengths, cache_lengths, ctx)
    var total_length = cache_table.total_length
    var max_full_context_length = cache_table.max_full_context_length
    var max_valid_length = max(1, cache_table.max_seq_length_batch)

    # The low-level kernels address the full final 64-row output tile.
    var padded_total_length = align_up(total_length, 64)

    var tensor_layout = row_major(
        padded_total_length, Idx[num_q_heads], Idx[head_size]
    )

    var scale = rsqrt(Float32(head_size))

    seed(0x9F3A + label.byte_length())

    var q = HostDeviceTileTensor[dtype](tensor_layout, ctx)
    random(q.host_tensor())
    q.to_device()
    var q_device = q.device_tensor().as_imm()

    var rel_logits = HostDeviceTileTensor[dtype](
        row_major(total_length, Idx[num_q_heads], Idx[extent]), ctx
    )
    random(rel_logits.host_tensor())
    rel_logits.to_device()
    var rel_logits_device = rel_logits.device_tensor().as_imm()

    var bias = TileView(rel_logits_device)
    var cache_lengths_view = TileView(
        cache_table.cache_lengths.device_tile_tensor().as_imm()
    )
    var row_offsets = TileView(
        cache_table.input_row_offsets.device_tile_tensor().as_imm()
    )
    var mask = RelativeLogitsMask[
        window,
        type_of(bias),
        type_of(cache_lengths_view),
        type_of(row_offsets),
    ](bias, cache_lengths_view, row_offsets, NoView())

    var num_paged_blocks = 0
    for i in range(batch_size):
        num_paged_blocks += ceildiv(
            cache_lengths[i] + valid_lengths[i], page_size
        )
    num_paged_blocks += 4

    var kv_blocks = HostDeviceTileTensor[dtype](
        row_major(
            Int64(num_paged_blocks),
            Idx[2],
            Int64(num_layers),
            Idx[page_size],
            Idx[kv_params.num_heads],
            Idx[head_size],
        ),
        ctx,
    )
    random(kv_blocks.host_tensor())
    kv_blocks.to_device()

    var paged_lut = PagedLookupTable[page_size].build(
        valid_lengths,
        cache_lengths,
        max_full_context_length,
        num_paged_blocks,
        ctx,
    )

    comptime Collection = PagedKVCacheCollection[
        dtype,
        kv_params,
        page_size,
        MutAnyOrigin,
        ImmutAnyOrigin,
        ImmutAnyOrigin,
        MutAnyOrigin,
    ]
    var kv_collection = Collection(
        kv_blocks.device_tensor().as_unsafe_any_origin(),
        cache_table.cache_lengths.device_tile_tensor(),
        paged_lut.device_tile_tensor(),
        UInt32(max_valid_length),
        UInt32(max_full_context_length),
    )
    var k_cache = kv_collection.get_key_cache(layer_idx)
    var v_cache = kv_collection.get_value_cache(layer_idx)

    var test_output = HostDeviceTileTensor[dtype](tensor_layout, ctx)
    flash_attention[ragged=True](
        test_output.device_tensor(),
        q_device,
        k_cache,
        v_cache,
        mask,
        cache_table.input_row_offsets.device_tile_tensor(),
        scale,
        ctx,
        num_partitions=num_partitions,
    )

    var reference_output = HostDeviceTileTensor[dtype](tensor_layout, ctx)
    mha_gpu_naive[ragged=True](
        q_device,
        k_cache,
        v_cache,
        mask,
        reference_output.device_tensor(),
        cache_table.input_row_offsets.device_tile_tensor(),
        scale,
        batch_size,
        max_valid_length,
        max_full_context_length,
        num_q_heads,
        head_size,
        group,
        ctx,
    )

    test_output.to_host()
    reference_output.to_host()
    var actual = test_output.host_tensor()
    var expected = reference_output.host_tensor()
    for t in range(total_length):
        for h in range(num_q_heads):
            for d in range(head_size):
                assert_almost_equal(
                    actual[t, h, d], expected[t, h, d], atol=0.06
                )

    comptime if token_sink:
        # Sink logits span the log-sum-exp of the scores, which for the scale
        # and the uniform inputs here is ~2.8 + ln(visible keys), so the sink
        # visibly moves long rows and leaves short ones nearly alone.
        var sink = HostDeviceTileTensor[dtype](
            row_major(total_length, Idx[num_q_heads]), ctx
        )
        random(sink.host_tensor(), min=0, max=12)
        sink.to_device()
        var sink_view = TileView(sink.device_tensor().as_imm())
        var sink_mask = RelativeLogitsMask[
            window,
            type_of(bias),
            type_of(cache_lengths_view),
            type_of(row_offsets),
            type_of(sink_view),
        ](bias, cache_lengths_view, row_offsets, sink_view)

        var sink_output = HostDeviceTileTensor[dtype](tensor_layout, ctx)
        flash_attention[ragged=True](
            sink_output.device_tensor(),
            q_device,
            k_cache,
            v_cache,
            sink_mask,
            cache_table.input_row_offsets.device_tile_tensor(),
            scale,
            ctx,
            num_partitions=num_partitions,
        )
        sink_output.to_host()
        var with_sink = sink_output.host_tensor()

        var q_host = q.host_tensor()
        var kv_host = kv_blocks.host_tensor()
        var bias_host = rel_logits.host_tensor()
        var sink_host = sink.host_tensor()
        var lut_cols = paged_lut.paged_lut.shape[1]
        var max_sink_effect = Float32(0)
        var checked = 0
        var row_start = 0
        for b in range(batch_size):
            var rows = valid_lengths[b]
            for i in range(rows):
                # Every row of short sequences; a stride, the first rows and
                # the last rows of long ones.
                if not (rows <= 64 or i % 7 == 0 or i < 2 or i >= rows - 4):
                    continue
                var t = row_start + i
                var q_pos = cache_lengths[b] + i
                var first_key = max(0, q_pos - window + 1) if window > 0 else 0
                for h in range(num_q_heads):
                    var kv_head = h // group
                    var scores = List[Float32](capacity=q_pos + 1 - first_key)
                    var row_max = sink_host[t, h].cast[DType.float32]()
                    var key_max = Float32(-1.0e30)
                    for j in range(first_key, q_pos + 1):
                        var block = Int(
                            paged_lut.paged_lut.host_ptr[
                                b * lut_cols + j // page_size
                            ]
                        )
                        var dot = Float32(0)
                        for d in range(head_size):
                            dot += (
                                q_host[t, h, d].cast[DType.float32]()
                                * kv_host[
                                    block, 0, 0, j % page_size, kv_head, d
                                ].cast[DType.float32]()
                            )
                        var score = dot * scale
                        if q_pos - j < extent:
                            score += bias_host[t, h, q_pos - j].cast[
                                DType.float32
                            ]()
                        scores.append(score)
                        key_max = max(key_max, score)
                    row_max = max(row_max, key_max)
                    var sink_mass = exp(
                        sink_host[t, h].cast[DType.float32]() - row_max
                    )
                    var denom = sink_mass
                    var denom_no_sink = Float32(0)
                    for p in range(len(scores)):
                        var e = exp(scores[p] - row_max)
                        scores[p] = e
                        denom += e
                        denom_no_sink += e
                    for d in range(head_size):
                        var acc = Float32(0)
                        for j in range(first_key, q_pos + 1):
                            var block = Int(
                                paged_lut.paged_lut.host_ptr[
                                    b * lut_cols + j // page_size
                                ]
                            )
                            acc += (
                                scores[j - first_key]
                                * kv_host[
                                    block, 1, 0, j % page_size, kv_head, d
                                ].cast[DType.float32]()
                            )
                        var want = acc / denom
                        var want_no_sink = acc / denom_no_sink
                        assert_almost_equal(
                            with_sink[t, h, d].cast[DType.float32](),
                            want,
                            atol=0.02,
                        )
                        max_sink_effect = max(
                            max_sink_effect, abs(want - want_no_sink)
                        )
                checked += 1
            row_start += rows
        # The check must be able to fail: the sink has to matter somewhere.
        assert_true(max_sink_effect > 0.1)
        print(t"  sink effect {max_sink_effect}, {checked} rows checked")

    print(t"PASSED [{label}]")


def main() raises:
    with DeviceContext() as ctx:
        # Prefill, global, GQA kv_heads=8, ragged batch of 2 unequal
        # sequences, extent < the longest sequence.
        execute_rel_logits_flash_attention_test[
            window=0,
            num_q_heads=32,
            kv_params=KVCacheStaticParams(num_heads=8, head_size=64),
            page_size=128,
            extent=8,
        ]([5, 9], [0, 0], "prefill+global+kv8", ctx)

        # Long cached prefixes exercise a nonzero start tile.
        execute_rel_logits_flash_attention_test[
            window=6,
            num_q_heads=32,
            kv_params=KVCacheStaticParams(num_heads=16, head_size=64),
            page_size=128,
            extent=4,
        ]([1, 1], [258, 513], "decode+window+kv16+long_prefix", ctx)

        execute_rel_logits_flash_attention_test[
            window=0,
            num_q_heads=32,
            kv_params=KVCacheStaticParams(num_heads=8, head_size=64),
            page_size=128,
            extent=0,
        ]([1], [0], "prefill+empty_extent", ctx)

        # Decode rows and a prompt in one batch, at Inkling's TP2 shapes. The
        # odd and even query positions cover both pair alignments of the
        # row-wise bias read.
        execute_rel_logits_flash_attention_test[
            window=0,
            num_q_heads=16,
            kv_params=KVCacheStaticParams(num_heads=4, head_size=128),
            page_size=128,
            extent=1024,
        ]([1, 1, 1, 300], [700, 1501, 33, 0], "mixed+global+extent1024", ctx)

        execute_rel_logits_flash_attention_test[
            window=512,
            num_q_heads=16,
            kv_params=KVCacheStaticParams(num_heads=4, head_size=128),
            page_size=128,
            extent=1024,
        ]([1, 1, 1, 300], [700, 1501, 33, 0], "mixed+window+extent1024", ctx)

        execute_rel_logits_flash_attention_test[
            window=0,
            num_q_heads=16,
            kv_params=KVCacheStaticParams(num_heads=4, head_size=128),
            page_size=128,
            extent=1024,
        ]([257], [901], "prefill+global+long_prefix", ctx)

        # TODO: Investigate MI355
        comptime if ctx.default_device_info == B200:
            execute_rel_logits_flash_attention_test[
                window=512,
                num_q_heads=32,
                kv_params=KVCacheStaticParams(num_heads=8, head_size=128),
                page_size=128,
                extent=512,
            ](
                [1, 1],
                [2048, 3072],
                "decode+window+splitk_p8",
                ctx,
                num_partitions=8,
            )

            execute_rel_logits_flash_attention_test[
                window=0,
                num_q_heads=32,
                kv_params=KVCacheStaticParams(num_heads=8, head_size=128),
                page_size=128,
                extent=1024,
            ](
                [1, 1],
                [2048, 4096],
                "decode+global+splitk_p4",
                ctx,
                num_partitions=4,
            )

            # GQA group 6 (Inkling 2) with per-token sink logits: a prompt
            # past the window and `extent`, decode rows beside a prompt, and
            # split-K decode.
            execute_rel_logits_flash_attention_test[
                window=1024,
                num_q_heads=12,
                kv_params=KVCacheStaticParams(num_heads=2, head_size=128),
                page_size=128,
                extent=1024,
                token_sink=True,
            ]([1150, 37], [0, 0], "group6+prefill+window+sink", ctx)

            execute_rel_logits_flash_attention_test[
                window=1024,
                num_q_heads=12,
                kv_params=KVCacheStaticParams(num_heads=2, head_size=128),
                page_size=128,
                extent=1024,
                token_sink=True,
            ](
                [1, 1, 1, 50],
                [1400, 1101, 33, 0],
                "group6+mixed+window+sink",
                ctx,
            )

            execute_rel_logits_flash_attention_test[
                window=0,
                num_q_heads=12,
                kv_params=KVCacheStaticParams(num_heads=2, head_size=128),
                page_size=128,
                extent=1024,
                token_sink=True,
            ](
                [1, 1],
                [2048, 1500],
                "group6+decode+global+sink+splitk_p4",
                ctx,
                num_partitions=4,
            )
