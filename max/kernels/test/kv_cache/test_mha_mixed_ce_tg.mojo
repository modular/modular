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

from std.collections import Set
from std.math import ceildiv, rsqrt
from std.random import random_ui64

from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import Coord, Idx, TileTensor, row_major
from layout._fillers import random
from std.memory import unsafe_memcpy
from nn.attention.cpu.mha import flash_attention_kv_cache
from nn.attention.mha_mask import CausalMask
from std.testing import assert_almost_equal, assert_not_equal


def execute_ragged_flash_attention() raises:
    comptime num_q_heads = 32
    comptime kv_params = KVCacheStaticParams(num_heads=8, head_size=128)
    comptime type = DType.float32
    comptime num_paged_blocks = 32
    comptime page_size = 512
    comptime PagedCollectionType = PagedKVCacheCollection[
        type,
        kv_params,
        page_size,
        MutAnyOrigin,
        ImmutAnyOrigin,
        ImmutAnyOrigin,
        MutAnyOrigin,
    ]
    var num_layers = 1
    var layer_idx = 0

    var true_ce_prompt_lens = [100, 200, 300, 400]
    var mixed_ce_prompt_lens = [50, 100, 150, 100]

    var true_ce_cache_lens = [0, 0, 0, 0]
    var mixed_ce_cache_lens = [50, 100, 150, 300]

    var batch_size = len(true_ce_prompt_lens)

    var true_ce_row_offsets_buf = List(length=batch_size + 1, fill=UInt32(0))
    var true_ce_row_offsets = TileTensor(
        Span(true_ce_row_offsets_buf), row_major(len(true_ce_row_offsets_buf))
    ).reshape(Coord(Int64(batch_size + 1)))
    var true_ce_cache_lengths_buf = List(length=batch_size, fill=UInt32(0))
    var true_ce_cache_lengths = TileTensor(
        Span(true_ce_cache_lengths_buf),
        row_major(len(true_ce_cache_lengths_buf)),
    ).reshape(Coord(Int64(batch_size)))
    var mixed_ce_row_offsets_buf = List(length=batch_size + 1, fill=UInt32(0))
    var mixed_ce_row_offsets = TileTensor(
        Span(mixed_ce_row_offsets_buf), row_major(len(mixed_ce_row_offsets_buf))
    ).reshape(Coord(Int64(batch_size + 1)))
    var mixed_ce_cache_lengths_buf = List(length=batch_size, fill=UInt32(0))
    var mixed_ce_cache_lengths = TileTensor(
        Span(mixed_ce_cache_lengths_buf),
        row_major(len(mixed_ce_cache_lengths_buf)),
    ).reshape(Coord(Int64(batch_size)))

    var true_ce_total_length = 0
    var mixed_ce_total_length = 0
    var true_ce_max_full_context_length = 0
    var mixed_ce_max_full_context_length = 0
    var true_ce_max_prompt_length = 0
    var mixed_ce_max_prompt_length = 0
    for i in range(batch_size):
        true_ce_row_offsets[i] = UInt32(true_ce_total_length)
        mixed_ce_row_offsets[i] = UInt32(mixed_ce_total_length)
        true_ce_cache_lengths[i] = UInt32(true_ce_cache_lens[i])
        mixed_ce_cache_lengths[i] = UInt32(mixed_ce_cache_lens[i])

        true_ce_max_full_context_length = max(
            true_ce_max_full_context_length,
            true_ce_cache_lens[i] + true_ce_prompt_lens[i],
        )
        mixed_ce_max_full_context_length = max(
            mixed_ce_max_full_context_length,
            mixed_ce_cache_lens[i] + mixed_ce_prompt_lens[i],
        )

        true_ce_max_prompt_length = max(
            true_ce_max_prompt_length, true_ce_prompt_lens[i]
        )
        mixed_ce_max_prompt_length = max(
            mixed_ce_max_prompt_length, mixed_ce_prompt_lens[i]
        )

        true_ce_total_length += true_ce_prompt_lens[i]
        mixed_ce_total_length += mixed_ce_prompt_lens[i]

    true_ce_row_offsets[batch_size] = UInt32(true_ce_total_length)
    mixed_ce_row_offsets[batch_size] = UInt32(mixed_ce_total_length)
    # CPU cache attention specializes its work count on the query head axis.
    var true_ce_q_ragged_buf = List(
        length=true_ce_total_length * num_q_heads * kv_params.head_size,
        fill=Scalar[type](0),
    )
    var true_ce_q_ragged = TileTensor(
        Span(true_ce_q_ragged_buf), row_major(len(true_ce_q_ragged_buf))
    ).reshape(
        Coord(
            Int64(true_ce_total_length),
            Idx[num_q_heads],
            Idx[kv_params.head_size],
        )
    )
    random(true_ce_q_ragged)

    var mixed_ce_q_ragged_buf = List(
        length=mixed_ce_total_length * num_q_heads * kv_params.head_size,
        fill=Scalar[type](0),
    )
    var mixed_ce_q_ragged = TileTensor(
        Span(mixed_ce_q_ragged_buf), row_major(len(mixed_ce_q_ragged_buf))
    ).reshape(
        Coord(
            Int64(mixed_ce_total_length),
            Idx[num_q_heads],
            Idx[kv_params.head_size],
        )
    )
    for bs_idx in range(batch_size):
        var mixed_ce_prompt_len = mixed_ce_prompt_lens[bs_idx]

        var true_ce_row_offset = true_ce_row_offsets[bs_idx]
        var mixed_ce_row_offset = mixed_ce_row_offsets[bs_idx]

        var mixed_ce_cache_len = mixed_ce_cache_lens[bs_idx]

        var true_ce_offset = true_ce_q_ragged.ptr + true_ce_q_ragged.layout(
            Coord(Int(true_ce_row_offset + UInt32(mixed_ce_cache_len)), 0, 0)
        )
        var mixed_ce_offset = mixed_ce_q_ragged.ptr + mixed_ce_q_ragged.layout(
            Coord(Int(mixed_ce_row_offset), 0, 0)
        )

        unsafe_memcpy(
            dest=mixed_ce_offset,
            src=true_ce_offset,
            count=mixed_ce_prompt_len * num_q_heads * kv_params.head_size,
        )

    # initialize reference output
    var mixed_ce_output_buf = List(
        length=mixed_ce_total_length * num_q_heads * kv_params.head_size,
        fill=Scalar[type](9999),
    )
    var mixed_ce_output = TileTensor(
        Span(mixed_ce_output_buf), row_major(len(mixed_ce_output_buf))
    ).reshape(
        Coord(
            Int64(mixed_ce_total_length),
            Idx[num_q_heads],
            Idx[kv_params.head_size],
        )
    )
    var true_ce_output_buf = List(
        length=true_ce_total_length * num_q_heads * kv_params.head_size,
        fill=Scalar[type](-9999),
    )
    var true_ce_output = TileTensor(
        Span(true_ce_output_buf), row_major(len(true_ce_output_buf))
    ).reshape(
        Coord(
            Int64(true_ce_total_length),
            Idx[num_q_heads],
            Idx[kv_params.head_size],
        )
    )

    # initialize our KVCache
    var kv_block_paged_buf = List(
        length=num_paged_blocks
        * 2
        * num_layers
        * page_size
        * kv_params.num_heads
        * kv_params.head_size,
        fill=Scalar[type](0),
    )
    comptime BlocksLayout = PagedCollectionType.blocks_tt_layout
    var native_blocks_shape = Coord[*BlocksLayout.shape_types]()
    native_blocks_shape[0] = Int64(num_paged_blocks)
    native_blocks_shape[2] = Int64(num_layers)
    var native_blocks_strides = Coord[*BlocksLayout.stride_types]()
    native_blocks_strides[1] = native_blocks_shape[2] * Int64(
        native_blocks_strides[2].value()
    )
    native_blocks_strides[0] = (
        Int64(native_blocks_shape[1].value()) * native_blocks_strides[1]
    )
    var kv_block_paged = TileTensor(
        Span(kv_block_paged_buf), row_major(len(kv_block_paged_buf))
    ).reshape(BlocksLayout(native_blocks_shape, native_blocks_strides))
    random(kv_block_paged)

    var paged_lut_buf = List(
        length=batch_size * ceildiv(true_ce_max_full_context_length, page_size),
        fill=UInt32(0),
    )
    var paged_lut = TileTensor(
        Span(paged_lut_buf), row_major(len(paged_lut_buf))
    ).reshape(
        Coord(
            Int64(batch_size),
            Int64(ceildiv(true_ce_max_full_context_length, page_size)),
        )
    )
    var paged_lut_set = Set[Int]()
    for bs in range(batch_size):
        var seq_len = true_ce_cache_lens[bs] + true_ce_prompt_lens[bs]

        for block_idx in range(0, ceildiv(seq_len, page_size)):
            var randval = Int(random_ui64(0, num_paged_blocks - 1))
            while randval in paged_lut_set:
                randval = Int(random_ui64(0, num_paged_blocks - 1))

            paged_lut_set.add(randval)
            paged_lut[bs, block_idx] = UInt32(randval)

    var true_ce_kv_collection = PagedCollectionType(
        kv_block_paged.as_unsafe_any_origin(),
        true_ce_cache_lengths.as_imm().as_unsafe_any_origin(),
        paged_lut.as_imm().as_unsafe_any_origin(),
        UInt32(true_ce_max_prompt_length),
        UInt32(true_ce_max_full_context_length),
    )

    var mixed_ce_kv_collection = PagedCollectionType(
        kv_block_paged.as_unsafe_any_origin(),
        mixed_ce_cache_lengths.as_imm().as_unsafe_any_origin(),
        paged_lut.as_imm().as_unsafe_any_origin(),
        UInt32(mixed_ce_max_prompt_length),
        UInt32(mixed_ce_max_full_context_length),
    )

    # "true CE" execution
    print("true")
    flash_attention_kv_cache(
        true_ce_q_ragged.as_imm(),
        true_ce_row_offsets.as_imm(),
        true_ce_row_offsets.as_imm(),
        true_ce_kv_collection.get_key_cache(layer_idx),
        true_ce_kv_collection.get_value_cache(layer_idx),
        CausalMask(),
        rsqrt(Float32(kv_params.head_size)),
        true_ce_output,
    )

    # "mixed CE" execution
    print("mixed")
    flash_attention_kv_cache(
        mixed_ce_q_ragged.as_imm(),
        mixed_ce_row_offsets.as_imm(),
        mixed_ce_row_offsets.as_imm(),
        mixed_ce_kv_collection.get_key_cache(layer_idx),
        mixed_ce_kv_collection.get_value_cache(layer_idx),
        CausalMask(),
        rsqrt(Float32(kv_params.head_size)),
        mixed_ce_output,
    )

    var true_ce_out = true_ce_output
    var mixed_ce_out = mixed_ce_output
    for bs in range(batch_size):
        var mixed_ce_prompt_len = mixed_ce_prompt_lens[bs]
        var mixed_ce_row_offset = mixed_ce_row_offsets[bs]
        var true_ce_row_offset = true_ce_row_offsets[bs]
        var mixed_ce_cache_len = mixed_ce_cache_lens[bs]

        var true_ce_ragged_offset = Int(
            true_ce_row_offset + UInt32(mixed_ce_cache_len)
        )
        var mixed_ce_ragged_offset = Int(mixed_ce_row_offset)
        for s in range(mixed_ce_prompt_len):
            for h in range(num_q_heads):
                for hd in range(kv_params.head_size):
                    try:
                        assert_not_equal(
                            true_ce_out[true_ce_ragged_offset + s, h, hd][0],
                            -9999,
                        )
                        assert_not_equal(
                            mixed_ce_out[mixed_ce_ragged_offset + s, h, hd][0],
                            9999,
                        )
                        assert_almost_equal(
                            true_ce_out[true_ce_ragged_offset + s, h, hd][0],
                            mixed_ce_out[mixed_ce_ragged_offset + s, h, hd][0],
                            atol=1e-3,
                        )
                    except e:
                        print(
                            "MISMATCH:",
                            bs,
                            s,
                            h,
                            hd,
                            true_ce_out[true_ce_ragged_offset + s, h, hd][0],
                            mixed_ce_out[mixed_ce_ragged_offset + s, h, hd][0],
                        )
                        raise e^


def main() raises:
    execute_ragged_flash_attention()
