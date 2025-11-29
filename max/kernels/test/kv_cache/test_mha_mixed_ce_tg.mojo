# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from collections import Set
from math import ceildiv, rsqrt
from random import random_ui64

from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from layout._fillers import random
from memory import alloc, memcpy
from nn.flash_attention import flash_attention_kv_cache
from nn.mha_mask import CausalMask
from testing import assert_almost_equal

from utils import IndexList


def execute_ragged_flash_attention():
    comptime num_q_heads = 32
    comptime kv_params = KVCacheStaticParams(num_heads=8, head_size=128)
    comptime type = DType.float32
    comptime num_paged_blocks = 32
    comptime page_size = 512
    comptime PagedCollectionType = PagedKVCacheCollection[
        type, kv_params, page_size
    ]
    var num_layers = 1
    var layer_idx = 0

    var true_ce_prompt_lens = [100, 200, 300, 400]
    var mixed_ce_prompt_lens = [50, 100, 150, 100]

    var true_ce_cache_lens = [0, 0, 0, 0]
    var mixed_ce_cache_lens = [50, 100, 150, 300]

    var batch_size = len(true_ce_prompt_lens)

    comptime layout_1d = Layout.row_major[1]()
    var true_ce_row_offsets = LayoutTensor[DType.uint32, layout_1d](
        alloc[Scalar[DType.uint32]](batch_size + 1),
        RuntimeLayout[layout_1d].row_major(IndexList[1](batch_size + 1)),
    ).fill(0)
    var true_ce_cache_lengths = LayoutTensor[DType.uint32, layout_1d](
        alloc[Scalar[DType.uint32]](batch_size),
        RuntimeLayout[layout_1d].row_major(IndexList[1](batch_size)),
    ).fill(0)
    var mixed_ce_row_offsets = LayoutTensor[DType.uint32, layout_1d](
        alloc[Scalar[DType.uint32]](batch_size + 1),
        RuntimeLayout[layout_1d].row_major(IndexList[1](batch_size + 1)),
    ).fill(0)
    var mixed_ce_cache_lengths = LayoutTensor[DType.uint32, layout_1d](
        alloc[Scalar[DType.uint32]](batch_size),
        RuntimeLayout[layout_1d].row_major(IndexList[1](batch_size)),
    ).fill(0)

    var true_ce_total_length = 0
    var mixed_ce_total_length = 0
    var true_ce_max_full_context_length = 0
    var mixed_ce_max_full_context_length = 0
    var true_ce_max_prompt_length = 0
    var mixed_ce_max_prompt_length = 0
    for i in range(batch_size):
        true_ce_row_offsets[i] = true_ce_total_length
        mixed_ce_row_offsets[i] = mixed_ce_total_length
        true_ce_cache_lengths[i] = true_ce_cache_lens[i]
        mixed_ce_cache_lengths[i] = mixed_ce_cache_lens[i]

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

    true_ce_row_offsets[batch_size] = true_ce_total_length
    mixed_ce_row_offsets[batch_size] = mixed_ce_total_length
    comptime layout_3d = Layout.row_major[3]()
    var true_ce_q_ragged = LayoutTensor[type, layout_3d](
        alloc[Scalar[type]](
            true_ce_total_length * num_q_heads * Int(kv_params.head_size)
        ),
        RuntimeLayout[layout_3d].row_major(
            IndexList[3](
                true_ce_total_length, num_q_heads, Int(kv_params.head_size)
            )
        ),
    )
    random(true_ce_q_ragged)

    var mixed_ce_q_ragged = LayoutTensor[type, layout_3d](
        alloc[Scalar[type]](
            mixed_ce_total_length * num_q_heads * Int(kv_params.head_size)
        ),
        RuntimeLayout[layout_3d].row_major(
            IndexList[3](
                mixed_ce_total_length, num_q_heads, Int(kv_params.head_size)
            )
        ),
    ).fill(0)
    for bs_idx in range(batch_size):
        true_ce_prompt_len = true_ce_prompt_lens[bs_idx]
        mixed_ce_prompt_len = mixed_ce_prompt_lens[bs_idx]

        true_ce_row_offset = true_ce_row_offsets[bs_idx]
        mixed_ce_row_offset = mixed_ce_row_offsets[bs_idx]

        mixed_ce_cache_len = mixed_ce_cache_lens[bs_idx]

        true_ce_offset = true_ce_q_ragged.ptr + true_ce_q_ragged._offset(
            IndexList[3](Int(true_ce_row_offset + mixed_ce_cache_len), 0, 0)
        )
        mixed_ce_offset = mixed_ce_q_ragged.ptr + mixed_ce_q_ragged._offset(
            IndexList[3](Int(mixed_ce_row_offset), 0, 0)
        )

        memcpy(
            dest=mixed_ce_offset,
            src=true_ce_offset,
            count=mixed_ce_prompt_len * num_q_heads * Int(kv_params.head_size),
        )

    # initialize reference output
    var mixed_ce_output = LayoutTensor[type, layout_3d](
        alloc[Scalar[type]](
            mixed_ce_total_length * num_q_heads * Int(kv_params.head_size)
        ),
        RuntimeLayout[layout_3d].row_major(
            IndexList[3](
                mixed_ce_total_length, num_q_heads, Int(kv_params.head_size)
            )
        ),
    ).fill(0)
    var true_ce_output = LayoutTensor[type, layout_3d](
        alloc[Scalar[type]](
            true_ce_total_length * num_q_heads * Int(kv_params.head_size)
        ),
        RuntimeLayout[layout_3d].row_major(
            IndexList[3](
                true_ce_total_length, num_q_heads, Int(kv_params.head_size)
            )
        ),
    ).fill(0)

    # initialize our KVCache
    comptime layout_6d = Layout.row_major[6]()
    var kv_block_paged = LayoutTensor[type, layout_6d](
        alloc[Scalar[type]](
            num_paged_blocks
            * 2
            * num_layers
            * page_size
            * Int(kv_params.num_heads)
            * Int(kv_params.head_size)
        ),
        RuntimeLayout[layout_6d].row_major(
            IndexList[6](
                num_paged_blocks,
                2,
                num_layers,
                page_size,
                Int(kv_params.num_heads),
                Int(kv_params.head_size),
            )
        ),
    ).fill(0)
    random(kv_block_paged)

    comptime layout_2d = Layout.row_major[2]()
    var paged_lut = LayoutTensor[DType.uint32, layout_2d](
        alloc[Scalar[DType.uint32]](
            batch_size * ceildiv(true_ce_max_full_context_length, page_size)
        ),
        RuntimeLayout[layout_2d].row_major(
            IndexList[2](
                batch_size,
                ceildiv(true_ce_max_full_context_length, page_size),
            )
        ),
    ).fill(0)
    paged_lut_set = Set[Int]()
    for bs in range(batch_size):
        seq_len = true_ce_cache_lens[bs] + true_ce_prompt_lens[bs]

        for block_idx in range(0, ceildiv(seq_len, page_size)):
            var randval = Int(random_ui64(0, num_paged_blocks - 1))
            while randval in paged_lut_set:
                randval = Int(random_ui64(0, num_paged_blocks - 1))

            paged_lut_set.add(randval)
            paged_lut[bs, block_idx] = randval

    true_ce_kv_collection = PagedCollectionType(
        LayoutTensor[
            kv_block_paged.dtype,
            Layout.row_major[6](),
            MutAnyOrigin,
        ](
            kv_block_paged.ptr,
            RuntimeLayout[Layout.row_major[6]()](
                kv_block_paged.runtime_layout.shape.value,
                kv_block_paged.runtime_layout.stride.value,
            ),
        ),
        LayoutTensor[
            true_ce_cache_lengths.dtype, Layout(UNKNOWN_VALUE), ImmutAnyOrigin
        ](
            true_ce_cache_lengths.ptr,
            RuntimeLayout[Layout(UNKNOWN_VALUE)](
                true_ce_cache_lengths.runtime_layout.shape.value,
                true_ce_cache_lengths.runtime_layout.stride.value,
            ),
        ),
        LayoutTensor[paged_lut.dtype, Layout.row_major[2](), ImmutAnyOrigin](
            paged_lut.ptr,
            RuntimeLayout[Layout.row_major[2]()](
                paged_lut.runtime_layout.shape.value,
                paged_lut.runtime_layout.stride.value,
            ),
        ),
        true_ce_max_prompt_length,
        true_ce_max_full_context_length,
    )

    mixed_ce_kv_collection = PagedCollectionType(
        LayoutTensor[
            kv_block_paged.dtype,
            Layout.row_major[6](),
            MutAnyOrigin,
        ](
            kv_block_paged.ptr,
            RuntimeLayout[Layout.row_major[6]()](
                kv_block_paged.runtime_layout.shape.value,
                kv_block_paged.runtime_layout.stride.value,
            ),
        ),
        LayoutTensor[
            mixed_ce_cache_lengths.dtype, Layout(UNKNOWN_VALUE), ImmutAnyOrigin
        ](
            mixed_ce_cache_lengths.ptr,
            RuntimeLayout[Layout(UNKNOWN_VALUE)](
                mixed_ce_cache_lengths.runtime_layout.shape.value,
                mixed_ce_cache_lengths.runtime_layout.stride.value,
            ),
        ),
        LayoutTensor[paged_lut.dtype, Layout.row_major[2](), ImmutAnyOrigin](
            paged_lut.ptr,
            RuntimeLayout[Layout.row_major[2]()](
                paged_lut.runtime_layout.shape.value,
                paged_lut.runtime_layout.stride.value,
            ),
        ),
        mixed_ce_max_prompt_length,
        mixed_ce_max_full_context_length,
    )

    # "true CE" execution
    print("true")
    flash_attention_kv_cache(
        true_ce_q_ragged,
        true_ce_row_offsets,
        true_ce_row_offsets,
        true_ce_kv_collection.get_key_cache(layer_idx),
        true_ce_kv_collection.get_value_cache(layer_idx),
        CausalMask(),
        rsqrt(Float32(kv_params.head_size)),
        true_ce_output,
    )

    # "mixed CE" execution
    print("mixed")
    flash_attention_kv_cache(
        mixed_ce_q_ragged,
        mixed_ce_row_offsets,
        mixed_ce_row_offsets,
        mixed_ce_kv_collection.get_key_cache(layer_idx),
        mixed_ce_kv_collection.get_value_cache(layer_idx),
        CausalMask(),
        rsqrt(Float32(kv_params.head_size)),
        mixed_ce_output,
    )

    true_ce_out = true_ce_output
    mixed_ce_out = mixed_ce_output
    for bs in range(batch_size):
        mixed_ce_prompt_len = mixed_ce_prompt_lens[bs]
        mixed_ce_row_offset = mixed_ce_row_offsets[bs]
        true_ce_row_offset = true_ce_row_offsets[bs]
        mixed_ce_cache_len = mixed_ce_cache_lens[bs]

        true_ce_ragged_offset = Int(true_ce_row_offset + mixed_ce_cache_len)
        mixed_ce_ragged_offset = Int(mixed_ce_row_offset)
        for s in range(mixed_ce_prompt_len):
            for h in range(num_q_heads):
                for hd in range(kv_params.head_size):
                    try:
                        assert_almost_equal(
                            true_ce_out[true_ce_ragged_offset + s, h, Int(hd)][
                                0
                            ],
                            mixed_ce_out[
                                mixed_ce_ragged_offset + s, h, Int(hd)
                            ][0],
                            atol=1e-3,
                        )
                    except e:
                        print(
                            "MISMATCH:",
                            bs,
                            s,
                            h,
                            hd,
                            true_ce_out[true_ce_ragged_offset + s, h, Int(hd)][
                                0
                            ],
                            mixed_ce_out[
                                mixed_ce_ragged_offset + s, h, Int(hd)
                            ][0],
                        )
                        raise e

    true_ce_q_ragged.ptr.free()
    mixed_ce_q_ragged.ptr.free()
    true_ce_row_offsets.ptr.free()
    mixed_ce_row_offsets.ptr.free()
    kv_block_paged.ptr.free()
    paged_lut.ptr.free()
    true_ce_output.ptr.free()
    mixed_ce_output.ptr.free()
    true_ce_cache_lengths.ptr.free()
    mixed_ce_cache_lengths.ptr.free()


def main():
    execute_ragged_flash_attention()
