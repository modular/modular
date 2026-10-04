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

from std.math import ceildiv

from max.gpu.host import DeviceContext
from kv_cache_test_utils import random_distinct
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import (
    Coord,
    Idx,
    ImmTileTensor,
    RowMajorLayout,
    TileTensor,
    row_major,
)
from layout._host_device_tile_tensor import HostDeviceTileTensor
from nn.kv_cache_ragged import kv_cache_2m_iadd_dispatch

from std.utils import IndexList

comptime _Collection[
    dtype: DType, num_heads: Int, head_dim: Int, page_size: Int
] = PagedKVCacheCollection[
    dtype,
    KVCacheStaticParams(num_heads=num_heads, head_size=head_dim),
    page_size,
    MutAnyOrigin,
    ImmutAnyOrigin,
    ImmutAnyOrigin,
    MutAnyOrigin,
]


def _create_kv_collection[
    dtype: DType, //, num_heads: Int, head_dim: Int, page_size: Int
](
    kv_block_paged: TileTensor[mut=True, dtype, _, _, linear_idx_type=_],
    cache_lengths: ImmTileTensor[
        .uint32, RowMajorLayout[Int64], ImmutAnyOrigin
    ],
    paged_lut: ImmTileTensor[
        .uint32, RowMajorLayout[Int64, Int64], ImmutAnyOrigin
    ],
    max_prompt_length: Int,
    max_full_context_length: Int,
) -> _Collection[dtype, num_heads, head_dim, page_size]:
    comptime Collection = _Collection[dtype, num_heads, head_dim, page_size]
    return Collection(
        kv_block_paged.as_unsafe_any_origin(),
        cache_lengths,
        paged_lut,
        UInt32(max_prompt_length),
        UInt32(max_full_context_length),
    )


def _verify_kv_cache[
    dtype: DType,
    num_heads: Int,
    head_dim: Int,
    page_size: Int,
    batch_size: Int,
](
    kv_collection_host: _Collection[dtype, num_heads, head_dim, page_size],
    prompt_lens: IndexList[batch_size],
    cache_lens: IndexList[batch_size],
    num_active_loras: Int,
    total_slice_length: Int,
    layer_idx: Int,
) raises:
    var k_cache_host = kv_collection_host.get_key_cache(layer_idx)
    var v_cache_host = kv_collection_host.get_value_cache(layer_idx)

    # First check that we didn't augment previous cache entries
    for i in range(batch_size):
        for c in range(cache_lens[i]):
            for h in range(num_heads):
                for d in range(head_dim):
                    var k_val = k_cache_host.load[width=1](i, h, c, d)
                    var v_val = v_cache_host.load[width=1](i, h, c, d)
                    if k_val != 1:
                        raise Error(
                            "Mismatch in output for k, expected 1, got "
                            + String(k_val)
                            + " in k_cache at index "
                            + String(IndexList[4](i, c, h, d))
                        )
                    if v_val != 1:
                        raise Error(
                            "Mismatch in output for v, expected 1, got "
                            + String(v_val)
                            + " in v_cache at index "
                            + String(IndexList[4](i, c, h, d))
                        )

    # Check that non-LoRA entries are not augmented
    for i in range(num_active_loras, batch_size):
        for c in range(prompt_lens[i]):
            var actual_len = c + cache_lens[i]
            for h in range(num_heads):
                for d in range(head_dim):
                    var k_val = k_cache_host.load[width=1](i, h, actual_len, d)
                    var v_val = v_cache_host.load[width=1](i, h, actual_len, d)
                    if k_val != 1:
                        raise Error(
                            "Mismatch in output for k, expected 1, got "
                            + String(k_val)
                            + " in k_cache at index "
                            + String(IndexList[4](i, h, actual_len, d))
                        )
                    if v_val != 1:
                        raise Error(
                            "Mismatch in output for v, expected 1, got "
                            + String(v_val)
                            + " in v_cache at index "
                            + String(IndexList[4](i, h, actual_len, d))
                        )

    # Check that the LoRA-augmented entries are correct
    var slice_row_offset = 0
    for i in range(num_active_loras):
        for c in range(prompt_lens[i]):
            var actual_len = c + cache_lens[i]
            var k_row_base = slice_row_offset * num_heads * head_dim
            for h in range(num_heads):
                for d in range(head_dim):
                    var k_val = k_cache_host.load[width=1](i, h, actual_len, d)
                    var k_idx = k_row_base + h * head_dim + d
                    var expected_k_val = 1 + k_idx
                    if k_val != Scalar[dtype](expected_k_val):
                        raise Error(
                            "Mismatch in output for k, expected "
                            + String(expected_k_val)
                            + ", got "
                            + String(k_val)
                            + " in k_cache at index "
                            + String(IndexList[4](i, h, actual_len, d))
                        )
            # V portion is stored in rows [total_slice_length, 2*total_slice_length)
            # of the input tensor, so the V row base starts at total_slice_length.
            var v_row_base = (
                (total_slice_length + slice_row_offset) * num_heads * head_dim
            )
            for h in range(num_heads):
                for d in range(head_dim):
                    var v_val = v_cache_host.load[width=1](i, h, actual_len, d)
                    var v_idx = v_row_base + h * head_dim + d
                    var expected_v_val = 1 + v_idx
                    if v_val != Scalar[dtype](expected_v_val):
                        raise Error(
                            "Mismatch in output for v, expected "
                            + String(expected_v_val)
                            + ", got "
                            + String(v_val)
                            + " in v_cache at index "
                            + String(IndexList[4](i, h, actual_len, d))
                        )
            slice_row_offset += 1


def test_kv_cache_2m_iadd_gpu[
    dtype: DType,
    num_heads: Int,
    head_dim: Int,
    page_size: Int,
    batch_size: Int,
](
    prompt_lens: IndexList[batch_size],
    cache_lens: IndexList[batch_size],
    num_active_loras: Int,
    ctx: DeviceContext,
) raises:
    comptime num_layers = 2
    assert (
        num_active_loras <= batch_size
    ), "num_active_loras must be less than or equal to batch_size"
    var cache_lengths = HostDeviceTileTensor[.uint32](
        row_major(Int64(batch_size)), ctx
    )
    var cache_lengths_host = cache_lengths.host_tensor()
    var input_row_offsets_slice = HostDeviceTileTensor[.uint32](
        row_major(Int64(num_active_loras + 1)), ctx
    )
    var input_row_offsets_slice_host = input_row_offsets_slice.host_tensor()
    var total_length = 0
    var total_slice_length = 0
    var max_full_context_length = 0
    var max_prompt_length = 0
    for i in range(batch_size):
        cache_lengths_host[i] = UInt32(cache_lens[i])
        max_full_context_length = max(
            max_full_context_length, cache_lens[i] + prompt_lens[i]
        )
        max_prompt_length = max(max_prompt_length, prompt_lens[i])

        if i < num_active_loras:
            input_row_offsets_slice_host[i] = UInt32(total_length)
            total_slice_length += prompt_lens[i]

        total_length += prompt_lens[i]

    input_row_offsets_slice_host[num_active_loras] = UInt32(total_slice_length)
    cache_lengths.to_device()
    input_row_offsets_slice.to_device()

    var num_paged_blocks = ceildiv(
        batch_size * max_full_context_length * 2, page_size
    )

    var lora_end_idx_host_ptr = ctx.enqueue_create_host_buffer[.int64](1)
    var batch_seq_len_host_ptr = ctx.enqueue_create_host_buffer[.int64](1)
    ctx.synchronize()
    var lora_end_idx_host = TileTensor(lora_end_idx_host_ptr, row_major[1]())
    lora_end_idx_host[0] = Int64(total_slice_length)

    var batch_seq_len_host = TileTensor(batch_seq_len_host_ptr, row_major[1]())
    batch_seq_len_host[0] = Int64(total_length)

    var kv_block_paged = HostDeviceTileTensor[dtype](
        row_major(
            Int64(num_paged_blocks),
            Idx[2],
            Int64(num_layers),
            Idx[page_size],
            Idx[num_heads],
            Idx[head_dim],
        ),
        ctx,
    )
    _ = kv_block_paged.host_tensor().fill(1)
    kv_block_paged.to_device()
    var paged_lut = HostDeviceTileTensor[.uint32](
        row_major(
            Int64(batch_size),
            Int64(ceildiv(max_full_context_length, page_size)),
        ),
        ctx,
    )
    var paged_lut_host = paged_lut.host_tensor()
    # Sample one distinct paged block per page across the whole batch up
    # front, then hand them out in iteration order. Total pages needed is
    # <= num_paged_blocks by construction.
    var total_pages = 0
    for bs in range(batch_size):
        total_pages += ceildiv(cache_lens[bs] + prompt_lens[bs], page_size)
    var paged_blocks = random_distinct(num_paged_blocks, total_pages)

    var page_pos = 0
    for bs in range(batch_size):
        var seq_len = cache_lens[bs] + prompt_lens[bs]

        for block_idx in range(0, ceildiv(seq_len, page_size)):
            paged_lut_host[bs, block_idx] = UInt32(paged_blocks[page_pos])
            page_pos += 1
    paged_lut.to_device()

    var kv_collection_device = _create_kv_collection[
        num_heads, head_dim, page_size
    ](
        kv_block_paged.device_tensor(),
        cache_lengths.device_tensor().as_imm().as_unsafe_any_origin(),
        paged_lut.device_tensor().as_imm().as_unsafe_any_origin(),
        max_prompt_length,
        max_full_context_length,
    )

    var a = HostDeviceTileTensor[dtype](
        row_major(Int64(2 * total_slice_length), Idx[num_heads * head_dim]),
        ctx,
    )
    var a_host = a.host_tensor()
    for i in range(a_host.num_elements()):
        a_host.raw_store(i, Scalar[dtype](i))
    a.to_device()

    var layer_idx = 1
    kv_cache_2m_iadd_dispatch[target="gpu"](
        a.device_tensor(),
        kv_collection_device,
        input_row_offsets_slice.device_tensor(),
        lora_end_idx_host,
        batch_seq_len_host,
        UInt32(layer_idx),
        ctx,
    )
    ctx.synchronize()
    kv_block_paged.to_host()

    var kv_collection_host = _create_kv_collection[
        num_heads, head_dim, page_size
    ](
        kv_block_paged.host_tensor(),
        cache_lengths.host_tensor().as_imm().as_unsafe_any_origin(),
        paged_lut.host_tensor().as_imm().as_unsafe_any_origin(),
        max_prompt_length,
        max_full_context_length,
    )

    _verify_kv_cache[dtype, num_heads, head_dim, page_size, batch_size](
        kv_collection_host,
        prompt_lens,
        cache_lens,
        num_active_loras,
        total_slice_length,
        layer_idx,
    )

    # The collections hold untracked views of these buffers.
    _ = kv_block_paged^
    _ = cache_lengths^
    _ = paged_lut^


def test_kv_cache_2m_iadd_cpu[
    dtype: DType,
    num_heads: Int,
    head_dim: Int,
    page_size: Int,
    batch_size: Int,
](
    prompt_lens: IndexList[batch_size],
    cache_lens: IndexList[batch_size],
    num_active_loras: Int,
    ctx: DeviceContext,
) raises:
    comptime num_layers = 2
    assert (
        num_active_loras <= batch_size
    ), "num_active_loras must be less than or equal to batch_size"
    var cache_lengths_host_ptr = List(length=batch_size, fill=UInt32(0))
    var cache_lengths_host = TileTensor(
        cache_lengths_host_ptr, row_major(Int64(batch_size))
    )

    var input_row_offsets_slice_host_ptr = List(
        length=num_active_loras + 1, fill=UInt32(0)
    )
    var input_row_offsets_slice_host = TileTensor(
        input_row_offsets_slice_host_ptr, row_major(num_active_loras + 1)
    )
    var total_length = 0
    var total_slice_length = 0
    var max_full_context_length = 0
    var max_prompt_length = 0
    for i in range(batch_size):
        cache_lengths_host[i] = UInt32(cache_lens[i])
        max_full_context_length = max(
            max_full_context_length, cache_lens[i] + prompt_lens[i]
        )
        max_prompt_length = max(max_prompt_length, prompt_lens[i])

        if i < num_active_loras:
            input_row_offsets_slice_host[i] = UInt32(total_length)
            total_slice_length += prompt_lens[i]

        total_length += prompt_lens[i]

    input_row_offsets_slice_host[num_active_loras] = UInt32(total_slice_length)

    var num_paged_blocks = ceildiv(
        batch_size * max_full_context_length * 2, page_size
    )

    var lora_end_idx_host_ptr = List(length=1, fill=Int64(0))
    var lora_end_idx_host = TileTensor(lora_end_idx_host_ptr, row_major(Idx[1]))
    lora_end_idx_host[0] = Int64(total_slice_length)

    var batch_seq_len_host_ptr = List(length=1, fill=Int64(0))
    var batch_seq_len_host = TileTensor(
        batch_seq_len_host_ptr, row_major(Idx[1])
    )
    batch_seq_len_host[0] = Int64(total_length)

    var kv_block_paged_size = (
        num_paged_blocks * 2 * num_layers * page_size * num_heads * head_dim
    )
    var kv_block_paged_host_ptr = List(
        length=kv_block_paged_size, fill=Scalar[dtype](0)
    )
    var kv_block_paged_host = TileTensor(
        kv_block_paged_host_ptr,
        row_major(
            Int64(num_paged_blocks),
            Idx[2],
            Int64(num_layers),
            Idx[page_size],
            Idx[num_heads],
            Idx[head_dim],
        ),
    )
    _ = kv_block_paged_host.fill(1)
    var paged_lut_cols = ceildiv(max_full_context_length, page_size)
    var paged_lut_host_ptr = List(
        length=batch_size * paged_lut_cols, fill=UInt32(0)
    )
    var paged_lut_host = TileTensor(
        paged_lut_host_ptr,
        row_major(Int64(batch_size), Int64(paged_lut_cols)),
    )
    # Sample one distinct paged block per page across the whole batch up
    # front, then hand them out in iteration order. Total pages needed is
    # <= num_paged_blocks by construction.
    var total_pages = 0
    for bs in range(batch_size):
        total_pages += ceildiv(cache_lens[bs] + prompt_lens[bs], page_size)
    var paged_blocks = random_distinct(num_paged_blocks, total_pages)

    var page_pos = 0
    for bs in range(batch_size):
        var seq_len = cache_lens[bs] + prompt_lens[bs]

        for block_idx in range(0, ceildiv(seq_len, page_size)):
            paged_lut_host[bs, block_idx] = UInt32(paged_blocks[page_pos])
            page_pos += 1

    var kv_collection_host = _create_kv_collection[
        num_heads, head_dim, page_size
    ](
        kv_block_paged_host,
        cache_lengths_host.as_imm().as_unsafe_any_origin(),
        paged_lut_host.as_imm().as_unsafe_any_origin(),
        max_prompt_length,
        max_full_context_length,
    )

    var a_size = 2 * total_slice_length * num_heads * head_dim
    var a_host_ptr = List(length=a_size, fill=Scalar[dtype](0))
    var a_host = TileTensor(
        a_host_ptr,
        row_major(2 * total_slice_length, Idx[num_heads * head_dim]),
    )
    for i in range(a_host.num_elements()):
        a_host.raw_store(i, Scalar[dtype](i))

    var layer_idx = 1
    kv_cache_2m_iadd_dispatch[target="cpu"](
        a_host,
        kv_collection_host,
        input_row_offsets_slice_host,
        lora_end_idx_host,
        batch_seq_len_host,
        UInt32(layer_idx),
        ctx,
    )

    _verify_kv_cache[dtype, num_heads, head_dim, page_size, batch_size](
        kv_collection_host,
        prompt_lens,
        cache_lens,
        num_active_loras,
        total_slice_length,
        layer_idx,
    )

    # The collection holds untracked views of these lists.
    _ = kv_block_paged_host_ptr^
    _ = cache_lengths_host_ptr^
    _ = paged_lut_host_ptr^


def main() raises:
    # CPU tests
    with DeviceContext(api="cpu") as cpu_ctx:
        test_kv_cache_2m_iadd_cpu[.float32, 8, 128, 128, 4](
            IndexList[4](10, 20, 30, 40),
            IndexList[4](40, 30, 20, 10),
            2,
            cpu_ctx,
        )
        test_kv_cache_2m_iadd_cpu[.float32, 8, 128, 128, 4](
            IndexList[4](10, 20, 30, 40),
            IndexList[4](40, 30, 20, 10),
            4,
            cpu_ctx,
        )
        test_kv_cache_2m_iadd_cpu[.float32, 8, 128, 128, 4](
            IndexList[4](10, 20, 30, 40),
            IndexList[4](40, 30, 20, 10),
            0,
            cpu_ctx,
        )
        test_kv_cache_2m_iadd_cpu[.float32, 8, 128, 128, 1](
            (10,),
            (40,),
            1,
            cpu_ctx,
        )

    # GPU tests
    with DeviceContext() as ctx:
        test_kv_cache_2m_iadd_gpu[.float32, 8, 128, 128, 4](
            IndexList[4](10, 20, 30, 40),
            IndexList[4](40, 30, 20, 10),
            2,
            ctx,
        )
        test_kv_cache_2m_iadd_gpu[.float32, 8, 128, 128, 4](
            IndexList[4](10, 20, 30, 40),
            IndexList[4](40, 30, 20, 10),
            4,
            ctx,
        )
        test_kv_cache_2m_iadd_gpu[.float32, 8, 128, 128, 4](
            IndexList[4](10, 20, 30, 40),
            IndexList[4](40, 30, 20, 10),
            0,
            ctx,
        )
        test_kv_cache_2m_iadd_gpu[.float32, 8, 128, 128, 1](
            (10,),
            (40,),
            1,
            ctx,
        )
