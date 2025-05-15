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
from math import ceildiv, isqrt
from random import random_ui64

from buffer import Dim, DimList, NDBuffer
from gpu.host import DeviceContext
from internal_utils import DeviceNDBuffer, HostNDBuffer, random
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from memory import UnsafePointer, memcpy
from nn.fused_qk_rope import fused_qk_rope_ragged
from testdata.fused_qk_rope_goldens import freqs_cis_table_input
from testing import assert_almost_equal

from utils import IndexList


def _init_device_ndbuffer_from_goldens[
    type: DType, //, shape: DimList
](goldens: List[Scalar[type]], ctx: DeviceContext) -> DeviceNDBuffer[
    type, len(shape), shape=shape
]:
    """Initializes a device buffer with a set of golden values."""
    host_tensor = HostNDBuffer[type, len(shape), shape=shape]()
    memcpy(dest=host_tensor.tensor.data, src=goldens.data, count=len(goldens))

    # Copy tensor to device.
    device_tensor = DeviceNDBuffer[
        host_tensor.type,
        host_tensor.rank,
        shape = host_tensor.shape,
    ](ctx=ctx)
    ctx.enqueue_copy(device_tensor.buffer, host_tensor.tensor.data)
    ctx.synchronize()

    # Ensure the host buffer outlives the copy.
    _ = host_tensor^

    return device_tensor


def execute_fused_qk_rope_ragged(
    ctx: DeviceContext,
):
    alias num_q_heads = 32
    alias kv_params = KVCacheStaticParams(num_heads=8, head_size=128)
    alias type = DType.float32
    alias num_paged_blocks = 32
    alias page_size = 128
    var num_layers = 1
    var layer_idx = 0

    alias max_seq_len = 1024

    var true_ce_prompt_lens = List[Int](100, 200, 300, 400)
    var mixed_ce_prompt_lens = List[Int](50, 100, 150, 100)

    var true_ce_cache_lens = List[Int](0, 0, 0, 0)
    var mixed_ce_cache_lens = List[Int](50, 100, 150, 300)

    var batch_size = len(true_ce_prompt_lens)

    var true_ce_row_offsets_host = HostNDBuffer[DType.uint32, 1](
        IndexList[1](batch_size + 1)
    )
    var true_ce_cache_lengths_host = HostNDBuffer[DType.uint32, 1](
        IndexList[1](batch_size)
    )
    var mixed_ce_row_offsets_host = HostNDBuffer[DType.uint32, 1](
        IndexList[1](batch_size + 1)
    )
    var mixed_ce_cache_lengths_host = HostNDBuffer[DType.uint32, 1](
        IndexList[1](batch_size)
    )

    var true_ce_total_length = 0
    var mixed_ce_total_length = 0
    var true_ce_max_cache_length = 0
    var mixed_ce_max_cache_length = 0
    var true_ce_max_full_context_length = 0
    var mixed_ce_max_full_context_length = 0
    var true_ce_max_prompt_length = 0
    var mixed_ce_max_prompt_length = 0
    for i in range(batch_size):
        true_ce_row_offsets_host.tensor[i] = true_ce_total_length
        mixed_ce_row_offsets_host.tensor[i] = mixed_ce_total_length
        true_ce_cache_lengths_host.tensor[i] = true_ce_cache_lens[i]
        mixed_ce_cache_lengths_host.tensor[i] = mixed_ce_cache_lens[i]

        true_ce_max_cache_length = max(
            true_ce_max_cache_length, true_ce_cache_lens[i]
        )
        mixed_ce_max_cache_length = max(
            mixed_ce_max_cache_length, mixed_ce_cache_lens[i]
        )
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

    true_ce_row_offsets_host.tensor[batch_size] = true_ce_total_length
    mixed_ce_row_offsets_host.tensor[batch_size] = mixed_ce_total_length
    true_ce_row_offsets_device = true_ce_row_offsets_host.copy_to_device(ctx)
    mixed_ce_row_offsets_device = mixed_ce_row_offsets_host.copy_to_device(ctx)
    true_ce_cache_lengths_device = true_ce_cache_lengths_host.copy_to_device(
        ctx
    )
    mixed_ce_cache_lengths_device = mixed_ce_cache_lengths_host.copy_to_device(
        ctx
    )
    true_ce_q_ragged_host = HostNDBuffer[
        type, 3, DimList(Dim(), num_q_heads, kv_params.head_size)
    ](IndexList[3](true_ce_total_length, num_q_heads, kv_params.head_size))
    random(true_ce_q_ragged_host.tensor)
    true_ce_q_ragged_device = true_ce_q_ragged_host.copy_to_device(ctx)

    mixed_ce_q_ragged_host = HostNDBuffer[
        type, 3, DimList(Dim(), num_q_heads, kv_params.head_size)
    ](IndexList[3](mixed_ce_total_length, num_q_heads, kv_params.head_size))
    for bs_idx in range(batch_size):
        true_ce_prompt_len = true_ce_prompt_lens[bs_idx]
        mixed_ce_prompt_len = mixed_ce_prompt_lens[bs_idx]

        true_ce_row_offset = true_ce_row_offsets_host.tensor[bs_idx]
        mixed_ce_row_offset = mixed_ce_row_offsets_host.tensor[bs_idx]

        mixed_ce_cache_len = mixed_ce_cache_lens[bs_idx]

        true_ce_offset = true_ce_q_ragged_host.tensor._offset(
            IndexList[3](Int(true_ce_row_offset + mixed_ce_cache_len), 0, 0)
        )
        mixed_ce_offset = mixed_ce_q_ragged_host.tensor._offset(
            IndexList[3](Int(mixed_ce_row_offset), 0, 0)
        )

        memcpy(
            mixed_ce_offset,
            true_ce_offset,
            mixed_ce_prompt_len * num_q_heads * kv_params.head_size,
        )

    mixed_ce_q_ragged_device = mixed_ce_q_ragged_host.copy_to_device(ctx)

    freqs_cis_table_dev = _init_device_ndbuffer_from_goldens[
        shape = DimList(max_seq_len, kv_params.head_size)
    ](freqs_cis_table_input[type](), ctx)

    # initialize reference output
    mixed_ce_output_host = HostNDBuffer[
        type, 3, DimList(Dim(), num_q_heads, kv_params.head_size)
    ](IndexList[3](mixed_ce_total_length, num_q_heads, kv_params.head_size))
    mixed_ce_output_device = mixed_ce_output_host.copy_to_device(ctx)
    true_ce_output_host = HostNDBuffer[
        type, 3, DimList(Dim(), num_q_heads, kv_params.head_size)
    ](IndexList[3](true_ce_total_length, num_q_heads, kv_params.head_size))
    true_ce_output_device = true_ce_output_host.copy_to_device(ctx)

    # initialize our KVCache
    true_ce_kv_block_paged_host = HostNDBuffer[type, 6](
        IndexList[6](
            num_paged_blocks,
            2,
            num_layers,
            page_size,
            kv_params.num_heads,
            kv_params.head_size,
        )
    )
    mixed_ce_kv_block_paged_host = HostNDBuffer[type, 6](
        IndexList[6](
            num_paged_blocks,
            2,
            num_layers,
            page_size,
            kv_params.num_heads,
            kv_params.head_size,
        )
    )
    random(true_ce_kv_block_paged_host.tensor)
    true_ce_kv_block_paged_device = true_ce_kv_block_paged_host.copy_to_device(
        ctx
    )
    # intentionally copy from true_ce so the contents of these blocks are consistent.
    mixed_ce_kv_block_paged_device = true_ce_kv_block_paged_host.copy_to_device(
        ctx
    )

    paged_lut_host = HostNDBuffer[DType.uint32, 2](
        IndexList[2](
            batch_size,
            ceildiv(true_ce_max_full_context_length, page_size),
        )
    )
    paged_lut_set = Set[Int]()
    for bs in range(batch_size):
        seq_len = true_ce_cache_lens[bs] + true_ce_prompt_lens[bs]

        for block_idx in range(0, ceildiv(seq_len, page_size)):
            var randval = Int(random_ui64(0, num_paged_blocks - 1))
            while randval in paged_lut_set:
                randval = Int(random_ui64(0, num_paged_blocks - 1))

            paged_lut_set.add(randval)
            paged_lut_host.tensor[bs, block_idx] = randval

    paged_lut_device = paged_lut_host.copy_to_device(ctx)

    var true_ce_k_cache_collection = PagedKVCacheCollection[
        type, kv_params, page_size
    ](
        true_ce_kv_block_paged_device.tensor,
        true_ce_cache_lengths_device.tensor,
        paged_lut_device.tensor,
        true_ce_max_prompt_length,
        true_ce_max_cache_length,
    )

    var mixed_ce_k_cache_collection = PagedKVCacheCollection[
        type, kv_params, page_size
    ](
        mixed_ce_kv_block_paged_device.tensor,
        mixed_ce_cache_lengths_device.tensor,
        paged_lut_device.tensor,
        mixed_ce_max_prompt_length,
        mixed_ce_max_cache_length,
    )

    # "true CE" execution
    print("true")
    var freqs_cis = rebind[
        NDBuffer[
            type,
            2,
            MutableAnyOrigin,
            shape = freqs_cis_table_dev.shape,
        ]
    ](freqs_cis_table_dev.tensor)
    fused_qk_rope_ragged[
        mixed_ce_k_cache_collection.CacheType, interleaved=False, target="gpu"
    ](
        true_ce_q_ragged_device.tensor,
        true_ce_row_offsets_device.tensor,
        true_ce_k_cache_collection,
        freqs_cis,
        layer_idx,
        output=true_ce_output_device.tensor,
        context=ctx,
    )

    # "mixed CE" execution
    print("mixed")
    fused_qk_rope_ragged[
        mixed_ce_k_cache_collection.CacheType, interleaved=False, target="gpu"
    ](
        mixed_ce_q_ragged_device.tensor,
        mixed_ce_row_offsets_device.tensor,
        mixed_ce_k_cache_collection,
        freqs_cis,
        layer_idx,
        output=mixed_ce_output_device.tensor,
        context=ctx,
    )
    ctx.enqueue_copy(
        mixed_ce_output_host.tensor.data, mixed_ce_output_device.buffer
    )
    ctx.enqueue_copy(
        true_ce_output_host.tensor.data, true_ce_output_device.buffer
    )
    ctx.enqueue_copy(
        true_ce_kv_block_paged_host.tensor.data,
        true_ce_kv_block_paged_device.buffer,
    )
    ctx.enqueue_copy(
        mixed_ce_kv_block_paged_host.tensor.data,
        mixed_ce_kv_block_paged_device.buffer,
    )
    ctx.synchronize()

    var true_ce_k_cache_collection_host = PagedKVCacheCollection[
        type, kv_params, page_size
    ](
        true_ce_kv_block_paged_host.tensor,
        true_ce_cache_lengths_host.tensor,
        paged_lut_host.tensor,
        true_ce_max_prompt_length,
        true_ce_max_cache_length,
    )
    var true_ce_k_cache = true_ce_k_cache_collection_host.get_key_cache(
        layer_idx
    )

    var mixed_ce_k_cache_collection_host = PagedKVCacheCollection[
        type, kv_params, page_size
    ](
        mixed_ce_kv_block_paged_host.tensor,
        mixed_ce_cache_lengths_host.tensor,
        paged_lut_host.tensor,
        mixed_ce_max_prompt_length,
        mixed_ce_max_cache_length,
    )
    var mixed_ce_k_cache = mixed_ce_k_cache_collection_host.get_key_cache(
        layer_idx
    )
    print("comparing Q")
    for bs_idx in range(batch_size):
        true_ce_batch_start_idx = Int(true_ce_row_offsets_host.tensor[bs_idx])
        mixed_ce_batch_start_idx = Int(mixed_ce_row_offsets_host.tensor[bs_idx])
        mixed_ce_cache_len = Int(mixed_ce_cache_lengths_host.tensor[bs_idx])

        for tok_idx in range(mixed_ce_prompt_lens[bs_idx]):
            for head_idx in range(num_q_heads):
                for head_dim in range(kv_params.head_size):
                    assert_almost_equal(
                        mixed_ce_output_host.tensor[
                            mixed_ce_batch_start_idx + tok_idx,
                            head_idx,
                            head_dim,
                        ],
                        true_ce_output_host.tensor[
                            true_ce_batch_start_idx
                            + mixed_ce_cache_len
                            + tok_idx,
                            head_idx,
                            head_dim,
                        ],
                    )

    print("comparing K")
    for bs_idx in range(batch_size):
        mixed_ce_cache_len = mixed_ce_cache_lens[bs_idx]

        for tok_idx in range(mixed_ce_prompt_lens[bs_idx]):
            for head_idx in range(kv_params.num_heads):
                for head_dim in range(kv_params.head_size):
                    assert_almost_equal(
                        true_ce_k_cache.load[width=1](
                            bs_idx,
                            head_idx,
                            mixed_ce_cache_len + tok_idx,
                            head_dim,
                        ),
                        mixed_ce_k_cache.load[width=1](
                            bs_idx,
                            head_idx,
                            mixed_ce_cache_len + tok_idx,
                            head_dim,
                        ),
                    )

    _ = true_ce_row_offsets_host^
    _ = true_ce_row_offsets_device^
    _ = mixed_ce_row_offsets_host^
    _ = mixed_ce_row_offsets_device^
    _ = true_ce_q_ragged_host^
    _ = true_ce_q_ragged_device^
    _ = mixed_ce_q_ragged_host^
    _ = mixed_ce_q_ragged_device^
    _ = true_ce_kv_block_paged_host^
    _ = true_ce_kv_block_paged_device^
    _ = mixed_ce_kv_block_paged_host^
    _ = mixed_ce_kv_block_paged_device^
    _ = paged_lut_host^
    _ = paged_lut_device^
    _ = true_ce_output_host^
    _ = true_ce_output_device^
    _ = mixed_ce_output_host^
    _ = mixed_ce_output_device^


# We test the fused_qk_rope_ragged kernel with rope_dim = 64 and q_head_size = 192
# and kv_params.head_size = 576 (shapes are chosen based on Deepseek models).
# For Q, we confirm that the only the last 64 elements in each head are correctly roped,
# and the first 128 elements in each head are simply copied from the input Q tensor.
# For KV cache, we confirm that the only the last 64 elements in each head are correctly roped,
# and the first 512 elements are left unchanged.
def execute_fused_qk_rope_ragged_mla(ctx: DeviceContext):
    alias num_q_heads = 16
    alias q_head_size = 192
    alias kv_params = KVCacheStaticParams(num_heads=1, head_size=576)
    alias kv_params_64 = KVCacheStaticParams(num_heads=1, head_size=64)
    alias type = DType.bfloat16
    alias num_paged_blocks = 2
    alias page_size = 128
    alias rope_dim = 64
    alias max_seq_len = 256
    alias num_layers = 1
    alias layer_idx = 0

    alias seq_len = 200
    alias batch_size = 1

    # create a random query tensor and KV cache with above params
    var q_ragged_host = HostNDBuffer[
        type, 3, DimList(Dim(), num_q_heads, q_head_size)
    ](IndexList[3](seq_len, num_q_heads, q_head_size))
    random(q_ragged_host.tensor)
    var q_ragged_device = q_ragged_host.copy_to_device(ctx)

    # create a query tensor that only has 64 elements in each head,
    # then copy the last 64 elements of each head from q_ragged_device
    # to the new tensor
    var q_ragged_host_64 = HostNDBuffer[
        type, 3, DimList(Dim(), num_q_heads, rope_dim)
    ](IndexList[3](seq_len, num_q_heads, rope_dim))
    for seq_idx in range(seq_len):
        for head_idx in range(num_q_heads):
            memcpy(
                q_ragged_host_64.tensor._offset(
                    IndexList[3](seq_idx, head_idx, 0)
                ),
                q_ragged_host.tensor._offset(
                    IndexList[3](seq_idx, head_idx, q_head_size - rope_dim)
                ),
                rope_dim,
            )
    var q_ragged_device_64 = q_ragged_host_64.copy_to_device(ctx)

    # create a random KV cache with above params
    var kv_block_paged_host = HostNDBuffer[type, 6](
        IndexList[6](
            num_paged_blocks,
            2,
            num_layers,
            page_size,
            kv_params.num_heads,
            kv_params.head_size,
        )
    )
    random(kv_block_paged_host.tensor)
    var kv_block_paged_device = kv_block_paged_host.copy_to_device(ctx)

    # create a KV cache that only has 64 elements in each head,
    # then copy the last 64 elements of each head from kv_block_paged_device
    # to the new tensor
    var kv_block_paged_host_64 = HostNDBuffer[type, 6](
        IndexList[6](
            num_paged_blocks,
            2,
            num_layers,
            page_size,
            kv_params.num_heads,
            rope_dim,
        )
    )
    for page_idx in range(num_paged_blocks):
        for kv_idx in range(2):
            for layer_idx in range(num_layers):
                for tok_idx in range(page_size):
                    for head_idx in range(kv_params.num_heads):
                        memcpy(
                            kv_block_paged_host_64.tensor._offset(
                                IndexList[6](
                                    page_idx,
                                    kv_idx,
                                    layer_idx,
                                    tok_idx,
                                    head_idx,
                                    0,
                                )
                            ),
                            kv_block_paged_host.tensor._offset(
                                IndexList[6](
                                    page_idx,
                                    kv_idx,
                                    layer_idx,
                                    tok_idx,
                                    head_idx,
                                    kv_params.head_size - rope_dim,
                                )
                            ),
                            rope_dim,
                        )
    var kv_block_paged_device_64 = kv_block_paged_host_64.copy_to_device(ctx)

    # create a random freqs_cis tensor with above params
    var freqs_cis_host = HostNDBuffer[type, 2, DimList(max_seq_len, rope_dim)](
        IndexList[2](max_seq_len, rope_dim)
    )
    random(freqs_cis_host.tensor)
    var freqs_cis_device = freqs_cis_host.copy_to_device(ctx)

    # create a output tensor with above params
    var output_host = HostNDBuffer[type, 3](
        IndexList[3](seq_len, num_q_heads, q_head_size)
    )
    var output_device = output_host.copy_to_device(ctx)

    # create a output tensor for reference
    var output_host_ref = HostNDBuffer[type, 3](
        IndexList[3](seq_len, num_q_heads, rope_dim)
    )
    var output_device_ref = output_host_ref.copy_to_device(ctx)

    # initialize our row_offsets, we only has 1 sequence
    var row_offsets_host = HostNDBuffer[DType.uint32, 1](
        IndexList[1](batch_size + 1)
    )
    row_offsets_host.tensor[0] = 0
    row_offsets_host.tensor[1] = seq_len
    var row_offsets_device = row_offsets_host.copy_to_device(ctx)

    # initialize our lut, we only has 2 pages
    var paged_lut_host = HostNDBuffer[DType.uint32, 2](
        IndexList[2](batch_size, 2)
    )
    paged_lut_host.tensor[0, 0] = 0
    paged_lut_host.tensor[0, 1] = 1
    var paged_lut_device = paged_lut_host.copy_to_device(ctx)

    # initialize our cache_lengths, we only has 1 sequence
    var cache_lengths_host = HostNDBuffer[DType.uint32, 1](
        IndexList[1](batch_size)
    )
    cache_lengths_host.tensor[0] = 0
    var cache_lengths_device = cache_lengths_host.copy_to_device(ctx)

    # intialize our max_prompt_length and max_cache_length
    var max_prompt_length = Int(seq_len)
    var max_cache_length = Int(0)

    # initialize our k_cache_collection
    var k_cache_collection = PagedKVCacheCollection[type, kv_params, page_size](
        kv_block_paged_device.tensor,
        cache_lengths_device.tensor,
        paged_lut_device.tensor,
        max_prompt_length,
        max_cache_length,
    )

    # initialize our k_cache_collection_64
    var k_cache_collection_64 = PagedKVCacheCollection[
        type, kv_params_64, page_size
    ](
        kv_block_paged_device_64.tensor,
        cache_lengths_device.tensor,
        paged_lut_device.tensor,
        max_prompt_length,
        max_cache_length,
    )

    # execute the kernel
    var freqs_cis = rebind[
        NDBuffer[
            type,
            2,
            __type_of(freqs_cis_device.tensor).origin,
            shape = DimList(max_seq_len, rope_dim),
        ]
    ](freqs_cis_device.tensor)
    fused_qk_rope_ragged[
        k_cache_collection.CacheType, interleaved=True, target="gpu"
    ](
        q_ragged_device.tensor,
        row_offsets_device.tensor,
        k_cache_collection,
        freqs_cis,
        layer_idx,
        output=output_device.tensor,
        context=ctx,
    )

    # execute the kernel for 64
    fused_qk_rope_ragged[
        k_cache_collection_64.CacheType, interleaved=True, target="gpu"
    ](
        q_ragged_device_64.tensor,
        row_offsets_device.tensor,
        k_cache_collection_64,
        freqs_cis,
        layer_idx,
        output=output_device_ref.tensor,
        context=ctx,
    )

    # copy the output back to host
    ctx.enqueue_copy(output_host.tensor.data, output_device.buffer)
    ctx.enqueue_copy(output_host_ref.tensor.data, output_device_ref.buffer)
    ctx.synchronize()

    # compare the output, the first 128 elements in each head should be the same
    # as in input Q tensor, the last 64 elements should be the same as the reference
    for seq_idx in range(seq_len):
        for head_idx in range(num_q_heads):
            for head_dim_idx in range(q_head_size - rope_dim):
                assert_almost_equal(
                    output_host.tensor[seq_idx, head_idx, head_dim_idx],
                    q_ragged_host.tensor[seq_idx, head_idx, head_dim_idx],
                )

            for head_dim_idx in range(rope_dim):
                assert_almost_equal(
                    output_host.tensor[
                        seq_idx, head_idx, q_head_size - rope_dim + head_dim_idx
                    ],
                    output_host_ref.tensor[seq_idx, head_idx, head_dim_idx],
                )

    # copy the kv_block_paged_device back to a new host buffer
    var kv_block_paged_host_copy = HostNDBuffer[type, 6](
        IndexList[6](
            num_paged_blocks,
            2,
            num_layers,
            page_size,
            kv_params.num_heads,
            kv_params.head_size,
        )
    )
    ctx.enqueue_copy(
        kv_block_paged_host_copy.tensor.data, kv_block_paged_device.buffer
    )
    # copy the kv_block_paged_device_64 back to the original host buffer
    ctx.enqueue_copy(
        kv_block_paged_host_64.tensor.data, kv_block_paged_device_64.buffer
    )
    ctx.synchronize()

    # compare the KV cache, the first 512 elements should be the same as the input
    # the last 64 elements should be the same as the reference
    for page_idx in range(num_paged_blocks):
        # only compare the K cache
        for kv_idx in range(1):
            for layer_idx in range(num_layers):
                for tok_idx in range(page_size):
                    if tok_idx + page_idx * page_size < seq_len:
                        for head_idx in range(kv_params.num_heads):
                            for head_dim_idx in range(
                                kv_params.head_size - rope_dim
                            ):
                                assert_almost_equal(
                                    kv_block_paged_host_copy.tensor[
                                        page_idx,
                                        kv_idx,
                                        layer_idx,
                                        tok_idx,
                                        head_idx,
                                        head_dim_idx,
                                    ],
                                    kv_block_paged_host.tensor[
                                        page_idx,
                                        kv_idx,
                                        layer_idx,
                                        tok_idx,
                                        head_idx,
                                        head_dim_idx,
                                    ],
                                )
                            for head_dim_idx in range(rope_dim):
                                assert_almost_equal(
                                    kv_block_paged_host_copy.tensor[
                                        page_idx,
                                        kv_idx,
                                        layer_idx,
                                        tok_idx,
                                        head_idx,
                                        kv_params.head_size
                                        - rope_dim
                                        + head_dim_idx,
                                    ],
                                    kv_block_paged_host_64.tensor[
                                        page_idx,
                                        kv_idx,
                                        layer_idx,
                                        tok_idx,
                                        head_idx,
                                        head_dim_idx,
                                    ],
                                )

    # clean up
    _ = q_ragged_host^
    _ = q_ragged_device^
    _ = q_ragged_host_64^
    _ = q_ragged_device_64^
    _ = kv_block_paged_host^
    _ = kv_block_paged_host_copy^
    _ = kv_block_paged_device^
    _ = kv_block_paged_host_64^
    _ = kv_block_paged_device_64^
    _ = freqs_cis_host^
    _ = freqs_cis_device^
    _ = output_host^
    _ = output_device^
    _ = output_host_ref^
    _ = output_device_ref^
    _ = row_offsets_host^
    _ = row_offsets_device^
    _ = paged_lut_host^
    _ = paged_lut_device^
    _ = cache_lengths_host^
    _ = cache_lengths_device^


def main():
    with DeviceContext() as ctx:
        execute_fused_qk_rope_ragged(ctx)
        execute_fused_qk_rope_ragged_mla(ctx)
