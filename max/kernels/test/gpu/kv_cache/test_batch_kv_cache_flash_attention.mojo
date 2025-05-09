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
from math import isqrt
from random import random_ui64, seed

from buffer import Dim, DimList, NDBuffer
from gpu.host import DeviceContext
from internal_utils import DeviceNDBuffer, HostNDBuffer, random
from kv_cache.types import (
    ContinuousBatchingKVCacheCollection,
    KVCacheStaticParams,
)
from memory import UnsafePointer
from nn.mha import flash_attention, mha_gpu_naive
from nn.mha_mask import MaterializedMask
from nn.mha_score_mod import IdentityScoreMod
from testing import assert_almost_equal

from utils import Index, IndexList

from tensor_internal import ManagedTensorSlice
from tensor_internal import IOUnknown
from tensor_internal.managed_tensor_slice import StaticTensorSpec

alias kv_params_replit = KVCacheStaticParams(num_heads=8, head_size=128)
alias replit_num_q_heads = 24

alias kv_params_llama3 = KVCacheStaticParams(num_heads=8, head_size=128)
alias llama_num_q_heads = 32


def execute_flash_attention[
    num_q_heads: Int, type: DType, kv_params: KVCacheStaticParams
](
    batch_size: Int,
    valid_length: NDBuffer[DType.uint32, 1],
    max_seq_len: Int,
    num_layers: Int,
    layer_idx: Int,
    cache_valid_length: NDBuffer[DType.uint32, 1],
    ctx: DeviceContext,
):
    alias num_blocks = 32
    alias CollectionType = ContinuousBatchingKVCacheCollection[type, kv_params]

    debug_assert(
        batch_size < num_blocks,
        "batch_size passed to unit test (",
        batch_size,
        ") is larger than configured num_blocks (",
        num_blocks,
        ")",
    )

    max_prompt_len = 0
    max_context_len = 0

    for i in range(batch_size):
        max_prompt_len = max(max_prompt_len, Int(valid_length[i]))
        max_context_len = max(
            max_context_len, Int(cache_valid_length[i] + valid_length[i])
        )

    # initialize q tensor
    q_host = HostNDBuffer[
        type, 4, DimList(Dim(), Dim(), num_q_heads, kv_params.head_size)
    ](
        IndexList[4](
            batch_size, max_prompt_len, num_q_heads, kv_params.head_size
        )
    )

    random(q_host.tensor)

    valid_lengths_device = DeviceNDBuffer[DType.uint32, 1](
        IndexList[1](batch_size),
        ctx=ctx,
    )
    ctx.enqueue_copy(valid_lengths_device.buffer, valid_length.data)

    q_device = DeviceNDBuffer[
        type, 4, DimList(Dim(), Dim(), num_q_heads, kv_params.head_size)
    ](
        IndexList[4](
            batch_size, max_prompt_len, num_q_heads, kv_params.head_size
        ),
        ctx=ctx,
    )
    ctx.enqueue_copy(q_device.buffer, q_host.tensor.data)

    # initialize mask tensor
    # TODO this should ideally create a triangular matrix
    # but the output should be consistent regardless.
    mask_host = HostNDBuffer[
        type, 4, DimList(Dim(), num_q_heads, Dim(), Dim())
    ](IndexList[4](batch_size, num_q_heads, max_prompt_len, max_context_len))
    random(mask_host.tensor)
    mask_device = mask_host.copy_to_device(ctx)

    # initialize reference output
    ref_output_host = HostNDBuffer[
        type, 4, DimList(Dim(), Dim(), num_q_heads, kv_params.head_size)
    ](
        IndexList[4](
            batch_size, max_prompt_len, num_q_heads, kv_params.head_size
        ),
    )
    ref_output_device = DeviceNDBuffer[
        type, 4, DimList(Dim(), Dim(), num_q_heads, kv_params.head_size)
    ](
        IndexList[4](
            batch_size, max_prompt_len, num_q_heads, kv_params.head_size
        ),
        ctx=ctx,
    )

    # initialize test output
    test_output_host = HostNDBuffer[
        type, 4, DimList(Dim(), Dim(), num_q_heads, kv_params.head_size)
    ](
        IndexList[4](
            batch_size, max_prompt_len, num_q_heads, kv_params.head_size
        ),
    )
    test_output_device = DeviceNDBuffer[
        type, 4, DimList(Dim(), Dim(), num_q_heads, kv_params.head_size)
    ](
        IndexList[4](
            batch_size, max_prompt_len, num_q_heads, kv_params.head_size
        ),
        ctx=ctx,
    )

    # initialize our KVCache
    var cache_lengths_dev = ctx.enqueue_create_buffer[DType.uint32](batch_size)

    ctx.enqueue_copy(cache_lengths_dev, cache_valid_length.data)
    var cache_lengths_device_nd = NDBuffer[DType.uint32, 1](
        cache_lengths_dev._unsafe_ptr(), Index(batch_size)
    )
    kv_block_host = HostNDBuffer[type, 6](
        IndexList[6](
            num_blocks,
            2,
            num_layers,
            max_seq_len,
            kv_params.num_heads,
            kv_params.head_size,
        ),
    )
    random(kv_block_host.tensor)
    kv_block_device = kv_block_host.copy_to_device(ctx)

    var lookup_table_host = HostNDBuffer[DType.uint32, 1](
        IndexList[1](batch_size),
    )

    var lookup_table_device = DeviceNDBuffer[DType.uint32, 1](
        IndexList[1](batch_size),
        ctx=ctx,
    )

    # hacky way to get random block indices
    var block_idx_set = Set[Int]()
    var idx = 0
    while len(block_idx_set) < batch_size:
        var randval = Int(random_ui64(0, num_blocks - 1))
        if randval in block_idx_set:
            continue
        block_idx_set.add(randval)
        lookup_table_host.tensor[idx] = UInt32(randval)
        idx += 1

    ctx.enqueue_copy(lookup_table_device.buffer, lookup_table_host.tensor.data)

    var kv_collection_device = CollectionType(
        kv_block_device.tensor,
        cache_lengths_device_nd,
        lookup_table_device.tensor,
        max_prompt_len,
        max_context_len,
    )

    var k_cache_device = kv_collection_device.get_key_cache(layer_idx)
    var v_cache_device = kv_collection_device.get_value_cache(layer_idx)

    flash_attention(
        test_output_device.tensor,
        q_device.tensor,
        k_cache_device,
        v_cache_device,
        MaterializedMask(mask_device.tensor, start_pos=cache_lengths_device_nd),
        IdentityScoreMod(),
        ManagedTensorSlice[
            io_spec=IOUnknown,
            static_spec = StaticTensorSpec[DType.uint32, 1].create_unknown(),
        ](valid_lengths_device.tensor),
        isqrt(Float32(kv_params.head_size)),
        ctx,
    )

    mha_gpu_naive(
        q_device.tensor,
        k_cache_device,
        v_cache_device,
        MaterializedMask(mask_device.tensor, start_pos=cache_lengths_device_nd),
        ref_output_device.tensor,
        ManagedTensorSlice[
            io_spec=IOUnknown,
            static_spec = StaticTensorSpec[DType.uint32, 1].create_unknown(),
        ](valid_lengths_device.tensor),
        isqrt(Float32(kv_params.head_size)),
        batch_size,
        max_prompt_len,
        max_context_len,
        num_q_heads,  # TODO fix this for GQA
        kv_params.head_size,
        num_q_heads // kv_params.num_heads,
        ctx,
    )

    ctx.enqueue_copy(test_output_host.tensor.data, test_output_device.buffer)
    ctx.enqueue_copy(ref_output_host.tensor.data, ref_output_device.buffer)
    ctx.synchronize()

    ref_out = ref_output_host.tensor
    test_out = test_output_host.tensor
    for bs in range(Int(batch_size)):
        for s in range(Int(valid_length[bs])):
            for h in range(Int(num_q_heads)):
                for hd in range(kv_params.head_size):
                    assert_almost_equal(
                        ref_out[bs, s, h, hd],
                        test_out[bs, s, h, hd],
                        atol=1e-5,
                        rtol=8e-3,
                    )

    _ = q_device^
    _ = q_host^
    _ = ref_output_device^
    _ = ref_output_host^
    _ = test_output_device^
    _ = test_output_host^
    _ = kv_block_device^
    _ = kv_block_host^
    _ = mask_host^
    _ = mask_device^
    _ = cache_lengths_dev^
    _ = valid_lengths_device^


def execute_flash_attention_suite(ctx: DeviceContext):
    alias types = (DType.float32, DType.bfloat16)
    var bs = 2
    var valid_length_ptr = UnsafePointer[UInt32].alloc(bs)
    var valid_length = NDBuffer[DType.uint32, 1](valid_length_ptr, Index(bs))

    var cache_valid_length_ptr = UnsafePointer[UInt32].alloc(bs)
    var cache_valid_length = NDBuffer[DType.uint32, 1](
        cache_valid_length_ptr, Index(bs)
    )

    @parameter
    for type_idx in range(len(types)):
        alias type = types[type_idx]

        # Replit context encoding [testing even query valid lengths].
        valid_length[0] = 128
        valid_length[1] = 64
        cache_valid_length[0] = 0
        cache_valid_length[1] = 0
        execute_flash_attention[replit_num_q_heads, type, kv_params_replit](
            bs, valid_length, 1024, 4, 3, cache_valid_length, ctx
        )

        # Replit context encoding [testing odd query valid length].
        valid_length[0] = 128
        valid_length[1] = 65
        cache_valid_length[0] = 0
        cache_valid_length[1] = 0
        execute_flash_attention[replit_num_q_heads, type, kv_params_replit](
            bs, valid_length, 1024, 4, 0, cache_valid_length, ctx
        )

        # Replit token gen [testing even cache valid lengths].
        valid_length[0] = 1
        valid_length[1] = 1
        cache_valid_length[0] = 200
        cache_valid_length[1] = 256

        execute_flash_attention[replit_num_q_heads, type, kv_params_replit](
            bs, valid_length, 1024, 4, 1, cache_valid_length, ctx
        )

        # Replit token gen [testing even cache valid lengths].
        valid_length[0] = 1
        valid_length[1] = 1
        cache_valid_length[0] = 200
        cache_valid_length[1] = 255

        execute_flash_attention[replit_num_q_heads, type, kv_params_replit](
            bs, valid_length, 1024, 4, 2, cache_valid_length, ctx
        )


def main():
    seed(42)
    with DeviceContext() as ctx:
        execute_flash_attention_suite(ctx)
