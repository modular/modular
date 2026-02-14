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

from memory import LegacyUnsafePointer

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]
from collections import Set
from math import exp2, iota, rsqrt
from random import random_ui64, seed

from bit import prev_power_of_two
from gpu.host import DeviceContext
from kv_cache.types import (
    ContinuousBatchingKVCacheCollection,
    KVCacheStaticParams,
)
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from layout._fillers import random
from nn.mha import flash_attention, mha_gpu_naive
from nn.mha_mask import CausalMask, MaterializedMask
from nn.mha_score_mod import AlibiScoreMod, IdentityScoreMod
from tensor import IOUnknown, ManagedTensorSlice
from tensor.managed_tensor_slice import StaticTensorSpec
from testing import assert_almost_equal

from utils import Index, IndexList
from utils.numerics import min_or_neg_inf

comptime kv_params_replit = KVCacheStaticParams(num_heads=8, head_size=128)
comptime replit_num_q_heads = 24

comptime kv_params_llama3 = KVCacheStaticParams(num_heads=8, head_size=128)
comptime llama_num_q_heads = 32


fn generate_alibi_bias[
    dtype: DType,
    width: Int,
    num_heads: Int,
](
    head_idx: SIMD[DType.int, width],
    q_idx: SIMD[DType.int, width],
    k_idx: SIMD[DType.int, width],
    max_prompt_len: Int = 0,
) -> SIMD[dtype, width]:
    comptime assert dtype.is_floating_point(), "dtype must be floating point"
    var scale: SIMD[dtype, width]

    @parameter
    if num_heads.is_power_of_two():
        scale = exp2(
            -((head_idx + 1).cast[dtype]() * 8.0 / Scalar[dtype](num_heads))
        )
    else:
        comptime floor_power_of_2 = prev_power_of_two(num_heads)
        if head_idx < Scalar[DType.int](floor_power_of_2):
            scale = exp2(
                -(
                    (head_idx + 1).cast[dtype]()
                    * 8.0
                    / Scalar[dtype](floor_power_of_2)
                )
            )
        else:
            scale = exp2(
                -(
                    (
                        (head_idx - Scalar[DType.int](floor_power_of_2)) * 2 + 1
                    ).cast[dtype]()
                    * 8.0
                    / Scalar[dtype](floor_power_of_2 * 2)
                )
            )
    var bias = (
        -(
            Scalar[DType.int](max_prompt_len - 1)
            - k_idx
            - iota[DType.int, width]()
        ).cast[dtype]()
        * scale
    )
    return bias


def execute_flash_attention[
    num_q_heads: Int,
    dtype: DType,
    kv_params: KVCacheStaticParams,
](
    batch_size: Int,
    valid_length: LayoutTensor[DType.uint32, Layout(UNKNOWN_VALUE)],
    max_seq_len: Int,
    cache_valid_length: LayoutTensor[DType.uint32, Layout(UNKNOWN_VALUE)],
    ctx: DeviceContext,
):
    comptime num_blocks = 32
    comptime CollectionType = ContinuousBatchingKVCacheCollection[
        dtype, kv_params
    ]

    debug_assert(
        batch_size < num_blocks,
        "batch_size passed to unit test (",
        batch_size,
        ") is larger than configured num_blocks (",
        num_blocks,
        ")",
    )

    # initialize our KVCache
    max_prompt_len = 0
    max_context_len = 0

    for i in range(batch_size):
        max_prompt_len = max(max_prompt_len, Int(valid_length[i]))
        max_context_len = max(
            max_context_len, Int(cache_valid_length[i] + valid_length[i])
        )

    var cache_lengths_dev = ctx.enqueue_create_buffer[DType.uint32](batch_size)

    ctx.enqueue_copy(cache_lengths_dev, cache_valid_length.ptr)
    var cache_lengths = LayoutTensor[
        DType.uint32, Layout(UNKNOWN_VALUE), ImmutAnyOrigin
    ](
        cache_lengths_dev.unsafe_ptr(),
        RuntimeLayout[Layout(UNKNOWN_VALUE)].row_major(Index(batch_size)),
    )
    var cache_lengths_lt = LayoutTensor[
        DType.uint32, Layout.row_major(UNKNOWN_VALUE)
    ](
        cache_lengths_dev.unsafe_ptr(),
        RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(
            Index(batch_size)
        ),
    )

    # Define layouts for q tensor
    comptime q_static_layout = Layout.row_major(
        UNKNOWN_VALUE, UNKNOWN_VALUE, num_q_heads, Int(kv_params.head_size)
    )
    var q_shape = IndexList[4](
        batch_size, max_prompt_len, num_q_heads, Int(kv_params.head_size)
    )
    var q_runtime_layout = RuntimeLayout[q_static_layout].row_major(q_shape)

    # Create device buffers for q
    var q_device = ctx.enqueue_create_buffer[dtype](q_shape.flattened_length())

    # Initialize q with random data
    with q_device.map_to_host() as q_host:
        var q_host_tensor = LayoutTensor[dtype, q_static_layout](
            q_host, q_runtime_layout
        )
        random(q_host_tensor)

    var valid_length_device = ctx.enqueue_create_buffer[DType.uint32](
        batch_size
    )
    ctx.enqueue_copy(valid_length_device, valid_length.ptr)

    # Define layouts for mask tensor
    comptime mask_static_layout = Layout.row_major(
        UNKNOWN_VALUE, num_q_heads, UNKNOWN_VALUE, UNKNOWN_VALUE
    )
    var mask_shape = IndexList[4](
        batch_size, num_q_heads, max_prompt_len, max_context_len
    )
    var mask_runtime_layout = RuntimeLayout[mask_static_layout].row_major(
        mask_shape
    )

    # Create device buffers for masks
    var mask_device = ctx.enqueue_create_buffer[DType.float32](
        mask_shape.flattened_length()
    )
    var mask_device_mod = ctx.enqueue_create_buffer[DType.float32](
        mask_shape.flattened_length()
    )

    # Initialize causal mask
    with mask_device.map_to_host() as mask_host:
        var mask_host_tensor = LayoutTensor[DType.float32, mask_static_layout](
            mask_host, mask_runtime_layout
        )
        for b in range(batch_size):
            for h in range(num_q_heads):
                for q_idx in range(max_prompt_len):
                    for k_idx in range(max_context_len):
                        mask_host_tensor[b, h, q_idx, k_idx] = (
                            0 if UInt32(q_idx) + cache_valid_length[b]
                            >= UInt32(k_idx) else min_or_neg_inf[
                                DType.float32
                            ]()
                        )

    # Initialize causal mask with alibi bias
    with mask_device_mod.map_to_host() as mask_host_mod:
        var mask_host_mod_tensor = LayoutTensor[
            DType.float32, mask_static_layout
        ](mask_host_mod, mask_runtime_layout)
        for b in range(batch_size):
            for h in range(num_q_heads):
                for q_idx in range(max_prompt_len):
                    for k_idx in range(max_context_len):
                        mask_host_mod_tensor[b, h, q_idx, k_idx] = (
                            generate_alibi_bias[DType.float32, 1, num_q_heads](
                                Scalar[DType.int](h),
                                Scalar[DType.int](q_idx),
                                Scalar[DType.int](k_idx),
                                max_context_len,
                            ) if UInt32(q_idx)
                            + cache_valid_length[b]
                            >= UInt32(k_idx) else min_or_neg_inf[
                                DType.float32
                            ]()
                        )

    # Define layouts for output tensors
    comptime output_static_layout = Layout.row_major(
        UNKNOWN_VALUE, UNKNOWN_VALUE, num_q_heads, Int(kv_params.head_size)
    )
    var output_shape = IndexList[4](
        batch_size, max_prompt_len, num_q_heads, Int(kv_params.head_size)
    )
    var output_runtime_layout = RuntimeLayout[output_static_layout].row_major(
        output_shape
    )

    # Create device buffers for outputs
    var ref_output_device = ctx.enqueue_create_buffer[dtype](
        output_shape.flattened_length()
    )
    var test_output_device = ctx.enqueue_create_buffer[dtype](
        output_shape.flattened_length()
    )

    # Define layouts for kv_block tensor
    comptime kv_block_static_layout = Layout.row_major(
        UNKNOWN_VALUE,
        2,
        1,
        UNKNOWN_VALUE,
        Int(kv_params.num_heads),
        Int(kv_params.head_size),
    )
    var kv_block_shape = IndexList[6](
        num_blocks,
        2,
        1,
        max_seq_len,
        Int(kv_params.num_heads),
        Int(kv_params.head_size),
    )
    var kv_block_runtime_layout = RuntimeLayout[
        kv_block_static_layout
    ].row_major(kv_block_shape)

    var kv_block_device = ctx.enqueue_create_buffer[dtype](
        kv_block_shape.flattened_length()
    )

    # Initialize kv_block with random data
    with kv_block_device.map_to_host() as kv_block_host:
        var kv_block_host_tensor = LayoutTensor[dtype, kv_block_static_layout](
            kv_block_host, kv_block_runtime_layout
        )
        random(kv_block_host_tensor)

    # Create lookup table
    var lookup_table_device = ctx.enqueue_create_buffer[DType.uint32](
        batch_size
    )

    # Initialize lookup table
    with lookup_table_device.map_to_host() as lookup_table_host:
        # hacky way to select random blocks.
        var block_idx_set = Set[Int]()
        var idx = 0
        while idx < batch_size:
            var randval = Int(random_ui64(0, num_blocks - 1))
            if randval in block_idx_set:
                continue

            block_idx_set.add(randval)
            lookup_table_host[idx] = UInt32(randval)
            idx += 1

    # Create layout tensors for GPU operations
    var q_tensor = LayoutTensor[dtype, q_static_layout](
        q_device, q_runtime_layout
    )
    var valid_length_tensor = LayoutTensor[
        DType.uint32, Layout.row_major(UNKNOWN_VALUE)
    ](
        valid_length_device,
        RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(
            Index(batch_size)
        ),
    )
    var mask_tensor = LayoutTensor[DType.float32, mask_static_layout](
        mask_device, mask_runtime_layout
    )
    var mask_mod_tensor = LayoutTensor[DType.float32, mask_static_layout](
        mask_device_mod, mask_runtime_layout
    )
    var ref_output_tensor = LayoutTensor[dtype, output_static_layout](
        ref_output_device, output_runtime_layout
    )
    var test_output_tensor = LayoutTensor[dtype, output_static_layout](
        test_output_device, output_runtime_layout
    )
    var kv_block_tensor = LayoutTensor[dtype, kv_block_static_layout](
        kv_block_device, kv_block_runtime_layout
    )
    var lookup_table_tensor = LayoutTensor[
        DType.uint32, Layout(UNKNOWN_VALUE), ImmutAnyOrigin
    ](
        lookup_table_device.unsafe_ptr(),
        RuntimeLayout[Layout(UNKNOWN_VALUE)].row_major(Index(batch_size)),
    )

    kv_collection_device = CollectionType(
        LayoutTensor[dtype, Layout.row_major[6](), MutAnyOrigin](
            kv_block_tensor.ptr,
            RuntimeLayout[Layout.row_major[6]()](
                kv_block_tensor.runtime_layout.shape.value,
                kv_block_tensor.runtime_layout.stride.value,
            ),
        ),
        cache_lengths,
        lookup_table_tensor,
        UInt32(max_prompt_len),
        UInt32(max_context_len),
    )

    k_cache_device = kv_collection_device.get_key_cache(0)
    v_cache_device = kv_collection_device.get_value_cache(0)

    flash_attention[use_score_mod=True](
        test_output_tensor,
        q_tensor,
        k_cache_device,
        v_cache_device,
        CausalMask(),
        AlibiScoreMod[num_q_heads](),
        valid_length_tensor,
        rsqrt(Float32(kv_params.head_size)),
        ctx,
    )

    # Here pass mask that includes bias in q_idx >= k_idx (to compare).
    mha_gpu_naive(
        q_tensor,
        k_cache_device,
        v_cache_device,
        MaterializedMask(
            LayoutTensor[DType.float32, mask_static_layout, ImmutAnyOrigin](
                mask_mod_tensor.ptr,
                RuntimeLayout[mask_static_layout].row_major(
                    mask_mod_tensor.runtime_layout.shape.value.canonicalize()
                ),
            ),
            start_pos=cache_lengths_lt.get_immutable(),
        ),
        ref_output_tensor,
        valid_length_tensor,
        rsqrt(Float32(kv_params.head_size)),
        batch_size,
        max_prompt_len,
        max_context_len,
        num_q_heads,
        Int(kv_params.head_size),
        num_q_heads // Int(kv_params.num_heads),
        ctx,
    )

    ctx.synchronize()

    # Verify results
    with test_output_device.map_to_host() as test_out_host:
        with ref_output_device.map_to_host() as ref_out_host:
            var test_out_tensor = LayoutTensor[dtype, output_static_layout](
                test_out_host, output_runtime_layout
            )
            var ref_out_tensor = LayoutTensor[dtype, output_static_layout](
                ref_out_host, output_runtime_layout
            )
            for bs in range(batch_size):
                for s in range(valid_length[bs]):
                    for h in range(num_q_heads):
                        for hd in range(kv_params.head_size):
                            var expect = ref_out_tensor[bs, s, h, Int(hd)]
                            var actual = test_out_tensor[bs, s, h, Int(hd)]
                            assert_almost_equal(
                                expect,
                                actual,
                                atol=1e-5,
                                rtol=8e-3,
                            )


def execute_flash_attention_suite(ctx: DeviceContext):
    var bs = 2
    var valid_length_ptr = UnsafePointer[UInt32].alloc(bs)
    var valid_length = LayoutTensor[DType.uint32, Layout(UNKNOWN_VALUE)](
        valid_length_ptr,
        RuntimeLayout[Layout(UNKNOWN_VALUE)].row_major(Index(bs)),
    )

    var cache_valid_length_ptr = UnsafePointer[UInt32].alloc(bs)
    var cache_valid_length = LayoutTensor[DType.uint32, Layout(UNKNOWN_VALUE)](
        cache_valid_length_ptr,
        RuntimeLayout[Layout(UNKNOWN_VALUE)].row_major(Index(bs)),
    )

    comptime dtype = DType.bfloat16

    # Replit & Llama3 context encoding [testing even query valid lengths].
    valid_length[0] = 128
    valid_length[1] = 64
    cache_valid_length[0] = 0
    cache_valid_length[1] = 0

    execute_flash_attention[
        replit_num_q_heads,
        dtype,
        kv_params_replit,
    ](bs, valid_length, 1024, cache_valid_length, ctx)

    execute_flash_attention[
        llama_num_q_heads,
        dtype,
        kv_params_llama3,
    ](bs, valid_length, 1024, cache_valid_length, ctx)

    # Replit & Llama3 context encoding [testing odd query valid length].
    valid_length[0] = 128
    valid_length[1] = 65
    cache_valid_length[0] = 0
    cache_valid_length[1] = 0

    execute_flash_attention[
        replit_num_q_heads,
        dtype,
        kv_params_replit,
    ](bs, valid_length, 1024, cache_valid_length, ctx)

    execute_flash_attention[
        llama_num_q_heads,
        dtype,
        kv_params_llama3,
    ](bs, valid_length, 1024, cache_valid_length, ctx)

    # Replit & Llama3 token gen [testing even cache valid lengths].
    valid_length[0] = 1
    valid_length[1] = 1
    cache_valid_length[0] = 200
    cache_valid_length[1] = 256

    execute_flash_attention[
        replit_num_q_heads,
        dtype,
        kv_params_replit,
    ](bs, valid_length, 1024, cache_valid_length, ctx)

    execute_flash_attention[
        llama_num_q_heads,
        dtype,
        kv_params_llama3,
    ](bs, valid_length, 1024, cache_valid_length, ctx)

    # Replit & Llama3 token gen [testing even cache valid lengths].
    valid_length[0] = 1
    valid_length[1] = 1
    cache_valid_length[0] = 200
    cache_valid_length[1] = 255

    execute_flash_attention[
        replit_num_q_heads,
        dtype,
        kv_params_replit,
    ](bs, valid_length, 1024, cache_valid_length, ctx)

    execute_flash_attention[
        llama_num_q_heads,
        dtype,
        kv_params_llama3,
    ](bs, valid_length, 1024, cache_valid_length, ctx)


def main():
    seed(42)
    with DeviceContext() as ctx:
        execute_flash_attention_suite(ctx)

    print("Success!")
