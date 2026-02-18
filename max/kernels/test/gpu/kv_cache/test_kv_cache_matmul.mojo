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

from collections import Set
from random import random_ui64, seed

from buffer import Dim, DimList, NDBuffer
from gpu.host import DeviceContext
from kv_cache.types import (
    ContinuousBatchingKVCacheCollection,
    KVCacheStaticParams,
)
from layout import LayoutTensor, Layout, RuntimeLayout, UNKNOWN_VALUE
from layout._fillers import random
from linalg.matmul.gpu import _matmul_gpu
from memory import LegacyUnsafePointer

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]
from nn.kv_cache import _fused_qkv_matmul_kv_cache_impl
from testing import assert_almost_equal

from utils import IndexList

comptime kv_params_replit = KVCacheStaticParams(num_heads=8, head_size=128)
comptime replit_num_q_heads = 24

comptime kv_params_llama3 = KVCacheStaticParams(num_heads=8, head_size=128)
comptime llama_num_q_heads = 32


def execute_fused_qkv_matmul[
    num_q_heads: Int, dtype: DType, kv_params: KVCacheStaticParams
](
    batch_size: Int,
    prompt_len: Int,
    max_seq_len: Int,
    cache_sizes: List[Int],
    num_layers: Int,
    layer_idx: Int,
    ctx: DeviceContext,
):
    comptime hidden_size = num_q_heads * Int(kv_params.head_size)
    comptime kv_hidden_size = kv_params.num_heads * kv_params.head_size
    comptime fused_hidden_size = (2 * Int(kv_hidden_size)) + hidden_size
    comptime num_blocks = 32
    comptime CollectionType = ContinuousBatchingKVCacheCollection[
        dtype, kv_params
    ]

    debug_assert(
        batch_size < num_blocks,
        "batch_size passed to unit test (",
        batch_size,
        ") is larger than configured max_batch_size (",
        num_blocks,
        ")",
    )

    # Compute sizes
    var hidden_state_size = batch_size * prompt_len * hidden_size
    var weight_size = fused_hidden_size * hidden_size
    var ref_output_size = batch_size * prompt_len * fused_hidden_size
    var test_output_size = batch_size * prompt_len * hidden_size
    var kv_block_size = (
        num_blocks
        * 2
        * num_layers
        * max_seq_len
        * Int(kv_params.num_heads)
        * Int(kv_params.head_size)
    )

    # Define layouts
    comptime hidden_state_layout = Layout.row_major(
        UNKNOWN_VALUE, UNKNOWN_VALUE, hidden_size
    )
    comptime weight_layout = Layout.row_major(fused_hidden_size, hidden_size)
    comptime ref_output_layout = Layout.row_major[2]()
    comptime test_output_layout = Layout.row_major[3]()
    comptime kv_block_layout = Layout.row_major[6]()
    comptime cache_len_layout = Layout(UNKNOWN_VALUE)

    # Define shapes
    var hidden_state_shape = IndexList[3](batch_size, prompt_len, hidden_size)
    var weight_shape = IndexList[2](fused_hidden_size, hidden_size)
    var ref_output_shape = IndexList[2](
        batch_size * prompt_len, fused_hidden_size
    )
    var test_output_shape = IndexList[3](batch_size, prompt_len, hidden_size)
    var kv_block_shape = IndexList[6](
        num_blocks,
        2,
        num_layers,
        max_seq_len,
        Int(kv_params.num_heads),
        Int(kv_params.head_size),
    )

    # Initialize hidden state
    var hidden_state_host_ptr = UnsafePointer[Scalar[dtype]].alloc(
        hidden_state_size
    )
    var hidden_state_host = LayoutTensor[dtype, hidden_state_layout](
        hidden_state_host_ptr,
        RuntimeLayout[hidden_state_layout].row_major(hidden_state_shape),
    )
    random(hidden_state_host)

    var hidden_state_device = ctx.enqueue_create_buffer[dtype](
        hidden_state_size
    )
    ctx.enqueue_copy(hidden_state_device, hidden_state_host_ptr)
    var hidden_state_device_2d = NDBuffer[
        dtype, 2, MutAnyOrigin, DimList(Dim(), hidden_size)
    ](
        hidden_state_device.unsafe_ptr(),
        IndexList[2](batch_size * prompt_len, hidden_size),
    )

    # Initialize the weights
    var weight_host_ptr = UnsafePointer[Scalar[dtype]].alloc(weight_size)
    var weight_host = LayoutTensor[dtype, weight_layout](
        weight_host_ptr,
        RuntimeLayout[weight_layout].row_major(weight_shape),
    )
    random(weight_host)

    var weight_device = ctx.enqueue_create_buffer[dtype](weight_size)
    ctx.enqueue_copy(weight_device, weight_host_ptr)

    # Initialize reference output
    var ref_output_host_ptr = UnsafePointer[Scalar[dtype]].alloc(
        ref_output_size
    )
    var ref_output_host = LayoutTensor[dtype, ref_output_layout](
        ref_output_host_ptr,
        RuntimeLayout[ref_output_layout].row_major(ref_output_shape),
    )
    var ref_output_device = ctx.enqueue_create_buffer[dtype](ref_output_size)

    # Initialize test output
    var test_output_host_ptr = UnsafePointer[Scalar[dtype]].alloc(
        test_output_size
    )
    var test_output_host = LayoutTensor[dtype, test_output_layout](
        test_output_host_ptr,
        RuntimeLayout[test_output_layout].row_major(test_output_shape),
    )
    var test_output_device = ctx.enqueue_create_buffer[dtype](test_output_size)

    # Initialize our KVCache
    var is_context_encoding = True
    var cache_lengths_host_ptr = UnsafePointer[Scalar[DType.uint32]].alloc(
        batch_size
    )
    for i in range(batch_size):
        cache_lengths_host_ptr[i] = UInt32(cache_sizes[i])
        if cache_lengths_host_ptr[i] != 0:
            is_context_encoding = False

    var cache_lengths_dev = ctx.enqueue_create_buffer[DType.uint32](batch_size)
    ctx.enqueue_copy(cache_lengths_dev, cache_lengths_host_ptr)

    var kv_block_host_ptr = UnsafePointer[Scalar[dtype]].alloc(kv_block_size)
    var kv_block_host = LayoutTensor[dtype, kv_block_layout](
        kv_block_host_ptr,
        RuntimeLayout[kv_block_layout].row_major(kv_block_shape),
    )
    var kv_block_device = ctx.enqueue_create_buffer[dtype](kv_block_size)

    var lookup_table_host_ptr = UnsafePointer[Scalar[DType.uint32]].alloc(
        batch_size
    )
    var lookup_table_device = ctx.enqueue_create_buffer[DType.uint32](
        batch_size
    )

    # Hacky way to get random block indices
    var block_idx_set = Set[Int]()
    var idx = 0
    while len(block_idx_set) < batch_size:
        var randval = Int(random_ui64(0, num_blocks - 1))
        if randval in block_idx_set:
            continue
        block_idx_set.add(randval)
        lookup_table_host_ptr[idx] = UInt32(randval)
        idx += 1

    ctx.enqueue_copy(lookup_table_device, lookup_table_host_ptr)

    # Create runtime layouts for KV collection
    var kv_block_runtime = RuntimeLayout[kv_block_layout].row_major(
        kv_block_shape
    )
    var cache_len_runtime = RuntimeLayout[cache_len_layout].row_major(
        IndexList[1](batch_size)
    )

    var kv_collection_device = CollectionType(
        LayoutTensor[dtype, kv_block_layout, MutAnyOrigin](
            kv_block_device.unsafe_ptr(),
            kv_block_runtime,
        ),
        LayoutTensor[DType.uint32, cache_len_layout, ImmutAnyOrigin](
            cache_lengths_dev.unsafe_ptr(),
            cache_len_runtime,
        ),
        LayoutTensor[DType.uint32, cache_len_layout, ImmutAnyOrigin](
            lookup_table_device.unsafe_ptr(),
            cache_len_runtime,
        ),
        UInt32(max_seq_len),
        UInt32(0 if is_context_encoding else max_seq_len),
    )
    var kv_collection_host = CollectionType(
        LayoutTensor[dtype, kv_block_layout, MutAnyOrigin](
            kv_block_host_ptr,
            kv_block_runtime,
        ),
        LayoutTensor[DType.uint32, cache_len_layout, ImmutAnyOrigin](
            cache_lengths_host_ptr,
            cache_len_runtime,
        ),
        LayoutTensor[DType.uint32, cache_len_layout, ImmutAnyOrigin](
            lookup_table_host_ptr,
            cache_len_runtime,
        ),
        UInt32(max_seq_len),
        UInt32(0 if is_context_encoding else max_seq_len),
    )

    # Create device tensors for kernel calls
    var hidden_state_device_tensor = LayoutTensor[
        dtype, hidden_state_layout, MutAnyOrigin
    ](
        hidden_state_device.unsafe_ptr(),
        RuntimeLayout[hidden_state_layout].row_major(hidden_state_shape),
    )
    var weight_device_tensor = LayoutTensor[dtype, weight_layout, MutAnyOrigin](
        weight_device.unsafe_ptr(),
        RuntimeLayout[weight_layout].row_major(weight_shape),
    )
    var test_output_device_tensor = LayoutTensor[
        dtype, test_output_layout, MutAnyOrigin
    ](
        test_output_device.unsafe_ptr(),
        RuntimeLayout[test_output_layout].row_major(test_output_shape),
    )

    # Create valid_lengths - all sequences have full prompt_len valid
    var valid_lengths_host_ptr = UnsafePointer[Scalar[DType.uint32]].alloc(
        batch_size
    )
    for i in range(batch_size):
        valid_lengths_host_ptr[i] = UInt32(prompt_len)

    var valid_lengths_device = ctx.enqueue_create_buffer[DType.uint32](
        batch_size
    )
    ctx.enqueue_copy(valid_lengths_device, valid_lengths_host_ptr)

    var valid_lengths_tensor = LayoutTensor[
        DType.uint32, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin
    ](
        valid_lengths_device.unsafe_ptr(),
        RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(
            IndexList[1](batch_size)
        ),
    )

    _fused_qkv_matmul_kv_cache_impl[target="gpu"](
        hidden_state_device_tensor,
        weight_device_tensor,
        kv_collection_device,
        UInt32(layer_idx),
        valid_lengths_tensor,
        test_output_device_tensor,
        ctx,
    )

    var ref_output_device_ndbuffer = NDBuffer[
        dtype, 2, MutAnyOrigin, DimList(Dim(), fused_hidden_size)
    ](
        ref_output_device.unsafe_ptr(),
        ref_output_shape,
    )
    var weight_device_ndbuffer = NDBuffer[
        dtype, 2, MutAnyOrigin, DimList(fused_hidden_size, hidden_size)
    ](
        weight_device.unsafe_ptr(),
        weight_shape,
    )

    _matmul_gpu[use_tensor_core=True, transpose_b=True](
        ref_output_device_ndbuffer,
        hidden_state_device_2d,
        weight_device_ndbuffer,
        ctx,
    )

    ctx.enqueue_copy(kv_block_host_ptr, kv_block_device)
    ctx.enqueue_copy(test_output_host_ptr, test_output_device)
    ctx.enqueue_copy(ref_output_host_ptr, ref_output_device)
    ctx.synchronize()

    k_cache_host = kv_collection_host.get_key_cache(layer_idx)
    v_cache_host = kv_collection_host.get_value_cache(layer_idx)
    for bs in range(batch_size):
        for s in range(prompt_len):
            for q_dim in range(hidden_size):
                assert_almost_equal(
                    ref_output_host[bs * prompt_len + s, q_dim],
                    test_output_host[bs, s, q_dim],
                )

            for k_dim in range(kv_hidden_size):
                head_idx = k_dim // kv_params.head_size
                head_dim_idx = k_dim % kv_params.head_size
                assert_almost_equal(
                    ref_output_host[
                        bs * prompt_len + s, hidden_size + Int(k_dim)
                    ],
                    k_cache_host.load[width=1](
                        bs,
                        Int(head_idx),
                        cache_sizes[bs] + s,
                        Int(head_dim_idx),
                    ),
                )

            for v_dim in range(kv_hidden_size):
                head_idx = v_dim // kv_params.head_size
                head_dim_idx = v_dim % kv_params.head_size
                assert_almost_equal(
                    ref_output_host[
                        bs * prompt_len + s,
                        hidden_size + Int(kv_hidden_size + v_dim),
                    ],
                    v_cache_host.load[width=1](
                        bs,
                        Int(head_idx),
                        cache_sizes[bs] + s,
                        Int(head_dim_idx),
                    ),
                )

    # Cleanup host memory
    hidden_state_host_ptr.free()
    weight_host_ptr.free()
    ref_output_host_ptr.free()
    test_output_host_ptr.free()
    cache_lengths_host_ptr.free()
    kv_block_host_ptr.free()
    lookup_table_host_ptr.free()
    valid_lengths_host_ptr.free()

    # Cleanup device buffers
    _ = hidden_state_device^
    _ = weight_device^
    _ = ref_output_device^
    _ = test_output_device^
    _ = cache_lengths_dev^
    _ = kv_block_device^
    _ = lookup_table_device^
    _ = valid_lengths_device^


def execute_fused_matmul_suite(ctx: DeviceContext):
    comptime dtypes = (DType.float32, DType.bfloat16)

    @parameter
    for dtype_idx in range(2):
        comptime dtype = dtypes[dtype_idx]
        for bs in [1, 16]:
            ce_cache_sizes = List[Int]()
            tg_cache_sizes = List[Int]()
            for _ in range(bs):
                tg_cache_sizes.append(Int(random_ui64(0, 100)))
                ce_cache_sizes.append(0)

            # llama3 context encoding
            execute_fused_qkv_matmul[
                llama_num_q_heads, dtype, kv_params_llama3
            ](bs, 128, 1024, ce_cache_sizes, 4, 1, ctx)

            execute_fused_qkv_matmul[
                llama_num_q_heads, dtype, kv_params_llama3
            ](bs, 512, 1024, ce_cache_sizes, 4, 0, ctx)

            # llama3 token gen
            execute_fused_qkv_matmul[
                llama_num_q_heads, dtype, kv_params_llama3
            ](bs, 1, 1024, tg_cache_sizes, 4, 3, ctx)

            execute_fused_qkv_matmul[
                llama_num_q_heads, dtype, kv_params_llama3
            ](bs, 1, 1024, tg_cache_sizes, 4, 0, ctx)


def main():
    seed(42)
    with DeviceContext() as ctx:
        execute_fused_matmul_suite(ctx)
