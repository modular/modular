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
from math import ceildiv
from random import random_ui64, seed

from buffer import NDBuffer
from gpu.host import DeviceBuffer, DeviceContext
from kv_cache.types import (
    KVCacheStaticParams,
    PagedKVCacheCollection,
)
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from layout._fillers import random
from linalg.fp8_quantization import naive_blockwise_scaled_fp8_matmul
from memory import memcpy, LegacyUnsafePointer as UnsafePointer
from nn.kv_cache_ragged import (
    _matmul_k_cache_ragged_scale_impl,
)
from testing import assert_almost_equal

from utils import IndexList

comptime kv_params_llama3 = KVCacheStaticParams(num_heads=8, head_size=128)
comptime llama_num_q_heads = 32


comptime block_scale = 128


def _initialize_ragged_inputs[
    dtype: DType, hidden_size: Int
](
    input_row_offsets_host_ptr: UnsafePointer[Scalar[DType.uint32]],
    batch_size: Int,
    prompt_lens: List[Int],
    ctx: DeviceContext,
) -> Tuple[
    DeviceBuffer[DType.uint32],
    DeviceBuffer[dtype],
    DeviceBuffer[dtype],
    Int,  # total_length
    Int,  # max_seq_length_batch
]:
    """Initializes input row offsets and hidden state ragged tensor inputs."""
    var total_length = 0
    var max_seq_length_batch = -1
    for i in range(batch_size):
        input_row_offsets_host_ptr[i] = total_length

        var curr_len = prompt_lens[i]
        total_length += curr_len
        if curr_len > max_seq_length_batch:
            max_seq_length_batch = curr_len

    input_row_offsets_host_ptr[batch_size] = total_length
    var input_row_offsets_device = ctx.enqueue_create_buffer[DType.uint32](
        batch_size + 1
    )
    ctx.enqueue_copy(input_row_offsets_device, input_row_offsets_host_ptr)

    # Initialize ragged hidden state.
    comptime hidden_state_layout = Layout.row_major(UNKNOWN_VALUE, hidden_size)
    var ragged_size = total_length * hidden_size
    var hidden_state_ragged_host_ptr = UnsafePointer[Scalar[dtype]].alloc(
        ragged_size
    )
    var hidden_state_ragged_host = LayoutTensor[dtype, hidden_state_layout](
        hidden_state_ragged_host_ptr,
        RuntimeLayout[hidden_state_layout].row_major(
            IndexList[2](total_length, hidden_size)
        ),
    )
    random(hidden_state_ragged_host)

    var hidden_state_ragged_device = ctx.enqueue_create_buffer[dtype](
        ragged_size
    )
    ctx.enqueue_copy(hidden_state_ragged_device, hidden_state_ragged_host_ptr)

    # Initialize padded hidden state.
    var padded_size = batch_size * max_seq_length_batch * hidden_size
    var hidden_state_padded_host_ptr = UnsafePointer[Scalar[dtype]].alloc(
        padded_size
    )

    # Copy over the ragged values to the padded tensor.
    # Don't worry about padded values, we won't read them.
    for bs in range(batch_size):
        var unpadded_seq_len = prompt_lens[bs]
        var ragged_start_idx = Int(input_row_offsets_host_ptr[bs])
        for s in range(unpadded_seq_len):
            var padded_ptr = (
                hidden_state_padded_host_ptr
                + (bs * max_seq_length_batch + s) * hidden_size
            )
            var ragged_ptr = (
                hidden_state_ragged_host_ptr
                + (ragged_start_idx + s) * hidden_size
            )
            memcpy(dest=padded_ptr, src=ragged_ptr, count=hidden_size)

    var hidden_state_padded_device = ctx.enqueue_create_buffer[dtype](
        padded_size
    )
    ctx.enqueue_copy(hidden_state_padded_device, hidden_state_padded_host_ptr)

    # Sync here so that HtoD transfers complete prior to host buffer dtor.
    ctx.synchronize()

    hidden_state_ragged_host_ptr.free()
    hidden_state_padded_host_ptr.free()

    return (
        input_row_offsets_device,
        hidden_state_ragged_device,
        hidden_state_padded_device,
        total_length,
        max_seq_length_batch,
    )


def execute_matmul_k_cache_ragged_scale[
    num_q_heads: Int,
    dtype: DType,
    weight_dtype: DType,
    scale_dtype: DType,
    kv_params: KVCacheStaticParams,
    rtol: Float64,
    atol: Float64,
](
    prompt_lens: List[Int],
    max_seq_length_cache: Int,
    cache_sizes: List[Int],
    num_layers: Int,
    layer_idx: Int,
    ctx: DeviceContext,
):
    """Tests the scaled KV cache matmul for key projections.

    This test follows the same pattern as execute_matmul_k_cache_ragged but
    includes input_scale and weight_scale parameters for scaled FP8 operations.
    """
    comptime hidden_size = num_q_heads * Int(kv_params.head_size)
    comptime kv_hidden_size = kv_params.num_heads * kv_params.head_size

    comptime num_paged_blocks = 32
    comptime page_size = 512
    comptime CollectionType = PagedKVCacheCollection[
        dtype, kv_params, page_size
    ]
    comptime layout_1d = Layout(UNKNOWN_VALUE)
    comptime kv_block_layout = Layout.row_major[6]()
    comptime paged_lut_layout = Layout.row_major[2]()
    comptime hidden_state_layout = Layout.row_major(UNKNOWN_VALUE, hidden_size)
    comptime weight_layout = Layout.row_major(Int(kv_hidden_size), hidden_size)
    comptime input_scale_rows = Int(ceildiv(hidden_size, block_scale))
    comptime weight_scale_rows = Int(ceildiv(kv_hidden_size, block_scale))
    comptime weight_scale_cols = Int(ceildiv(hidden_size, block_scale))
    comptime input_scale_layout = Layout.row_major(
        input_scale_rows, UNKNOWN_VALUE
    )
    comptime weight_scale_layout = Layout.row_major(
        weight_scale_rows, weight_scale_cols
    )
    comptime ref_output_layout = Layout.row_major(
        UNKNOWN_VALUE, Int(kv_hidden_size)
    )

    var batch_size = len(prompt_lens)

    debug_assert(
        len(prompt_lens) == len(cache_sizes),
        "expected prompt_lens and cache_sizes size to be equal",
    )

    var cache_lengths_host_ptr = UnsafePointer[Scalar[DType.uint32]].alloc(
        batch_size
    )

    var kv_block_size = (
        num_paged_blocks
        * 2
        * num_layers
        * page_size
        * Int(kv_params.num_heads)
        * Int(kv_params.head_size)
    )
    var kv_block_shape = IndexList[6](
        num_paged_blocks,
        2,
        num_layers,
        page_size,
        Int(kv_params.num_heads),
        Int(kv_params.head_size),
    )
    var kv_block_host_ptr = UnsafePointer[Scalar[dtype]].alloc(kv_block_size)

    var total_length = 0
    var max_full_context_length = 0
    var max_seq_length_batch = 0
    for i in range(batch_size):
        cache_lengths_host_ptr[i] = cache_sizes[i]
        max_full_context_length = max(
            max_full_context_length, Int(cache_sizes[i] + prompt_lens[i])
        )
        max_seq_length_batch = max(max_seq_length_batch, prompt_lens[i])
        total_length += prompt_lens[i]

    var cache_lengths_device = ctx.enqueue_create_buffer[DType.uint32](
        batch_size
    )
    ctx.enqueue_copy(cache_lengths_device, cache_lengths_host_ptr)

    var paged_lut_cols = ceildiv(max_full_context_length, page_size)
    var paged_lut_size = batch_size * paged_lut_cols
    var paged_lut_shape = IndexList[2](batch_size, paged_lut_cols)
    var paged_lut_host_ptr = UnsafePointer[Scalar[DType.uint32]].alloc(
        paged_lut_size
    )
    var paged_lut_host = LayoutTensor[DType.uint32, paged_lut_layout](
        paged_lut_host_ptr,
        RuntimeLayout[paged_lut_layout].row_major(paged_lut_shape),
    )

    var paged_lut_set = Set[Int]()
    for bs in range(batch_size):
        var seq_len = cache_sizes[bs] + prompt_lens[bs]

        for block_idx in range(0, ceildiv(seq_len, page_size)):
            var randval = Int(random_ui64(0, num_paged_blocks - 1))
            while randval in paged_lut_set:
                randval = Int(random_ui64(0, num_paged_blocks - 1))

            paged_lut_set.add(randval)
            paged_lut_host[bs, block_idx] = randval

    var paged_lut_device = ctx.enqueue_create_buffer[DType.uint32](
        paged_lut_size
    )
    ctx.enqueue_copy(paged_lut_device, paged_lut_host_ptr)
    var kv_block_device = ctx.enqueue_create_buffer[dtype](kv_block_size)

    # Create runtime layouts
    var kv_block_runtime = RuntimeLayout[kv_block_layout].row_major(
        kv_block_shape
    )
    var cache_len_runtime = RuntimeLayout[layout_1d].row_major(
        IndexList[1](batch_size)
    )
    var paged_lut_runtime = RuntimeLayout[paged_lut_layout].row_major(
        paged_lut_shape
    )

    var kv_collection_device = CollectionType(
        LayoutTensor[dtype, kv_block_layout, MutAnyOrigin](
            kv_block_device.unsafe_ptr(),
            kv_block_runtime,
        ),
        LayoutTensor[DType.uint32, layout_1d, ImmutAnyOrigin](
            cache_lengths_device.unsafe_ptr(),
            cache_len_runtime,
        ),
        LayoutTensor[DType.uint32, paged_lut_layout, ImmutAnyOrigin](
            paged_lut_device.unsafe_ptr(),
            paged_lut_runtime,
        ),
        max_seq_length_batch,
        max_full_context_length,
    )

    var k_cache_device = kv_collection_device.get_key_cache(layer_idx)

    var kv_collection_host = CollectionType(
        LayoutTensor[dtype, kv_block_layout, MutAnyOrigin](
            kv_block_host_ptr,
            kv_block_runtime,
        ),
        LayoutTensor[DType.uint32, layout_1d, ImmutAnyOrigin](
            cache_lengths_host_ptr,
            cache_len_runtime,
        ),
        LayoutTensor[DType.uint32, paged_lut_layout, ImmutAnyOrigin](
            paged_lut_host_ptr,
            paged_lut_runtime,
        ),
        max_seq_length_batch,
        max_full_context_length,
    )

    var k_cache_host = kv_collection_host.get_key_cache(layer_idx)

    # Initialize input row offsets and hidden states.
    var input_row_offsets_host_ptr = UnsafePointer[Scalar[DType.uint32]].alloc(
        batch_size + 1
    )
    var init_result = _initialize_ragged_inputs[weight_dtype, hidden_size](
        input_row_offsets_host_ptr, batch_size, prompt_lens, ctx
    )
    var input_row_offsets_device = init_result[0]
    var hidden_state_ragged_device = init_result[1]
    var hidden_state_padded_device = init_result[2]
    var ragged_total_length = init_result[3]
    var init_max_seq_length_batch = init_result[4]

    # Initialize the weights.
    var weight_size = Int(kv_hidden_size) * hidden_size
    var weight_shape = IndexList[2](Int(kv_hidden_size), hidden_size)
    var weight_host_ptr = UnsafePointer[Scalar[weight_dtype]].alloc(weight_size)
    var weight_host = LayoutTensor[weight_dtype, weight_layout](
        weight_host_ptr,
        RuntimeLayout[weight_layout].row_major(weight_shape),
    )
    random(weight_host)

    var weight_device = ctx.enqueue_create_buffer[weight_dtype](weight_size)
    ctx.enqueue_copy(weight_device, weight_host_ptr)

    # Initialize scales for blockwise scaling.
    var input_scale_cols = ragged_total_length
    var input_scale_size = input_scale_rows * input_scale_cols
    var input_scale_shape = IndexList[2](input_scale_rows, input_scale_cols)
    var weight_scale_size = weight_scale_rows * weight_scale_cols
    var weight_scale_shape = IndexList[2](weight_scale_rows, weight_scale_cols)

    var input_scale_host_ptr = UnsafePointer[Scalar[scale_dtype]].alloc(
        input_scale_size
    )
    var input_scale_host = LayoutTensor[scale_dtype, input_scale_layout](
        input_scale_host_ptr,
        RuntimeLayout[input_scale_layout].row_major(input_scale_shape),
    )
    var weight_scale_host_ptr = UnsafePointer[Scalar[scale_dtype]].alloc(
        weight_scale_size
    )
    var weight_scale_host = LayoutTensor[scale_dtype, weight_scale_layout](
        weight_scale_host_ptr,
        RuntimeLayout[weight_scale_layout].row_major(weight_scale_shape),
    )

    random(input_scale_host)
    random(weight_scale_host)

    var input_scale_device = ctx.enqueue_create_buffer[scale_dtype](
        input_scale_size
    )
    ctx.enqueue_copy(input_scale_device, input_scale_host_ptr)
    var weight_scale_device = ctx.enqueue_create_buffer[scale_dtype](
        weight_scale_size
    )
    ctx.enqueue_copy(weight_scale_device, weight_scale_host_ptr)

    # Initialize reference output.
    var ref_output_size = ragged_total_length * Int(kv_hidden_size)
    var ref_output_shape = IndexList[2](
        ragged_total_length, Int(kv_hidden_size)
    )
    var ref_output_host_ptr = UnsafePointer[Scalar[dtype]].alloc(
        ref_output_size
    )
    var ref_output_host = LayoutTensor[dtype, ref_output_layout](
        ref_output_host_ptr,
        RuntimeLayout[ref_output_layout].row_major(ref_output_shape),
    )
    var ref_output_device = ctx.enqueue_create_buffer[dtype](ref_output_size)

    # Create device LayoutTensors for kernel calls
    var hidden_state_ragged_tensor = LayoutTensor[
        weight_dtype, hidden_state_layout, MutAnyOrigin
    ](
        hidden_state_ragged_device.unsafe_ptr(),
        RuntimeLayout[hidden_state_layout].row_major(
            IndexList[2](ragged_total_length, hidden_size)
        ),
    )
    var input_row_offsets_tensor = LayoutTensor[
        DType.uint32, layout_1d, ImmutAnyOrigin
    ](
        input_row_offsets_device.unsafe_ptr(),
        RuntimeLayout[layout_1d].row_major(IndexList[1](batch_size + 1)),
    )
    var weight_device_tensor = LayoutTensor[
        weight_dtype, weight_layout, MutAnyOrigin
    ](
        weight_device.unsafe_ptr(),
        RuntimeLayout[weight_layout].row_major(weight_shape),
    )
    var input_scale_device_tensor = LayoutTensor[
        scale_dtype, input_scale_layout, MutAnyOrigin
    ](
        input_scale_device.unsafe_ptr(),
        RuntimeLayout[input_scale_layout].row_major(input_scale_shape),
    )
    var weight_scale_device_tensor = LayoutTensor[
        scale_dtype, weight_scale_layout, MutAnyOrigin
    ](
        weight_scale_device.unsafe_ptr(),
        RuntimeLayout[weight_scale_layout].row_major(weight_scale_shape),
    )

    # Execute test with scaled implementation.
    _matmul_k_cache_ragged_scale_impl[
        target="gpu",
        scales_granularity_mnk = IndexList[3](1, block_scale, block_scale),
    ](
        hidden_state_ragged_tensor,
        input_row_offsets_tensor,
        weight_device_tensor,
        input_scale_device_tensor,
        weight_scale_device_tensor,
        k_cache_device,
        ctx,
    )

    # Execute reference using naive blockwise scaled matmul.
    # Create weight_ref buffer and copy weight data
    var weight_ref_host_ptr = UnsafePointer[Scalar[weight_dtype]].alloc(
        weight_size
    )
    var weight_ref_device = ctx.enqueue_create_buffer[weight_dtype](weight_size)

    ctx.enqueue_copy(weight_ref_host_ptr, weight_device)
    ctx.synchronize()
    ctx.enqueue_copy(weight_ref_device, weight_ref_host_ptr)

    # Create scale tensors for reference computation
    var ref_input_scale_host_ptr = UnsafePointer[Scalar[scale_dtype]].alloc(
        input_scale_size
    )
    var ref_weight_scale_host_ptr = UnsafePointer[Scalar[scale_dtype]].alloc(
        weight_scale_size
    )

    # Fill with the same scale values
    for i in range(input_scale_rows):
        for j in range(input_scale_cols):
            ref_input_scale_host_ptr[
                i * input_scale_cols + j
            ] = input_scale_host_ptr[i * input_scale_cols + j]
    for i in range(weight_scale_rows):
        for j in range(weight_scale_cols):
            ref_weight_scale_host_ptr[
                i * weight_scale_cols + j
            ] = weight_scale_host_ptr[i * weight_scale_cols + j]

    var ref_input_scale_device = ctx.enqueue_create_buffer[scale_dtype](
        input_scale_size
    )
    ctx.enqueue_copy(ref_input_scale_device, ref_input_scale_host_ptr)
    var ref_weight_scale_device = ctx.enqueue_create_buffer[scale_dtype](
        weight_scale_size
    )
    ctx.enqueue_copy(ref_weight_scale_device, ref_weight_scale_host_ptr)

    # Create NDBuffers for naive_blockwise_scaled_fp8_matmul
    var ref_output_ndbuffer = NDBuffer[dtype, 2](
        ref_output_device.unsafe_ptr(),
        ref_output_shape,
    )
    var hidden_state_ragged_ndbuffer = NDBuffer[weight_dtype, 2](
        hidden_state_ragged_device.unsafe_ptr(),
        IndexList[2](ragged_total_length, hidden_size),
    )
    var weight_ref_ndbuffer = NDBuffer[weight_dtype, 2](
        weight_ref_device.unsafe_ptr(),
        weight_shape,
    )
    var ref_input_scale_ndbuffer = NDBuffer[scale_dtype, 2](
        ref_input_scale_device.unsafe_ptr(),
        input_scale_shape,
    )
    var ref_weight_scale_ndbuffer = NDBuffer[scale_dtype, 2](
        ref_weight_scale_device.unsafe_ptr(),
        weight_scale_shape,
    )

    # Use naive blockwise scaled matmul as reference
    naive_blockwise_scaled_fp8_matmul[
        BLOCK_DIM=16,
        transpose_b=True,
        scales_granularity_mnk = IndexList[3](1, block_scale, block_scale),
    ](
        ref_output_ndbuffer,
        hidden_state_ragged_ndbuffer,
        weight_ref_ndbuffer,
        ref_input_scale_ndbuffer,
        ref_weight_scale_ndbuffer,
        ctx,
    )

    ctx.enqueue_copy(kv_block_host_ptr, kv_block_device)
    ctx.enqueue_copy(ref_output_host_ptr, ref_output_device)
    ctx.synchronize()

    # Verify results
    for bs in range(batch_size):
        var prompt_len = prompt_lens[bs]
        for s in range(prompt_len):
            for k_dim in range(kv_hidden_size):
                var head_idx = k_dim // kv_params.head_size
                var head_dim_idx = k_dim % kv_params.head_size
                var a = ref_output_host[
                    Int(input_row_offsets_host_ptr[bs]) + s, Int(k_dim)
                ]
                var b = k_cache_host.load[width=1](
                    bs,
                    Int(head_idx),
                    cache_sizes[bs] + s,
                    Int(head_dim_idx),
                )
                assert_almost_equal(a, b, atol=atol, rtol=rtol)

    # Cleanup host memory
    cache_lengths_host_ptr.free()
    kv_block_host_ptr.free()
    paged_lut_host_ptr.free()
    input_row_offsets_host_ptr.free()
    weight_host_ptr.free()
    input_scale_host_ptr.free()
    weight_scale_host_ptr.free()
    ref_output_host_ptr.free()
    weight_ref_host_ptr.free()
    ref_input_scale_host_ptr.free()
    ref_weight_scale_host_ptr.free()

    # Cleanup device buffers
    _ = hidden_state_ragged_device^
    _ = hidden_state_padded_device^
    _ = weight_device^
    _ = weight_ref_device^
    _ = input_scale_device^
    _ = weight_scale_device^
    _ = ref_input_scale_device^
    _ = ref_weight_scale_device^
    _ = ref_output_device^
    _ = kv_block_device^
    _ = paged_lut_device^
    _ = cache_lengths_device^
    _ = input_row_offsets_device^


def execute_fused_matmul_suite_float8_e4m3fn(ctx: DeviceContext):
    """Test suite specifically for FP8 scaled matmul operations."""
    comptime dtype = DType.float8_e4m3fn
    comptime rtol = 1e-2
    comptime atol = 1e-2
    for bs in [1, 16]:
        var ce_cache_sizes = List[Int]()
        var ce_seq_lens = List[Int]()
        var tg_cache_sizes = List[Int]()
        var tg_seq_lens = List[Int]()
        for _ in range(bs):
            tg_seq_lens.append(1)
            # TODO increase sizes here to ensure we cross page boundary.
            tg_cache_sizes.append(Int(random_ui64(512, 700)))
            ce_seq_lens.append(Int(random_ui64(512, 700)))
            ce_cache_sizes.append(0)

        # Context encoding test
        execute_matmul_k_cache_ragged_scale[
            llama_num_q_heads,
            DType.float32,
            dtype,
            dtype,
            kv_params_llama3,
            rtol,
            atol,
        ](ce_seq_lens, 1024, ce_cache_sizes, 4, 1, ctx)

        # Token generation test
        execute_matmul_k_cache_ragged_scale[
            llama_num_q_heads,
            DType.float32,
            dtype,
            dtype,
            kv_params_llama3,
            rtol,
            atol,
        ](tg_seq_lens, 1024, tg_cache_sizes, 4, 3, ctx)


def main():
    seed(42)
    with DeviceContext() as ctx:
        execute_fused_matmul_suite_float8_e4m3fn(ctx)
