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
"""MLA FP8 index kernel for computing attention scores with paged KV cache."""

from sys import size_of
from math import ceildiv

from layout import Layout, RuntimeLayout, UNKNOWN_VALUE
from layout.layout_tensor import LayoutTensor
from layout._tile_tensor import TileTensor
from layout._coord import Idx
from layout._layout import row_major

from gpu import block_idx, thread_idx
from gpu.host import DeviceContext, FuncAttribute

from kv_cache.types import KVCacheT, KVCollectionT

from nn.index_fp8 import fp8_index_kernel, IndexSmemStorage
from nn.mha_mask import MHAMask, MaskName
from nn.mha_operand import KVCacheMHAOperand, KVCacheScalesMHAOperand
from nn.mha_score_mod import ScoreModTrait, IdentityScoreMod
from nn.mha_utils import dispatch_mask_and_score_mod
from nn.topk import topk_gpu

from utils.index import Index


# ===----------------------------------------------------------------------=== #
# Mask application kernel
# ===----------------------------------------------------------------------=== #


fn apply_mask_kernel[
    output_layout: Layout,
    valid_length_layout: Layout,
    mask_t: MHAMask,
](
    output: LayoutTensor[DType.float32, output_layout, MutAnyOrigin],
    valid_length: LayoutTensor[
        DType.uint32, valid_length_layout, ImmutAnyOrigin
    ],
    mask: mask_t,
    max_num_keys: Int,
):
    """Apply causal mask to the output scores."""
    var batch_idx = block_idx.x
    var seq_idx = block_idx.y * 16 + thread_idx.x
    var key_idx = block_idx.z * 16 + thread_idx.y

    var start_of_seq = valid_length[batch_idx][0]
    var end_of_seq = valid_length[batch_idx + 1][0]
    var seq_len = end_of_seq - start_of_seq

    if seq_idx >= UInt(seq_len) or key_idx >= UInt(max_num_keys):
        return

    var global_seq_idx = start_of_seq + UInt32(seq_idx)
    var current_val = output.ptr[
        Int(global_seq_idx) * max_num_keys + Int(key_idx)
    ]

    # Apply mask: coord = [batch, head, q_idx, k_idx]
    # For causal mask: q_idx >= k_idx means visible
    var coord = Index(Int(batch_idx), 0, Int(seq_idx), Int(key_idx))
    var masked_val = mask.mask(coord, current_val)

    output.ptr[Int(global_seq_idx) * max_num_keys + Int(key_idx)] = masked_val


fn fill_invalid_topk_kernel[
    input_row_offsets_layout: Layout,
    cache_lengths_layout: Layout,
    use_causal_mask: Bool,
](
    output_indices: UnsafePointer[Scalar[DType.int32], MutAnyOrigin],
    input_row_offsets: LayoutTensor[
        DType.uint32, input_row_offsets_layout, ImmutAnyOrigin
    ],
    cache_lengths: LayoutTensor[
        DType.uint32, cache_lengths_layout, ImmutAnyOrigin
    ],
    total_seq_len: Int,
    top_k: Int,
    effective_k: Int,
):
    """Fill invalid positions with -1 in topk output.

    topk_gpu has already written valid indices to positions [0, effective_k)
    in output_indices (which has top_k stride). This kernel fills positions
    that should be -1:
    - Positions [effective_k, top_k) when top_k > max_num_keys
    - Positions where k_idx >= num_keys for that token
    - Positions where the index VALUE >= num_keys (topk selected an invalid key)

    Output shape: [total_seq_len, top_k].

    With causal masking, each token can only see keys up to its position:
        num_keys = cache_len + local_seq_idx + 1
    Without causal masking, each token can see all keys in the batch:
        num_keys = cache_len + seq_len
    """
    var token_idx = Int(block_idx.x)
    var k_idx = Int(thread_idx.x)

    if token_idx >= total_seq_len or k_idx >= top_k:
        return

    # Output index: [token_idx, k_idx] with top_k stride
    var out_idx = token_idx * top_k + k_idx

    # If k_idx >= effective_k, we don't have computed values - fill with -1
    if k_idx >= effective_k:
        output_indices[out_idx] = -1
        return

    # Find which batch this token belongs to by scanning row_offsets
    var batch_idx = 0
    var batch_size = input_row_offsets.dim[0]() - 1
    for b in range(batch_size):
        var q_end = Int(input_row_offsets[b + 1][0])
        if token_idx < q_end:
            batch_idx = b
            break

    var q_start = Int(input_row_offsets[batch_idx][0])
    var q_end = Int(input_row_offsets[batch_idx + 1][0])
    var seq_len = q_end - q_start
    var local_seq_idx = token_idx - q_start

    var cache_len = Int(cache_lengths[batch_idx][0])

    # Compute num_keys based on mask type
    var num_keys: Int

    @parameter
    if use_causal_mask:
        # Causal: only keys up to current position are valid
        num_keys = cache_len + local_seq_idx + 1
    else:
        # No causal mask: all keys in the batch are valid
        num_keys = cache_len + seq_len

    # Get the index value that topk wrote
    var idx_val = Int(output_indices[out_idx])

    # Fill with -1 if:
    # 1. This position is beyond the number of valid keys (k_idx >= num_keys)
    # 2. The index VALUE points beyond valid keys (idx_val >= num_keys)
    #    This can happen because topk operates on max_num_keys which may be
    #    larger than num_keys for this specific token/batch
    if k_idx >= num_keys or idx_val >= num_keys or idx_val < 0:
        output_indices[out_idx] = -1


# ===----------------------------------------------------------------------=== #
# Main function: mla_indexer_ragged_float8_paged
# ===----------------------------------------------------------------------=== #


@always_inline
fn mla_indexer_ragged_float8_paged[
    dtype: DType,
    q_layout: Layout,
    qs_layout: Layout,
    output_layout: Layout,
    KCollectionT: KVCollectionT,
    num_heads: Int,
    depth: Int,
    top_k: Int,
    mask_str: StaticString,
](
    output_indices: LayoutTensor[
        mut=True,
        DType.int32,
        output_layout,
        address_space = AddressSpace.GENERIC,
        ...,
    ],
    q: LayoutTensor[dtype, q_layout, address_space = AddressSpace.GENERIC, ...],
    q_s: LayoutTensor[
        DType.float32, qs_layout, address_space = AddressSpace.GENERIC, ...
    ],
    input_row_offsets: LayoutTensor[
        DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    k_collection: KCollectionT,
    layer_idx: UInt32,
    ctx: DeviceContext,
) raises:
    """Compute FP8 indexed attention scores using paged KV cache and return top-k indices.

    This function:
    1. Computes FP8 matmul between q and cached k (with scales), aggregated across heads
    2. Applies the specified mask (causal, etc.)
    3. Computes top-k indices per token (scores are summed across all heads)

    Args:
        output_indices: Dense output tensor for top-k indices [total_seq_len, top_k].
            Invalid positions (where there are fewer than top_k valid keys due to
            causal masking or shorter sequences) are filled with -1.
        q: Query tensor [total_seq_len, num_heads, head_dim] in FP8.
        q_s: Query scales [total_seq_len, num_heads] in float32.
        input_row_offsets: Ragged row offsets for queries [batch_size + 1].
        k_collection: KV collection containing cached K values and K scales.
            K scales are accessed via k_cache.scales (quantization_granularity=head_size).
        layer_idx: Layer index for retrieving cache.
        ctx: Device context.
    """
    # Verify that k_collection has scales enabled (required for MLA k_s).
    # For MLA, scales should have head_dim_granularity == 1 (one scale per token
    # per head), which requires quantization_granularity >= depth (head_size).
    comptime CacheType = KCollectionT.CacheType
    comptime assert (
        CacheType.quantization_enabled
    ), "k_collection must have quantization/scales enabled for MLA k_s values"
    comptime assert (
        CacheType.scale_dtype != DType.invalid
    ), "k_collection must have valid scale_dtype for MLA k_s values"
    comptime assert CacheType.quantization_granularity >= depth, (
        "k_collection.quantization_granularity must be >= depth (head_dim) for"
        " MLA (requires one scale per token per head, i.e. head_dim_granularity"
        " == 1)"
    )

    # Only NULL (no mask) and CAUSAL masks are supported
    comptime assert (
        mask_str == MaskName.NULL.name or mask_str == MaskName.CAUSAL.name
    ), "mask_str must be either MaskName.NULL or MaskName.CAUSAL"

    var batch_size = input_row_offsets.dim[0]() - 1
    var total_seq_len = q.dim[0]()

    var k_cache = k_collection.get_key_cache(Int(layer_idx))

    # max_new_tokens is used for grid dimensions (maximum possible new tokens)
    var max_new_tokens = Int(k_cache.max_prompt_length())

    # This is an approximation of the max number of keys per token,
    # but it's only used to compute the grid dim so an approximation is fine.
    var max_num_keys = Int(k_cache.max_context_length()) + max_new_tokens

    # Allocate intermediate scores buffer: [total_seq_len, max_num_keys]
    # Initialize to -inf so invalid positions don't appear in top-k
    var scores_size = total_seq_len * max_num_keys
    var scores_buf = ctx.enqueue_create_buffer[DType.float32](scores_size)
    scores_buf.enqueue_fill(-Float32.MAX)

    # Reshape scores as [total_seq_len, max_num_keys] for topk
    comptime scores_layout = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)
    var scores_runtime_layout = RuntimeLayout[scores_layout].row_major(
        Index(total_seq_len, max_num_keys)
    )
    var scores_tensor = LayoutTensor[
        DType.float32, scores_layout, MutAnyOrigin
    ](scores_buf.unsafe_ptr(), scores_runtime_layout)

    # Create valid_length tensor from input_row_offsets
    var valid_length = LayoutTensor[
        DType.uint32,
        type_of(input_row_offsets).layout,
        ImmutAnyOrigin,
    ](
        input_row_offsets.ptr,
        RuntimeLayout[type_of(input_row_offsets).layout].row_major(
            input_row_offsets.runtime_layout.shape.value.canonicalize()
        ),
    )

    var k_operand = KVCacheMHAOperand(k_cache)
    var ks_operand = KVCacheScalesMHAOperand(k_cache)

    comptime block_tile_shape: InlineArray[Int, 2] = [512, 128]
    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime smem_use = size_of[IndexSmemStorage[dtype, num_heads, depth, BN]]()
    comptime smem_available = ctx.default_device_info.shared_memory_per_multiprocessor - 1024

    # fp8_index_kernel computes scores aggregated across heads.
    # Output is [total_seq_len, max_num_keys] with one score per (token, key) pair.
    comptime kernel = fp8_index_kernel[
        dtype,
        scores_layout,
        q.layout,
        qs_layout,
        type_of(k_operand),
        type_of(ks_operand),
        block_tile_shape,
        type_of(valid_length).layout,
        num_heads,
        depth,
    ]

    ctx.enqueue_function[kernel, kernel](
        scores_tensor,
        q,
        q_s,
        k_operand,
        ks_operand,
        valid_length,
        grid_dim=(
            batch_size,
            max_new_tokens,
            ceildiv(max_num_keys, BM),
        ),
        block_dim=(16, 8, 1),
        shared_mem_bytes=smem_use,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(smem_available)
        ),
    )

    # Apply mask for prefill (seq_len > 1)
    @parameter
    if mask_str != MaskName.NULL.name:
        if max_new_tokens > 1:

            @always_inline
            @parameter
            fn apply_mask_dispatch[
                mask_t: MHAMask, score_mod_t: ScoreModTrait
            ](mask: mask_t, score_mod: score_mod_t) raises:
                comptime mask_kernel = apply_mask_kernel[
                    scores_layout,
                    type_of(valid_length).layout,
                    mask_t,
                ]

                ctx.enqueue_function[mask_kernel, mask_kernel](
                    scores_tensor,
                    valid_length,
                    mask,
                    max_num_keys,
                    grid_dim=(
                        batch_size,
                        ceildiv(max_new_tokens, 16),
                        ceildiv(max_num_keys, 16),
                    ),
                    block_dim=(16, 16, 1),
                )

            dispatch_mask_and_score_mod[
                mask_str,
                IdentityScoreMod.name_str,
                apply_mask_dispatch,
            ]()

    # Compute top-k indices from scores [total_seq_len, max_num_keys]
    var scores_tile = TileTensor(
        scores_buf.unsafe_ptr(),
        row_major((Idx(total_seq_len), Idx(max_num_keys))),
    )

    # Compute effective_k - the actual number of values we can select.
    # If top_k > max_num_keys, we can only select max_num_keys values.
    var effective_k = min(top_k, max_num_keys)

    # Create temp buffer for topk values (only need effective_k per row for computation)
    var topk_vals_buf = ctx.enqueue_create_buffer[DType.float32](
        total_seq_len * effective_k
    )
    var topk_vals_tile = TileTensor(
        topk_vals_buf.unsafe_ptr(),
        row_major((Idx(total_seq_len), Idx(effective_k))),
    )

    # Output indices tile - use top_k stride to match output buffer layout.
    # topk_gpu will write effective_k values at the start of each row.
    var topk_idxs_tile = TileTensor(
        output_indices.ptr.bitcast[Scalar[DType.int32]](),
        row_major((Idx(total_seq_len), Idx(top_k))),
    )

    topk_gpu[sampling=False, largest=True](
        ctx,
        effective_k,
        scores_tile,
        topk_vals_tile,
        topk_idxs_tile,
    )

    # Fill invalid positions with -1:
    # - Positions [effective_k, top_k) when top_k > max_num_keys
    # - Positions where k_idx >= num_keys for that token (causal masking)
    var cache_lengths = k_cache.cache_lengths_nd()

    # Determine if causal masking is used (any mask except NULL)
    comptime use_causal_mask = mask_str != MaskName.NULL.name

    comptime fill_kernel = fill_invalid_topk_kernel[
        type_of(valid_length).layout,
        type_of(cache_lengths).layout,
        use_causal_mask,
    ]

    var block_size = ceildiv(top_k, 32) * 32
    block_size = min(block_size, 1024)  # Cap at max threads per block

    ctx.enqueue_function[fill_kernel, fill_kernel](
        output_indices.ptr.bitcast[Scalar[DType.int32]](),
        valid_length,
        cache_lengths,
        total_seq_len,
        top_k,
        effective_k,
        grid_dim=(total_seq_len, 1, 1),
        block_dim=(block_size, 1, 1),
    )

    _ = scores_buf
    _ = topk_vals_buf
