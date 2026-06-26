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

from std.sys import size_of
from std.math import ceildiv

from layout import (
    Idx,
    TensorLayout,
    TileTensor,
    row_major,
)

from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext, FuncAttribute

from kv_cache.types import KVCollectionT

from nn.index_fp8 import fp8_index_kernel, IndexSmemStorage
from nn.attention.mha_mask import MHAMask, MaskName
from nn.attention.mha_operand import KVCacheMHAOperand, KVCacheScalesMHAOperand
from nn.attention.mha_utils import dispatch_mask
from nn.topk import topk_gpu

from std.utils.index import Index


# ===----------------------------------------------------------------------=== #
# Mask application kernel
# ===----------------------------------------------------------------------=== #


@__name(t"mla_apply_mask")
def apply_mask_kernel[
    mask_t: MHAMask,
    ScoresLayoutType: TensorLayout,
    scores_origin: MutOrigin,
    VLLayoutType: TensorLayout,
    vl_origin: ImmutOrigin,
    CLLayoutType: TensorLayout,
](
    output: TileTensor[DType.float32, ScoresLayoutType, scores_origin],
    valid_length: TileTensor[DType.uint32, VLLayoutType, vl_origin],
    cache_lengths: TileTensor[DType.uint32, CLLayoutType, ImmutAnyOrigin],
    mask: mask_t,
    max_num_keys: Int,
):
    """Apply causal mask to the output scores."""
    var batch_idx = block_idx.x
    var seq_idx = block_idx.y * 16 + thread_idx.x
    var key_idx = block_idx.z * 16 + thread_idx.y

    var start_of_seq = valid_length.raw_load(batch_idx)
    var end_of_seq = valid_length.raw_load(batch_idx + 1)
    var seq_len = end_of_seq - start_of_seq

    if seq_idx >= Int(seq_len) or key_idx >= Int(max_num_keys):
        return

    var global_seq_idx = start_of_seq + UInt32(seq_idx)
    var current_val = output.raw_load(
        Int(global_seq_idx) * max_num_keys + key_idx
    )

    # Apply mask: coord = [batch, head, query_idx, key_idx]
    # key_idx indexes the full KV cache, so the query index must be its
    # absolute position cache_len + seq_idx.
    var cache_len = Int(cache_lengths[batch_idx])
    var coord = Index(batch_idx, 0, cache_len + seq_idx, key_idx)
    var masked_val = mask.mask(coord, current_val)

    output.raw_store(Int(global_seq_idx) * max_num_keys + key_idx, masked_val)


@__name(t"mla_fill_invalid_topk_{use_causal_mask}")
def fill_invalid_topk_kernel[
    IROLayoutType: TensorLayout,
    iro_origin: ImmutOrigin,
    cache_lengths_layout: TensorLayout,
    use_causal_mask: Bool,
](
    output_indices: UnsafePointer[Scalar[DType.int32], MutAnyOrigin],
    topk_indices: UnsafePointer[Scalar[DType.int32], MutAnyOrigin],
    input_row_offsets: TileTensor[DType.uint32, IROLayoutType, iro_origin],
    cache_lengths: TileTensor[
        DType.uint32, cache_lengths_layout, ImmutAnyOrigin
    ],
    total_seq_len: Int,
    top_k: Int,
    effective_k: Int,
):
    """Scatter the compact top-k indices into the top_k-strided output, with
    invalid positions set to -1.

    topk_gpu wrote valid indices to positions [0, effective_k) of the compact
    `topk_indices` buffer (row stride effective_k). This kernel copies them
    into `output_indices` (row stride top_k), setting positions that should be
    -1:
    - Positions [effective_k, top_k): no computed value.
    - Positions where k_idx >= num_keys for that token.
    - Positions where the index VALUE >= num_keys (topk selected an invalid key).

    Output shape: [total_seq_len, top_k].

    With causal masking, each token can only see keys up to its position:
        num_keys = cache_len + local_seq_idx + 1
    Without causal masking, each token can see all keys in the batch:
        num_keys = cache_len + seq_len
    """
    comptime assert cache_lengths.flat_rank == 1

    var token_idx = block_idx.x

    if token_idx >= total_seq_len:
        return

    # Token-level quantities (independent of k): find which batch this token
    # belongs to and how many keys it may attend to.
    var batch_idx = 0
    var batch_size = Int(input_row_offsets.dim[0]()) - 1
    for b in range(batch_size):
        var q_end_b = Int(input_row_offsets.raw_load(b + 1))
        if token_idx < q_end_b:
            batch_idx = b
            break

    var q_start = Int(input_row_offsets.raw_load(batch_idx))
    var q_end = Int(input_row_offsets.raw_load(batch_idx + 1))
    var seq_len = q_end - q_start
    var local_seq_idx = token_idx - q_start

    var cache_len = Int(cache_lengths[batch_idx])

    # Compute num_keys based on mask type
    var num_keys: Int

    comptime if use_causal_mask:
        # Causal: only keys up to current position are valid
        num_keys = cache_len + local_seq_idx + 1
    else:
        # No causal mask: all keys in the batch are valid
        num_keys = cache_len + seq_len

    # Cover ALL top_k output columns. The launch caps block_dim at 1024, which
    # is smaller than top_k when top_k > 1024 (e.g. the indexer's top_k=2048),
    # so each thread strides across multiple columns. Without this grid-stride
    # loop, columns [block_dim, top_k) were never written (garbage, not -1).
    var k_idx = Int(thread_idx.x)
    while k_idx < top_k:
        # Output index: [token_idx, k_idx] with top_k stride
        var out_idx = Int(token_idx) * top_k + k_idx

        if k_idx >= effective_k:
            # No computed value at this position: must be -1.
            output_indices[out_idx] = -1
        else:
            # Read topk's selection from the compact (effective_k-strided)
            # buffer, then invalidate it if it is out of range:
            # 1. position beyond the valid keys (k_idx >= num_keys)
            # 2. the index VALUE points beyond valid keys (idx_val >= num_keys),
            #    which can happen because topk operates on
            #    max_num_keys >= num_keys for this token/batch.
            var idx_val = Int(
                topk_indices[Int(token_idx) * effective_k + k_idx]
            )
            if k_idx >= num_keys or idx_val >= num_keys or idx_val < 0:
                output_indices[out_idx] = -1
            else:
                output_indices[out_idx] = Int32(idx_val)

        k_idx += Int(block_dim.x)


# ===----------------------------------------------------------------------=== #
# Main function: mla_indexer_ragged_float8_paged
# ===----------------------------------------------------------------------=== #


@always_inline
def mla_indexer_ragged_float8_paged[
    dtype: DType,
    KCollectionT: KVCollectionT,
    num_heads: Int,
    depth: Int,
    top_k: Int,
    mask_str: StaticString,
](
    output_indices: TileTensor[DType.int32, ...],
    q: TileTensor[mut=False, dtype, ...],
    q_s: TileTensor[DType.float32, ...],
    input_row_offsets: TileTensor[mut=False, DType.uint32, ...],
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

    var batch_size = Int(input_row_offsets.dim[0]()) - 1
    var total_seq_len = Int(q.dim[0]())
    if total_seq_len == 0:
        return

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

    var scores_tile = TileTensor(
        scores_buf,
        row_major(total_seq_len, max_num_keys),
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
        type_of(scores_tile).LayoutType,
        type_of(q).LayoutType,
        type_of(q_s).LayoutType,
        type_of(k_operand),
        type_of(ks_operand),
        block_tile_shape,
        type_of(input_row_offsets.as_immut()).LayoutType,
        num_heads,
        depth,
    ]

    ctx.enqueue_function[kernel](
        scores_tile,
        q.as_immut(),
        q_s,
        k_operand,
        ks_operand,
        input_row_offsets.as_immut(),
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

    # Per-batch KV cache lengths (cached-prefix length).  Needed both by the
    # causal mask below (to map local query index → absolute position) and by
    # fill_invalid_topk below.
    var cache_lengths = k_cache.cache_lengths_nd()

    # Apply mask for prefill (seq_len > 1)
    comptime if mask_str != MaskName.NULL.name:
        if max_new_tokens > 1:

            @always_inline
            @parameter
            def apply_mask_dispatch[mask_t: MHAMask](mask: mask_t) raises:
                comptime mask_kernel = apply_mask_kernel[
                    mask_t,
                    scores_tile.LayoutType,
                    scores_tile.origin,
                    input_row_offsets.LayoutType,
                    ImmutOrigin(input_row_offsets.origin),
                    type_of(cache_lengths).LayoutType,
                ]

                ctx.enqueue_function[mask_kernel](
                    scores_tile,
                    input_row_offsets.as_immut(),
                    cache_lengths,
                    mask,
                    max_num_keys,
                    grid_dim=(
                        batch_size,
                        ceildiv(max_new_tokens, 16),
                        ceildiv(max_num_keys, 16),
                    ),
                    block_dim=(16, 16, 1),
                )

            dispatch_mask[mask_str, apply_mask_dispatch]()

    # Compute effective_k - the actual number of values we can select.
    # If top_k > max_num_keys, we can only select max_num_keys values.
    var effective_k = min(top_k, max_num_keys)

    # topk_gpu strides its index output by effective_k, so when effective_k <
    # top_k we use a compact buffer here and scatter into the top_k-strided
    # output_indices in fill_invalid (writing directly would misplace rows).
    var topk_vals_buf = ctx.enqueue_create_buffer[DType.float32](
        total_seq_len * effective_k
    )
    var topk_vals_tile = TileTensor(
        topk_vals_buf,
        row_major(total_seq_len, effective_k),
    )

    var topk_idxs_buf = ctx.enqueue_create_buffer[DType.int32](
        total_seq_len * effective_k
    )
    var topk_idxs_tile = TileTensor(
        topk_idxs_buf,
        row_major(total_seq_len, effective_k),
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
    # (cache_lengths hoisted above for the causal mask launch.)

    # Determine if causal masking is used (any mask except NULL)
    comptime use_causal_mask = mask_str != MaskName.NULL.name

    comptime fill_kernel = fill_invalid_topk_kernel[
        input_row_offsets.LayoutType,
        ImmutOrigin(input_row_offsets.origin),
        type_of(cache_lengths).LayoutType,
        use_causal_mask,
    ]

    var block_size = ceildiv(top_k, 32) * 32
    block_size = min(block_size, 1024)  # Cap at max threads per block

    ctx.enqueue_function[fill_kernel](
        rebind[UnsafePointer[Scalar[DType.int32], MutAnyOrigin]](
            output_indices.ptr
        ),
        rebind[UnsafePointer[Scalar[DType.int32], MutAnyOrigin]](
            topk_idxs_tile.ptr
        ),
        input_row_offsets.as_immut(),
        cache_lengths,
        total_seq_len,
        top_k,
        effective_k,
        grid_dim=(total_seq_len, 1, 1),
        block_dim=(block_size, 1, 1),
    )

    _ = scores_buf
    _ = topk_vals_buf
    _ = topk_idxs_buf
