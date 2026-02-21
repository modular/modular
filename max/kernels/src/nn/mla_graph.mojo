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


from math import align_up, ceildiv

from sys import simd_width_of, size_of
from utils.index import Index, IndexList

from algorithm.functional import _elementwise_impl_gpu
from gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_idx,
    global_idx,
    grid_dim,
    thread_idx,
)
from gpu.primitives.grid_controls import PDL, pdl_launch_attributes
from gpu.host import DeviceContext, get_gpu_target
from layout._coord import Coord, CoordLike, Idx, coord_to_index_list
from layout._layout import TensorLayout, Layout as TileLayout, row_major
from layout._tile_tensor import TileTensor
from linalg.bmm import _batched_matmul_gpu, batched_matmul_dynamic_scaled_fp8
from linalg.matmul import matmul
from utils.index import StaticTuple
from utils.numerics import get_accum_type
from linalg.fp8_quantization import (
    matmul_dynamic_scaled_fp8,
    quantize_dynamic_scaled_fp8,
    batched_quantize_dynamic_scaled_fp8,
)
from nn._ragged_utils import get_batch_and_token_idx_from_row_offsets
from nn.fused_qk_rope import rope_k_cache, rope_q_proj
from nn.kv_cache import KVCollectionT, KVCacheT
from nn.kv_cache_ragged import (
    generic_flare_mla_decode_kv_cache_ragged,
    generic_flare_mla_prefill_kv_cache_ragged,
)
from nn.mla import _k_cache_to_buffer
from nn.normalization import _rms_norm_warp_tiling_subkernel


# ===-----------------------------------------------------------------------===#
# Manually fused MLA RoPE and RMSNorm kernel
# ===-----------------------------------------------------------------------===#


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(block_size))
)
fn fused_rope_rmsnorm_kernel[
    dtype: DType,
    freq_dtype: DType,
    gamma_dtype: DType,
    QRopeOutputLayoutType: TensorLayout,
    QRopeLayoutType: TensorLayout,
    InputRowOffsetsLayoutType: TensorLayout,
    FreqsCisLayoutType: TensorLayout,
    GammaLayoutType: TensorLayout,
    cache_t: KVCacheT,
    block_size: Int,
    n_rope_blocks: Int,
    n_rms_blocks: Int,
](
    q_rope_output: TileTensor[
        mut=True, dtype, QRopeOutputLayoutType, MutExternalOrigin
    ],
    q_rope: TileTensor[dtype, QRopeLayoutType, ImmutExternalOrigin],
    input_row_offsets: TileTensor[
        DType.uint32, InputRowOffsetsLayoutType, ImmutExternalOrigin
    ],
    freqs_cis: TileTensor[freq_dtype, FreqsCisLayoutType, ImmutExternalOrigin],
    gamma: TileTensor[gamma_dtype, GammaLayoutType, ImmutExternalOrigin],
    k_cache: cache_t,
    epsilon: Float32,
) -> None:
    """Fused GPU kernel that applies RoPE to query projections and RMSNorm to KV
    cache.

    This kernel processes tokens in parallel across GPU blocks, with separate
    block groups handling RoPE and RMSNorm operations. The RoPE blocks apply
    rotary position embeddings to both the query rope part (in-place) and the
    key cache rope part (in-place). The RMSNorm blocks normalize the first
    `kv_norm_dim` elements of the key cache entries.

    Parameters:
        dtype: Data type of query tensors.
        freq_dtype: Data type of frequency cosine/sine values.
        gamma_dtype: Data type of RMSNorm gamma weights.
        QRopeOutputLayoutType: Layout types of the output query rope tensor.
        QRopeLayoutType: Layout types of the input query rope tensor.
        InputRowOffsetsLayoutType: Layout types of the row offset indices tensor.
        FreqsCisLayoutType: Layout types of the frequency tensor.
        GammaLayoutType: Layout types of the gamma weights tensor.
        cache_t: Type of the KV cache.
        block_size: Number of threads per block.
        n_rope_blocks: Number of blocks allocated for RoPE computation.
        n_rms_blocks: Number of blocks allocated for RMSNorm computation.

    Args:
        q_rope_output: Output tensor for RoPE-applied query projections.
            Shape: [tot_seq_len, num_heads, rope_dim].
        q_rope: Input query rope projections. Shape: [tot_seq_len, num_heads, rope_dim].
        input_row_offsets: Row offsets indicating request boundaries.
            Shape: [num_batches + 1].
        freqs_cis: Precomputed RoPE frequency values. Shape: [max_seq_len, rope_dim].
        gamma: RMSNorm gamma weights. Shape: [kv_norm_dim].
        k_cache: Key cache to apply RoPE and RMSNorm to.
        epsilon: Small constant for numerical stability in RMSNorm.
    """
    comptime assert (
        cache_t.kv_params.num_heads == 1
    ), "num_heads should be 1 for MLA"
    comptime assert q_rope_output.flat_rank == 3
    comptime assert q_rope.flat_rank == 3
    comptime assert input_row_offsets.flat_rank == 1
    comptime assert freqs_cis.flat_rank == 2
    comptime assert gamma.flat_rank == 1

    comptime num_q_heads = q_rope.static_shape[1]
    comptime rope_dim = q_rope.static_shape[2]
    comptime kv_norm_dim = gamma.static_shape[0]

    var worker_idx = block_idx.y
    var num_workers = grid_dim.y
    var num_tokens = q_rope.dim(0)

    with PDL():
        for global_token_idx in range(worker_idx, num_tokens, num_workers):
            var batch_idx, token_idx = get_batch_and_token_idx_from_row_offsets(
                input_row_offsets, global_token_idx
            )
            var post_seq_idx = k_cache.cache_length(batch_idx) + token_idx

            # First n_rope_blocks blocks of this worker process RoPE.
            if block_idx.x < UInt(n_rope_blocks):
                comptime q_width = simd_width_of[dtype]()
                comptime assert (
                    rope_dim % q_width == 0
                ), "rope_dim should be divisible by q_width"

                var head_idx, head_dim_idx = divmod(
                    Int(global_idx.x) * q_width, rope_dim
                )
                var f_c = freqs_cis.load[width=q_width](
                    (Idx(post_seq_idx), Idx(head_dim_idx))
                )

                if head_idx < num_q_heads:
                    rope_q_proj[interleaved=True](
                        q_rope,
                        q_rope_output,
                        Index(global_token_idx, head_idx, head_dim_idx),
                        f_c,
                        rope_dim,
                    )
                elif head_idx == num_q_heads:
                    rope_k_cache[interleaved=True](
                        k_cache,
                        batch_idx,
                        0,  # num_k_heads is 1 for MLA
                        post_seq_idx,
                        head_dim_idx + kv_norm_dim,
                        f_c,
                        rope_dim,
                    )

            # The last block of this worker processes RMSNorm.
            else:
                comptime k_dtype = cache_t.dtype
                comptime k_width = simd_width_of[k_dtype]()
                comptime accum_type = get_accum_type[k_dtype]()
                comptime warps_per_block = block_size // WARP_SIZE

                comptime assert (
                    kv_norm_dim % k_width == 0
                ), "kv_norm_dim should be divisible by k_width"

                var vec_data = SIMD[accum_type, k_width](0)

                var idx = Int(thread_idx.x) * k_width
                if idx < kv_norm_dim:
                    vec_data = k_cache.load[width=k_width](
                        batch_idx,
                        0,  # num_k_heads is 1 for MLA
                        post_seq_idx,
                        idx,
                    ).cast[accum_type]()

                var norm_val = _rms_norm_warp_tiling_subkernel[
                    warps_per_block,
                    False,  # Do not multiply the gamma before casting.
                ](
                    global_token_idx,
                    idx,
                    vec_data,
                    gamma,
                    epsilon.cast[accum_type](),
                    0.0,
                    kv_norm_dim,
                )

                if idx < kv_norm_dim:
                    k_cache.store(
                        batch_idx,
                        0,  # num_k_heads is 1 for MLA
                        post_seq_idx,
                        idx,
                        norm_val.cast[k_dtype](),
                    )


@always_inline
fn mla_fused_rope_rmsnorm[
    dtype: DType,
    freq_dtype: DType,
    gamma_dtype: DType,
    collection_t: KVCollectionT,
    //,
](
    q_rope_output: TileTensor[mut=True, dtype, ...],
    q_rope: TileTensor[dtype, ...],
    input_row_offsets: TileTensor[DType.uint32, ...],
    freqs_cis: TileTensor[freq_dtype, ...],
    gamma: TileTensor[gamma_dtype, ...],
    kv_collection: collection_t,
    layer_idx: UInt32,
    epsilon: Float32,
    ctx: DeviceContext,
) raises:
    """Launches the fused RoPE and RMSNorm kernel for MLA attention.

    This function fuses two operations:
    1. RoPE applied to query and key cache rope parts.
    2. RMSNorm applied to the non-rope portion of the key cache.

    Parameters:
        dtype: Data type of query tensors.
        freq_dtype: Data type of frequency cosine/sine values.
        gamma_dtype: Data type of RMSNorm gamma weights.
        collection_t: Type of the KV cache collection.

    Args:
        q_rope_output: Output tensor for RoPE-applied query projections.
            Shape: [tot_seq_len, num_heads, rope_dim].
        q_rope: Input query rope projections. Shape: [tot_seq_len, num_heads, rope_dim].
        input_row_offsets: Row offsets indicating request boundaries.
            Shape: [num_batches + 1].
        freqs_cis: Precomputed RoPE frequency values. Shape: [max_seq_len, rope_dim].
        gamma: RMSNorm gamma weights. Shape: [kv_norm_dim].
        kv_collection: Paged KV cache collection.
        layer_idx: Index of the current transformer layer.
        epsilon: Small constant for numerical stability in RMSNorm.
        ctx: Device context for kernel execution.
    """
    comptime hw_info = ctx.default_device_info
    comptime sm_count = hw_info.sm_count
    comptime num_q_heads = q_rope.static_shape[1]
    comptime num_k_heads = 1  # Fixed to 1 for MLA.
    comptime rope_dim = q_rope.static_shape[2]
    comptime kv_norm_dim = gamma.static_shape[0]

    comptime assert (
        q_rope_output.static_shape[2] == rope_dim
    ), "q_rope_output and q_rope must have the same head_size"
    comptime assert (
        q_rope_output.rank == 3 and q_rope.rank == 3
    ), "q_rope_output and q_rope must be rank 3"
    comptime assert rope_dim + kv_norm_dim == Int(
        collection_t.kv_params.head_size
    ), "rope_dim + kv_norm_dim must be equal to kvcache head_size"

    # Default block size used by the `elementwise` function on Blackwell.
    comptime block_size = 128
    comptime kernel_simd_width = simd_width_of[
        dtype, target = get_gpu_target()
    ]()
    comptime n_rope_elems = (num_q_heads + num_k_heads) * rope_dim

    # Make sure that we can use one block to process the rmsnorm.
    comptime assert kv_norm_dim <= block_size * kernel_simd_width, (
        "kv_norm_dim must be less than or equal to block_size *"
        " kernel_simd_width"
    )
    comptime n_rope_blocks = ceildiv(
        ceildiv(n_rope_elems, kernel_simd_width), block_size
    )
    comptime n_rms_blocks = 1  # Fixed to 1 for MLA.

    var max_workers = (
        sm_count
        * (hw_info.max_thread_block_size // block_size)
        // (n_rope_blocks + n_rms_blocks)
    )
    var num_workers = min(max_workers, Int(q_rope.dim(0)))

    var k_cache = kv_collection.get_key_cache(Int(layer_idx))

    comptime kernel = fused_rope_rmsnorm_kernel[
        dtype,
        freq_dtype,
        gamma_dtype,
        q_rope_output.LayoutType,
        q_rope.LayoutType,
        input_row_offsets.LayoutType,
        freqs_cis.LayoutType,
        gamma.LayoutType,
        type_of(k_cache),
        block_size,
        n_rope_blocks,
        n_rms_blocks,
    ]

    ctx.enqueue_function[kernel, kernel](
        q_rope_output,
        q_rope.as_immut(),
        input_row_offsets.as_immut(),
        freqs_cis.as_immut(),
        gamma.as_immut(),
        k_cache,
        epsilon,
        grid_dim=(n_rope_blocks + n_rms_blocks, num_workers, 1),
        block_dim=block_size,
        attributes=pdl_launch_attributes(),
    )


# ===-----------------------------------------------------------------------===#
# Manually fused MLA prefill branch (FP8)
# ===-----------------------------------------------------------------------===#


fn mla_prefill_branch_fp8[
    dtype: DType,
    fp8_dtype: DType,
    fp8_scale_dtype: DType,
    collection_t: KVCollectionT,
    //,
    m_scale_granularity: Int,
    n_scale_granularity: Int,
    k_scale_granularity: Int,
    mask_str: StaticString,
    score_mod_str: StaticString,
    target: StaticString = "cpu",
](
    output: TileTensor[
        mut=True, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    q: TileTensor[dtype, address_space = AddressSpace.GENERIC, ...],
    input_row_offsets: TileTensor[
        DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    freqs_cis: TileTensor[_, address_space = AddressSpace.GENERIC, ...],
    kv_norm_gamma: TileTensor[_, address_space = AddressSpace.GENERIC, ...],
    kv_collection: collection_t,
    layer_idx: UInt32,
    scale: Float32,
    epsilon: Float32,
    buffer_row_offsets: TileTensor[
        DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    cache_offsets: TileTensor[
        mut=True, DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    buffer_length: Int,
    w_k: TileTensor[fp8_dtype, address_space = AddressSpace.GENERIC, ...],
    w_k_scale: TileTensor[
        fp8_scale_dtype, address_space = AddressSpace.GENERIC, ...
    ],
    w_uv: TileTensor[fp8_dtype, address_space = AddressSpace.GENERIC, ...],
    w_uv_scale: TileTensor[
        fp8_scale_dtype, address_space = AddressSpace.GENERIC, ...
    ],
    ctx: DeviceContext,
) raises:
    """
    This is a manually fused kernel that performs the following operations:
    - Apply RoPE to the query and the key cache (in-place).
    - Apply RMSNorm to the non-rope portion of the key cache (in-place).
    - Copy the KV latent values from PagedKVCache to a contiguous buffer.
    - Quantize the KV latent values to fp8.
    - Up-project the latent KV values to full K and V through two matmuls.
    - Perform MLA prefill.

    Parameters:
        dtype: Data type of the input and output tensors.
        fp8_dtype: Data type of the fp8 input and output tensors.
        fp8_scale_dtype: Data type of the fp8 scale input and output tensors.
        collection_t: Type of the KV collection.
        m_scale_granularity: Granularity of the scale for M dimension of the
            matrix multiplication.
        n_scale_granularity: Granularity of the scale for N dimension of the
            matrix multiplication.
        k_scale_granularity: Granularity of the scale for K dimension of the
            matrix multiplication.
        mask_str: Mask variant.
        score_mod_str: Positional encoding variant.
        target: Target device.

    Args:
        output: Output tensor of shape [tot_seq_len, num_heads, v_head_dim].
        q: Combined query tensor containing both nope and rope parts. Shape:
            [tot_seq_len, num_heads, qk_nope_head_dim + qk_rope_head_dim].
        input_row_offsets: Indicates where each request starts and ends in
            `q`. Shape: [num_batches + 1].
        freqs_cis: Precomputed RoPE frequency values for rotary position
            embeddings. Shape: [max_seq_len, qk_rope_head_dim].
        kv_norm_gamma: RMSNorm gamma weights for normalizing the KV cache.
            Shape: [kv_lora_rank].
        kv_collection: Paged KV Cache object.
        layer_idx: Layer index.
        scale: Scale for the attention calculation.
        epsilon: Small constant for numerical stability in RMSNorm.
        buffer_row_offsets: Indicates where each request's KV latent values
            should be stored in the contiguous K buffer. This is a 1D tensor
            of shape [num_batches + 1].
        cache_offsets: Indicates the starting token position in the KV cache
            from which to copy KV latent values for each request. This is a 1D
            tensor of shape [num_batches + 1].
        buffer_length: The total number of tokens in the KV cache. Scalar.
        w_k: Weight matrix for up-projecting the latent cache to full K. Shape:
            [num_heads * qk_nope_head_dim, kv_latent_dim].
        w_k_scale: Scale tensor for `w_k`.
        w_uv: Weight tensor for projecting latent values to V. Shape:
            [num_heads, v_head_dim, kv_latent_dim].
        w_uv_scale: Scale tensor for `w_uv`.
        ctx: Device context.
    """
    comptime kv_params = collection_t.kv_params
    comptime assert kv_params.is_mla, "kv_params.is_mla should be true"
    comptime assert kv_params.num_heads == 1, "kv_params.num_heads should be 1"

    comptime num_heads = q.static_shape[1]
    comptime q_head_dim = q.static_shape[2]
    comptime qk_rope_head_dim = freqs_cis.static_shape[1]
    comptime qk_nope_head_dim = q_head_dim - qk_rope_head_dim
    comptime v_head_dim = output.static_shape[2]

    comptime assert w_k.shape_known, "w_k's shape should be static"
    comptime assert (
        w_k.static_shape[0] == num_heads * qk_nope_head_dim
    ), "w_k.shape[0] should be equal to num_heads * qk_nope_head_dim"
    comptime kv_latent_dim = w_k.static_shape[1]
    comptime assert w_uv.shape_known, "w_uv's shape should be static"
    comptime assert (
        w_uv.static_shape[0] == num_heads
    ), "w_uv.shape[0] should be equal to num_heads"
    comptime assert (
        w_uv.static_shape[1] == v_head_dim
    ), "w_uv.shape[1] should be equal to v_head_dim"
    comptime assert (
        w_uv.static_shape[2] == kv_latent_dim
    ), "w_uv.shape[2] should be equal to kv_latent_dim"

    comptime assert m_scale_granularity == 1, "m_scale_granularity should be 1"
    comptime assert (
        n_scale_granularity == k_scale_granularity == 128
    ), "n, k scale_granularity should be 128"

    # Return early if we have no tokens to process.
    if buffer_length == 0:
        return

    var seq_len = Int(q.dim(0))

    # =========================================================================#
    # QK RoPE and K cache RMSNorm                                              #
    # =========================================================================#

    # Create a view of the `q` tensor that only contains the last
    # qk_rope_head_dim columns of each Q head.
    var q_rope = TileTensor(
        q.ptr + qk_nope_head_dim,
        TileLayout(
            (Idx(seq_len), Idx[num_heads](), Idx[qk_rope_head_dim]()),
            (Idx[num_heads * q_head_dim](), Idx[q_head_dim](), Idx(1)),
        ),
    )

    # In-place update of the rope part of the `q` tensor
    var q_rope_mut = TileTensor(
        q_rope.ptr.mut_cast[True](),
        TileLayout(
            (Idx(seq_len), Idx[num_heads](), Idx[qk_rope_head_dim]()),
            (Idx[num_heads * q_head_dim](), Idx[q_head_dim](), Idx(1)),
        ),
    )

    mla_fused_rope_rmsnorm(
        q_rope_mut,
        q_rope.as_any_origin(),  # hack aliasing.
        input_row_offsets,
        freqs_cis,
        kv_norm_gamma,
        kv_collection,
        layer_idx,
        epsilon,
        ctx,
    )

    # =========================================================================#
    # Up-project the latent KV cache to full K and V                           #
    # =========================================================================#

    # First, dump the k cache to a contiguous buffer
    # allocate a buffer for raw latent KV values
    var k_latent_buf = ctx.enqueue_create_buffer[dtype](
        buffer_length * kv_latent_dim
    )
    var k_latent = TileTensor(
        k_latent_buf,
        row_major((Idx(buffer_length), Idx[kv_latent_dim]())),
    )

    # copy the k cache to the latent buffer
    var k_cache = kv_collection.get_key_cache(Int(layer_idx))
    _k_cache_to_buffer(
        buffer_row_offsets.to_layout_tensor(),
        cache_offsets.to_layout_tensor(),
        k_cache,
        Int32(buffer_length),
        k_latent.to_layout_tensor(),
        ctx,
    )

    # quantize the latent KV values to fp8
    # allocate buffers for fp8 latent KV values and scales
    # TODO: Fused the _k_cache_to_buffer with the quantize_dynamic_scaled_fp8
    var fp8_k_latent_buf = ctx.enqueue_create_buffer[fp8_dtype](
        buffer_length * kv_latent_dim
    )
    var fp8_k_latent = TileTensor(
        fp8_k_latent_buf,
        row_major((Idx(buffer_length), Idx[kv_latent_dim]())),
    )

    # the scales are stored in a transposed, padded format
    comptime scales_m_padding = 16 // size_of[fp8_scale_dtype]()
    var scales_padded_m = align_up(buffer_length, scales_m_padding)
    var fp8_k_latent_scale_buf = ctx.enqueue_create_buffer[fp8_scale_dtype](
        scales_padded_m * kv_latent_dim // k_scale_granularity
    )
    var fp8_k_latent_scale = TileTensor(
        fp8_k_latent_scale_buf,
        row_major(
            (Idx[kv_latent_dim // k_scale_granularity](), Idx(scales_padded_m))
        ),
    )

    @__copy_capture(k_latent)
    @always_inline
    @parameter
    fn input_fn[
        width: Int, alignment: Int
    ](row: Int, col: Int) -> SIMD[k_latent.dtype, width]:
        return k_latent.load[width=width]((Idx(row), Idx(col)))

    quantize_dynamic_scaled_fp8[
        input_fn, k_scale_granularity, k_latent.static_shape[1]
    ](
        fp8_k_latent._to_ndbuffer(),
        fp8_k_latent_scale._to_ndbuffer(),
        1200.0,
        ctx,
        Int(k_latent.dim[0]()),
    )

    # Up-project latent KV values to K.
    var k_buf = ctx.enqueue_create_buffer[dtype](
        buffer_length * num_heads * qk_nope_head_dim
    )
    var k_flat = TileTensor(
        k_buf,
        row_major((Idx(buffer_length), Idx[num_heads * qk_nope_head_dim]())),
    )
    matmul_dynamic_scaled_fp8[
        input_scale_granularity="block",
        weight_scale_granularity="block",
        m_scale_granularity=m_scale_granularity,
        n_scale_granularity=n_scale_granularity,
        k_scale_granularity=k_scale_granularity,
        transpose_b=True,
        target=target,
    ](
        k_flat,
        fp8_k_latent,
        w_k,
        fp8_k_latent_scale,
        w_k_scale,
        ctx,
    )

    # Reuse decode's rank-3 w_uv by flattening [H, Dv, K] -> [H*Dv, K].
    var w_v = TileTensor(
        w_uv.ptr,
        row_major((Idx[num_heads * v_head_dim](), Idx[kv_latent_dim]())),
    )
    var w_v_scale = TileTensor(
        w_uv_scale.ptr,
        row_major(
            (
                Idx[num_heads * (v_head_dim // n_scale_granularity)](),
                Idx[kv_latent_dim // k_scale_granularity](),
            )
        ),
    )

    # Up-project latent KV values to V.
    var v_buf = ctx.enqueue_create_buffer[dtype](
        buffer_length * num_heads * v_head_dim
    )
    var v_flat = TileTensor(
        v_buf,
        row_major((Idx(buffer_length), Idx[num_heads * v_head_dim]())),
    )
    matmul_dynamic_scaled_fp8[
        input_scale_granularity="block",
        weight_scale_granularity="block",
        m_scale_granularity=m_scale_granularity,
        n_scale_granularity=n_scale_granularity,
        k_scale_granularity=k_scale_granularity,
        transpose_b=True,
        target=target,
    ](
        v_flat,
        fp8_k_latent,
        w_v,
        fp8_k_latent_scale,
        w_v_scale,
        ctx,
    )

    var k = TileTensor(
        k_buf,
        row_major(
            (Idx(buffer_length), Idx[num_heads](), Idx[qk_nope_head_dim]())
        ),
    )
    var v = TileTensor(
        v_buf,
        row_major((Idx(buffer_length), Idx[num_heads](), Idx[v_head_dim]())),
    )

    generic_flare_mla_prefill_kv_cache_ragged[
        target=target,
        mask_str=mask_str,
        score_mod_str=score_mod_str,
    ](
        q.to_layout_tensor(),
        k.to_layout_tensor(),
        v.to_layout_tensor(),
        buffer_row_offsets.to_layout_tensor(),
        cache_offsets.to_layout_tensor(),
        input_row_offsets.to_layout_tensor(),
        kv_collection,
        layer_idx,
        scale,
        output.to_layout_tensor(),
        ctx,
    )


# ===-----------------------------------------------------------------------===#
# Manually fused MLA decode branch (FP8)
# ===-----------------------------------------------------------------------===#


@always_inline
fn quantize_and_bmm_fp8_helper[
    dtype: DType,
    fp8_dtype: DType,
    fp8_scale_dtype: DType,
    m_scale_granularity: Int,
    n_scale_granularity: Int,
    k_scale_granularity: Int,
    target: StaticString = "cpu",
](
    c: TileTensor[mut=True, dtype, address_space = AddressSpace.GENERIC, ...],
    a: TileTensor[dtype, address_space = AddressSpace.GENERIC, ...],
    b: TileTensor[fp8_dtype, address_space = AddressSpace.GENERIC, ...],
    b_scales: TileTensor[
        fp8_scale_dtype, address_space = AddressSpace.GENERIC, ...
    ],
    ctx: DeviceContext,
) raises:
    """
    Helper function to quantize and perform a batched matrix multiplication.
    This function uses the transposed view of the input tensor `a`.
    """

    comptime B = a.static_shape[1]
    comptime K = a.static_shape[2]
    comptime N = b.static_shape[1]

    var m = Int(a.dim(0))

    # allocate buffers for quantized a and its scales
    var fp8_a_buf = ctx.enqueue_create_buffer[fp8_dtype](B * m * K)
    var fp8_a = TileTensor(fp8_a_buf, row_major((Idx[B](), Idx(m), Idx[K]())))

    # the scales are stored in a transposed, padded format
    comptime scales_m_padding = 16 // size_of[fp8_scale_dtype]()
    var scales_padded_m = align_up(m, scales_m_padding)
    var fp8_a_scale_buf = ctx.enqueue_create_buffer[fp8_scale_dtype](
        B * (K // k_scale_granularity) * scales_padded_m
    )
    var fp8_a_scale = TileTensor(
        fp8_a_scale_buf,
        row_major(
            (Idx[B](), Idx[K // k_scale_granularity](), Idx(scales_padded_m))
        ),
    )

    @parameter
    @__copy_capture(a)
    @always_inline
    fn input_fn[
        width: Int, alignment: Int
    ](batch: Int, row: Int, col: Int) capturing -> SIMD[dtype, width]:
        # First transpose the q_nope tensor from [row, batch, col] to [batch, row, col].
        comptime assert a.flat_rank == 3
        return a.load[width=width]((Idx(row), Idx(batch), Idx(col)))

    batched_quantize_dynamic_scaled_fp8[
        input_fn=input_fn,
        group_size_or_per_token=k_scale_granularity,
        num_cols=K,
    ](
        fp8_a._to_ndbuffer(),
        fp8_a_scale._to_ndbuffer(),
        1200.0,
        ctx,
        num_rows=m,
        batch_size=B,
    )

    batched_matmul_dynamic_scaled_fp8[
        input_scale_granularity="block",
        weight_scale_granularity="block",
        m_scale_granularity=m_scale_granularity,
        n_scale_granularity=n_scale_granularity,
        k_scale_granularity=k_scale_granularity,
        transpose_b=True,
        target=target,
    ](
        c.to_layout_tensor(),
        fp8_a.to_layout_tensor(),
        b.to_layout_tensor(),
        fp8_a_scale.to_layout_tensor(),
        b_scales.to_layout_tensor(),
        ctx,
    )


fn mla_decode_branch_fp8[
    dtype: DType,
    fp8_dtype: DType,
    fp8_scale_dtype: DType,
    collection_t: KVCollectionT,
    //,
    m_scale_granularity: Int,
    n_scale_granularity: Int,
    k_scale_granularity: Int,
    mask_str: StaticString,
    score_mod_str: StaticString,
    target: StaticString = "cpu",
](
    output: TileTensor[
        mut=True, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    q: TileTensor[dtype, address_space = AddressSpace.GENERIC, ...],
    input_row_offsets: TileTensor[
        DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    freqs_cis: TileTensor[_, address_space = AddressSpace.GENERIC, ...],
    kv_norm_gamma: TileTensor[_, address_space = AddressSpace.GENERIC, ...],
    kv_collection: collection_t,
    layer_idx: UInt32,
    scale: Float32,
    epsilon: Float32,
    w_uk: TileTensor[fp8_dtype, address_space = AddressSpace.GENERIC, ...],
    w_uk_scale: TileTensor[
        fp8_scale_dtype, address_space = AddressSpace.GENERIC, ...
    ],
    w_uv: TileTensor[fp8_dtype, address_space = AddressSpace.GENERIC, ...],
    w_uv_scale: TileTensor[
        fp8_scale_dtype, address_space = AddressSpace.GENERIC, ...
    ],
    ctx: DeviceContext,
) raises:
    """
    This is a manually fused kernel that performs the following operations:
    - Apply RoPE to the query and the key cache (in-place).
    - Apply RMSNorm to the non-rope portion of the key cache (in-place).
    - Project q_nope to kv_latent_dim through a fp8 batched matmul:
        q_nope_proj = q_nope_t @ w_uk.
    - Concatenate q_nope_proj and q_rope:
        q_full = concat(q_nope_proj, q_rope, axis=2).
    - Perform MLA decode.
    - Project raw_output to v_head_dim through another fp8 batched matmul:
        output = raw_output_t @ w_uv.

    Parameters:
        dtype: Data type of the input and output tensors.
        fp8_dtype: Data type of the fp8 input and output tensors.
        fp8_scale_dtype: Data type of the fp8 scale input and output tensors.
        collection_t: Type of the KV collection.
        m_scale_granularity: Granularity of the scale for M dimension of the
            matrix multiplication.
        n_scale_granularity: Granularity of the scale for N dimension of the
            matrix multiplication.
        k_scale_granularity: Granularity of the scale for K dimension of the
            matrix multiplication.
        mask_str: Mask variant.
        score_mod_str: Positional encoding variant.
        target: Target device.

    Args:
        output: Output tensor of shape [tot_seq_len, num_heads, v_head_dim].
        q: Combined query tensor containing both nope and rope parts. Shape:
            [tot_seq_len, num_heads, qk_nope_head_dim + qk_rope_head_dim].
        input_row_offsets: Indicates where each request starts and ends in
            `q`. Shape: [num_batches + 1].
        freqs_cis: Precomputed RoPE frequency values for rotary position
            embeddings. Shape: [max_seq_len, qk_rope_head_dim].
        kv_norm_gamma: RMSNorm gamma weights for normalizing the KV cache.
            Shape: [kv_lora_rank].
        kv_collection: Paged KV Cache object.
        layer_idx: Layer index.
        scale: Scale for the attention calculation.
        epsilon: Small constant for numerical stability in RMSNorm.
        w_uk: Weight matrix for projecting the non-rope part of each query head to
            KV latent space. Shape: [num_heads, kv_latent_dim, qk_nope_head_dim].
        w_uk_scale: The scale for the w_uk weight matrix. Shape varies
            depending on the float8_config.
        w_uv: Weight matrix for projecting the output of the attention back to
            each head's original space. Shape: [num_heads, v_head_dim, kv_latent_dim].
        w_uv_scale: The scale for the w_uv weight matrix. Shape varies
            depending on the float8_config.
        ctx: Device context.
    """

    comptime kv_params = collection_t.kv_params
    comptime assert kv_params.is_mla, "kv_params.is_mla should be true"
    comptime assert kv_params.num_heads == 1, "kv_params.num_heads should be 1"

    comptime num_heads = q.static_shape[1]
    comptime q_head_dim = q.static_shape[2]
    comptime qk_rope_head_dim = freqs_cis.static_shape[1]
    comptime qk_nope_head_dim = q_head_dim - qk_rope_head_dim
    comptime v_head_dim = output.static_shape[2]
    comptime k_cache_dim = Int(kv_params.head_size)

    comptime assert (
        w_uk.shape_known and w_uv.shape_known
    ), "w_uk and w_uv's shapes should be static"
    comptime assert (
        w_uk.static_shape[2] == qk_nope_head_dim
    ), "w_uk.static_shape[2] should be equal to qk_nope_head_dim"
    comptime assert (
        w_uv.static_shape[1] == v_head_dim
    ), "w_uv.static_shape[1] should be equal to v_head_dim"
    comptime kv_latent_dim = w_uk.static_shape[1]
    comptime assert (
        kv_latent_dim + qk_rope_head_dim == k_cache_dim
    ), "kv_latent_dim + qk_rope_head_dim should be equal to kv_params.head_size"

    var seq_len = Int(q.dim(0))

    if seq_len == 0:
        return

    # First, create a input buffer for the mla decode kernel
    var mla_decode_input_buf = ctx.enqueue_create_buffer[dtype](
        seq_len * num_heads * k_cache_dim
    )
    var mla_decode_input = TileTensor(
        mla_decode_input_buf,
        row_major((Idx(seq_len), Idx[num_heads](), Idx[k_cache_dim]())),
    )

    # =========================================================================#
    # Project the non-rope part of each query head to kv_latent_dim            #
    # =========================================================================#

    # The first qk_nope_head_dim columns of each Q head will be up-projected to
    # kv_latent_dim through a fp8 batched matmul.

    # We start by create a view of the input tensor `q` that only contains the
    # first qk_nope_head_dim columns of each Q head.
    var q_nope = TileTensor(
        q.ptr,
        TileLayout(
            (Idx(seq_len), Idx[num_heads](), Idx[qk_nope_head_dim]()),
            (Idx[num_heads * q_head_dim](), Idx[q_head_dim](), Idx(1)),
        ),
    )

    # Then create a view of the mla_decode_input tensor that only contains the
    # first kv_latent_dim columns of each Q head.
    var mla_decode_input_nope = TileTensor(
        mla_decode_input.ptr,
        TileLayout(
            (Idx[num_heads](), Idx(seq_len), Idx[kv_latent_dim]()),
            (Idx[k_cache_dim](), Idx[num_heads * k_cache_dim](), Idx(1)),
        ),
    )

    # Proceed with the fp8 batched matmul
    # This helper function uses the transposed view of the input tensor `q_nope`.
    quantize_and_bmm_fp8_helper[
        m_scale_granularity=m_scale_granularity,
        n_scale_granularity=n_scale_granularity,
        k_scale_granularity=k_scale_granularity,
        target=target,
    ](mla_decode_input_nope, q_nope, w_uk, w_uk_scale, ctx)

    # =========================================================================#
    # QK RoPE and K cache RMSNorm                                              #
    # =========================================================================#

    # Create a view of the `q` tensor that only contains the last
    # qk_rope_head_dim columns of each Q head.
    var q_rope = TileTensor(
        q.ptr + qk_nope_head_dim,
        TileLayout(
            (Idx(seq_len), Idx[num_heads](), Idx[qk_rope_head_dim]()),
            (Idx[num_heads * q_head_dim](), Idx[q_head_dim](), Idx(1)),
        ),
    )

    # Create a view of the `mla_decode_input` tensor that only contains the last
    # qk_rope_head_dim columns of each Q head.
    var mla_decode_input_rope = TileTensor(
        mla_decode_input.ptr + kv_latent_dim,
        TileLayout(
            (Idx(seq_len), Idx[num_heads](), Idx[qk_rope_head_dim]()),
            (Idx[num_heads * k_cache_dim](), Idx[k_cache_dim](), Idx(1)),
        ),
    )

    mla_fused_rope_rmsnorm(
        mla_decode_input_rope,
        q_rope,
        input_row_offsets,
        freqs_cis,
        kv_norm_gamma,
        kv_collection,
        layer_idx,
        epsilon,
        ctx,
    )

    # =========================================================================#
    # MLA decode                                                               #
    # =========================================================================#

    var raw_output_buf = ctx.enqueue_create_buffer[dtype](
        seq_len * num_heads * kv_latent_dim
    )
    var raw_output = TileTensor(
        raw_output_buf,
        row_major((Idx(seq_len), Idx[num_heads](), Idx[kv_latent_dim]())),
    )

    generic_flare_mla_decode_kv_cache_ragged[
        target=target,
        mask_str=mask_str,
        score_mod_str=score_mod_str,
    ](
        mla_decode_input.to_layout_tensor(),
        input_row_offsets.to_layout_tensor(),
        kv_collection,
        layer_idx,
        scale,
        raw_output.to_layout_tensor(),
        ctx,
    )

    # Create a view of the output tensor with logical shape
    # [num_heads, seq_len, v_head_dim], and map directly to
    # [seq_len, num_heads, v_head_dim] physical memory.
    var output_t = TileTensor(
        output.ptr,
        TileLayout(
            (Idx[num_heads](), Idx(seq_len), Idx[v_head_dim]()),
            (Idx[v_head_dim](), Idx[num_heads * v_head_dim](), Idx(1)),
        ),
    )

    # Another batched matmul to project the raw output to the original space
    # This helper function uses the transposed view of the input tensor `raw_output`.
    quantize_and_bmm_fp8_helper[
        dtype=dtype,
        fp8_dtype=fp8_dtype,
        fp8_scale_dtype=fp8_scale_dtype,
        m_scale_granularity=m_scale_granularity,
        n_scale_granularity=n_scale_granularity,
        k_scale_granularity=k_scale_granularity,
        target=target,
    ](output_t, raw_output, w_uv, w_uv_scale, ctx)


# ===-----------------------------------------------------------------------===#
# MLA prefill-decode graph (FP8)
# ===-----------------------------------------------------------------------===#


@always_inline
fn mla_prefill_decode_graph_fp8[
    dtype: DType,
    fp8_dtype: DType,
    fp8_scale_dtype: DType,
    collection_t: KVCollectionT,
    //,
    m_scale_granularity: Int,
    n_scale_granularity: Int,
    k_scale_granularity: Int,
    mask_str: StaticString,
    score_mod_str: StaticString,
    target: StaticString = "cpu",
](
    output: TileTensor[
        mut=True, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    q: TileTensor[dtype, address_space = AddressSpace.GENERIC, ...],
    input_row_offsets: TileTensor[
        DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    freqs_cis: TileTensor[_, address_space = AddressSpace.GENERIC, ...],
    kv_norm_gamma: TileTensor[_, address_space = AddressSpace.GENERIC, ...],
    kv_collection: collection_t,
    layer_idx: UInt32,
    scale: Float32,
    epsilon: Float32,
    buffer_row_offsets: TileTensor[
        DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    cache_offsets: TileTensor[
        mut=True, DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    buffer_length: Int,
    max_seq_len: Int,
    w_k: TileTensor[fp8_dtype, address_space = AddressSpace.GENERIC, ...],
    w_k_scale: TileTensor[
        fp8_scale_dtype, address_space = AddressSpace.GENERIC, ...
    ],
    w_uk: TileTensor[fp8_dtype, address_space = AddressSpace.GENERIC, ...],
    w_uk_scale: TileTensor[
        fp8_scale_dtype, address_space = AddressSpace.GENERIC, ...
    ],
    w_uv: TileTensor[fp8_dtype, address_space = AddressSpace.GENERIC, ...],
    w_uv_scale: TileTensor[
        fp8_scale_dtype, address_space = AddressSpace.GENERIC, ...
    ],
    ctx: DeviceContext,
) raises:
    """
    This is a manually fused kernel that performs the following operations:
    - Perform MLA prefill or decode based on the maximum sequence length.
    """

    var seq_len = q.dim(0)

    if seq_len == 0:
        return

    # TODO: Remove this once prefill and decode branches support FP8 KV cache KERN-2394.
    comptime assert (
        collection_t.dtype == dtype
    ), "This KVCache DType is not supported."

    if max_seq_len == 1:
        mla_decode_branch_fp8[
            m_scale_granularity=m_scale_granularity,
            n_scale_granularity=n_scale_granularity,
            k_scale_granularity=k_scale_granularity,
            mask_str=mask_str,
            score_mod_str=score_mod_str,
            target=target,
        ](
            output,
            q,
            input_row_offsets,
            freqs_cis,
            kv_norm_gamma,
            kv_collection,
            layer_idx,
            scale,
            epsilon,
            w_uk,
            w_uk_scale,
            w_uv,
            w_uv_scale,
            ctx,
        )

    else:
        mla_prefill_branch_fp8[
            m_scale_granularity=m_scale_granularity,
            n_scale_granularity=n_scale_granularity,
            k_scale_granularity=k_scale_granularity,
            mask_str=mask_str,
            score_mod_str=score_mod_str,
            target=target,
        ](
            output,
            q,
            input_row_offsets,
            freqs_cis,
            kv_norm_gamma,
            kv_collection,
            layer_idx,
            scale,
            epsilon,
            buffer_row_offsets,
            cache_offsets,
            buffer_length,
            w_k,
            w_k_scale,
            w_uv,
            w_uv_scale,
            ctx,
        )


# ===-----------------------------------------------------------------------===#
# Manually fused MLA prefill branch (BF16)
# ===-----------------------------------------------------------------------===#


fn mla_prefill_branch_bf16[
    dtype: DType,
    collection_t: KVCollectionT,
    //,
    mask_str: StaticString,
    score_mod_str: StaticString,
    target: StaticString = "cpu",
](
    output: TileTensor[
        mut=True, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    q: TileTensor[dtype, address_space = AddressSpace.GENERIC, ...],
    input_row_offsets: TileTensor[
        DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    freqs_cis: TileTensor[_, address_space = AddressSpace.GENERIC, ...],
    kv_norm_gamma: TileTensor[_, address_space = AddressSpace.GENERIC, ...],
    kv_collection: collection_t,
    layer_idx: UInt32,
    scale: Float32,
    epsilon: Float32,
    buffer_row_offsets: TileTensor[
        DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    cache_offsets: TileTensor[
        mut=True, DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    buffer_length: Int,
    w_k: TileTensor[dtype, address_space = AddressSpace.GENERIC, ...],
    w_uv: TileTensor[dtype, address_space = AddressSpace.GENERIC, ...],
    ctx: DeviceContext,
) raises:
    """BF16 MLA prefill path.

    Applies RoPE and RMSNorm, up-projects latent KV to full K and V, then runs
    prefill attention.
    """
    comptime kv_params = collection_t.kv_params
    comptime assert kv_params.is_mla, "kv_params.is_mla should be true"
    comptime assert kv_params.num_heads == 1, "kv_params.num_heads should be 1"

    comptime num_heads = q.static_shape[1]
    comptime q_head_dim = q.static_shape[2]
    comptime qk_rope_head_dim = freqs_cis.static_shape[1]
    comptime qk_nope_head_dim = q_head_dim - qk_rope_head_dim
    comptime v_head_dim = output.static_shape[2]

    comptime assert w_k.shape_known, "w_k's shape should be static"
    comptime assert (
        w_k.static_shape[0] == num_heads * qk_nope_head_dim
    ), "w_k.shape[0] should be equal to num_heads * qk_nope_head_dim"
    comptime kv_latent_dim = w_k.static_shape[1]
    comptime assert w_uv.shape_known, "w_uv's shape should be static"
    comptime assert (
        w_uv.static_shape[0] == num_heads
    ), "w_uv.shape[0] should be equal to num_heads"
    comptime assert (
        w_uv.static_shape[1] == v_head_dim
    ), "w_uv.shape[1] should be equal to v_head_dim"
    comptime assert (
        w_uv.static_shape[2] == kv_latent_dim
    ), "w_uv.shape[2] should be equal to kv_latent_dim"

    if buffer_length == 0:
        return

    var seq_len = Int(q.dim(0))
    if seq_len == 0:
        return

    # =========================================================================#
    # QK RoPE and K cache RMSNorm                                              #
    # =========================================================================#

    # Create a view of the `q` tensor that only contains the last
    # qk_rope_head_dim columns of each Q head.
    var q_rope = TileTensor(
        q.ptr + qk_nope_head_dim,
        TileLayout(
            (Idx(seq_len), Idx[num_heads](), Idx[qk_rope_head_dim]()),
            (Idx[num_heads * q_head_dim](), Idx[q_head_dim](), Idx(1)),
        ),
    )

    # In-place update of the rope part of the `q` tensor
    var q_rope_mut = TileTensor(
        q_rope.ptr.mut_cast[True](),
        TileLayout(
            (Idx(seq_len), Idx[num_heads](), Idx[qk_rope_head_dim]()),
            (Idx[num_heads * q_head_dim](), Idx[q_head_dim](), Idx(1)),
        ),
    )

    mla_fused_rope_rmsnorm(
        q_rope_mut,
        q_rope.as_any_origin(),  # hack aliasing.
        input_row_offsets,
        freqs_cis,
        kv_norm_gamma,
        kv_collection,
        layer_idx,
        epsilon,
        ctx,
    )

    # allocate buffers for latent KV
    var k_latent_buf = ctx.enqueue_create_buffer[dtype](
        buffer_length * kv_latent_dim
    )
    var k_latent = TileTensor(
        k_latent_buf,
        row_major((Idx(buffer_length), Idx[kv_latent_dim]())),
    )

    var buffer_length_int = Int(buffer_length)
    var k_cache = kv_collection.get_key_cache(Int(layer_idx))

    _k_cache_to_buffer(
        buffer_row_offsets.to_layout_tensor(),
        cache_offsets.to_layout_tensor(),
        k_cache,
        Int32(buffer_length_int),
        k_latent.to_layout_tensor(),
        ctx,
    )

    var k_buf = ctx.enqueue_create_buffer[dtype](
        buffer_length * num_heads * qk_nope_head_dim
    )
    var k_flat = TileTensor(
        k_buf,
        row_major((Idx(buffer_length), Idx[num_heads * qk_nope_head_dim]())),
    )
    matmul[target=target, transpose_b=True](
        k_flat.to_layout_tensor(),
        k_latent.to_layout_tensor(),
        w_k.to_layout_tensor(),
        Optional(ctx),
    )

    var w_v = TileTensor(
        w_uv.ptr,
        row_major((Idx[num_heads * v_head_dim](), Idx[kv_latent_dim]())),
    )
    var v_buf = ctx.enqueue_create_buffer[dtype](
        buffer_length * num_heads * v_head_dim
    )
    var v_flat = TileTensor(
        v_buf,
        row_major((Idx(buffer_length), Idx[num_heads * v_head_dim]())),
    )
    matmul[target=target, transpose_b=True](
        v_flat.to_layout_tensor(),
        k_latent.to_layout_tensor(),
        w_v.to_layout_tensor(),
        Optional(ctx),
    )

    var k = TileTensor(
        k_buf,
        row_major(
            (Idx(buffer_length), Idx[num_heads](), Idx[qk_nope_head_dim]())
        ),
    )
    var v = TileTensor(
        v_buf,
        row_major((Idx(buffer_length), Idx[num_heads](), Idx[v_head_dim]())),
    )

    generic_flare_mla_prefill_kv_cache_ragged[
        target=target,
        mask_str=mask_str,
        score_mod_str=score_mod_str,
    ](
        q.to_layout_tensor(),
        k.to_layout_tensor(),
        v.to_layout_tensor(),
        buffer_row_offsets.to_layout_tensor(),
        cache_offsets.to_layout_tensor(),
        input_row_offsets.to_layout_tensor(),
        kv_collection,
        layer_idx,
        scale,
        output.to_layout_tensor(),
        ctx,
    )


# ===-----------------------------------------------------------------------===#
# Manually fused MLA decode branch (BF16)
# ===-----------------------------------------------------------------------===#


fn mla_decode_branch_bf16[
    dtype: DType,
    collection_t: KVCollectionT,
    //,
    mask_str: StaticString,
    score_mod_str: StaticString,
    target: StaticString = "cpu",
](
    output: TileTensor[
        mut=True, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    q: TileTensor[dtype, address_space = AddressSpace.GENERIC, ...],
    input_row_offsets: TileTensor[
        DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    freqs_cis: TileTensor[_, address_space = AddressSpace.GENERIC, ...],
    kv_norm_gamma: TileTensor[_, address_space = AddressSpace.GENERIC, ...],
    kv_collection: collection_t,
    layer_idx: UInt32,
    scale: Float32,
    epsilon: Float32,
    w_uk: TileTensor[dtype, address_space = AddressSpace.GENERIC, ...],
    w_uv: TileTensor[dtype, address_space = AddressSpace.GENERIC, ...],
    ctx: DeviceContext,
) raises:
    """BF16 MLA decode path.

    Applies RoPE and RMSNorm, projects q_nope to latent space, concatenates with
    q_rope, and runs decode.
    """
    comptime kv_params = collection_t.kv_params
    comptime assert kv_params.is_mla, "kv_params.is_mla should be true"
    comptime assert kv_params.num_heads == 1, "kv_params.num_heads should be 1"

    comptime num_heads = q.static_shape[1]
    comptime q_head_dim = q.static_shape[2]
    comptime qk_rope_head_dim = freqs_cis.static_shape[1]
    comptime qk_nope_head_dim = q_head_dim - qk_rope_head_dim
    comptime v_head_dim = output.static_shape[2]
    comptime k_cache_dim = Int(kv_params.head_size)

    comptime assert (
        w_uk.shape_known and w_uv.shape_known
    ), "w_uk and w_uv's shapes should be static"
    comptime assert (
        w_uk.static_shape[2] == qk_nope_head_dim
    ), "w_uk.static_shape[2] should be equal to qk_nope_head_dim"
    comptime kv_latent_dim = w_uk.static_shape[1]
    comptime assert (
        w_uv.static_shape[2] == kv_latent_dim
    ), "w_uv.static_shape[2] should be equal to kv_latent_dim"
    comptime assert (
        w_uv.static_shape[1] == v_head_dim
    ), "w_uv.static_shape[1] should be equal to v_head_dim"

    var seq_len = Int(q.dim(0))
    if seq_len == 0:
        return

    # First, create a input buffer for the mla decode kernel
    var mla_decode_input_buf = ctx.enqueue_create_buffer[dtype](
        seq_len * num_heads * k_cache_dim
    )
    var mla_decode_input = TileTensor(
        mla_decode_input_buf,
        row_major((Idx(seq_len), Idx[num_heads](), Idx[k_cache_dim]())),
    )

    # =========================================================================#
    # QK RoPE and K cache RMSNorm                                              #
    # =========================================================================#

    # Create a view of the `q` tensor that only contains the last
    # qk_rope_head_dim columns of each Q head.
    var q_rope = TileTensor(
        q.ptr + qk_nope_head_dim,
        TileLayout(
            (Idx(seq_len), Idx[num_heads](), Idx[qk_rope_head_dim]()),
            (Idx[num_heads * q_head_dim](), Idx[q_head_dim](), Idx(1)),
        ),
    )

    # Create a view of the `mla_decode_input` tensor that only contains the last
    # qk_rope_head_dim columns of each Q head.
    var mla_decode_input_rope = TileTensor(
        mla_decode_input.ptr + kv_latent_dim,
        TileLayout(
            (Idx(seq_len), Idx[num_heads](), Idx[qk_rope_head_dim]()),
            (Idx[num_heads * k_cache_dim](), Idx[k_cache_dim](), Idx(1)),
        ),
    )

    mla_fused_rope_rmsnorm(
        mla_decode_input_rope,
        q_rope,
        input_row_offsets,
        freqs_cis,
        kv_norm_gamma,
        kv_collection,
        layer_idx,
        epsilon,
        ctx,
    )

    # =========================================================================#
    # Project the non-rope part of each query head to kv_latent_dim            #
    # =========================================================================#

    # Create a view of the `q` tensor that only contains the first
    # qk_nope_head_dim columns of each Q head. Also transposed to
    # [num_heads, seq_len, qk_nope_head_dim].
    var q_nope_t = TileTensor(
        q.ptr,
        TileLayout(
            (Idx[num_heads](), Idx(seq_len), Idx[qk_nope_head_dim]()),
            (Idx[q_head_dim](), Idx[num_heads * q_head_dim](), Idx(1)),
        ),
    )

    # Then create a view of the mla_decode_input tensor that only contains the
    # first kv_latent_dim columns of each Q head.
    var mla_decode_input_nope = TileTensor(
        mla_decode_input.ptr,
        TileLayout(
            (Idx[num_heads](), Idx(seq_len), Idx[kv_latent_dim]()),
            (Idx[k_cache_dim](), Idx[num_heads * k_cache_dim](), Idx(1)),
        ),
    )

    _batched_matmul_gpu[transpose_b=True](
        mla_decode_input_nope, q_nope_t, w_uk, ctx
    )

    # Perform MLA decode
    var raw_output_buf = ctx.enqueue_create_buffer[dtype](
        seq_len * num_heads * kv_latent_dim
    )
    var raw_output = TileTensor(
        raw_output_buf,
        row_major((Idx(seq_len), Idx[num_heads](), Idx[kv_latent_dim]())),
    )
    generic_flare_mla_decode_kv_cache_ragged[
        target=target,
        mask_str=mask_str,
        score_mod_str=score_mod_str,
    ](
        mla_decode_input.to_layout_tensor(),
        input_row_offsets.to_layout_tensor(),
        kv_collection,
        layer_idx,
        scale,
        raw_output.to_layout_tensor(),
        ctx,
    )

    # Create a view of the raw output tensor with logical shape
    # [num_heads, seq_len, kv_latent_dim], and map directly to
    # [seq_len, num_heads, kv_latent_dim] physical memory.
    var raw_output_t = TileTensor(
        raw_output_buf,
        TileLayout(
            (Idx[num_heads](), Idx(seq_len), Idx[kv_latent_dim]()),
            (Idx[kv_latent_dim](), Idx[num_heads * kv_latent_dim](), Idx(1)),
        ),
    )

    # Create a view of the output tensor with logical shape
    # [num_heads, seq_len, v_head_dim], and map directly to
    # [seq_len, num_heads, v_head_dim] physical memory.
    var output_t = TileTensor(
        output.ptr,
        TileLayout(
            (Idx[num_heads](), Idx(seq_len), Idx[v_head_dim]()),
            (Idx[v_head_dim](), Idx[num_heads * v_head_dim](), Idx(1)),
        ),
    )

    _batched_matmul_gpu[transpose_b=True](output_t, raw_output_t, w_uv, ctx)


# ===-----------------------------------------------------------------------===#
# MLA prefill-decode graph (BF16)
# ===-----------------------------------------------------------------------===#


@always_inline
fn mla_prefill_decode_graph_bf16[
    dtype: DType,
    collection_t: KVCollectionT,
    //,
    mask_str: StaticString,
    score_mod_str: StaticString,
    target: StaticString = "cpu",
](
    output: TileTensor[
        mut=True, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    q: TileTensor[dtype, address_space = AddressSpace.GENERIC, ...],
    input_row_offsets: TileTensor[
        DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    freqs_cis: TileTensor[_, address_space = AddressSpace.GENERIC, ...],
    kv_norm_gamma: TileTensor[_, address_space = AddressSpace.GENERIC, ...],
    kv_collection: collection_t,
    layer_idx: UInt32,
    scale: Float32,
    epsilon: Float32,
    buffer_row_offsets: TileTensor[
        mut=False, DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    cache_offsets: TileTensor[
        mut=True, DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    buffer_length: Int,
    max_seq_len: Int,
    w_k: TileTensor[dtype, address_space = AddressSpace.GENERIC, ...],
    w_uk: TileTensor[dtype, address_space = AddressSpace.GENERIC, ...],
    w_uv: TileTensor[dtype, address_space = AddressSpace.GENERIC, ...],
    ctx: DeviceContext,
) raises:
    """BF16 MLA prefill/decode graph.

    Dispatches to prefill or decode based on max sequence length in the batch.
    """
    var seq_len = q.dim(0)

    if seq_len == 0:
        return

    # TODO(SERVOPT-967): generalize to number of draft tokens.
    if max_seq_len == 1:
        mla_decode_branch_bf16[
            mask_str=mask_str,
            score_mod_str=score_mod_str,
            target=target,
        ](
            output,
            q,
            input_row_offsets,
            freqs_cis,
            kv_norm_gamma,
            kv_collection,
            layer_idx,
            scale,
            epsilon,
            w_uk,
            w_uv,
            ctx,
        )
    else:
        mla_prefill_branch_bf16[
            mask_str=mask_str,
            score_mod_str=score_mod_str,
            target=target,
        ](
            output,
            q,
            input_row_offsets,
            freqs_cis,
            kv_norm_gamma,
            kv_collection,
            layer_idx,
            scale,
            epsilon,
            buffer_row_offsets,
            cache_offsets,
            buffer_length,
            w_k,
            w_uv,
            ctx,
        )
