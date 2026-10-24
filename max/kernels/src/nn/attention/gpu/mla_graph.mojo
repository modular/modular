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

"""Provides manually fused GPU graph kernels for Multi-head Latent Attention (MLA).

Defines the fused RoPE and RMSNorm kernel, the FP8 prefill and decode branches,
and the combined prefill-decode graph that up-projects the latent KV cache and
performs MLA attention with dynamic FP8 scaling.
"""


from std.collections import Optional, OptionalReg
from std.math import abs, align_up, ceildiv
from std.memory import bitcast

from std.sys import simd_width_of, size_of
from std.sys.info import align_of
from std.utils.index import Index, IndexList

from max.algorithm.functional import _elementwise_impl_gpu
from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_idx,
    global_idx,
    grid_dim,
    lane_id,
    thread_idx,
)
import max.gpu.primitives.block as block
from max.gpu.primitives.grid_controls import (
    PDL,
    PDLLevel,
    pdl_launch_attributes,
)
from max.gpu.host import DeviceAttribute, DeviceBuffer, DeviceContext
from max.gpu.host.info import _is_sm10x_gpu
from max.gpu.primitives.warp import shuffle_xor
from layout import (
    Coord,
    Idx,
    TensorEngine,
    TensorLayout,
    TileTensor,
    row_major,
)
from layout.layout import *
from layout.tile_layout import Layout as TileLayout
from linalg.bmm import _batched_matmul_gpu, batched_matmul_dynamic_scaled_fp8
from linalg.matmul import matmul
from std.utils.index import StaticTuple
from std.utils.numerics import get_accum_type
from linalg.fp8_quantization import (
    matmul_dynamic_scaled_fp8,
    quantize_dynamic_scaled_fp8,
    batched_quantize_dynamic_scaled_fp8,
)
from internal_utils.fp8_utils import cast_saturating
from linalg.fp4_utils import (
    NVFP4_SF_DTYPE,
    NVFP4_SF_VECTOR_SIZE,
    nvfp4_kv_group_scale,
    nvfp4_kv_quantize_x8,
)
from nn._ragged_utils import get_batch_and_token_idx_from_row_offsets
from nn.fused_qk_rope import rope_k_cache, rope_q_proj, rope_value
from kv_cache.types import KVCacheStaticParams
from nn.kv_cache import KVCollectionT, KVCacheT
from nn.kv_cache_ragged import (
    generic_flare_mla_decode_kv_cache_ragged,
    generic_flare_mla_prefill_kv_cache_ragged,
)
from nn.attention.gpu.mla import (
    _k_cache_to_buffer,
    flare_mla_decoding,
    flare_mla_decoding_rows,
    mla_decode_max_seq_len,
)
from nn.attention.gpu.mha_decode_partition_heuristic import (
    amd_sparse_mla_decode_num_partitions,
)
from nn.attention.mha_operand import KVCacheMHAOperand, SparseRowsMHAOperand
from nn.attention.gpu.nvidia.sm100.mla_decode_utils import (
    SparseRowLayout,
    assert_sparse_row_packed,
)
from nn.attention.mha_mask import PaddingMask, SparseRowsDecodeMask
from nn.attention.mha_utils import (
    MHAConfig,
    NullPointer,
    OptionalPointer,
    as_optional_reg,
    null_pointer,
    unread_pointer,
)
from nn.attention.gpu.nvidia.sm100.mla_prefill import (
    mla_sm100_prefill_sparse,
    mla_sm100_prefill_sparse_fp8,
)
from nn.attention.gpu.nvidia.sm100.mla_nvfp4_kv_to_fp8 import (
    mla_prefill_sparse_nvfp4_via_fp8,
    nvfp4_prefill_converts_once,
    nvfp4_prefill_scratch_rows,
)
from nn.attention.gpu.nvidia.sm100.mla_prefill_sparse_qkv_fp8 import (
    mla_prefill_sparse_qkv_fp8,
)
from nn.attention.gpu.nvidia.sm100.mla_prefill_sparse_utils import (
    MLASparseConfig,
)
from nn.normalization import _rms_norm_warp_tiling_subkernel


def _mla_nvfp4_cache[cache_t: KVCacheT]() -> Bool:
    """Whether this latent cache is packed NVFP4 with E4M3 scales."""
    return (
        cache_t.dtype == .uint8
        and cache_t.quantization_enabled
        and cache_t.scale_dtype == .float8_e4m3fn
    )


def _mla_cache_row_elems[kv_params: KVCacheStaticParams, dtype: DType]() -> Int:
    """The logical elements of one latent-cache row.

    A packed NVFP4 (`uint8`) row stores two e2m1 values per byte, except for
    the FP8 rope bytes at its end.
    """
    comptime if dtype == DType.uint8:
        return kv_params.nvfp4_row_elems()
    return kv_params.head_size


def _mla_native_fp8_cache[cache_t: KVCacheT]() -> Bool:
    """Whether the SM100 FP8-Q sparse kernels read this latent cache directly:
    a unit-scale FP8 cache, or packed NVFP4 that they dequantize to FP8."""
    comptime unit_fp8 = (
        cache_t.dtype == .float8_e4m3fn and not cache_t.quantization_enabled
    )
    return unit_fp8 or _mla_nvfp4_cache[cache_t]()


def _mla_sparse_row_topk[cache_t: KVCacheT](indices_stride: Int) -> Int:
    """The top-k slots of a sparse index row `indices_stride` entries wide.

    An NVFP4 cache pages its scales apart from its values, so the indexer
    follows each row's value rows with the matching scale-pool rows.
    """
    comptime if _mla_nvfp4_cache[cache_t]():
        return indices_stride // 2
    return indices_stride


# NVFP4 latent-cache write helper
# ===-----------------------------------------------------------------------===#


@always_inline
def _store_nvfp4_latent_slice[
    dtype: DType, cache_t: KVCacheT, width: Int
](
    k_cache: cache_t,
    batch_idx: Int,
    post_seq_idx: Int,
    col: Int,
    row: SIMD[dtype, width],
    active: Bool,
):
    """Quantizes one thread's slice of an MLA latent row to NVFP4 and stores it.

    Targets NVIDIA SM100. Mirrors the packing and scale conventions of
    `nvfp4_kv_write_row` in `nn/kv_cache.mojo`: packed E2M1 with element 0 in
    the low nibble, one E4M3 scale per 16 elements, per-tensor scale fixed at
    1. A 16-element group spans `16 / width` adjacent lanes, which exchange
    their amax with a pair-scoped shuffle so the helper is safe to call from
    a divergent branch as long as the lanes of one group branch together.

    Parameters:
        dtype: Compute dtype the row was rounded to before quantization.
        cache_t: Type of the paged key cache (a `uint8` NVFP4 cache).
        width: Elements held per thread.

    Args:
        k_cache: The NVFP4 key cache.
        batch_idx: Batch entry of the token.
        post_seq_idx: Cache position the row is written to.
        col: First element column of this thread's slice within the row.
        row: The normed and roped slice, already rounded to `dtype`.
        active: Whether the slice is a real part of the row. Inactive lanes
            must hold zeros and only ever share a group with each other.
    """
    comptime group = NVFP4_SF_VECTOR_SIZE
    comptime assert (
        cache_t.quantization_enabled and cache_t.scale_dtype == NVFP4_SF_DTYPE
    ), "an NVFP4 KV cache needs E4M3 scales"
    comptime assert (
        cache_t.quantization_granularity * 2 == group
    ), "NVFP4 KV scales cover 16 elements (8 packed bytes)"
    comptime assert (
        width % 8 == 0 and group % width == 0
    ), "the per-thread width must tile a 16-element scale group in 8s"

    # A group spans `lanes` adjacent threads.
    comptime lanes = group // width
    var x = row.cast[DType.float32]()
    var amax = abs(x).reduce_max()
    comptime if lanes > 1:
        var group_mask = UInt((1 << lanes) - 1) << UInt(
            lane_id() & ~(lanes - 1)
        )
        comptime for k in range(4):
            comptime if (1 << k) < lanes:
                amax = max(amax, shuffle_xor(group_mask, amax, UInt32(1 << k)))
    var scale = nvfp4_kv_group_scale(amax)

    var packed = SIMD[DType.uint32, width // 8]()
    comptime for w in range(width // 8):
        packed[w] = nvfp4_kv_quantize_x8(x.slice[8, offset=w * 8](), scale)

    if active:
        k_cache.store(
            bs=batch_idx,
            head_idx=0,  # num_k_heads is 1 for MLA
            tok_idx=post_seq_idx,
            head_dim_idx=col // 2,
            val=rebind[SIMD[cache_t.dtype, width // 2]](
                bitcast[DType.uint8, width // 2](packed)
            ),
        )
        if (col // width) % lanes == 0:
            k_cache.store_scale(
                bs=batch_idx,
                head_idx=0,
                tok_idx=post_seq_idx,
                head_dim_idx=col // 2,
                scales=SIMD[NVFP4_SF_DTYPE, 1](scale.cast[NVFP4_SF_DTYPE]()),
            )


# Manually fused MLA RoPE and RMSNorm kernel
# ===-----------------------------------------------------------------------===#


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(block_size))
)
@__name(t"fused_rope_rmsnorm_{dtype}")
def fused_rope_rmsnorm_kernel[
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
        mut=True, dtype, QRopeOutputLayoutType, MutUntrackedOrigin
    ],
    q_rope: TileTensor[dtype, QRopeLayoutType, ImmUntrackedOrigin],
    input_row_offsets: TileTensor[
        .uint32, InputRowOffsetsLayoutType, ImmUntrackedOrigin
    ],
    freqs_cis: TileTensor[freq_dtype, FreqsCisLayoutType, ImmUntrackedOrigin],
    gamma: TileTensor[gamma_dtype, GammaLayoutType, ImmUntrackedOrigin],
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

    # Evidence asserts for TileTensor load/store Coord constraints.
    comptime assert (
        TileTensor[freq_dtype, FreqsCisLayoutType, ImmUntrackedOrigin].flat_rank
        >= 2
    )
    comptime assert (
        TileTensor[gamma_dtype, GammaLayoutType, ImmUntrackedOrigin].flat_rank
        >= 1
    )

    comptime num_q_heads = q_rope.static_shape[1]
    comptime rope_dim = q_rope.static_shape[2]
    comptime kv_norm_dim = gamma.static_shape[0]

    var worker_idx = block_idx.y
    var num_workers = grid_dim.y
    var num_tokens = q_rope.dim(0)

    with PDL():
        for global_token_idx in range(worker_idx, Int(num_tokens), num_workers):
            var batch_idx, token_idx = get_batch_and_token_idx_from_row_offsets(
                input_row_offsets, global_token_idx
            )
            var post_seq_idx = k_cache.cache_length(batch_idx) + token_idx

            # First n_rope_blocks blocks of this worker process RoPE.
            if block_idx.x < n_rope_blocks:
                comptime q_width = simd_width_of[dtype]()
                comptime assert (
                    rope_dim % q_width == 0
                ), "rope_dim should be divisible by q_width"

                var head_idx, head_dim_idx = divmod(
                    global_idx.x * q_width, rope_dim
                )
                var f_c = freqs_cis.load[width=q_width](
                    (post_seq_idx, head_dim_idx)
                )

                if head_idx < num_q_heads:
                    rope_q_proj[interleaved=True](
                        q_rope,
                        q_rope_output,
                        (global_token_idx, head_idx, head_dim_idx),
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
                var gamma_val = SIMD[gamma_dtype, k_width](0)

                var idx = thread_idx.x * k_width
                if idx < kv_norm_dim:
                    vec_data = k_cache.load[width=k_width](
                        batch_idx,
                        0,  # num_k_heads is 1 for MLA
                        post_seq_idx,
                        idx,
                    ).cast[accum_type]()
                    # Prefetch gamma before reduction.
                    gamma_val = gamma.load[
                        width=k_width,
                        alignment=align_of[SIMD[gamma_dtype, k_width]](),
                    ](Coord(idx))

                var norm_val = _rms_norm_warp_tiling_subkernel[
                    warps_per_block,
                    False,  # Do not multiply the gamma before casting.
                ](
                    global_token_idx,
                    idx,
                    vec_data,
                    gamma_val,
                    epsilon,
                    0.0,
                    kv_norm_dim,
                )

                if idx < kv_norm_dim:
                    k_cache.store(
                        batch_idx,
                        0,  # num_k_heads is 1 for MLA
                        post_seq_idx,
                        idx,
                        cast_saturating[k_dtype](norm_val),
                    )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(block_size))
)
@__name(t"fused_rope_rmsnorm_quantization_{dtype}_{out_rope_dtype}")
def fused_rope_rmsnorm_quantization_kernel[
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
    out_rope_dtype: DType,
    KVInputFnType: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int](IndexList[2]) -> SIMD[DType.bfloat16, width],
    *,
    QRopeOutputEngine: TensorEngine,
    QRopeEngine: TensorEngine,
    InputRowOffsetsEngine: TensorEngine,
    FreqsCisEngine: TensorEngine,
    GammaEngine: TensorEngine,
](
    q_rope_output: TileTensor[
        mut=True,
        out_rope_dtype,
        QRopeOutputLayoutType,
        MutUntrackedOrigin,
        Engine=QRopeOutputEngine,
    ],
    q_rope: TileTensor[
        dtype, QRopeLayoutType, ImmUntrackedOrigin, Engine=QRopeEngine
    ],
    input_row_offsets: TileTensor[
        .uint32,
        InputRowOffsetsLayoutType,
        ImmUntrackedOrigin,
        Engine=InputRowOffsetsEngine,
    ],
    freqs_cis: TileTensor[
        freq_dtype,
        FreqsCisLayoutType,
        ImmUntrackedOrigin,
        Engine=FreqsCisEngine,
    ],
    gamma: TileTensor[
        gamma_dtype, GammaLayoutType, ImmUntrackedOrigin, Engine=GammaEngine
    ],
    k_cache: cache_t,
    epsilon: Float32,
    kv_input_fn: KVInputFnType,
) -> None:
    """Fused GPU kernel that applies RoPE to query projections and RMSNorm to KV
    cache, reading the inputs from a KV buffer and quantizing the final results
    before writing to the KVCache object.

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
        out_rope_dtype: Data type of the RoPE output.
        KVInputFnType: Type of the closure that loads the KV latent values.
        QRopeOutputEngine: Storage engine backing the `q_rope_output` tile.
        QRopeEngine: Storage engine backing the `q_rope` tile.
        InputRowOffsetsEngine: Storage engine backing the `input_row_offsets` tile.
        FreqsCisEngine: Storage engine backing the `freqs_cis` tile.
        GammaEngine: Storage engine backing the `gamma` tile.

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
        kv_input_fn: Closure that loads the KV latent values. Shape:
            [tot_seq_len, cache_head_dim]. Where cache_head_dim = kv_lora_rank
            + qk_rope_head_dim.
    """
    comptime assert (
        cache_t.kv_params.num_heads == 1
    ), "num_heads should be 1 for MLA"
    comptime assert q_rope_output.flat_rank == 3
    comptime assert q_rope.flat_rank == 3
    comptime assert input_row_offsets.flat_rank == 1
    comptime assert freqs_cis.flat_rank == 2
    comptime assert gamma.flat_rank == 1

    # Evidence asserts for TileTensor load/store Coord constraints.
    comptime assert (
        TileTensor[freq_dtype, FreqsCisLayoutType, ImmUntrackedOrigin].flat_rank
        >= 2
    )
    comptime assert (
        TileTensor[gamma_dtype, GammaLayoutType, ImmUntrackedOrigin].flat_rank
        >= 1
    )

    comptime num_q_heads = q_rope.static_shape[1]
    comptime rope_dim = q_rope.static_shape[2]
    comptime kv_norm_dim = gamma.static_shape[0]
    comptime cache_dtype = cache_t.dtype
    comptime k_width = simd_width_of[dtype]()

    var worker_idx = block_idx.y
    var num_workers = grid_dim.y
    var num_tokens = q_rope.dim(0)

    with PDL():
        for global_token_idx in range(worker_idx, Int(num_tokens), num_workers):
            var batch_idx, token_idx = get_batch_and_token_idx_from_row_offsets(
                input_row_offsets, global_token_idx
            )
            var post_seq_idx = k_cache.cache_length(batch_idx) + token_idx

            # First n_rope_blocks blocks of this worker process RoPE.
            if block_idx.x < n_rope_blocks:
                comptime q_width = simd_width_of[dtype]()
                comptime assert (
                    rope_dim % q_width == 0
                ), "rope_dim should be divisible by q_width"

                var head_idx, head_dim_idx = divmod(
                    global_idx.x * q_width, rope_dim
                )
                var f_c = freqs_cis.load[width=q_width](
                    (post_seq_idx, head_dim_idx)
                )

                if head_idx < num_q_heads:
                    # One `alignment` covers both the load and the store, so it
                    # must hold for the narrower of the two element types.
                    rope_q_proj[
                        interleaved=True,
                        alignment=align_of[SIMD[out_rope_dtype, q_width]](),
                    ](
                        q_rope,
                        q_rope_output,
                        (global_token_idx, head_idx, head_dim_idx),
                        f_c,
                        rope_dim,
                    )
                elif head_idx == num_q_heads:
                    var val = kv_input_fn[width=k_width](
                        {global_token_idx, kv_norm_dim + head_dim_idx}
                    )
                    var roped_val = rope_value(val, f_c).cast[dtype]()
                    comptime if cache_dtype == DType.uint8:
                        # NVFP4: the rope dims stay FP8, exactly as an FP8
                        # cache stores them, in the bytes after the packed
                        # latent. Rope dominates the attention scores, so e2m1
                        # there costs accuracy.
                        comptime assert (
                            cache_t.kv_params.fp8_tail_size == rope_dim
                        ), "an NVFP4 MLA row ends in its FP8 rope dims"
                        k_cache.store(
                            batch_idx,
                            0,  # num_k_heads is 1 for MLA
                            post_seq_idx,
                            kv_norm_dim // 2 + head_dim_idx,
                            rebind[SIMD[cache_dtype, k_width]](
                                bitcast[DType.uint8, k_width](
                                    cast_saturating[DType.float8_e4m3fn](
                                        roped_val
                                    )
                                )
                            ),
                        )
                    else:
                        k_cache.store(
                            batch_idx,
                            0,  # num_k_heads is 1 for MLA
                            post_seq_idx,
                            head_dim_idx + kv_norm_dim,
                            cast_saturating[cache_dtype](roped_val),
                        )

            # The last block of this worker processes RMSNorm.
            else:
                comptime accum_type = get_accum_type[dtype]()
                comptime warps_per_block = block_size // WARP_SIZE

                comptime assert (
                    kv_norm_dim % k_width == 0
                ), "kv_norm_dim should be divisible by k_width"

                var vec_data = SIMD[accum_type, k_width](0)
                var gamma_val = SIMD[gamma_dtype, k_width](0)

                var idx = thread_idx.x * k_width
                if idx < kv_norm_dim:
                    vec_data = kv_input_fn[width=k_width](
                        {global_token_idx, idx}
                    ).cast[accum_type]()
                    # Prefetch gamma before reduction.
                    gamma_val = gamma.load[
                        width=k_width,
                        alignment=align_of[SIMD[gamma_dtype, k_width]](),
                    ](Coord(idx))

                var norm_val = _rms_norm_warp_tiling_subkernel[
                    warps_per_block,
                    False,  # Do not multiply the gamma before casting.
                ](
                    global_token_idx,
                    idx,
                    vec_data,
                    gamma_val,
                    epsilon,
                    0.0,
                    kv_norm_dim,
                )

                comptime if cache_dtype == DType.uint8:
                    # NVFP4 quantizes the row rounded to the compute dtype,
                    # and every thread takes part in the group-amax shuffle.
                    _store_nvfp4_latent_slice(
                        k_cache,
                        batch_idx,
                        post_seq_idx,
                        idx,
                        norm_val.cast[dtype](),
                        idx < kv_norm_dim,
                    )
                else:
                    if idx < kv_norm_dim:
                        k_cache.store(
                            batch_idx,
                            0,  # num_k_heads is 1 for MLA
                            post_seq_idx,
                            idx,
                            cast_saturating[cache_dtype](norm_val),
                        )


@inline(.always)
def mla_fused_rope_rmsnorm_quantization[
    dtype: DType,
    freq_dtype: DType,
    gamma_dtype: DType,
    collection_t: KVCollectionT,
    out_rope_dtype: DType,
    KVInputFnType: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int](IndexList[2]) -> SIMD[DType.bfloat16, width],
    //,
](
    q_rope_output: TileTensor[mut=True, out_rope_dtype, ...],
    q_rope: TileTensor[mut=False, dtype, ...],
    input_row_offsets: TileTensor[mut=False, .uint32, ...],
    freqs_cis: TileTensor[mut=False, freq_dtype, ...],
    gamma: TileTensor[mut=False, gamma_dtype, ...],
    kv_collection: collection_t,
    layer_idx: UInt32,
    epsilon: Float32,
    kv_input_fn: KVInputFnType,
    ctx: DeviceContext,
) raises:
    """Launches the fused RoPE and RMSNorm kernel for MLA attention.

    This function fuses three operations:
    1. RoPE applied to query and key cache rope parts.
    2. RMSNorm applied to the non-rope portion of the key cache.
    3. Quantization of the final results before writing to the KVCache object.

    Parameters:
        dtype: Data type of query tensors.
        freq_dtype: Data type of frequency cosine/sine values.
        gamma_dtype: Data type of RMSNorm gamma weights.
        collection_t: Type of the KV cache collection.
        out_rope_dtype: Data type of the RoPE output values.
        KVInputFnType: Type of the closure that loads the KV latent values
            (inferred).

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
        kv_input_fn: Closure that loads the KV latent values. Shape:
            [tot_seq_len, cache_head_dim]. Where cache_head_dim = kv_lora_rank
            + qk_rope_head_dim. It runs on the device, so it must copy what
            it captures.
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
    comptime cache_row_elems = _mla_cache_row_elems[
        collection_t.kv_params, collection_t.dtype
    ]()
    comptime assert (
        rope_dim + kv_norm_dim == cache_row_elems
    ), "rope_dim + kv_norm_dim must be equal to kvcache head_size"

    # Default block size used by the `elementwise` function on Blackwell.
    comptime block_size = 128
    comptime kernel_simd_width = simd_width_of[dtype, target=ctx.target]()
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

    comptime kernel = fused_rope_rmsnorm_quantization_kernel[
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
        out_rope_dtype,
        KVInputFnType=KVInputFnType,
        QRopeOutputEngine=q_rope_output.Engine,
        QRopeEngine=q_rope.Engine,
        InputRowOffsetsEngine=input_row_offsets.Engine,
        FreqsCisEngine=freqs_cis.Engine,
        GammaEngine=gamma.Engine,
    ]

    ctx.enqueue_function[kernel](
        q_rope_output,
        q_rope.as_imm(),
        input_row_offsets.as_imm(),
        freqs_cis.as_imm(),
        gamma.as_imm(),
        k_cache,
        epsilon,
        host_args=(kv_input_fn,),
        grid_dim=(n_rope_blocks + n_rms_blocks, num_workers, 1),
        block_dim=block_size,
        attributes=pdl_launch_attributes(PDLLevel.ON),
    )


# ===-----------------------------------------------------------------------===#
# Manually fused MLA prefill branch (FP8)
# ===-----------------------------------------------------------------------===#


def mla_prefill_branch_fp8[
    dtype: DType,
    fp8_dtype: DType,
    fp8_scale_dtype: DType,
    collection_t: KVCollectionT,
    KVInputFnType: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int](IndexList[2]) -> SIMD[DType.bfloat16, width],
    //,
    m_scale_granularity: Int,
    n_scale_granularity: Int,
    k_scale_granularity: Int,
    mask_str: StaticString,
    target: StaticString = "cpu",
](
    output: TileTensor[mut=True, dtype, address_space=.GENERIC, ...],
    q: TileTensor[dtype, address_space=.GENERIC, ...],
    input_row_offsets: TileTensor[.uint32, address_space=.GENERIC, ...],
    freqs_cis: TileTensor[_, address_space=.GENERIC, ...],
    kv_norm_gamma: TileTensor[_, address_space=.GENERIC, ...],
    kv_collection: collection_t,
    layer_idx: UInt32,
    scale: Float32,
    epsilon: Float32,
    buffer_row_offsets: TileTensor[.uint32, address_space=.GENERIC, ...],
    cache_offsets: TileTensor[mut=True, .uint32, address_space=.GENERIC, ...],
    buffer_length: Int,
    w_k: TileTensor[fp8_dtype, address_space=.GENERIC, ...],
    w_k_scale: TileTensor[fp8_scale_dtype, address_space=.GENERIC, ...],
    w_uv: TileTensor[fp8_dtype, address_space=.GENERIC, ...],
    w_uv_scale: TileTensor[fp8_scale_dtype, address_space=.GENERIC, ...],
    kv_input_fn: KVInputFnType,
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
        KVInputFnType: Type of the closure that loads the KV latent
            values (inferred).
        m_scale_granularity: Granularity of the scale for M dimension of the
            matrix multiplication.
        n_scale_granularity: Granularity of the scale for N dimension of the
            matrix multiplication.
        k_scale_granularity: Granularity of the scale for K dimension of the
            matrix multiplication.
        mask_str: Mask variant.
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
        kv_input_fn: Closure that loads the KV latent values. Shape:
            [tot_seq_len, cache_head_dim]. Where cache_head_dim = kv_lora_rank
            + qk_rope_head_dim. It runs on the device, so it must copy what
            it captures.
        ctx: Device context.
    """
    comptime kv_params = collection_t.kv_params
    comptime assert kv_params.is_mla, "kv_params.is_mla should be true"
    comptime assert kv_params.num_heads == 1, "kv_params.num_heads should be 1"

    comptime num_heads = q.static_shape[1]
    comptime q_head_dim = q.static_shape[2]
    comptime v_head_dim = output.static_shape[2]
    comptime k_cache_dim = kv_params.head_size

    comptime assert w_k.shape_known, "w_k's shape should be static"
    comptime kv_latent_dim = w_k.static_shape[1]
    # Derive the rotary width from cache+latent, not freqs_cis, so a NoPE model
    # with no rotary table can still reach this op.
    comptime qk_rope_head_dim = k_cache_dim - kv_latent_dim
    comptime qk_nope_head_dim = q_head_dim - qk_rope_head_dim
    comptime assert (
        qk_rope_head_dim == 0 or freqs_cis.static_shape[1] == qk_rope_head_dim
    ), "freqs_cis's width should be k_cache_dim - kv_latent_dim."
    comptime assert (
        w_k.static_shape[0] == num_heads * qk_nope_head_dim
    ), "w_k.shape[0] should be equal to num_heads * qk_nope_head_dim"
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
        n_scale_granularity == k_scale_granularity
        and n_scale_granularity in (64, 128)
    ), (
        "n_scale_granularity and k_scale_granularity must be equal and in"
        " (64, 128)"
    )

    # Return early if we have no tokens to process.
    if buffer_length == 0:
        return

    var seq_len = Int(q.dim(0))

    # =========================================================================#
    # QK RoPE and K cache RMSNorm                                              #
    # =========================================================================#

    # Create a view of the `q` tensor that only contains the last
    # qk_rope_head_dim columns of each Q head. The slice carries runtime
    # extents, so reshape restores the static trailing dims.
    var q_rope = q[:, :, qk_nope_head_dim:].reshape(
        TileLayout(
            (seq_len, Idx[num_heads], Idx[qk_rope_head_dim]),
            (Idx[num_heads * q_head_dim], Idx[q_head_dim], Idx[1]),
        )
    )

    # In-place update of the rope part of the `q` tensor
    var q_rope_mut = TileTensor(
        q_rope.ptr.unsafe_mut_cast[True](),
        q_rope.layout,
    )

    mla_fused_rope_rmsnorm_quantization(
        q_rope_mut,
        q_rope.as_unsafe_any_origin(),  # hack aliasing.
        input_row_offsets,
        freqs_cis,
        kv_norm_gamma,
        kv_collection,
        layer_idx,
        epsilon,
        kv_input_fn,
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
        row_major(buffer_length, Idx[kv_latent_dim]),
    )

    # copy the k cache to the latent buffer
    var k_cache = kv_collection.get_key_cache(Int(layer_idx))
    _k_cache_to_buffer(
        buffer_row_offsets,
        cache_offsets,
        k_cache,
        Int32(buffer_length),
        k_latent,
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
        row_major(buffer_length, Idx[kv_latent_dim]),
    )

    # the scales are stored in a transposed, padded format
    comptime scales_m_padding = 16 // size_of[fp8_scale_dtype]()
    var scales_padded_m = align_up(buffer_length, scales_m_padding)
    var fp8_k_latent_scale_buf = ctx.enqueue_create_buffer[fp8_scale_dtype](
        scales_padded_m * ceildiv(kv_latent_dim, k_scale_granularity)
    )
    var fp8_k_latent_scale = TileTensor(
        fp8_k_latent_scale_buf,
        row_major(
            (Idx[ceildiv(kv_latent_dim, k_scale_granularity)], scales_padded_m)
        ),
    )

    @inline(.always)
    def input_fn[
        width: Int, alignment: Int
    ](row: Int, col: Int) {var k_latent} -> SIMD[k_latent.dtype, width]:
        return k_latent.load[width=width]((row, col))

    quantize_dynamic_scaled_fp8[
        in_dtype=k_latent.dtype,
        group_size_or_per_token=k_scale_granularity,
        num_cols=k_latent.static_shape[1],
    ](
        input_fn,
        fp8_k_latent,
        fp8_k_latent_scale,
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
        row_major(buffer_length, Idx[num_heads * qk_nope_head_dim]),
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
    var w_v = w_uv.reshape(
        row_major(Idx[num_heads * v_head_dim], Idx[kv_latent_dim]),
    )
    var w_v_scale = w_uv_scale.reshape(
        row_major(
            (
                Idx[num_heads * (v_head_dim // n_scale_granularity)],
                Idx[kv_latent_dim // k_scale_granularity],
            )
        ),
    )

    # Up-project latent KV values to V.
    var v_buf = ctx.enqueue_create_buffer[dtype](
        buffer_length * num_heads * v_head_dim
    )
    var v_flat = TileTensor(
        v_buf,
        row_major(buffer_length, Idx[num_heads * v_head_dim]),
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
        row_major((buffer_length, Idx[num_heads], Idx[qk_nope_head_dim])),
    )
    var v = TileTensor(
        v_buf,
        row_major(buffer_length, Idx[num_heads], Idx[v_head_dim]),
    )

    # On AMD a unit-scale FP8 latent cache needs FP8 Q/K/V together for the
    # validated FP8xFP8 path; bf16 Q hits an unsupported fp8xbf16 MFMA.
    comptime stage_prefill_fp8_amd = (
        ctx.target.is_amd_gpu()
        and dtype == .bfloat16
        and collection_t.CacheType.dtype == .float8_e4m3fn
        and not collection_t.CacheType.quantization_enabled
    )
    comptime if stage_prefill_fp8_amd:
        var q_fp8_buf = ctx.enqueue_create_buffer[.float8_e4m3fn](
            seq_len * num_heads * q_head_dim
        )
        var q_fp8 = TileTensor(
            q_fp8_buf,
            row_major(seq_len, Idx[num_heads], Idx[q_head_dim]),
        )
        var q_view = q.as_imm().bitcast[.bfloat16]()
        convert_bf16_to_fp8_e4m3fn(q_view, q_fp8, ctx)

        var k_fp8_buf = ctx.enqueue_create_buffer[.float8_e4m3fn](
            buffer_length * num_heads * qk_nope_head_dim
        )
        var k_fp8 = TileTensor(
            k_fp8_buf,
            row_major((buffer_length, Idx[num_heads], Idx[qk_nope_head_dim])),
        )
        var k_view = k.as_imm().bitcast[.bfloat16]()
        convert_bf16_to_fp8_e4m3fn(k_view, k_fp8, ctx)

        var v_fp8_buf = ctx.enqueue_create_buffer[.float8_e4m3fn](
            buffer_length * num_heads * v_head_dim
        )
        var v_fp8 = TileTensor(
            v_fp8_buf,
            row_major(buffer_length, Idx[num_heads], Idx[v_head_dim]),
        )
        var v_view = v.as_imm().bitcast[.bfloat16]()
        convert_bf16_to_fp8_e4m3fn(v_view, v_fp8, ctx)

        generic_flare_mla_prefill_kv_cache_ragged[
            target=target,
            mask_str=mask_str,
        ](
            q_fp8,
            k_fp8,
            v_fp8,
            buffer_row_offsets,
            cache_offsets,
            input_row_offsets,
            kv_collection,
            layer_idx,
            scale,
            output,
            ctx,
        )
    else:
        generic_flare_mla_prefill_kv_cache_ragged[
            target=target,
            mask_str=mask_str,
        ](
            q,
            k,
            v,
            buffer_row_offsets,
            cache_offsets,
            input_row_offsets,
            kv_collection,
            layer_idx,
            scale,
            output,
            ctx,
        )


# ===-----------------------------------------------------------------------===#
# Manually fused MLA decode branch (FP8)
# ===-----------------------------------------------------------------------===#


@inline(.always)
def quantize_and_bmm_fp8_helper[
    dtype: DType,
    c_dtype: DType,
    fp8_dtype: DType,
    fp8_scale_dtype: DType,
    m_scale_granularity: Int,
    n_scale_granularity: Int,
    k_scale_granularity: Int,
    target: StaticString = "cpu",
](
    c: TileTensor[mut=True, c_dtype, address_space=.GENERIC, ...],
    a: TileTensor[dtype, address_space=.GENERIC, ...],
    b: TileTensor[fp8_dtype, address_space=.GENERIC, ...],
    b_scales: TileTensor[fp8_scale_dtype, address_space=.GENERIC, ...],
    ctx: DeviceContext,
) raises:
    """
    Helper function to quantize and perform a batched matrix multiplication.
    This function uses the transposed view of the input tensor `a`.

    Parameters:
        dtype: Data type of the input tensor `a`.
        c_dtype: Data type of the output tensor `c`. When this is an FP8 type
            the matmul epilogue quantizes as it stores, so no separate
            conversion pass is needed downstream.
        fp8_dtype: Data type of the FP8 quantized tensors.
        fp8_scale_dtype: Data type of the FP8 scale tensors.
        m_scale_granularity: Granularity of the scale for the M dimension of
            the batched matrix multiplication.
        n_scale_granularity: Granularity of the scale for the N dimension of
            the batched matrix multiplication.
        k_scale_granularity: Granularity of the scale for the K dimension of
            the batched matrix multiplication, also the quantization group
            size for `a`.
        target: Target device for the batched matrix multiplication (defaults
            to "cpu").

    Args:
        c: Output tensor for the batched matrix multiplication result. Shape:
            [batch, m, n].
        a: Input tensor to quantize to FP8 and use as the left operand, loaded
            via a transposed view. Shape: [m, batch, k].
        b: FP8 weight tensor used as the right operand. Shape: [batch, n, k].
        b_scales: Scale tensor for `b`.
        ctx: Device context for buffer allocation and kernel execution.
    """

    # Evidence assert for TileTensor load Coord constraint.
    comptime assert type_of(a).flat_rank >= 3

    comptime B = a.static_shape[1]
    comptime K = a.static_shape[2]
    comptime N = b.static_shape[1]

    var m = Int(a.dim(0))

    # allocate buffers for quantized a and its scales
    var fp8_a_buf = ctx.enqueue_create_buffer[fp8_dtype](B * m * K)
    var fp8_a = TileTensor(fp8_a_buf, row_major(Idx[B], m, Idx[K]))

    # the scales are stored in a transposed, padded format
    comptime scales_m_padding = 16 // size_of[fp8_scale_dtype]()
    var scales_padded_m = align_up(m, scales_m_padding)
    var fp8_a_scale_buf = ctx.enqueue_create_buffer[fp8_scale_dtype](
        B * ceildiv(K, k_scale_granularity) * scales_padded_m
    )
    var fp8_a_scale = TileTensor(
        fp8_a_scale_buf,
        row_major(
            (Idx[B], Idx[ceildiv(K, k_scale_granularity)], scales_padded_m)
        ),
    )

    @inline(.always)
    def input_fn[
        width: Int, alignment: Int
    ](batch: Int, row: Int, col: Int) {var a} -> SIMD[dtype, width]:
        # First transpose the q_nope tensor from [row, batch, col] to [batch, row, col].
        comptime assert a.flat_rank == 3
        return a.load[width=width]((row, batch, col))

    batched_quantize_dynamic_scaled_fp8[
        in_dtype=dtype,
        group_size_or_per_token=k_scale_granularity,
        num_cols=K,
    ](
        input_fn,
        fp8_a,
        fp8_a_scale,
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
        c,
        fp8_a,
        b,
        fp8_a_scale,
        b_scales,
        ctx,
    )


# AMD sparse MLA decode gathers each token's top-k rows into scratch with the
# kernel below unless the caller supplies a split-K partition count, in which
# case it reads them in place, as prefill always does, through
# `SparseRowsMHAOperand`.
comptime _AMD_SPARSE_MLA_GATHER_VEC = 16
# Tokens gathered per pass on the BF16 AMD sparse MLA arm: 256 tokens of
# 2051 x 576 BF16 rows is ~600 MB of scratch.
comptime _AMD_SPARSE_MLA_MAX_GATHER_TOKENS = 256


@__name(t"amd_sparse_mla_gather_kv_rows_{k_cache_dim}_{rows_per_block}")
def _amd_sparse_mla_gather_kv_rows[
    CacheT: KVCacheT,
    k_cache_dim: Int,
    rows_per_block: Int,
    has_topk_lengths: Bool = True,
](
    k_cache: CacheT,
    blocks_base: MutPointer[Scalar[CacheT.dtype], MutAnyOrigin],
    d_indices: MutPointer[Int32, MutAnyOrigin],
    topk_lengths: MutPointer[Int32, MutAnyOrigin],
    scratch: MutPointer[Scalar[CacheT.dtype], MutAnyOrigin],
    indices_stride: Int32,
):
    """Gathers selected MLA latent rows from a paged KV cache into scratch.

    Grid: `(ceildiv(indices_stride, rows_per_block), batch_size)`; each
    block packs `rows_per_block` (batch, slot) rows, copied (or
    zero-filled) one `k_cache_dim`-wide row by `thread_idx.x` vector
    lanes of `_AMD_SPARSE_MLA_GATHER_VEC` elements each (`thread_idx.y`
    is the row), so a wavefront moves 64 x 16 B at a stride of 16 B
    instead of 64 scalar bytes. Slots at or past `topk_lengths[batch]`,
    and any `-1` (invalid) index, are zero-filled -- correctness relies on
    the caller masking these columns out (a zero row participates in, and
    would corrupt, an unmasked softmax). Without `has_topk_lengths`, every
    token keeps the full `indices_stride` and `topk_lengths` is not read.
    """
    comptime assert CacheT.kv_params.num_heads == 1, (
        "AMD sparse MLA gather only supports a single-head (absorbed"
        " latent) KV cache."
    )
    comptime assert (
        k_cache_dim % _AMD_SPARSE_MLA_GATHER_VEC == 0
    ), "AMD sparse MLA gather vector lanes must tile k_cache_dim exactly."

    var indices_stride_i = Int(indices_stride)
    var batch_idx = block_idx.y
    var col = block_idx.x * rows_per_block + thread_idx.y
    if col < indices_stride_i:
        var vec_off = thread_idx.x * _AMD_SPARSE_MLA_GATHER_VEC
        var slot = batch_idx * indices_stride_i + col
        var dst = scratch.unsafe_offset(slot * k_cache_dim).unsafe_offset(
            vec_off
        )

        var encoded = d_indices[slot]
        var topk_len = indices_stride_i
        comptime if has_topk_lengths:
            topk_len = Int(topk_lengths[batch_idx])
        if col < topk_len and encoded >= 0:
            var virtual_row = Int(k_cache.get_tma_row(encoded))
            var src = blocks_base.unsafe_offset(
                virtual_row * k_cache_dim
            ).unsafe_offset(vec_off)
            dst.unsafe_store(
                src.unsafe_load[width=_AMD_SPARSE_MLA_GATHER_VEC]()
            )
        else:
            dst.unsafe_store(SIMD[CacheT.dtype, _AMD_SPARSE_MLA_GATHER_VEC](0))


@__name(t"amd_sparse_mla_gather_kv_rows_lengths_{k_cache_dim}_{rows_per_block}")
def _amd_sparse_mla_gather_kv_rows_and_lengths[
    CacheT: KVCacheT,
    k_cache_dim: Int,
    rows_per_block: Int,
    has_topk_lengths: Bool = True,
](
    k_cache: CacheT,
    blocks_base: MutPointer[Scalar[CacheT.dtype], MutAnyOrigin],
    d_indices: MutPointer[Int32, MutAnyOrigin],
    topk_lengths: MutPointer[Int32, MutAnyOrigin],
    scratch: MutPointer[Scalar[CacheT.dtype], MutAnyOrigin],
    valid_lengths: MutPointer[UInt32, MutAnyOrigin],
    indices_stride: Int32,
    selection_lag: Int32,
):
    """Gathers like `_amd_sparse_mla_gather_kv_rows` and writes each token's
    valid length for `PaddingMask`, replacing a separate scan launch.

    The length is closed-form, as in SM100 sparse decode: a plain top-k row
    holds `min(top_k, visible)` valid slots followed by `-1` padding, so a
    row selected `selection_lag` tokens earlier holds
    `min(top_k, visible - selection_lag)`. As on SM100,
    `assert_sparse_row_packed` checks that contract in assert-enabled builds
    only for a lagged row; at lag 0 it relies on the indexer's packed-row
    contract. One thread per token writes the length.
    """
    _amd_sparse_mla_gather_kv_rows[
        CacheT,
        k_cache_dim,
        rows_per_block,
        has_topk_lengths=has_topk_lengths,
    ](k_cache, blocks_base, d_indices, topk_lengths, scratch, indices_stride)

    if block_idx.x == 0 and thread_idx.x == 0 and thread_idx.y == 0:
        var batch_idx = block_idx.y
        var stride = Int(indices_stride)
        var topk_len = stride
        comptime if has_topk_lengths:
            topk_len = min(max(Int(topk_lengths[batch_idx]), 0), stride)
        # Decode reads one new token per sequence, so it sees the cached
        # keys plus itself.
        var visible = k_cache.cache_length(batch_idx) + 1
        assert_sparse_row_packed[SparseRowLayout()](
            d_indices,
            stride,
            batch_idx,
            visible,
            Int(selection_lag),
            topk_len,
        )
        valid_lengths[batch_idx] = UInt32(
            min(topk_len, max(visible - Int(selection_lag), 0))
        )


comptime _AMD_SPARSE_MLA_ROW_VALID_LENGTHS_THREADS = 256


@__name("amd_sparse_mla_row_valid_lengths")
def _amd_sparse_mla_row_valid_lengths[
    BLOCK_SIZE: Int = _AMD_SPARSE_MLA_ROW_VALID_LENGTHS_THREADS,
](
    d_indices: MutPointer[Int32, MutAnyOrigin],
    topk_lengths: MutPointer[Int32, MutAnyOrigin],
    valid_lengths: MutPointer[UInt32, MutAnyOrigin],
    indices_stride: Int32,
    zero_row: MutPointer[UInt8, MutAnyOrigin],
    zero_row_bytes: Int32,
):
    """Writes each token's valid prefix length for `PaddingMask`.

    The caller's `topk_lengths` is the constant `index_topk` (the Python
    graph broadcasts it, which is exact for SM100 -- see
    `mla_decode_sparse_kv_fp8`'s causal slot bound -- but names no real
    count), so masking with it leaves every zero-filled invalid slot visible
    to the softmax. The indexer's bounded top-k writes the invalid slots as a
    `-1` suffix, so the first negative index is the row's true length.

    Taking the FIRST negative rather than counting the non-negatives matters:
    if a `-1` ever appeared mid-row the count would overcount and re-admit a
    zero row, while the first-negative truncates instead. Truncating loses
    keys; admitting a zero row corrupts the softmax denominator.

    The length is also capped at `topk_lengths[row]`, so a slot past the
    caller's top-k length is never admitted even when its index is valid.

    Block 0 also clears `zero_row`, the row `SparseRowsMHAOperand` reads for
    every slot past a token's valid length, which saves a separate fill
    launch per layer.
    """
    var row = block_idx.x
    if row == 0:
        for i in range(thread_idx.x, Int(zero_row_bytes), BLOCK_SIZE):
            zero_row[i] = 0
    var base = row * Int(indices_stride)

    # Strided scan: thread `t` examines columns `t, t+BLK, t+2*BLK, ...`
    # and keeps the first negative it sees (or `indices_stride` if none).
    var tid = thread_idx.x
    var stride = Int(indices_stride)
    var local_length = stride
    for col in range(tid, stride, BLOCK_SIZE):
        if d_indices[base + col] < 0:
            local_length = col
            break

    # Block-reduce the per-thread partial minima with MIN: same reduction as
    # the serial first-negative scan, identical result for any input.
    var block_min = block.min[
        block_size=BLOCK_SIZE,
        broadcast=False,
    ](
        SIMD[DType.int32, 1](local_length)
    )[0]

    if tid == 0:
        var topk_len = min(max(Int(topk_lengths[row]), 0), stride)
        valid_lengths[row] = UInt32(min(Int(block_min), topk_len))


@__name("amd_sparse_mla_compact_indices")
def _amd_sparse_mla_compact_indices[
    BLOCK_SIZE: Int = _AMD_SPARSE_MLA_ROW_VALID_LENGTHS_THREADS,
](
    d_indices: MutPointer[Int32, MutAnyOrigin],
    topk_lengths: MutPointer[Int32, MutAnyOrigin],
    compact_indices: MutPointer[Int32, MutAnyOrigin],
    valid_lengths: MutPointer[UInt32, MutAnyOrigin],
    indices_stride: Int32,
):
    """Moves each row's valid indices to the front and records their count.

    `_amd_sparse_mla_row_valid_lengths` assumes the invalid slots form a `-1`
    suffix. A k-pooled selection breaks that: unselected pools leave `-1`
    holes inside the row, and the always-selected tail columns after them
    hold the query's most recent positions, its own included. Truncating at
    the first `-1` would drop those. Compacting keeps every valid key in
    front of a `-1` suffix, so the count is an exact `PaddingMask` length.
    Order within the row is preserved; attention does not depend on it.

    Only the first `topk_lengths[row]` slots are read, so the count never
    exceeds the columns the gather fills.
    """
    var row = block_idx.x
    var stride = Int(indices_stride)
    var base = row * stride
    var tid = thread_idx.x
    var scan_len = min(max(Int(topk_lengths[row]), 0), stride)

    var written = Int32(0)
    for chunk in range(0, scan_len, BLOCK_SIZE):
        var col = chunk + tid
        var index = Int32(-1)
        if col < scan_len:
            index = d_indices[base + col]
        var is_valid = Int32(1) if index >= 0 else Int32(0)
        var offset = block.prefix_sum[block_size=BLOCK_SIZE, exclusive=True](
            is_valid
        )
        var chunk_valid = block.sum[block_size=BLOCK_SIZE, broadcast=True](
            is_valid
        )
        if is_valid == 1:
            compact_indices[base + Int(written + offset)] = index
        written += chunk_valid

    for col in range(Int(written) + tid, stride, BLOCK_SIZE):
        compact_indices[base + col] = -1
    if tid == 0:
        valid_lengths[row] = UInt32(written)


def mla_decode_branch_fp8[
    dtype: DType,
    fp8_dtype: DType,
    fp8_scale_dtype: DType,
    collection_t: KVCollectionT,
    KVInputFnType: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int](IndexList[2]) -> SIMD[DType.bfloat16, width],
    //,
    m_scale_granularity: Int,
    n_scale_granularity: Int,
    k_scale_granularity: Int,
    mask_str: StaticString,
    target: StaticString = "cpu",
    sparse_mla: Bool = False,
    # Read-once shared-index MTP fold; threaded to flare_mla_decoding (fp8-KV
    # only; bf16 branch is parity plumbing, always False).
    fold_shared_index: Bool = False,
    # How the sparse index producer packs each row; see flare_mla_decoding.
    sparse_row: SparseRowLayout = SparseRowLayout(),
    # Off keeps Q in `dtype`, so the two stagings can be compared.
    fp8_q: Bool = True,
    # Whether `extra_k` is supplied; see `flare_mla_decoding`.
    has_extra_k: Bool = False,
    physical_indices: Bool = True,
    # Presence of `attn_sink_ptr` / `topk_lengths` + `extra_topk_lengths`
    # (inferred from those arguments).
    AttnSinkPtrType: OptionalPointer = NullPointer[DType.float32],
    TopkLengthsPtrType: OptionalPointer = NullPointer[DType.int32],
    # Set only by a caller that knows the sink is the inert -1e38 constant; the
    # AMD sparse arm never reads it, so a live sink must fail loudly.
    attn_sink_inert: Bool = False,
](
    output: TileTensor[mut=True, dtype, address_space=.GENERIC, ...],
    q: TileTensor[dtype, address_space=.GENERIC, ...],
    input_row_offsets: TileTensor[.uint32, address_space=.GENERIC, ...],
    freqs_cis: TileTensor[_, address_space=.GENERIC, ...],
    kv_norm_gamma: TileTensor[_, address_space=.GENERIC, ...],
    kv_collection: collection_t,
    layer_idx: UInt32,
    scale: Float32,
    epsilon: Float32,
    w_uk: TileTensor[fp8_dtype, address_space=.GENERIC, ...],
    w_uk_scale: TileTensor[fp8_scale_dtype, address_space=.GENERIC, ...],
    w_uv: TileTensor[fp8_dtype, address_space=.GENERIC, ...],
    w_uv_scale: TileTensor[fp8_scale_dtype, address_space=.GENERIC, ...],
    scalar_args_buf: TileTensor[.int64, address_space=.GENERIC, ...],
    kv_input_fn: KVInputFnType,
    ctx: DeviceContext,
    d_indices: OptionalReg[MutPointer[Int32, MutAnyOrigin]] = None,
    indices_stride: Int = 0,
    topk_lengths: TopkLengthsPtrType = null_pointer[TopkLengthsPtrType](),
    attn_sink_ptr: AttnSinkPtrType = null_pointer[AttnSinkPtrType](),
    extra_k: OptionalReg[collection_t.CacheType] = None,
    extra_d_indices: OptionalReg[MutPointer[Int32, MutAnyOrigin]] = None,
    extra_indices_stride: Int = 0,
    extra_topk_lengths: TopkLengthsPtrType = unread_pointer[
        TopkLengthsPtrType
    ](),
    extra_scales_ptr: OptionalReg[MutPointer[Float32, MutAnyOrigin]] = None,
    # Capturable-graph scalar forwarded from the MoGG op input list.
    num_partitions_in: Optional[Int] = None,
    # Logical sparse indices for position-based causal masking; `None` keeps
    # the prior slot-count behavior. See mla_decode_utils.mojo.
    logical_indices: OptionalReg[MutPointer[Int32, MutAnyOrigin]] = None,
    # See flare_mla_decoding.
    selection_lag: Int = 0,
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
        KVInputFnType: Type of the closure that loads the KV latent
            values (inferred).
        m_scale_granularity: Granularity of the scale for M dimension of the
            matrix multiplication.
        n_scale_granularity: Granularity of the scale for N dimension of the
            matrix multiplication.
        k_scale_granularity: Granularity of the scale for K dimension of the
            matrix multiplication.
        mask_str: Mask variant.
        target: Target device.
        sparse_mla: Whether to use sparse MLA.
        fold_shared_index: Whether to enable the default-off read-once
            shared-index fold on the sparse FP8 decode path, which packs the
            folded MTP query positions into one CTA and gathers their single
            shared top-k list once.
        sparse_row: How the sparse index producer packs each row, which
            bounds each row by its valid length (defaults to plain top-k).
        fp8_q: Whether to stage Q in the FP8 cache dtype where the sparse
            decode supports it, so the producers below quantize as they store.
            Off stages Q in `dtype` and routes to the sparse kernel that reads
            a wider Q.
        has_extra_k: Whether `extra_k` is supplied (defaults to False); see
            `flare_mla_decoding`.
        physical_indices: Whether `d_indices` holds paged physical rows rather
            than token positions (defaults to True); see `flare_mla_decoding`.
        AttnSinkPtrType: `OptionalPointer` type of `attn_sink_ptr`
            (inferred, defaults to `NullPointer`).
        TopkLengthsPtrType: `OptionalPointer` type of `topk_lengths` and
            `extra_topk_lengths` (inferred, defaults to `NullPointer`).
        attn_sink_inert: Whether the caller guarantees `attn_sink_ptr` holds
            the inert -1e38 constant. The AMD sparse arm never reads the sink
            and rejects one otherwise.

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
            depending on the FP8 block-scaling granularity used for weight
            quantization.
        w_uv: Weight matrix for projecting the output of the attention back to
            each head's original space. Shape: [num_heads, v_head_dim, kv_latent_dim].
        w_uv_scale: The scale for the w_uv weight matrix. Shape varies
            depending on the FP8 block-scaling granularity used for weight
            quantization.
        scalar_args_buf: Packed MLA dispatch metadata buffer.
        kv_input_fn: Closure that loads the KV latent values. Shape:
            [tot_seq_len, cache_head_dim]. Where cache_head_dim = kv_lora_rank
            + qk_rope_head_dim. It runs on the device, so it must copy what
            it captures.
        ctx: Device context.
        d_indices: Sparse decode packed indices (null when dense).
        indices_stride: Row stride in ``d_indices``.
        topk_lengths: Per-batch valid top-k counts.
        attn_sink_ptr: Optional per-batch attention sink weights.
        extra_k: Optional second key cache operand (see ``flare_mla_decoding``).
        extra_d_indices: Extra-stream sparse indices.
        extra_indices_stride: Stride for ``extra_d_indices``.
        extra_topk_lengths: Extra-stream per-batch lengths.
        extra_scales_ptr: Extra-stream scales.
        num_partitions_in: Capturable-graph num_partitions override.
        logical_indices: Logical sparse indices for position-based causal
            masking; `None` keeps the prior slot-count behavior.
        selection_lag: Positions between the token each sparse row was
            computed for and the token that reads it.
    """

    comptime kv_params = collection_t.kv_params
    comptime assert kv_params.is_mla, "kv_params.is_mla should be true"
    comptime assert kv_params.num_heads == 1, "kv_params.num_heads should be 1"

    comptime num_heads = q.static_shape[1]
    comptime q_head_dim = q.static_shape[2]
    comptime v_head_dim = output.static_shape[2]
    comptime k_cache_dim = kv_params.head_size

    comptime assert (
        w_uk.shape_known and w_uv.shape_known
    ), "w_uk and w_uv's shapes should be static"
    comptime kv_latent_dim = w_uk.static_shape[1]
    comptime qk_rope_head_dim = k_cache_dim - kv_latent_dim
    comptime qk_nope_head_dim = q_head_dim - qk_rope_head_dim
    comptime assert (
        qk_rope_head_dim == 0 or freqs_cis.static_shape[1] == qk_rope_head_dim
    ), "freqs_cis's width should be k_cache_dim - kv_latent_dim."
    comptime assert (
        w_uk.static_shape[2] == qk_nope_head_dim
    ), "w_uk.static_shape[2] should be equal to qk_nope_head_dim"
    comptime assert (
        w_uv.static_shape[1] == v_head_dim
    ), "w_uv.static_shape[1] should be equal to v_head_dim"

    var seq_len = Int(q.dim(0))

    if seq_len == 0:
        return

    comptime if ctx.target.is_amd_gpu() and sparse_mla:
        # Gather top-k rows into scratch, reuse the dense decode kernel (top-k
        # selection gives causality). Guards reject unsupported features first.
        if d_indices is None:
            raise Error("AMD sparse MLA decode requires d_indices.")
        if extra_k is not None:
            raise Error(
                "AMD sparse MLA decode does not support a second (extra)"
                " KV stream."
            )
        if not AttnSinkPtrType.is_null and not attn_sink_inert:
            raise Error(
                "AMD sparse MLA decode does not consume attention sinks; a"
                " caller whose sink is inert must say so with"
                " `attn_sink_inert`."
            )
        if logical_indices is not None:
            raise Error(
                "AMD sparse MLA decode enforces causality via top-k"
                " selection, not logical position; logical_indices is not"
                " honored."
            )
        comptime assert not fold_shared_index, (
            "The read-once shared-index MTP fold is not implemented for"
            " AMD sparse MLA decode."
        )
        comptime assert physical_indices, (
            "AMD sparse MLA decode gathers paged physical rows; logical"
            " d_indices are not supported."
        )
        comptime assert not sparse_row.is_bounded(), (
            "AMD sparse MLA decode supports plain top-k rows only (no k-pool"
            " row layout)."
        )

        # This arm treats every `q` row as its own S=1 sequence (batch =
        # seq_len); a ragged S>1 batch (MTP/spec-decode) runs wrong geometry.
        var num_batches = Int(input_row_offsets.dim(0)) - 1
        if seq_len != num_batches:
            raise Error(
                "AMD sparse MLA decode only supports one query token per"
                " sequence (S=1 decode); got "
                + String(seq_len)
                + " query tokens across "
                + String(num_batches)
                + " sequences. Multi-token decode (MTP / speculative-"
                "decode verify, S>1) is not implemented on this path."
            )

    # Stage Q in the FP8 cache dtype where the kernel consumes it natively:
    # SM100 sparse decode, and AMD dense decode/prefill (no fp8xbf16 MFMA).
    comptime native_fp8_sparse = (
        fp8_q
        and sparse_mla
        and dtype == .bfloat16
        and _mla_native_fp8_cache[collection_t.CacheType]()
        and _is_sm10x_gpu(ctx.default_device_info)
    )
    # The AMD arm needs FP8 Q whether it runs over the KV cache or a gathered
    # scratch of the same dtype, so this does not gate on `sparse_mla`.
    comptime stage_fp8_amd = (
        ctx.target.is_amd_gpu()
        and collection_t.CacheType.dtype == .float8_e4m3fn
        and not collection_t.CacheType.quantization_enabled
    )
    comptime mla_decode_input_dtype = (
        DType.float8_e4m3fn if native_fp8_sparse else (
            collection_t.CacheType.dtype if stage_fp8_amd else dtype
        )
    )

    # First, create a input buffer for the mla decode kernel
    var mla_decode_input_buf = ctx.enqueue_create_buffer[
        mla_decode_input_dtype
    ](seq_len * num_heads * k_cache_dim)
    var mla_decode_input = TileTensor(
        mla_decode_input_buf,
        row_major(seq_len, Idx[num_heads], Idx[k_cache_dim]),
    )

    # =========================================================================#
    # Project the non-rope part of each query head to kv_latent_dim            #
    # =========================================================================#

    # The first qk_nope_head_dim columns of each Q head will be up-projected to
    # kv_latent_dim through a fp8 batched matmul.

    # We start by create a view of the input tensor `q` that only contains the
    # first qk_nope_head_dim columns of each Q head.
    var q_nope = q.reshape(
        TileLayout(
            (seq_len, Idx[num_heads], Idx[qk_nope_head_dim]),
            (Idx[num_heads * q_head_dim], Idx[q_head_dim], Idx[1]),
        )
    )

    # Then create a view of the mla_decode_input tensor that only contains the
    # first kv_latent_dim columns of each Q head.
    var mla_decode_input_nope = mla_decode_input.reshape(
        TileLayout(
            (Idx[num_heads], seq_len, Idx[kv_latent_dim]),
            (Idx[k_cache_dim], Idx[num_heads * k_cache_dim], Idx[1]),
        )
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
    # qk_rope_head_dim columns of each Q head. The slice carries runtime
    # extents, so reshape restores the static trailing dims.
    var q_rope = q[:, :, qk_nope_head_dim:].reshape(
        TileLayout(
            (seq_len, Idx[num_heads], Idx[qk_rope_head_dim]),
            (Idx[num_heads * q_head_dim], Idx[q_head_dim], Idx[1]),
        )
    )

    # Create a view of the `mla_decode_input` tensor that only contains the last
    # qk_rope_head_dim columns of each Q head.
    var mla_decode_input_rope = mla_decode_input[:, :, kv_latent_dim:].reshape(
        TileLayout(
            (seq_len, Idx[num_heads], Idx[qk_rope_head_dim]),
            (Idx[num_heads * k_cache_dim], Idx[k_cache_dim], Idx[1]),
        )
    )

    mla_fused_rope_rmsnorm_quantization(
        mla_decode_input_rope,
        q_rope,
        input_row_offsets,
        freqs_cis,
        kv_norm_gamma,
        kv_collection,
        layer_idx,
        epsilon,
        kv_input_fn,
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
        row_major(seq_len, Idx[num_heads], Idx[kv_latent_dim]),
    )

    comptime if ctx.target.is_amd_gpu() and sparse_mla:
        # Guards for this arm (d_indices presence, extra_k,
        # attn_sink_ptr, logical_indices, fold_shared_index, S=1) already
        # ran above, before the up-projection/RoPE preamble.
        var batch_size = seq_len
        var k_cache = kv_collection.get_key_cache(Int(layer_idx))

        # The graph omits `topk_lengths` when every token keeps the full top-k
        # stride. The kernels then cap at the stride and never read the
        # pointer, which saves a fill launch per layer.
        comptime has_topk_lengths = not TopkLengthsPtrType.is_null
        var decode_topk_lengths = MutPointer[
            Int32, MutAnyOrigin
        ].unsafe_dangling()
        comptime if has_topk_lengths:
            decode_topk_lengths = rebind[MutPointer[Int32, MutAnyOrigin]](
                as_optional_reg(topk_lengths).value()
            )

        # View decode input/output as rank-4 [batch,1,heads,depth]; decode has
        # q_seq_len==1 (batch==seq_len), so a size-1 axis, no data movement.
        var q_padded = mla_decode_input.reshape(
            TileLayout(
                (batch_size, Idx[1], Idx[num_heads], Idx[k_cache_dim]),
                (
                    Idx[num_heads * k_cache_dim],
                    Idx[num_heads * k_cache_dim],
                    Idx[k_cache_dim],
                    Idx[1],
                ),
            )
        )
        var raw_output_padded = raw_output.reshape(
            TileLayout(
                (batch_size, Idx[1], Idx[num_heads], Idx[kv_latent_dim]),
                (
                    Idx[num_heads * kv_latent_dim],
                    Idx[num_heads * kv_latent_dim],
                    Idx[kv_latent_dim],
                    Idx[1],
                ),
            )
        )

        # With a split-K partition count from the caller, read the selected
        # rows in place. A decode workgroup's time is a fixed chain (index
        # reads, row reads, softmax, store) more than row bandwidth, so
        # partitions of a few dozen rows run one sequence on many CUs at once.
        # The valid length is derived in-kernel, so the decode and its split-K
        # reduce are the only launches. Without a count the dispatcher would
        # split a 2048-row selection only 8 ways, where in place loses to
        # gather-then-decode (+3.6 us per layer on GLM-5.3, MI355X), so keep
        # the gather.
        if case .Some(num_partitions) = num_partitions_in:
            var k_rows = SparseRowsMHAOperand[
                row_width=k_cache_dim, predicated=True
            ](
                KVCacheMHAOperand(k_cache),
                k_cache.blocks_raw_ptr(),
                d_indices.value(),
                indices_stride,
                decode_topk_lengths,
                has_topk_lengths,
                selection_lag,
            )
            flare_mla_decoding_rows[
                config=MHAConfig[mla_decode_input_dtype](num_heads, k_cache_dim)
            ](
                raw_output_padded,
                q_padded,
                k_rows,
                indices_stride,
                SparseRowsDecodeMask(k_rows),
                scale,
                ctx,
                scalar_args_buf,
                num_partitions=num_partitions,
            )
        else:
            var scratch_buf = ctx.enqueue_create_buffer[mla_decode_input_dtype](
                batch_size * indices_stride * k_cache_dim
            )
            # 576 / 16 = 36 vector lanes per row x 7 rows = 252 threads
            # (4 wavefronts); grid.x shrinks 7x vs one block per row.
            comptime GATHER_ROWS_PER_BLOCK = 7
            var k_cache_blocks_base = k_cache.blocks_raw_ptr()
            # `topk_lengths` is the constant `index_topk`, so masking with it marks
            # every tile NO_MASK; the gather writes each row's real length.
            var valid_lengths_buf = ctx.enqueue_create_buffer[DType.uint32](
                batch_size
            )
            ctx.enqueue_function[
                _amd_sparse_mla_gather_kv_rows_and_lengths[
                    type_of(k_cache),
                    k_cache_dim,
                    GATHER_ROWS_PER_BLOCK,
                    has_topk_lengths=has_topk_lengths,
                ]
            ](
                k_cache,
                k_cache_blocks_base,
                d_indices.value(),
                decode_topk_lengths,
                scratch_buf.unsafe_ptr(),
                valid_lengths_buf.unsafe_ptr(),
                Int32(indices_stride),
                Int32(selection_lag),
                grid_dim=(
                    ceildiv(indices_stride, GATHER_ROWS_PER_BLOCK),
                    batch_size,
                    1,
                ),
                block_dim=(
                    k_cache_dim // _AMD_SPARSE_MLA_GATHER_VEC,
                    GATHER_ROWS_PER_BLOCK,
                    1,
                ),
            )

            var k_scratch = TileTensor(
                scratch_buf,
                row_major(batch_size, indices_stride, Idx[1], Idx[k_cache_dim]),
            )
            var amd_lengths_tt = TileTensor(
                valid_lengths_buf.unsafe_ptr(), row_major(batch_size)
            )
            var padding_mask = PaddingMask(amd_lengths_tt)

            flare_mla_decoding[
                config=MHAConfig[mla_decode_input_dtype](
                    num_heads, k_cache_dim
                ),
                decoding_warp_split_k=False,
            ](
                raw_output_padded,
                q_padded,
                k_scratch,
                padding_mask,
                scale,
                ctx,
                scalar_args_buf,
                num_partitions=amd_sparse_mla_decode_num_partitions(
                    batch_size,
                    indices_stride,
                    num_heads,
                    ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT),
                ),
            )
    else:
        generic_flare_mla_decode_kv_cache_ragged[
            target=target,
            mask_str=mask_str,
            sparse_mla=sparse_mla,
            fold_shared_index=fold_shared_index,
            sparse_row=sparse_row,
            has_extra_k=has_extra_k,
            physical_indices=physical_indices,
        ](
            mla_decode_input,
            input_row_offsets,
            kv_collection,
            layer_idx,
            scale,
            raw_output,
            scalar_args_buf,
            ctx,
            d_indices=d_indices,
            indices_stride=indices_stride,
            topk_lengths=topk_lengths,
            attn_sink_ptr=attn_sink_ptr,
            extra_k=extra_k,
            extra_d_indices=extra_d_indices,
            extra_indices_stride=extra_indices_stride,
            extra_topk_lengths=extra_topk_lengths,
            extra_scales_ptr=extra_scales_ptr,
            num_partitions_in=num_partitions_in,
            logical_indices=logical_indices,
            selection_lag=selection_lag,
        )

    # Create a view of the output tensor with logical shape
    # [num_heads, seq_len, v_head_dim], and map directly to
    # [seq_len, num_heads, v_head_dim] physical memory.
    var output_t = output.reshape(
        TileLayout(
            (Idx[num_heads], seq_len, Idx[v_head_dim]),
            (Idx[v_head_dim], Idx[num_heads * v_head_dim], Idx[1]),
        )
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


@inline(.always)
def mla_prefill_branch_sparse_fp8[
    dtype: DType,
    fp8_dtype: DType,
    fp8_scale_dtype: DType,
    collection_t: KVCollectionT,
    KVInputFnType: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int](IndexList[2]) -> SIMD[DType.bfloat16, width],
    //,
    m_scale_granularity: Int,
    n_scale_granularity: Int,
    k_scale_granularity: Int,
    indices_stride: Int,
    target: StaticString = "cpu",
    # See `mla_decode_branch_fp8`; this path does not read the sink either.
    attn_sink_inert: Bool = False,
](
    output: TileTensor[mut=True, dtype, address_space=.GENERIC, ...],
    q: TileTensor[dtype, address_space=.GENERIC, ...],
    input_row_offsets: TileTensor[.uint32, address_space=.GENERIC, ...],
    freqs_cis: TileTensor[_, address_space=.GENERIC, ...],
    kv_norm_gamma: TileTensor[_, address_space=.GENERIC, ...],
    kv_collection: collection_t,
    layer_idx: UInt32,
    scale: Float32,
    epsilon: Float32,
    w_uk: TileTensor[fp8_dtype, address_space=.GENERIC, ...],
    w_uk_scale: TileTensor[fp8_scale_dtype, address_space=.GENERIC, ...],
    w_uv: TileTensor[fp8_dtype, address_space=.GENERIC, ...],
    w_uv_scale: TileTensor[fp8_scale_dtype, address_space=.GENERIC, ...],
    scalar_args_buf: TileTensor[.int64, address_space=.GENERIC, ...],
    kv_input_fn: KVInputFnType,
    ctx: DeviceContext,
    d_indices: MutPointer[Int32, MutAnyOrigin],
    topk_lengths: MutPointer[Int32, MutAnyOrigin],
    attn_sink_ptr: OptionalReg[MutPointer[Float32, MutAnyOrigin]],
) raises:
    """Sparse MLA prefill branch (DSv3.2/GLM absorbed shape, FP8 weights).

    Reuses `mla_decode_branch_fp8`'s absorbed-Q construction (q_nope up-proj via
    `w_uk` + RoPE/RMSNorm) and the identical `w_uv` output up-projection. On
    SM100 the attention call swaps to `mla_sm100_prefill_sparse[_fp8]` over the
    paged latent cache; the caller (`.fp8.sparse` op) has already remapped
    `d_indices` from logical to physical rows, so they are passed straight
    through. On AMD (no native gather4/TMA sparse prefill kernel), it instead
    runs the dense AMD decode kernel over each query token's selected rows,
    read in place through `SparseRowsMHAOperand`. `d_indices`/`topk_lengths`
    are already `[total_q, indices_stride]`/`[total_q]` (per query token, not
    per sequence), so every row of `q` runs as its own single-token decode
    batch, with no prefill-specific kernel.
    """
    comptime kv_params = collection_t.kv_params
    comptime assert kv_params.is_mla, "kv_params.is_mla should be true"
    comptime assert kv_params.num_heads == 1, "kv_params.num_heads should be 1"

    comptime num_heads = q.static_shape[1]
    comptime q_head_dim = q.static_shape[2]
    comptime v_head_dim = output.static_shape[2]
    comptime k_cache_dim = kv_params.head_size

    comptime assert (
        w_uk.shape_known and w_uv.shape_known
    ), "w_uk and w_uv's shapes should be static"
    comptime kv_latent_dim = w_uk.static_shape[1]
    comptime qk_rope_head_dim = k_cache_dim - kv_latent_dim
    comptime qk_nope_head_dim = q_head_dim - qk_rope_head_dim
    comptime assert (
        qk_rope_head_dim == 0 or freqs_cis.static_shape[1] == qk_rope_head_dim
    ), "freqs_cis's width should be k_cache_dim - kv_latent_dim."
    comptime assert (
        w_uk.static_shape[2] == qk_nope_head_dim
    ), "w_uk.static_shape[2] should be equal to qk_nope_head_dim"
    comptime assert (
        w_uv.static_shape[1] == v_head_dim
    ), "w_uv.static_shape[1] should be equal to v_head_dim"

    var seq_len = Int(q.dim(0))
    if seq_len == 0:
        return

    comptime if ctx.target.is_amd_gpu():
        # Runs before the preamble, so an unsupported combo fails before launch.
        if attn_sink_ptr is not None and not attn_sink_inert:
            raise Error(
                "AMD sparse MLA prefill does not consume attention sinks; a"
                " caller whose sink is inert must say so with"
                " `attn_sink_inert`."
            )

    # Same mixed-precision MMA restriction as the decode arm: stage the absorbed
    # Q in the cache dtype so the cache rows and Q agree.
    comptime stage_fp8_amd = (
        ctx.target.is_amd_gpu()
        and collection_t.CacheType.dtype == .float8_e4m3fn
        and not collection_t.CacheType.quantization_enabled
    )
    comptime mla_decode_input_dtype = (
        collection_t.CacheType.dtype if stage_fp8_amd else dtype
    )

    var mla_decode_input_buf = ctx.enqueue_create_buffer[
        mla_decode_input_dtype
    ](seq_len * num_heads * k_cache_dim)
    var mla_decode_input = TileTensor(
        mla_decode_input_buf,
        row_major(seq_len, Idx[num_heads], Idx[k_cache_dim]),
    )

    var q_nope = q.reshape(
        TileLayout(
            (seq_len, Idx[num_heads], Idx[qk_nope_head_dim]),
            (Idx[num_heads * q_head_dim], Idx[q_head_dim], Idx[1]),
        )
    )
    var mla_decode_input_nope = mla_decode_input.reshape(
        TileLayout(
            (Idx[num_heads], seq_len, Idx[kv_latent_dim]),
            (Idx[k_cache_dim], Idx[num_heads * k_cache_dim], Idx[1]),
        )
    )
    quantize_and_bmm_fp8_helper[
        m_scale_granularity=m_scale_granularity,
        n_scale_granularity=n_scale_granularity,
        k_scale_granularity=k_scale_granularity,
        target=target,
    ](mla_decode_input_nope, q_nope, w_uk, w_uk_scale, ctx)

    var q_rope = q[:, :, qk_nope_head_dim:].reshape(
        TileLayout(
            (seq_len, Idx[num_heads], Idx[qk_rope_head_dim]),
            (Idx[num_heads * q_head_dim], Idx[q_head_dim], Idx[1]),
        )
    )
    var mla_decode_input_rope = mla_decode_input[:, :, kv_latent_dim:].reshape(
        TileLayout(
            (seq_len, Idx[num_heads], Idx[qk_rope_head_dim]),
            (Idx[num_heads * k_cache_dim], Idx[k_cache_dim], Idx[1]),
        )
    )
    mla_fused_rope_rmsnorm_quantization(
        mla_decode_input_rope,
        q_rope,
        input_row_offsets,
        freqs_cis,
        kv_norm_gamma,
        kv_collection,
        layer_idx,
        epsilon,
        kv_input_fn,
        ctx,
    )

    var raw_output_buf = ctx.enqueue_create_buffer[dtype](
        seq_len * num_heads * kv_latent_dim
    )
    var raw_output = TileTensor(
        raw_output_buf,
        row_major(seq_len, Idx[num_heads], Idx[kv_latent_dim]),
    )

    # `d_indices` / `topk_lengths` are int32 buffers reinterpreted as uint32:
    # invalid `-1` slots become 0xFFFFFFFF and are rejected by the kernel's
    # `idx >= 0` gather producer.
    var indices_tt = TileTensor(
        d_indices.unsafe_bitcast[UInt32](),
        row_major(seq_len * indices_stride),
    )
    var topk_lengths_tt = TileTensor(
        topk_lengths.unsafe_bitcast[UInt32](),
        row_major(seq_len),
    )
    var attn_sink_opt = Optional[ImmPointer[Float32, ImmutAnyOrigin]](None)
    if case .Some(attn_sink_ptr) = attn_sink_ptr:
        attn_sink_opt = ImmPointer[Float32, ImmutAnyOrigin](attn_sink_ptr)

    var k_cache = kv_collection.get_key_cache(Int(layer_idx))

    comptime if ctx.target.is_amd_gpu():
        # Run the dense decode kernel over each query token's top-k rows, read
        # in place; index buffers are per-token, so each `q` row is S=1.
        var batch_size = seq_len
        # View decode input/output as rank-4 [batch,1,heads,depth]; one slot
        # per gathered row (q_seq_len==1), so a size-1 axis, no data movement.
        var q_padded = mla_decode_input.reshape(
            TileLayout(
                (batch_size, Idx[1], Idx[num_heads], Idx[k_cache_dim]),
                (
                    Idx[num_heads * k_cache_dim],
                    Idx[num_heads * k_cache_dim],
                    Idx[k_cache_dim],
                    Idx[1],
                ),
            )
        )
        var raw_output_padded = raw_output.reshape(
            TileLayout(
                (batch_size, Idx[1], Idx[num_heads], Idx[kv_latent_dim]),
                (
                    Idx[num_heads * kv_latent_dim],
                    Idx[num_heads * kv_latent_dim],
                    Idx[kv_latent_dim],
                    Idx[1],
                ),
            )
        )
        # `topk_lengths` is the constant `index_topk`, so derive each gathered
        # row's real length from its `-1` suffix; confined to this AMD arm.
        var valid_lengths_buf = ctx.enqueue_create_buffer[DType.uint32](
            batch_size
        )
        # Slots past a token's valid length read this row; the launch below
        # clears it.
        var zero_row_buf = ctx.enqueue_create_buffer[type_of(k_cache).dtype](
            k_cache_dim
        )
        ctx.enqueue_function[
            _amd_sparse_mla_row_valid_lengths[
                _AMD_SPARSE_MLA_ROW_VALID_LENGTHS_THREADS
            ]
        ](
            d_indices,
            topk_lengths,
            valid_lengths_buf.unsafe_ptr(),
            Int32(indices_stride),
            zero_row_buf.unsafe_ptr().unsafe_bitcast[UInt8](),
            Int32(k_cache_dim * size_of[type_of(k_cache).dtype]()),
            grid_dim=(batch_size, 1, 1),
            block_dim=(_AMD_SPARSE_MLA_ROW_VALID_LENGTHS_THREADS, 1, 1),
        )
        var amd_lengths_tt = TileTensor(
            valid_lengths_buf.unsafe_ptr(), row_major(batch_size)
        )
        var padding_mask = PaddingMask(amd_lengths_tt)

        # Read the selected rows in place instead of gathering a copy.
        var k_rows = SparseRowsMHAOperand[row_width=k_cache_dim](
            KVCacheMHAOperand(k_cache),
            k_cache.blocks_raw_ptr(),
            zero_row_buf.unsafe_ptr(),
            d_indices,
            valid_lengths_buf.unsafe_ptr(),
            indices_stride,
        )
        flare_mla_decoding_rows[
            config=MHAConfig[mla_decode_input_dtype](num_heads, k_cache_dim)
        ](
            raw_output_padded,
            q_padded,
            k_rows,
            indices_stride,
            padding_mask,
            scale,
            ctx,
            scalar_args_buf,
        )
    else:
        # SM100 runs the native-FP8 sparse prefill kernel when heads fit its
        # tiles (128, or a multiple of 8 in (0,64]); Q is cast to FP8.
        comptime native_fp8_sparse = (
            dtype == .bfloat16
            and _mla_native_fp8_cache[collection_t.CacheType]()
            and _is_sm10x_gpu(ctx.default_device_info)
            and (num_heads == 128 or (num_heads <= 64 and num_heads % 8 == 0))
        )
        comptime if (
            collection_t.CacheType.dtype.is_float8()
            or collection_t.CacheType.dtype == .uint8
        ):
            comptime assert native_fp8_sparse or (
                collection_t.CacheType.dtype != .uint8
            ), "an NVFP4 latent cache needs the SM100 FP8-Q sparse prefill"
            comptime if native_fp8_sparse:
                var q_fp8_buf = ctx.enqueue_create_buffer[.float8_e4m3fn](
                    seq_len * num_heads * k_cache_dim
                )
                var q_fp8 = TileTensor(
                    q_fp8_buf,
                    row_major(seq_len, Idx[num_heads], Idx[k_cache_dim]),
                )
                var mla_decode_input_bf16 = mla_decode_input.bitcast[
                    DType.bfloat16
                ]().reshape(
                    row_major(seq_len, Idx[num_heads], Idx[k_cache_dim])
                )
                convert_bf16_to_fp8_e4m3fn(mla_decode_input_bf16, q_fp8, ctx)

                comptime cta_group = 2 if num_heads == 128 else 1
                comptime b_topk = 128 if cta_group == 2 else 64
                comptime num_mbars = 2 if cta_group == 2 else 4
                comptime config = MLASparseConfig[
                    DType.bfloat16,
                    b_topk_=b_topk,
                    num_mbars_=num_mbars,
                    cta_group_=cta_group,
                ](
                    num_q_heads=num_heads,
                    num_kv_heads=1,
                    # Tile stays laid out for 576. The kernel zero-fills the NoPE tail.
                    qk_depth=kv_latent_dim + 64,
                    input_qk_depth=k_cache_dim,
                    v_depth=kv_latent_dim,
                    indices_stride=indices_stride,
                    group=num_heads,
                )
                mla_prefill_sparse_qkv_fp8[
                    config=config,
                    group=num_heads,
                    q_depth=k_cache_dim,
                    scale_block_size=0,
                ](
                    raw_output,
                    q_fp8,
                    k_cache,
                    indices_tt,
                    topk_lengths_tt,
                    attn_sink_opt,
                    scale,
                    Int32(indices_stride),
                    ctx,
                )
            else:
                # FP8 cache + bf16 Q: run the FP8 sparse-prefill kernel over the
                # quantized cache at unit scale (no dequant scales yet, so
                # scale_block_size=0 and scales_ptr is a non-null dummy).
                var dummy_scales = (
                    raw_output_buf.unsafe_ptr()
                    .unsafe_bitcast[Float32]()
                    .as_unsafe_any_origin()
                )
                mla_sm100_prefill_sparse_fp8[
                    num_q_heads=num_heads,
                    qk_depth=k_cache_dim,
                    v_depth=kv_latent_dim,
                    indices_stride=indices_stride,
                    scale_block_size=0,
                ](
                    raw_output,
                    mla_decode_input,
                    k_cache,
                    indices_tt,
                    topk_lengths_tt,
                    attn_sink_opt,
                    dummy_scales,
                    scale,
                    ctx,
                )
        else:
            mla_sm100_prefill_sparse[
                num_q_heads=num_heads,
                qk_depth=k_cache_dim,
                v_depth=kv_latent_dim,
                indices_stride=indices_stride,
            ](
                raw_output,
                mla_decode_input,
                k_cache,
                indices_tt,
                topk_lengths_tt,
                attn_sink_opt,
                scale,
                ctx,
            )

    var output_t = output.reshape(
        TileLayout(
            (Idx[num_heads], seq_len, Idx[v_head_dim]),
            (Idx[v_head_dim], Idx[num_heads * v_head_dim], Idx[1]),
        )
    )
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


@inline(.always)
def _sparse_prefill_rows[
    physical_indices: Bool
](
    d_indices: OptionalReg[MutPointer[Int32, MutAnyOrigin]],
    prefill_d_indices: OptionalReg[MutPointer[Int32, MutAnyOrigin]],
) raises -> MutPointer[Int32, MutAnyOrigin]:
    """Returns the paged physical rows the sparse prefill branch gathers."""
    if case .Some(prefill_d_indices) = prefill_d_indices:
        return prefill_d_indices
    comptime if not physical_indices:
        raise Error(
            "sparse prefill needs prefill_d_indices when d_indices holds token"
            " positions"
        )
    return d_indices.value()


def mla_graph_takes_decode_branch[
    cache_dtype: DType, num_heads: Int, sparse_mla: Bool = False
](max_seq_len: Int) -> Bool:
    """Returns whether `mla_prefill_decode_graph_*` takes its decode branch.

    Parameters:
        cache_dtype: Element type of the latent KV cache.
        num_heads: Number of query attention heads.
        sparse_mla: Whether top-k sparse attention is in use; on AMD it
            routes S>1 to prefill.

    Args:
        max_seq_len: Maximum query length in the batch.

    Returns:
        True when `max_seq_len` fits the decode branch's query fold.
    """
    return (
        max_seq_len
        <= mla_decode_max_seq_len[cache_dtype, num_heads, sparse_mla]()
    )


def mla_prefill_decode_graph_fp8[
    dtype: DType,
    fp8_dtype: DType,
    fp8_scale_dtype: DType,
    collection_t: KVCollectionT,
    KVInputFnType: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int](IndexList[2]) -> SIMD[DType.bfloat16, width],
    //,
    m_scale_granularity: Int,
    n_scale_granularity: Int,
    k_scale_granularity: Int,
    mask_str: StaticString,
    target: StaticString = "cpu",
    sparse_mla: Bool = False,
    sparse_indices_stride: Int = 0,
    # Read-once shared-index MTP fold; threaded to flare_mla_decoding (fp8-KV
    # only; bf16 branch is parity plumbing, always False).
    fold_shared_index: Bool = False,
    # Stage Q in the FP8 cache dtype where the sparse decode supports it; see
    # `mla_decode_branch_fp8`.
    fp8_q: Bool = True,
    # Whether `extra_k` is supplied; see `flare_mla_decoding`.
    has_extra_k: Bool = False,
    physical_indices: Bool = True,
    # How the sparse index producer packs each row; see flare_mla_decoding.
    sparse_row: SparseRowLayout = SparseRowLayout(),
    # Presence of `attn_sink_ptr` / `topk_lengths` + `extra_topk_lengths`
    # (inferred from those arguments).
    AttnSinkPtrType: OptionalPointer = NullPointer[DType.float32],
    TopkLengthsPtrType: OptionalPointer = NullPointer[DType.int32],
    # Forwarded to both branches; see `mla_decode_branch_fp8`.
    attn_sink_inert: Bool = False,
](
    output: TileTensor[mut=True, dtype, address_space=.GENERIC, ...],
    q: TileTensor[dtype, address_space=.GENERIC, ...],
    input_row_offsets: TileTensor[.uint32, address_space=.GENERIC, ...],
    freqs_cis: TileTensor[_, address_space=.GENERIC, ...],
    kv_norm_gamma: TileTensor[_, address_space=.GENERIC, ...],
    kv_collection: collection_t,
    layer_idx: UInt32,
    scale: Float32,
    epsilon: Float32,
    buffer_row_offsets: TileTensor[.uint32, address_space=.GENERIC, ...],
    cache_offsets: TileTensor[mut=True, .uint32, address_space=.GENERIC, ...],
    buffer_length: Int,
    max_seq_len: Int,
    w_k: TileTensor[fp8_dtype, address_space=.GENERIC, ...],
    w_k_scale: TileTensor[fp8_scale_dtype, address_space=.GENERIC, ...],
    w_uk: TileTensor[fp8_dtype, address_space=.GENERIC, ...],
    w_uk_scale: TileTensor[fp8_scale_dtype, address_space=.GENERIC, ...],
    w_uv: TileTensor[fp8_dtype, address_space=.GENERIC, ...],
    w_uv_scale: TileTensor[fp8_scale_dtype, address_space=.GENERIC, ...],
    scalar_args_buf: TileTensor[.int64, address_space=.GENERIC, ...],
    kv_input_fn: KVInputFnType,
    ctx: DeviceContext,
    d_indices: OptionalReg[MutPointer[Int32, MutAnyOrigin]] = None,
    indices_stride: Int = 0,
    topk_lengths: TopkLengthsPtrType = null_pointer[TopkLengthsPtrType](),
    attn_sink_ptr: AttnSinkPtrType = null_pointer[AttnSinkPtrType](),
    extra_k: OptionalReg[collection_t.CacheType] = None,
    extra_d_indices: OptionalReg[MutPointer[Int32, MutAnyOrigin]] = None,
    extra_indices_stride: Int = 0,
    extra_topk_lengths: TopkLengthsPtrType = unread_pointer[
        TopkLengthsPtrType
    ](),
    extra_scales_ptr: OptionalReg[MutPointer[Float32, MutAnyOrigin]] = None,
    # Capturable-graph scalar forwarded from the MoGG op input list.
    num_partitions_in: Optional[Int] = None,
    prefill_d_indices: OptionalReg[MutPointer[Int32, MutAnyOrigin]] = None,
    selection_lag: Int = 0,
) raises:
    """
    This is a manually fused kernel that performs the following operations:
    - Perform MLA prefill or decode based on the maximum sequence length.

    Parameters:
        dtype: Data type of the input and output tensors (inferred).
        fp8_dtype: Data type of the fp8 input and output tensors (inferred).
        fp8_scale_dtype: Data type of the fp8 scale input and output tensors
            (inferred).
        collection_t: Type of the KV collection (inferred).
        KVInputFnType: Type of the closure that loads the KV latent
            values (inferred).
        m_scale_granularity: Granularity of the scale for M dimension of the
            matrix multiplication.
        n_scale_granularity: Granularity of the scale for N dimension of the
            matrix multiplication.
        k_scale_granularity: Granularity of the scale for K dimension of the
            matrix multiplication.
        mask_str: Mask variant.
        target: Target device (defaults to "cpu").
        sparse_mla: Whether to use sparse MLA (defaults to False).
        sparse_indices_stride: Row stride of the sparse decode index buffer
            (defaults to 0).
        fold_shared_index: Whether to use the read-once shared-index MTP
            fold threaded to `flare_mla_decoding` (defaults to False).
        fp8_q: Whether to stage Q in the FP8 cache dtype on the sparse decode
            path (defaults to True); see `mla_decode_branch_fp8`.
        has_extra_k: Whether `extra_k` is supplied (defaults to False); see
            `flare_mla_decoding`.
        physical_indices: Whether `d_indices` holds paged physical rows rather
            than token positions for the decode branch (defaults to True); see
            `flare_mla_decoding` and `prefill_d_indices`.
        sparse_row: How the sparse index producer packs each row, which
            bounds each decode row by its valid length (defaults to plain
            top-k).
        AttnSinkPtrType: `OptionalPointer` type of `attn_sink_ptr`
            (inferred, defaults to `NullPointer`).
        TopkLengthsPtrType: `OptionalPointer` type of `topk_lengths` and
            `extra_topk_lengths` (inferred, defaults to `NullPointer`).
        attn_sink_inert: Whether the caller guarantees an inert attention
            sink; see `mla_decode_branch_fp8`.

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
        max_seq_len: Maximum sequence length in the batch, used to select
            prefill versus decode.
        w_k: Weight matrix for up-projecting the latent cache to full K. Shape:
            [num_heads * qk_nope_head_dim, kv_latent_dim].
        w_k_scale: Scale tensor for `w_k`.
        w_uk: Weight matrix for projecting the non-rope part of each query head
            to KV latent space. Shape: [num_heads, kv_latent_dim,
            qk_nope_head_dim].
        w_uk_scale: The scale for the `w_uk` weight matrix. Shape varies
            depending on the FP8 block-scaling granularity used for weight
            quantization.
        w_uv: Weight tensor for projecting latent values to V. Shape:
            [num_heads, v_head_dim, kv_latent_dim].
        w_uv_scale: Scale tensor for `w_uv`.
        scalar_args_buf: Packed MLA dispatch metadata buffer.
        kv_input_fn: Closure that loads the KV latent values. Shape:
            [tot_seq_len, cache_head_dim]. Where cache_head_dim = kv_lora_rank
            + qk_rope_head_dim. It runs on the device, so it must copy what
            it captures.
        ctx: Device context.
        d_indices: Sparse decode packed indices (null when dense, defaults to
            None).
        indices_stride: Row stride in `d_indices` (defaults to 0).
        topk_lengths: Per-batch valid top-k counts (defaults to None).
        attn_sink_ptr: Optional per-batch attention sink weights (defaults to
            None).
        extra_k: Optional second key cache operand (see
            `flare_mla_decoding`, defaults to None).
        extra_d_indices: Extra-stream sparse indices (defaults to None).
        extra_indices_stride: Stride for `extra_d_indices` (defaults to 0).
        extra_topk_lengths: Extra-stream per-batch lengths (defaults to None).
        extra_scales_ptr: Extra-stream scales (defaults to None).
        num_partitions_in: Capturable-graph num_partitions override (defaults
            to None).
        prefill_d_indices: Paged physical rows the sparse prefill branch
            gathers in place of `d_indices` (defaults to None, meaning
            `d_indices`). Required without `physical_indices`, since only
            the decode kernels decode token positions.
        selection_lag: Positions between the token each sparse row was
            computed for and the token that reads it; read by the decode
            branch only.
    """

    var seq_len = q.dim(0)

    if seq_len == 0:
        return

    # When running verification with MTP we want to use the decode branch.
    # AMD sparse is the exception: its decode arm is S=1-only, so `sparse_mla`
    # sends S>1 to the sparse prefill arm instead.
    if mla_graph_takes_decode_branch[
        collection_t.CacheType.dtype, q.static_shape[1], sparse_mla
    ](max_seq_len):
        mla_decode_branch_fp8[
            m_scale_granularity=m_scale_granularity,
            n_scale_granularity=n_scale_granularity,
            k_scale_granularity=k_scale_granularity,
            mask_str=mask_str,
            target=target,
            sparse_mla=sparse_mla,
            fold_shared_index=fold_shared_index,
            fp8_q=fp8_q,
            has_extra_k=has_extra_k,
            physical_indices=physical_indices,
            sparse_row=sparse_row,
            attn_sink_inert=attn_sink_inert,
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
            scalar_args_buf,
            kv_input_fn,
            ctx,
            d_indices,
            indices_stride,
            topk_lengths,
            attn_sink_ptr,
            extra_k,
            extra_d_indices,
            extra_indices_stride,
            extra_topk_lengths,
            extra_scales_ptr,
            num_partitions_in,
            selection_lag=selection_lag,
        )

    else:
        comptime if sparse_mla:
            # Sparse MLA prefill is implemented for SM100 and (via the
            # in-place decode composition, see
            # `mla_prefill_branch_sparse_fp8`'s AMD arm) AMD; any other
            # architecture reaching this branch would instantiate SM100-only
            # tcgen05 kernels and fail with an unhelpful message. Fail fast
            # and name the real reason.
            comptime assert (
                _is_sm10x_gpu(ctx.default_device_info)
                or ctx.target.is_amd_gpu()
            ), (
                "Sparse MLA prefill is implemented only for SM100 (B200) and"
                " AMD (gfx950). This architecture would otherwise attempt to"
                " compile SM100-only tcgen05 kernels."
            )
            # The sparse prefill kernels read a per-token top-k length and
            # branch on the sink at runtime, so hand them that form. Absent
            # lengths mean every token keeps the full top-k stride.
            var full_topk: Optional[DeviceBuffer[DType.int32]] = None
            var prefill_topk_lengths: MutPointer[Int32, MutAnyOrigin]
            comptime if TopkLengthsPtrType.is_null:
                full_topk = ctx.enqueue_create_buffer[DType.int32](Int(seq_len))
                full_topk.value().enqueue_fill(
                    Int32(
                        _mla_sparse_row_topk[collection_t.CacheType](
                            sparse_indices_stride
                        )
                    )
                )
                prefill_topk_lengths = (
                    full_topk.value()
                    .unsafe_ptr()
                    .unsafe_origin_cast[MutAnyOrigin]()
                )
            else:
                prefill_topk_lengths = rebind[MutPointer[Int32, MutAnyOrigin]](
                    as_optional_reg(topk_lengths).value()
                )
            var prefill_attn_sink = rebind[
                OptionalReg[MutPointer[Float32, MutAnyOrigin]]
            ](as_optional_reg(attn_sink_ptr))
            # Sparse MLA prefill for BOTH bf16 and fp8 latent caches: the
            # branch comptime-dispatches the attention kernel on the cache
            # dtype (fp8 cache => mla_sm100_prefill_sparse_fp8 read at unit
            # scale, mirroring sparse decode; bf16 cache =>
            # mla_sm100_prefill_sparse). Replaces the dense FP8-cache fallback
            # that previously ran here.
            mla_prefill_branch_sparse_fp8[
                m_scale_granularity=m_scale_granularity,
                n_scale_granularity=n_scale_granularity,
                k_scale_granularity=k_scale_granularity,
                indices_stride=sparse_indices_stride,
                target=target,
                attn_sink_inert=attn_sink_inert,
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
                scalar_args_buf,
                kv_input_fn,
                ctx,
                _sparse_prefill_rows[physical_indices](
                    d_indices, prefill_d_indices
                ),
                prefill_topk_lengths,
                prefill_attn_sink,
            )
            _ = full_topk^
        else:
            # Dense prefill for NON-sparse MLA only. Sparse MLA (both bf16 and
            # fp8 caches) is handled by the sparse branch above.
            mla_prefill_branch_fp8[
                m_scale_granularity=m_scale_granularity,
                n_scale_granularity=n_scale_granularity,
                k_scale_granularity=k_scale_granularity,
                mask_str=mask_str,
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
                kv_input_fn,
                ctx,
            )


@inline(.always)
def convert_bf16_to_fp8_e4m3fn(
    input_buffer: TileTensor[mut=False, .bfloat16, ...],
    output_buffer: TileTensor[mut=True, .float8_e4m3fn, ...],
    context: DeviceContext,
) raises:
    """Convert bfloat16 weights to E4M3FN format.

    Args:
        input_buffer: Input tensor in bfloat16 format.
        output_buffer: Output tensor to store E4M3FN format.
        context: Device context for kernel execution.
    """
    # Runtime assertions for dynamic dimensions
    comptime assert (
        input_buffer.rank == output_buffer.rank
    ), "Input and output must have the same rank"

    @inline(.always)
    def convert_kernel[width: Int, alignment: Int = 1](idx: Coord) {var}:
        comptime assert idx.rank == 2 or idx.rank == 3, "rank should be 2 or 3"

        output_buffer.store(
            idx,
            cast_saturating[.float8_e4m3fn](input_buffer.load[width](idx)),
        )

    comptime target_simd_width = simd_width_of[
        DType.bfloat16, target=context.target
    ]()

    comptime if input_buffer.rank == 2:
        _elementwise_impl_gpu[
            simd_width=target_simd_width,
            trace_description="mla_bf16_to_fp8_convert",
        ](
            convert_kernel,
            shape=(
                Int(input_buffer.dim[0]()),
                Int(input_buffer.dim[1]()),
            ),
            ctx=context,
        )
    else:
        _elementwise_impl_gpu[
            simd_width=target_simd_width,
            trace_description="mla_bf16_to_fp8_convert",
        ](
            convert_kernel,
            shape=(
                Int(input_buffer.dim[0]()),
                Int(input_buffer.dim[1]()),
                Int(input_buffer.dim[2]()),
            ),
            ctx=context,
        )


# ===-----------------------------------------------------------------------===#
# Manually fused MLA prefill branch (BF16)
# ===-----------------------------------------------------------------------===#


def mla_prefill_branch_bf16[
    collection_t: KVCollectionT,
    KVInputFnType: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int](IndexList[2]) -> SIMD[DType.bfloat16, width],
    //,
    mask_str: StaticString,
    target: StaticString = "cpu",
](
    output: TileTensor[mut=True, .bfloat16, address_space=.GENERIC, ...],
    q: TileTensor[.bfloat16, address_space=.GENERIC, ...],
    input_row_offsets: TileTensor[.uint32, address_space=.GENERIC, ...],
    freqs_cis: TileTensor[_, address_space=.GENERIC, ...],
    kv_norm_gamma: TileTensor[_, address_space=.GENERIC, ...],
    kv_collection: collection_t,
    layer_idx: UInt32,
    scale: Float32,
    epsilon: Float32,
    buffer_row_offsets: TileTensor[.uint32, address_space=.GENERIC, ...],
    cache_offsets: TileTensor[mut=True, .uint32, address_space=.GENERIC, ...],
    buffer_length: Int,
    w_k: TileTensor[.bfloat16, address_space=.GENERIC, ...],
    w_uv: TileTensor[.bfloat16, address_space=.GENERIC, ...],
    kv_input_fn: KVInputFnType,
    ctx: DeviceContext,
) raises:
    """BF16 MLA prefill path.

    Applies RoPE and RMSNorm, up-projects latent KV to full K and V, then runs
    prefill attention.

    Parameters:
        collection_t: Type of the KV collection (inferred).
        KVInputFnType: Type of the closure that loads the KV latent
            values (inferred).
        mask_str: Mask variant.
        target: Target device (defaults to "cpu").

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
        w_uv: Weight tensor for projecting latent values to V. Shape:
            [num_heads, v_head_dim, kv_latent_dim].
        kv_input_fn: Closure that loads the KV latent values. Shape:
            [tot_seq_len, cache_head_dim]. Where cache_head_dim = kv_lora_rank
            + qk_rope_head_dim. It runs on the device, so it must copy what
            it captures.
        ctx: Device context.
    """
    comptime kv_params = collection_t.kv_params
    comptime assert kv_params.is_mla, "kv_params.is_mla should be true"
    comptime assert kv_params.num_heads == 1, "kv_params.num_heads should be 1"

    comptime num_heads = q.static_shape[1]
    comptime q_head_dim = q.static_shape[2]
    comptime v_head_dim = output.static_shape[2]
    comptime k_cache_dim = kv_params.head_size

    comptime assert w_k.shape_known, "w_k's shape should be static"
    comptime kv_latent_dim = w_k.static_shape[1]
    comptime qk_rope_head_dim = k_cache_dim - kv_latent_dim
    comptime qk_nope_head_dim = q_head_dim - qk_rope_head_dim
    comptime assert (
        qk_rope_head_dim == 0 or freqs_cis.static_shape[1] == qk_rope_head_dim
    ), "freqs_cis's width should be k_cache_dim - kv_latent_dim."
    comptime assert (
        w_k.static_shape[0] == num_heads * qk_nope_head_dim
    ), "w_k.shape[0] should be equal to num_heads * qk_nope_head_dim"
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
    # qk_rope_head_dim columns of each Q head. The slice carries runtime
    # extents, so reshape restores the static trailing dims.
    var q_rope = q[:, :, qk_nope_head_dim:].reshape(
        TileLayout(
            (seq_len, Idx[num_heads], Idx[qk_rope_head_dim]),
            (Idx[num_heads * q_head_dim], Idx[q_head_dim], Idx[1]),
        )
    )

    # In-place update of the rope part of the `q` tensor
    var q_rope_mut = TileTensor(
        q_rope.ptr.unsafe_mut_cast[True](),
        q_rope.layout,
    )

    mla_fused_rope_rmsnorm_quantization(
        q_rope_mut,
        q_rope.as_unsafe_any_origin(),  # hack aliasing.
        input_row_offsets,
        freqs_cis,
        kv_norm_gamma,
        kv_collection,
        layer_idx,
        epsilon,
        kv_input_fn,
        ctx,
    )

    # allocate buffers for latent KV
    var k_latent_buf = ctx.enqueue_create_buffer[.bfloat16](
        buffer_length * kv_latent_dim
    )
    var k_latent = TileTensor(
        k_latent_buf,
        row_major(buffer_length, Idx[kv_latent_dim]),
    )

    var buffer_length_int = Int(buffer_length)
    var k_cache = kv_collection.get_key_cache(Int(layer_idx))

    _k_cache_to_buffer(
        buffer_row_offsets,
        cache_offsets,
        k_cache,
        Int32(buffer_length_int),
        k_latent,
        ctx,
    )

    comptime if collection_t.CacheType.dtype.is_float8():
        # Allocate FP8 buffers for K and V
        var k_fp8_buf = ctx.enqueue_create_buffer[.float8_e4m3fn](
            buffer_length * num_heads * qk_nope_head_dim
        )
        var k_fp8_flat = TileTensor(
            k_fp8_buf,
            row_major((buffer_length, Idx[num_heads * qk_nope_head_dim])),
        )

        var v_fp8_buf = ctx.enqueue_create_buffer[.float8_e4m3fn](
            buffer_length * num_heads * v_head_dim
        )
        var v_fp8_flat = TileTensor(
            v_fp8_buf,
            row_major(buffer_length, Idx[num_heads * v_head_dim]),
        )

        # K matmul with internal FP8 conversion
        matmul[
            target=target,
            transpose_b=True,
        ](
            k_fp8_flat,
            k_latent,
            w_k,
            Optional(ctx),
        )

        var w_v = w_uv.reshape(
            row_major(Idx[num_heads * v_head_dim], Idx[kv_latent_dim]),
        )

        # V matmul with internal FP8 conversion
        matmul[
            target=target,
            transpose_b=True,
        ](
            v_fp8_flat,
            k_latent,
            w_v,
            Optional(ctx),
        )

        # Create 3D views for attention kernel
        var k_fp8 = TileTensor(
            k_fp8_buf,
            row_major((buffer_length, Idx[num_heads], Idx[qk_nope_head_dim])),
        )
        var v_fp8 = TileTensor(
            v_fp8_buf,
            row_major((buffer_length, Idx[num_heads], Idx[v_head_dim])),
        )

        # Allocate FP8 buffer for Q and convert
        var q_fp8_buf = ctx.enqueue_create_buffer[.float8_e4m3fn](
            seq_len * num_heads * q_head_dim
        )
        var q_fp8 = TileTensor(
            q_fp8_buf,
            row_major(seq_len, Idx[num_heads], Idx[q_head_dim]),
        )
        convert_bf16_to_fp8_e4m3fn(q, q_fp8, ctx)

        # Pass FP8 tensors to attention kernel
        generic_flare_mla_prefill_kv_cache_ragged[
            target=target,
            mask_str=mask_str,
        ](
            q_fp8,
            k_fp8,
            v_fp8,
            buffer_row_offsets,
            cache_offsets,
            input_row_offsets,
            kv_collection,
            layer_idx,
            scale,
            output,
            ctx,
        )
    else:
        # Standard BF16 path
        var k_buf = ctx.enqueue_create_buffer[.bfloat16](
            buffer_length * num_heads * qk_nope_head_dim
        )
        var k_flat = TileTensor(
            k_buf,
            row_major((buffer_length, Idx[num_heads * qk_nope_head_dim])),
        )
        matmul[target=target, transpose_b=True](
            k_flat,
            k_latent,
            w_k,
            Optional(ctx),
        )

        var w_v = w_uv.reshape(
            row_major(Idx[num_heads * v_head_dim], Idx[kv_latent_dim]),
        )
        var v_buf = ctx.enqueue_create_buffer[.bfloat16](
            buffer_length * num_heads * v_head_dim
        )
        var v_flat = TileTensor(
            v_buf,
            row_major(buffer_length, Idx[num_heads * v_head_dim]),
        )
        matmul[target=target, transpose_b=True](
            v_flat,
            k_latent,
            w_v,
            Optional(ctx),
        )

        var k = TileTensor(
            k_buf,
            row_major((buffer_length, Idx[num_heads], Idx[qk_nope_head_dim])),
        )
        var v = TileTensor(
            v_buf,
            row_major((buffer_length, Idx[num_heads], Idx[v_head_dim])),
        )

        generic_flare_mla_prefill_kv_cache_ragged[
            target=target,
            mask_str=mask_str,
        ](
            q,
            k,
            v,
            buffer_row_offsets,
            cache_offsets,
            input_row_offsets,
            kv_collection,
            layer_idx,
            scale,
            output,
            ctx,
        )


# ===-----------------------------------------------------------------------===#
# Manually fused MLA decode branch (BF16)
# ===-----------------------------------------------------------------------===#


def mla_decode_branch_bf16[
    collection_t: KVCollectionT,
    KVInputFnType: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int](IndexList[2]) -> SIMD[DType.bfloat16, width],
    //,
    mask_str: StaticString,
    target: StaticString = "cpu",
    sparse_mla: Bool = False,
    physical_indices: Bool = True,
    # How the sparse index producer packs each row; see flare_mla_decoding.
    sparse_row: SparseRowLayout = SparseRowLayout(),
    # Presence of `attn_sink_ptr` / `topk_lengths` + `extra_topk_lengths`
    # (inferred from those arguments).
    AttnSinkPtrType: OptionalPointer = NullPointer[DType.float32],
    TopkLengthsPtrType: OptionalPointer = NullPointer[DType.int32],
](
    output: TileTensor[mut=True, .bfloat16, address_space=.GENERIC, ...],
    q: TileTensor[.bfloat16, address_space=.GENERIC, ...],
    input_row_offsets: TileTensor[.uint32, address_space=.GENERIC, ...],
    freqs_cis: TileTensor[_, address_space=.GENERIC, ...],
    kv_norm_gamma: TileTensor[_, address_space=.GENERIC, ...],
    kv_collection: collection_t,
    layer_idx: UInt32,
    scale: Float32,
    epsilon: Float32,
    w_uk: TileTensor[.bfloat16, address_space=.GENERIC, ...],
    w_uv: TileTensor[.bfloat16, address_space=.GENERIC, ...],
    scalar_args_buf: TileTensor[.int64, address_space=.GENERIC, ...],
    kv_input_fn: KVInputFnType,
    ctx: DeviceContext,
    d_indices: OptionalReg[MutPointer[Int32, MutAnyOrigin]] = None,
    indices_stride: Int = 0,
    topk_lengths: TopkLengthsPtrType = null_pointer[TopkLengthsPtrType](),
    attn_sink_ptr: AttnSinkPtrType = null_pointer[AttnSinkPtrType](),
    # Capturable-graph scalar forwarded from the MoGG op input list.
    num_partitions_in: Optional[Int] = None,
    # See flare_mla_decoding.
    selection_lag: Int = 0,
) raises:
    """BF16 MLA decode path.

    Applies RoPE and RMSNorm, projects q_nope to latent space, concatenates with
    q_rope, and runs decode.

    Parameters:
        collection_t: Type of the KV collection (inferred).
        KVInputFnType: Type of the closure that loads the KV latent
            values (inferred).
        mask_str: Mask variant.
        target: Target device (defaults to "cpu").
        sparse_mla: Whether to use sparse MLA (defaults to False).
        physical_indices: Whether `d_indices` holds paged physical rows rather
            than token positions (defaults to True); see `flare_mla_decoding`.
        sparse_row: How the sparse index producer packs each row, which
            bounds each row by its valid length (defaults to plain top-k).
        AttnSinkPtrType: `OptionalPointer` type of `attn_sink_ptr`
            (inferred, defaults to `NullPointer`).
        TopkLengthsPtrType: `OptionalPointer` type of `topk_lengths` and
            `extra_topk_lengths` (inferred, defaults to `NullPointer`).

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
        w_uk: Weight matrix for projecting the non-rope part of each query head
            to KV latent space. Shape: [num_heads, kv_latent_dim,
            qk_nope_head_dim].
        w_uv: Weight matrix for projecting the output of the attention back to
            each head's original space. Shape: [num_heads, v_head_dim,
            kv_latent_dim].
        scalar_args_buf: Packed MLA dispatch metadata buffer.
        kv_input_fn: Closure that loads the KV latent values. Shape:
            [tot_seq_len, cache_head_dim]. Where cache_head_dim = kv_lora_rank
            + qk_rope_head_dim. It runs on the device, so it must copy what
            it captures.
        ctx: Device context.
        d_indices: Sparse decode packed indices (null when dense).
        indices_stride: Row stride in `d_indices` (defaults to 0).
        topk_lengths: Per-batch valid top-k counts.
        attn_sink_ptr: Optional per-batch attention sink weights.
        num_partitions_in: Capturable-graph num_partitions override.
        selection_lag: Positions between the token each sparse row was
            computed for and the token that reads it.
    """
    comptime kv_params = collection_t.kv_params
    comptime assert kv_params.is_mla, "kv_params.is_mla should be true"
    comptime assert kv_params.num_heads == 1, "kv_params.num_heads should be 1"

    comptime num_heads = q.static_shape[1]
    comptime q_head_dim = q.static_shape[2]
    comptime v_head_dim = output.static_shape[2]
    comptime k_cache_dim = _mla_cache_row_elems[
        kv_params, collection_t.CacheType.dtype
    ]()

    comptime assert (
        w_uk.shape_known and w_uv.shape_known
    ), "w_uk and w_uv's shapes should be static"
    comptime kv_latent_dim = w_uk.static_shape[1]
    comptime qk_rope_head_dim = k_cache_dim - kv_latent_dim
    comptime qk_nope_head_dim = q_head_dim - qk_rope_head_dim
    comptime assert (
        qk_rope_head_dim == 0 or freqs_cis.static_shape[1] == qk_rope_head_dim
    ), "freqs_cis's width should be k_cache_dim - kv_latent_dim."
    comptime assert (
        w_uk.static_shape[2] == qk_nope_head_dim
    ), "w_uk.static_shape[2] should be equal to qk_nope_head_dim"
    comptime assert (
        w_uv.static_shape[2] == kv_latent_dim
    ), "w_uv.static_shape[2] should be equal to kv_latent_dim"
    comptime assert (
        w_uv.static_shape[1] == v_head_dim
    ), "w_uv.static_shape[1] should be equal to v_head_dim"

    var seq_len = Int(q.dim(0))
    if seq_len == 0:
        return

    # Sparse decode over a unit-scale FP8 latent cache on SM100 stages Q in
    # FP8 so the dispatch routes to the native-FP8 sparse kernel
    # (MLA_SM100_Decode_Sparse_QKV_FP8, SWIZZLE_64B FP8 Q TMA). Other sparse
    # configs keep bf16 Q: the BF16/converter sparse kernels' Q TMA uses
    # SWIZZLE_128B over BK=576 elements, which requires bf16 (1152 B).
    # Non-sparse bf16 decode may still stage Q in the KV cache dtype when the
    # cache is FP8.
    comptime native_fp8_sparse = (
        sparse_mla
        and _mla_native_fp8_cache[collection_t.CacheType]()
        and _is_sm10x_gpu(ctx.default_device_info)
    )
    comptime mla_decode_input_dtype = (
        DType.float8_e4m3fn if native_fp8_sparse else (
            DType.bfloat16 if sparse_mla else collection_t.CacheType.dtype
        )
    )
    var mla_decode_input_buf = ctx.enqueue_create_buffer[
        mla_decode_input_dtype
    ](seq_len * num_heads * k_cache_dim)
    var mla_decode_input = TileTensor(
        mla_decode_input_buf,
        row_major(seq_len, Idx[num_heads], Idx[k_cache_dim]),
    )

    # =========================================================================#
    # QK RoPE and K cache RMSNorm                                              #
    # =========================================================================#

    # Create a view of the `q` tensor that only contains the last
    # qk_rope_head_dim columns of each Q head. The slice carries runtime
    # extents, so reshape restores the static trailing dims.
    var q_rope = q[:, :, qk_nope_head_dim:].reshape(
        TileLayout(
            (seq_len, Idx[num_heads], Idx[qk_rope_head_dim]),
            (Idx[num_heads * q_head_dim], Idx[q_head_dim], Idx[1]),
        )
    )

    # Create a view of the `mla_decode_input` tensor that only contains the last
    # qk_rope_head_dim columns of each Q head.
    var mla_decode_input_rope = mla_decode_input[:, :, kv_latent_dim:].reshape(
        TileLayout(
            (seq_len, Idx[num_heads], Idx[qk_rope_head_dim]),
            (Idx[num_heads * k_cache_dim], Idx[k_cache_dim], Idx[1]),
        )
    )

    mla_fused_rope_rmsnorm_quantization(
        mla_decode_input_rope,
        q_rope,
        input_row_offsets,
        freqs_cis,
        kv_norm_gamma,
        kv_collection,
        layer_idx,
        epsilon,
        kv_input_fn,
        ctx,
    )

    # =========================================================================#
    # Project the non-rope part of each query head to kv_latent_dim            #
    # =========================================================================#

    # Create a view of the `q` tensor that only contains the first
    # qk_nope_head_dim columns of each Q head. Also transposed to
    # [num_heads, seq_len, qk_nope_head_dim].
    var q_nope_t = q.reshape(
        TileLayout(
            (Idx[num_heads], seq_len, Idx[qk_nope_head_dim]),
            (Idx[q_head_dim], Idx[num_heads * q_head_dim], Idx[1]),
        )
    )

    # Then create a view of the mla_decode_input tensor that only contains the
    # first kv_latent_dim columns of each Q head.
    var mla_decode_input_nope = mla_decode_input.reshape(
        TileLayout(
            (Idx[num_heads], seq_len, Idx[kv_latent_dim]),
            (Idx[k_cache_dim], Idx[num_heads * k_cache_dim], Idx[1]),
        )
    )

    _batched_matmul_gpu[transpose_b=True](
        mla_decode_input_nope, q_nope_t, w_uk, ctx
    )

    # Perform MLA decode
    var raw_output_buf = ctx.enqueue_create_buffer[.bfloat16](
        seq_len * num_heads * kv_latent_dim
    )
    var raw_output = TileTensor(
        raw_output_buf,
        row_major(seq_len, Idx[num_heads], Idx[kv_latent_dim]),
    )

    comptime if ctx.target.is_amd_gpu() and sparse_mla:
        # gfx950 has no sparse MLA kernel, so gather each token's selected
        # latent rows into a dense scratch and run the dense decode kernel on
        # it, as `mla_decode_branch_fp8`'s AMD arm does. Each `q` row carries
        # its own selection, so a prefill routed here works the same as decode.
        comptime assert (
            collection_t.CacheType.dtype == DType.bfloat16
        ), "the BF16 AMD sparse MLA arm gathers from a BF16 latent cache"
        comptime assert (
            AttnSinkPtrType.is_null
        ), "AMD sparse MLA decode does not consume attention sinks"
        comptime assert (
            not TopkLengthsPtrType.is_null
        ), "AMD sparse MLA decode requires topk_lengths"
        comptime assert physical_indices, (
            "the AMD sparse MLA arm gathers paged physical rows; the graph op"
            " remaps logical indices before calling it"
        )
        if d_indices is None:
            raise Error("AMD sparse MLA decode requires d_indices.")

        var k_cache = kv_collection.get_key_cache(Int(layer_idx))
        var k_cache_blocks_base = k_cache.blocks_raw_ptr()
        var topk_lengths_ptr = rebind[MutPointer[Int32, MutAnyOrigin]](
            as_optional_reg(topk_lengths).value()
        )

        var compact_buf = ctx.enqueue_create_buffer[DType.int32](
            seq_len * indices_stride
        )
        var valid_lengths_buf = ctx.enqueue_create_buffer[DType.uint32](seq_len)
        ctx.enqueue_function[
            _amd_sparse_mla_compact_indices[
                _AMD_SPARSE_MLA_ROW_VALID_LENGTHS_THREADS
            ]
        ](
            d_indices.value(),
            topk_lengths_ptr,
            compact_buf.unsafe_ptr(),
            valid_lengths_buf.unsafe_ptr(),
            Int32(indices_stride),
            grid_dim=(seq_len, 1, 1),
            block_dim=(_AMD_SPARSE_MLA_ROW_VALID_LENGTHS_THREADS, 1, 1),
        )

        # A gathered row is `indices_stride * k_cache_dim` elements (~2.4 MB
        # for GLM-5.3-Flash), so a whole prefill chunk would not fit; bound
        # the scratch to `_AMD_SPARSE_MLA_MAX_GATHER_TOKENS` tokens.
        # TODO(vprabhakar): a fused gather-attend kernel would drop the
        # scratch and the per-chunk launches.
        var chunk_tokens = min(seq_len, _AMD_SPARSE_MLA_MAX_GATHER_TOKENS)
        var scratch_buf = ctx.enqueue_create_buffer[mla_decode_input_dtype](
            chunk_tokens * indices_stride * k_cache_dim
        )
        comptime GATHER_ROWS_PER_BLOCK = 7
        for token0 in range(0, seq_len, chunk_tokens):
            var tokens = min(chunk_tokens, seq_len - token0)
            ctx.enqueue_function[
                _amd_sparse_mla_gather_kv_rows[
                    type_of(k_cache),
                    k_cache_dim,
                    GATHER_ROWS_PER_BLOCK,
                ]
            ](
                k_cache,
                k_cache_blocks_base,
                compact_buf.unsafe_ptr().unsafe_offset(token0 * indices_stride),
                topk_lengths_ptr.unsafe_offset(token0),
                scratch_buf.unsafe_ptr(),
                Int32(indices_stride),
                grid_dim=(
                    ceildiv(indices_stride, GATHER_ROWS_PER_BLOCK),
                    tokens,
                    1,
                ),
                block_dim=(
                    k_cache_dim // _AMD_SPARSE_MLA_GATHER_VEC,
                    GATHER_ROWS_PER_BLOCK,
                    1,
                ),
            )
            var k_scratch = TileTensor(
                scratch_buf,
                row_major(tokens, indices_stride, Idx[1], Idx[k_cache_dim]),
            )
            var q_tokens = TileTensor(
                mla_decode_input.ptr.unsafe_offset(
                    token0 * num_heads * k_cache_dim
                ),
                TileLayout(
                    (tokens, Idx[1], Idx[num_heads], Idx[k_cache_dim]),
                    (
                        Idx[num_heads * k_cache_dim],
                        Idx[num_heads * k_cache_dim],
                        Idx[k_cache_dim],
                        Idx[1],
                    ),
                ),
            )
            var out_tokens = TileTensor(
                raw_output.ptr.unsafe_offset(
                    token0 * num_heads * kv_latent_dim
                ),
                TileLayout(
                    (tokens, Idx[1], Idx[num_heads], Idx[kv_latent_dim]),
                    (
                        Idx[num_heads * kv_latent_dim],
                        Idx[num_heads * kv_latent_dim],
                        Idx[kv_latent_dim],
                        Idx[1],
                    ),
                ),
            )
            var lengths_tt = TileTensor(
                valid_lengths_buf.unsafe_ptr().unsafe_offset(token0),
                row_major(tokens),
            )
            # Sized for this token chunk rather than taken from the graph; at
            # decode the chunk is the whole batch, matching the capture key.
            flare_mla_decoding[
                config=MHAConfig[mla_decode_input_dtype](
                    num_heads, k_cache_dim
                ),
                decoding_warp_split_k=False,
            ](
                out_tokens,
                q_tokens,
                k_scratch,
                PaddingMask(lengths_tt),
                scale,
                ctx,
                scalar_args_buf,
                num_partitions=amd_sparse_mla_decode_num_partitions(
                    tokens,
                    indices_stride,
                    num_heads,
                    ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT),
                ),
            )
        _ = compact_buf^
        _ = valid_lengths_buf^
        _ = scratch_buf^
    else:
        generic_flare_mla_decode_kv_cache_ragged[
            target=target,
            mask_str=mask_str,
            sparse_mla=sparse_mla,
            physical_indices=physical_indices,
            sparse_row=sparse_row,
        ](
            mla_decode_input,
            input_row_offsets,
            kv_collection,
            layer_idx,
            scale,
            raw_output,
            scalar_args_buf,
            ctx,
            d_indices=d_indices,
            indices_stride=indices_stride,
            topk_lengths=topk_lengths,
            attn_sink_ptr=attn_sink_ptr,
            num_partitions_in=num_partitions_in,
            selection_lag=selection_lag,
        )

    # Create a view of the raw output tensor with logical shape
    # [num_heads, seq_len, kv_latent_dim], and map directly to
    # [seq_len, num_heads, kv_latent_dim] physical memory.
    var raw_output_t = TileTensor(
        raw_output_buf,
        TileLayout(
            (Idx[num_heads], seq_len, Idx[kv_latent_dim]),
            (Idx[kv_latent_dim], Idx[num_heads * kv_latent_dim], Idx[1]),
        ),
    )

    # Create a view of the output tensor with logical shape
    # [num_heads, seq_len, v_head_dim], and map directly to
    # [seq_len, num_heads, v_head_dim] physical memory.
    var output_t = output.reshape(
        TileLayout(
            (Idx[num_heads], seq_len, Idx[v_head_dim]),
            (Idx[v_head_dim], Idx[num_heads * v_head_dim], Idx[1]),
        )
    )

    _batched_matmul_gpu[transpose_b=True](output_t, raw_output_t, w_uv, ctx)


# ===-----------------------------------------------------------------------===#
# Manually fused MLA sparse prefill branch (BF16)
# ===-----------------------------------------------------------------------===#


@inline(.always)
def mla_prefill_branch_sparse_bf16[
    collection_t: KVCollectionT,
    KVInputFnType: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int](IndexList[2]) -> SIMD[DType.bfloat16, width],
    //,
    indices_stride: Int,
    target: StaticString = "cpu",
    # See `nvfp4_prefill_converts_once`; 0 always runs the NVFP4 kernel.
    nvfp4_convert_ratio: Int = 0,
    nvfp4_convert_max_rows: Int = 0,
](
    output: TileTensor[mut=True, .bfloat16, address_space=.GENERIC, ...],
    q: TileTensor[.bfloat16, address_space=.GENERIC, ...],
    input_row_offsets: TileTensor[.uint32, address_space=.GENERIC, ...],
    freqs_cis: TileTensor[_, address_space=.GENERIC, ...],
    kv_norm_gamma: TileTensor[_, address_space=.GENERIC, ...],
    kv_collection: collection_t,
    layer_idx: UInt32,
    scale: Float32,
    epsilon: Float32,
    w_uk: TileTensor[.bfloat16, address_space=.GENERIC, ...],
    w_uv: TileTensor[.bfloat16, address_space=.GENERIC, ...],
    kv_input_fn: KVInputFnType,
    ctx: DeviceContext,
    d_indices: MutPointer[Int32, MutAnyOrigin],
    topk_lengths: MutPointer[Int32, MutAnyOrigin],
    attn_sink_ptr: OptionalReg[MutPointer[Float32, MutAnyOrigin]],
) raises:
    """Sparse MLA prefill branch (DSv3.2/GLM absorbed shape, BF16 weights).

    BF16 analogue of `mla_prefill_branch_sparse_fp8`: reuses
    `mla_decode_branch_bf16`'s absorbed-Q construction (q_nope up-proj via `w_uk`
    + RoPE/RMSNorm) and the identical `w_uv` output up-projection, and swaps the
    attention call to the existing `mla_sm100_prefill_sparse` kernel over the
    paged BF16 latent cache. The caller (`.sparse` op) has already remapped
    `d_indices` from logical to physical rows, so they are passed straight
    through. Only supported for a BF16 KV cache.
    """
    comptime kv_params = collection_t.kv_params
    comptime assert kv_params.is_mla, "kv_params.is_mla should be true"
    comptime assert kv_params.num_heads == 1, "kv_params.num_heads should be 1"

    comptime num_heads = q.static_shape[1]
    comptime q_head_dim = q.static_shape[2]
    comptime v_head_dim = output.static_shape[2]
    comptime k_cache_dim = _mla_cache_row_elems[
        kv_params, collection_t.CacheType.dtype
    ]()

    comptime assert (
        w_uk.shape_known and w_uv.shape_known
    ), "w_uk and w_uv's shapes should be static"
    comptime kv_latent_dim = w_uk.static_shape[1]
    comptime qk_rope_head_dim = k_cache_dim - kv_latent_dim
    comptime qk_nope_head_dim = q_head_dim - qk_rope_head_dim
    comptime assert (
        qk_rope_head_dim == 0 or freqs_cis.static_shape[1] == qk_rope_head_dim
    ), "freqs_cis's width should be k_cache_dim - kv_latent_dim."
    comptime assert (
        w_uk.static_shape[2] == qk_nope_head_dim
    ), "w_uk.static_shape[2] should be equal to qk_nope_head_dim"
    comptime assert (
        w_uv.static_shape[1] == v_head_dim
    ), "w_uv.static_shape[1] should be equal to v_head_dim"

    var seq_len = Int(q.dim(0))
    if seq_len == 0:
        return

    # SM100 stages Q in FP8 and runs the native-FP8 sparse prefill kernel when
    # heads fit its tiles (128, or a multiple of 8 in (0,64]); else keep bf16 Q.
    comptime native_fp8_sparse = (
        _mla_native_fp8_cache[collection_t.CacheType]()
        and _is_sm10x_gpu(ctx.default_device_info)
        and (num_heads == 128 or (num_heads <= 64 and num_heads % 8 == 0))
    )
    comptime q_staging_dtype = (
        DType.float8_e4m3fn if native_fp8_sparse else DType.bfloat16
    )
    var mla_decode_input_buf = ctx.enqueue_create_buffer[q_staging_dtype](
        seq_len * num_heads * k_cache_dim
    )
    var mla_decode_input = TileTensor(
        mla_decode_input_buf,
        row_major(seq_len, Idx[num_heads], Idx[k_cache_dim]),
    )

    # Transposed view [num_heads, seq_len, qk_nope_head_dim] of the first
    # qk_nope_head_dim columns of each Q head.
    var q_nope_t = q.reshape(
        TileLayout(
            (Idx[num_heads], seq_len, Idx[qk_nope_head_dim]),
            (Idx[q_head_dim], Idx[num_heads * q_head_dim], Idx[1]),
        )
    )
    var mla_decode_input_nope = mla_decode_input.reshape(
        TileLayout(
            (Idx[num_heads], seq_len, Idx[kv_latent_dim]),
            (Idx[k_cache_dim], Idx[num_heads * k_cache_dim], Idx[1]),
        )
    )
    _batched_matmul_gpu[transpose_b=True](
        mla_decode_input_nope, q_nope_t, w_uk, ctx
    )

    var q_rope = q[:, :, qk_nope_head_dim:].reshape(
        TileLayout(
            (seq_len, Idx[num_heads], Idx[qk_rope_head_dim]),
            (Idx[num_heads * q_head_dim], Idx[q_head_dim], Idx[1]),
        )
    )
    var mla_decode_input_rope = mla_decode_input[:, :, kv_latent_dim:].reshape(
        TileLayout(
            (seq_len, Idx[num_heads], Idx[qk_rope_head_dim]),
            (Idx[num_heads * k_cache_dim], Idx[k_cache_dim], Idx[1]),
        )
    )
    mla_fused_rope_rmsnorm_quantization(
        mla_decode_input_rope,
        q_rope,
        input_row_offsets,
        freqs_cis,
        kv_norm_gamma,
        kv_collection,
        layer_idx,
        epsilon,
        kv_input_fn,
        ctx,
    )

    var raw_output_buf = ctx.enqueue_create_buffer[.bfloat16](
        seq_len * num_heads * kv_latent_dim
    )
    var raw_output = TileTensor(
        raw_output_buf,
        row_major(seq_len, Idx[num_heads], Idx[kv_latent_dim]),
    )

    # `d_indices` / `topk_lengths` are int32 buffers reinterpreted as uint32:
    # invalid `-1` slots become 0xFFFFFFFF and are rejected by the kernel's
    # `idx >= 0` gather producer.
    var indices_tt = TileTensor(
        d_indices.unsafe_bitcast[UInt32](),
        row_major(seq_len * indices_stride),
    )
    var topk_lengths_tt = TileTensor(
        topk_lengths.unsafe_bitcast[UInt32](),
        row_major(seq_len),
    )
    var attn_sink_opt = Optional[ImmPointer[Float32, ImmutAnyOrigin]](None)
    if case .Some(attn_sink_ptr) = attn_sink_ptr:
        attn_sink_opt = ImmPointer[Float32, ImmutAnyOrigin](attn_sink_ptr)

    var k_cache = kv_collection.get_key_cache(Int(layer_idx))
    comptime if (
        collection_t.CacheType.dtype.is_float8()
        or collection_t.CacheType.dtype == .uint8
    ):
        comptime assert native_fp8_sparse or (
            collection_t.CacheType.dtype != .uint8
        ), "an NVFP4 latent cache needs the SM100 FP8-Q sparse prefill"
        comptime if native_fp8_sparse:
            # Native-FP8 sparse prefill: Q was staged in FP8 above, and the
            # kernel reads the unit-scale FP8 latent cache with no dequant.
            # Config knobs follow the kernel's validated tactics: head=128
            # runs the 2-CTA f8f6f4 tile at b_topk=128/num_mbars=2; head<=64
            # runs the single-CTA shared-KV tile at b_topk=64 (its only
            # correct PV mn-major layout) with num_mbars=4.
            comptime cta_group = 2 if num_heads == 128 else 1
            comptime b_topk = 128 if cta_group == 2 else 64
            comptime num_mbars = 2 if cta_group == 2 else 4
            comptime config = MLASparseConfig[
                DType.bfloat16,
                b_topk_=b_topk,
                num_mbars_=num_mbars,
                cta_group_=cta_group,
            ](
                num_q_heads=num_heads,
                num_kv_heads=1,
                # Tile stays laid out for 576. The kernel zero-fills the NoPE tail.
                qk_depth=kv_latent_dim + 64,
                input_qk_depth=k_cache_dim,
                v_depth=kv_latent_dim,
                indices_stride=indices_stride,
                group=num_heads,
            )
            var converted = False
            comptime if (
                _mla_nvfp4_cache[collection_t.CacheType]()
                and nvfp4_convert_ratio > 0
                and (indices_stride // 2) % 4 == 0
            ):
                if nvfp4_prefill_converts_once(
                    nvfp4_prefill_scratch_rows[
                        collection_t.CacheType.page_size_
                    ](
                        Int(input_row_offsets.dim(0)) - 1,
                        Int(k_cache.max_context_length()),
                    ),
                    seq_len,
                    nvfp4_convert_ratio,
                    nvfp4_convert_max_rows,
                ):
                    mla_prefill_sparse_nvfp4_via_fp8[
                        config=config, group=num_heads, q_depth=k_cache_dim
                    ](
                        raw_output,
                        mla_decode_input,
                        k_cache,
                        d_indices,
                        indices_stride,
                        topk_lengths_tt,
                        input_row_offsets,
                        attn_sink_opt,
                        scale,
                        ctx,
                    )
                    converted = True
            if not converted:
                mla_prefill_sparse_qkv_fp8[
                    config=config,
                    group=num_heads,
                    q_depth=k_cache_dim,
                    scale_block_size=0,
                ](
                    raw_output,
                    mla_decode_input,
                    k_cache,
                    indices_tt,
                    topk_lengths_tt,
                    attn_sink_opt,
                    scale,
                    Int32(indices_stride),
                    ctx,
                )
        else:
            # FP8 cache + bf16 Q: run the FP8 sparse-prefill kernel over the
            # quantized cache at unit scale (no dequant scales stored yet, so
            # scale_block_size=0 and scales_ptr is a non-null dummy).
            var dummy_scales = (
                raw_output_buf.unsafe_ptr()
                .unsafe_bitcast[Float32]()
                .as_unsafe_any_origin()
            )
            mla_sm100_prefill_sparse_fp8[
                num_q_heads=num_heads,
                qk_depth=k_cache_dim,
                v_depth=kv_latent_dim,
                indices_stride=indices_stride,
                scale_block_size=0,
            ](
                raw_output,
                mla_decode_input,
                k_cache,
                indices_tt,
                topk_lengths_tt,
                attn_sink_opt,
                dummy_scales,
                scale,
                ctx,
            )
    else:
        mla_sm100_prefill_sparse[
            num_q_heads=num_heads,
            qk_depth=k_cache_dim,
            v_depth=kv_latent_dim,
            indices_stride=indices_stride,
        ](
            raw_output,
            mla_decode_input,
            k_cache,
            indices_tt,
            topk_lengths_tt,
            attn_sink_opt,
            scale,
            ctx,
        )

    var raw_output_t = TileTensor(
        raw_output_buf,
        TileLayout(
            (Idx[num_heads], seq_len, Idx[kv_latent_dim]),
            (Idx[kv_latent_dim], Idx[num_heads * kv_latent_dim], Idx[1]),
        ),
    )
    var output_t = output.reshape(
        TileLayout(
            (Idx[num_heads], seq_len, Idx[v_head_dim]),
            (Idx[v_head_dim], Idx[num_heads * v_head_dim], Idx[1]),
        )
    )
    _batched_matmul_gpu[transpose_b=True](output_t, raw_output_t, w_uv, ctx)


# ===-----------------------------------------------------------------------===#
# MLA prefill-decode graph (BF16)
# ===-----------------------------------------------------------------------===#


@inline(.always)
def mla_prefill_decode_graph_bf16[
    collection_t: KVCollectionT,
    KVInputFnType: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int](IndexList[2]) -> SIMD[DType.bfloat16, width],
    //,
    mask_str: StaticString,
    target: StaticString = "cpu",
    sparse_mla: Bool = False,
    sparse_indices_stride: Int = 0,
    # Whether the sparse indices are paged physical rows rather than token
    # positions; see `flare_mla_decoding`.
    physical_indices: Bool = True,
    # How the sparse index producer packs each row; see flare_mla_decoding.
    sparse_row: SparseRowLayout = SparseRowLayout(),
    # Presence of `attn_sink_ptr` / `topk_lengths` + `extra_topk_lengths`
    # (inferred from those arguments).
    AttnSinkPtrType: OptionalPointer = NullPointer[DType.float32],
    TopkLengthsPtrType: OptionalPointer = NullPointer[DType.int32],
    nvfp4_convert_ratio: Int = 0,
    nvfp4_convert_max_rows: Int = 0,
](
    output: TileTensor[mut=True, .bfloat16, address_space=.GENERIC, ...],
    q: TileTensor[.bfloat16, address_space=.GENERIC, ...],
    input_row_offsets: TileTensor[.uint32, address_space=.GENERIC, ...],
    freqs_cis: TileTensor[_, address_space=.GENERIC, ...],
    kv_norm_gamma: TileTensor[_, address_space=.GENERIC, ...],
    kv_collection: collection_t,
    layer_idx: UInt32,
    scale: Float32,
    epsilon: Float32,
    buffer_row_offsets: TileTensor[
        mut=False, .uint32, address_space=.GENERIC, ...
    ],
    cache_offsets: TileTensor[mut=True, .uint32, address_space=.GENERIC, ...],
    buffer_length: Int,
    max_seq_len: Int,
    w_k: TileTensor[.bfloat16, address_space=.GENERIC, ...],
    w_uk: TileTensor[.bfloat16, address_space=.GENERIC, ...],
    w_uv: TileTensor[.bfloat16, address_space=.GENERIC, ...],
    scalar_args_buf: TileTensor[.int64, address_space=.GENERIC, ...],
    kv_input_fn: KVInputFnType,
    ctx: DeviceContext,
    d_indices: OptionalReg[MutPointer[Int32, MutAnyOrigin]] = None,
    indices_stride: Int = 0,
    topk_lengths: TopkLengthsPtrType = null_pointer[TopkLengthsPtrType](),
    attn_sink_ptr: AttnSinkPtrType = null_pointer[AttnSinkPtrType](),
    # Capturable-graph scalar forwarded from the MoGG op input list.
    num_partitions_in: Optional[Int] = None,
    prefill_d_indices: OptionalReg[MutPointer[Int32, MutAnyOrigin]] = None,
    selection_lag: Int = 0,
) raises:
    """BF16 MLA prefill/decode graph.

    Dispatches to prefill or decode based on max sequence length in the batch.

    Parameters:
        collection_t: Type of the KV collection (inferred).
        KVInputFnType: Type of the closure that loads the KV latent
            values (inferred).
        mask_str: Mask variant.
        target: Target device (defaults to "cpu").
        sparse_mla: Whether to use sparse MLA (defaults to False).
        sparse_indices_stride: Row stride of the sparse decode index buffer
            (defaults to 0).
        physical_indices: Whether `d_indices` holds paged physical rows rather
            than token positions for the decode branch (defaults to True); see
            `flare_mla_decoding` and `prefill_d_indices`.
        sparse_row: How the sparse index producer packs each row, which
            bounds each decode row by its valid length (defaults to plain
            top-k).
        AttnSinkPtrType: `OptionalPointer` type of `attn_sink_ptr`
            (inferred, defaults to `NullPointer`).
        TopkLengthsPtrType: `OptionalPointer` type of `topk_lengths` and
            `extra_topk_lengths` (inferred, defaults to `NullPointer`).
        nvfp4_convert_ratio: Most context rows per query token at which
            sparse prefill over an NVFP4 cache converts the context to FP8
            once (defaults to 0, never); see `nvfp4_prefill_converts_once`.
        nvfp4_convert_max_rows: Most rows that conversion may allocate
            (defaults to 0).

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
        max_seq_len: Maximum sequence length in the batch, used to select
            prefill versus decode.
        w_k: Weight matrix for up-projecting the latent cache to full K. Shape:
            [num_heads * qk_nope_head_dim, kv_latent_dim].
        w_uk: Weight matrix for projecting the non-rope part of each query head
            to KV latent space. Shape: [num_heads, kv_latent_dim,
            qk_nope_head_dim].
        w_uv: Weight matrix for projecting the output of the attention back to
            each head's original space. Shape: [num_heads, v_head_dim,
            kv_latent_dim].
        scalar_args_buf: Packed MLA dispatch metadata buffer.
        kv_input_fn: Closure that loads the KV latent values. Shape:
            [tot_seq_len, cache_head_dim]. Where cache_head_dim = kv_lora_rank
            + qk_rope_head_dim. It runs on the device, so it must copy what
            it captures.
        ctx: Device context.
        d_indices: Optional device pointer to packed int32 physical KV row
            indices for sparse decode (defaults to None).
        indices_stride: Stride between batch rows in `d_indices` (defaults
            to 0).
        topk_lengths: Optional per-batch valid top-k counts (defaults to
            None).
        attn_sink_ptr: Optional per-batch attention sink weights (defaults
            to None).
        num_partitions_in: Capturable-graph num_partitions override (defaults
            to None).
        prefill_d_indices: Paged physical rows the sparse prefill branch
            gathers in place of `d_indices` (defaults to None, meaning
            `d_indices`). Required without `physical_indices`, since only
            the decode kernels decode token positions.
        selection_lag: Positions between the token each sparse row was
            computed for and the token that reads it; read by the decode
            branch only.
    """
    var seq_len = q.dim(0)

    if seq_len == 0:
        return

    # The fold runs in the cache dtype (the decode branch quantizes Q to
    # `collection_t.CacheType.dtype`), so the decode-vs-prefill threshold keys on
    # the cache dtype, not the BF16 compute dtype: an FP8 cache with
    # num_heads<=AMD_MLA_DECODE_FOLD_MAX_NUM_HEADS routes S>1 to the decode fold
    # (prefill can't serve MTP); a BF16 cache or num_heads>16 returns 1 and routes
    # S>1 to prefill (no large-head decode fold exists), as does AMD sparse at
    # any head count -- its decode arm gathers one slot per row of `q` and is
    # correct only at S=1. Mirrors `mla_prefill_decode_graph_fp8`.
    if mla_graph_takes_decode_branch[
        collection_t.CacheType.dtype, q.static_shape[1], sparse_mla
    ](max_seq_len):
        mla_decode_branch_bf16[
            mask_str=mask_str,
            target=target,
            sparse_mla=sparse_mla,
            physical_indices=physical_indices,
            sparse_row=sparse_row,
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
            scalar_args_buf,
            kv_input_fn,
            ctx,
            d_indices,
            indices_stride,
            topk_lengths,
            attn_sink_ptr,
            num_partitions_in,
            selection_lag=selection_lag,
        )
    else:
        comptime if sparse_mla:
            # Sparse MLA prefill is implemented only for SM100; a non-SM100 build
            # that reaches this branch would instantiate tcgen05 kernels and fail
            # with an unhelpful message. Fail fast and name the real reason.
            comptime assert _is_sm10x_gpu(ctx.default_device_info), (
                "Sparse MLA prefill is implemented only for SM100 (B200). This"
                " architecture would otherwise attempt to compile SM100-only"
                " tcgen05 kernels."
            )
            # The sparse prefill kernels read a per-token top-k length and
            # branch on the sink at runtime, so hand them that form. Absent
            # lengths mean every token keeps the full top-k stride.
            var full_topk: Optional[DeviceBuffer[DType.int32]] = None
            var prefill_topk_lengths: MutPointer[Int32, MutAnyOrigin]
            comptime if TopkLengthsPtrType.is_null:
                full_topk = ctx.enqueue_create_buffer[DType.int32](Int(seq_len))
                full_topk.value().enqueue_fill(
                    Int32(
                        _mla_sparse_row_topk[collection_t.CacheType](
                            sparse_indices_stride
                        )
                    )
                )
                prefill_topk_lengths = (
                    full_topk.value()
                    .unsafe_ptr()
                    .unsafe_origin_cast[MutAnyOrigin]()
                )
            else:
                prefill_topk_lengths = rebind[MutPointer[Int32, MutAnyOrigin]](
                    as_optional_reg(topk_lengths).value()
                )
            var prefill_attn_sink = rebind[
                OptionalReg[MutPointer[Float32, MutAnyOrigin]]
            ](as_optional_reg(attn_sink_ptr))
            # Sparse MLA prefill for BOTH bf16 and fp8 latent caches: the
            # branch comptime-dispatches the attention kernel on the cache
            # dtype (fp8 cache => mla_sm100_prefill_sparse_fp8 read at unit
            # scale, mirroring sparse decode; bf16 cache =>
            # mla_sm100_prefill_sparse). Replaces the dense FP8-cache fallback
            # that previously ran here.
            mla_prefill_branch_sparse_bf16[
                indices_stride=sparse_indices_stride,
                target=target,
                nvfp4_convert_ratio=nvfp4_convert_ratio,
                nvfp4_convert_max_rows=nvfp4_convert_max_rows,
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
                kv_input_fn,
                ctx,
                _sparse_prefill_rows[physical_indices](
                    d_indices, prefill_d_indices
                ),
                prefill_topk_lengths,
                prefill_attn_sink,
            )
            _ = full_topk^
        else:
            # Dense prefill for NON-sparse MLA only. Sparse MLA (both bf16 and
            # fp8 caches) is handled by the sparse branch above.
            mla_prefill_branch_bf16[
                mask_str=mask_str,
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
                kv_input_fn,
                ctx,
            )
