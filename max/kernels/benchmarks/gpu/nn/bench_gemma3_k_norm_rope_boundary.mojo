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

from std.math import ceildiv, rsqrt
from std.random import seed
from std.sys import get_defined_dtype, get_defined_int
from std.sys.info import align_of, simd_width_of

from std.benchmark import Bench, BenchConfig, Bencher, BenchId
from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_idx_uint as block_idx,
    syncwarp,
    thread_idx_uint as thread_idx,
)
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from std.gpu.primitives import warp
from std.gpu.primitives.grid_controls import PDLLevel, pdl_launch_attributes
from std.memory import stack_allocation
from std.testing import assert_almost_equal

from internal_utils import arg_parse
from kv_cache.types import KVCacheStaticParams, KVCacheT, PagedKVCacheCollection
from layout import (
    Coord,
    Idx,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TensorLayout,
    TileTensor,
    UNKNOWN_VALUE,
    row_major,
)
from layout._fillers import random
from nn._ragged_utils import get_batch_from_row_offsets
from nn.fused_qk_rope import _rope_complex_mul_half
from nn.kv_cache import rms_norm_kv_cache_ragged_paged
from nn.normalization import _rms_norm_warp_tiling_subkernel
from nn.rope import _rope_k_cache_ragged, k_rms_norm_rope_ragged
from std.utils.numerics import get_accum_type
from std.utils.static_tuple import StaticTuple

from std.utils import IndexList


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(block_size))
)
def _gemma3_k_norm_rope_ragged_kernel[
    dtype: DType,
    freq_dtype: DType,
    KCacheType: KVCacheT,
    OffsetsLayoutType: TensorLayout,
    FreqLayoutType: TensorLayout,
    GammaLayoutType: TensorLayout,
    block_size: Int,
    warps_per_block: Int,
](
    input_row_offsets: TileTensor[
        DType.uint32, OffsetsLayoutType, MutAnyOrigin
    ],
    k_cache: KCacheType,
    freqs_cis: TileTensor[freq_dtype, FreqLayoutType, MutAnyOrigin],
    gamma: TileTensor[dtype, GammaLayoutType, MutAnyOrigin],
    epsilon: Scalar[dtype],
    weight_offset: Scalar[dtype],
    total_rows: Int,
):
    comptime assert input_row_offsets.flat_rank == 1
    comptime assert freqs_cis.flat_rank == 2
    comptime assert gamma.flat_rank == 1

    comptime num_heads = Int(KCacheType.kv_params.num_heads)
    comptime head_dim = Int(KCacheType.kv_params.head_size)
    comptime simd_width = simd_width_of[dtype]()
    comptime wide_align = align_of[SIMD[dtype, simd_width]]()
    comptime accum_type = get_accum_type[dtype]()
    comptime half_warp_size = WARP_SIZE // 2
    comptime assert head_dim == 128, "Only 128-column BF16 key rows are supported"
    comptime assert gamma.static_shape[0] == head_dim
    comptime assert freqs_cis.static_shape[1] == head_dim
    comptime assert head_dim == half_warp_size * simd_width

    var tid = thread_idx.x
    var warp_idx = tid // UInt(WARP_SIZE)
    var sub_warp_idx = (tid % UInt(WARP_SIZE)) // UInt(half_warp_size)
    var local_tid = tid % UInt(half_warp_size)
    var row = block_idx.x * UInt(warps_per_block * 2) + warp_idx * 2 + sub_warp_idx
    var col = local_tid * UInt(simd_width)

    if row < UInt(total_rows):
        var flat_row = Int(row)
        var global_token_idx = flat_row // num_heads
        var head_idx = flat_row % num_heads
        var batch_idx = get_batch_from_row_offsets(
            input_row_offsets, global_token_idx
        )
        var token_idx = Int(
            UInt32(global_token_idx) - input_row_offsets[batch_idx]
        )
        var cache_token_idx = token_idx + k_cache.cache_length(batch_idx)

        var epsilon_accum = epsilon.cast[accum_type]()
        var weight_offset_accum = weight_offset.cast[accum_type]()
        var vec_data = k_cache.load[width=simd_width](
            batch_idx, head_idx, cache_token_idx, Int(col)
        ).cast[accum_type]()
        var gamma_val = gamma.load[width=simd_width, alignment=wide_align](
            Coord(Idx(Int(col)))
        )

        var norm_val = _rms_norm_warp_tiling_subkernel[
            warps_per_block,
            True,
            rows_per_warp=2,
        ](
            flat_row,
            Int(col),
            vec_data,
            gamma_val,
            epsilon_accum,
            weight_offset_accum,
            head_dim,
        )

        var sub_warp_mask = (
            (UInt(1) << UInt(half_warp_size)) - UInt(1)
        ) << (sub_warp_idx * UInt(half_warp_size))
        var partner_lane = (
            sub_warp_idx * UInt(half_warp_size)
            + (local_tid % UInt(half_warp_size // 2))
            + UInt(half_warp_size // 2)
        )
        var norm_parts = norm_val.split()
        var norm_lo_parts = norm_parts[0].split()
        var norm_hi_parts = norm_parts[1].split()
        var partner_norm_lo = warp.shuffle_idx(
            sub_warp_mask,
            norm_lo_parts[0],
            UInt32(partner_lane),
        ).join(
            warp.shuffle_idx(
                sub_warp_mask,
                norm_lo_parts[1],
                UInt32(partner_lane),
            )
        )
        var partner_norm_hi = warp.shuffle_idx(
            sub_warp_mask,
            norm_hi_parts[0],
            UInt32(partner_lane),
        ).join(
            warp.shuffle_idx(
                sub_warp_mask,
                norm_hi_parts[1],
                UInt32(partner_lane),
            )
        )
        var partner_norm = rebind[type_of(norm_val)](
            partner_norm_lo.join(partner_norm_hi)
        )

        if local_tid < UInt(half_warp_size // 2):
            var re_offset = Int(local_tid) * simd_width
            var im_offset = re_offset + head_dim // 2
            var freq = freqs_cis.load[width=simd_width * 2, alignment=1](
                Coord(Idx(cache_token_idx), Idx(re_offset * 2))
            )
            comptime cache_dtype = KCacheType.dtype
            var rope_val = _rope_complex_mul_half[
                accum_type,
                freq_dtype,
                simd_width,
                simd_width * 2,
            ](
                norm_val.cast[accum_type](),
                partner_norm.cast[accum_type](),
                freq,
            )
            k_cache.store(
                batch_idx,
                head_idx,
                cache_token_idx,
                re_offset,
                rope_val[0].cast[cache_dtype](),
            )
            k_cache.store(
                batch_idx,
                head_idx,
                cache_token_idx,
                im_offset,
                rope_val[1].cast[cache_dtype](),
            )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(block_size))
)
def _gemma3_k_norm_rope_decode_kernel[
    dtype: DType,
    freq_dtype: DType,
    KCacheType: KVCacheT,
    FreqLayoutType: TensorLayout,
    GammaLayoutType: TensorLayout,
    block_size: Int,
    warps_per_block: Int,
](
    k_cache: KCacheType,
    freqs_cis: TileTensor[freq_dtype, FreqLayoutType, MutAnyOrigin],
    gamma: TileTensor[dtype, GammaLayoutType, MutAnyOrigin],
    epsilon: Scalar[dtype],
    weight_offset: Scalar[dtype],
    total_rows: Int,
):
    comptime assert freqs_cis.flat_rank == 2
    comptime assert gamma.flat_rank == 1

    comptime num_heads = Int(KCacheType.kv_params.num_heads)
    comptime head_dim = Int(KCacheType.kv_params.head_size)
    comptime simd_width = simd_width_of[dtype]()
    comptime wide_align = align_of[SIMD[dtype, simd_width]]()
    comptime accum_type = get_accum_type[dtype]()
    comptime half_warp_size = WARP_SIZE // 2
    comptime assert head_dim == 128, "Only 128-column BF16 key rows are supported"
    comptime assert gamma.static_shape[0] == head_dim
    comptime assert freqs_cis.static_shape[1] == head_dim
    comptime assert head_dim == half_warp_size * simd_width

    var tid = thread_idx.x
    var warp_idx = tid // UInt(WARP_SIZE)
    var sub_warp_idx = (tid % UInt(WARP_SIZE)) // UInt(half_warp_size)
    var local_tid = tid % UInt(half_warp_size)
    var row = block_idx.x * UInt(warps_per_block * 2) + warp_idx * 2 + sub_warp_idx
    var col = local_tid * UInt(simd_width)

    if row < UInt(total_rows):
        var flat_row = Int(row)
        var global_token_idx = flat_row // num_heads
        var head_idx = flat_row % num_heads
        var batch_idx = global_token_idx
        var cache_token_idx = k_cache.cache_length(batch_idx)

        var epsilon_accum = epsilon.cast[accum_type]()
        var weight_offset_accum = weight_offset.cast[accum_type]()
        var vec_data = k_cache.load[width=simd_width](
            batch_idx, head_idx, cache_token_idx, Int(col)
        ).cast[accum_type]()
        var gamma_val = gamma.load[width=simd_width, alignment=wide_align](
            Coord(Idx(Int(col)))
        )

        var norm_val = _rms_norm_warp_tiling_subkernel[
            warps_per_block,
            True,
            rows_per_warp=2,
        ](
            flat_row,
            Int(col),
            vec_data,
            gamma_val,
            epsilon_accum,
            weight_offset_accum,
            head_dim,
        )

        var sub_warp_mask = (
            (UInt(1) << UInt(half_warp_size)) - UInt(1)
        ) << (sub_warp_idx * UInt(half_warp_size))
        var partner_lane = (
            sub_warp_idx * UInt(half_warp_size)
            + (local_tid % UInt(half_warp_size // 2))
            + UInt(half_warp_size // 2)
        )
        var norm_parts = norm_val.split()
        var norm_lo_parts = norm_parts[0].split()
        var norm_hi_parts = norm_parts[1].split()
        var partner_norm_lo = warp.shuffle_idx(
            sub_warp_mask,
            norm_lo_parts[0],
            UInt32(partner_lane),
        ).join(
            warp.shuffle_idx(
                sub_warp_mask,
                norm_lo_parts[1],
                UInt32(partner_lane),
            )
        )
        var partner_norm_hi = warp.shuffle_idx(
            sub_warp_mask,
            norm_hi_parts[0],
            UInt32(partner_lane),
        ).join(
            warp.shuffle_idx(
                sub_warp_mask,
                norm_hi_parts[1],
                UInt32(partner_lane),
            )
        )
        var partner_norm = rebind[type_of(norm_val)](
            partner_norm_lo.join(partner_norm_hi)
        )

        if local_tid < UInt(half_warp_size // 2):
            var re_offset = Int(local_tid) * simd_width
            var im_offset = re_offset + head_dim // 2
            var freq = freqs_cis.load[width=simd_width * 2, alignment=1](
                Coord(Idx(cache_token_idx), Idx(re_offset * 2))
            )
            comptime cache_dtype = KCacheType.dtype
            var rope_val = _rope_complex_mul_half[
                accum_type,
                freq_dtype,
                simd_width,
                simd_width * 2,
            ](
                norm_val.cast[accum_type](),
                partner_norm.cast[accum_type](),
                freq,
            )
            k_cache.store(
                batch_idx,
                head_idx,
                cache_token_idx,
                re_offset,
                rope_val[0].cast[cache_dtype](),
            )
            k_cache.store(
                batch_idx,
                head_idx,
                cache_token_idx,
                im_offset,
                rope_val[1].cast[cache_dtype](),
            )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(block_size))
)
def _gemma3_k_norm_rope_ragged_wide_kernel[
    dtype: DType,
    freq_dtype: DType,
    KCacheType: KVCacheT,
    OffsetsLayoutType: TensorLayout,
    FreqLayoutType: TensorLayout,
    GammaLayoutType: TensorLayout,
    block_size: Int,
    warps_per_block: Int,
](
    input_row_offsets: TileTensor[
        DType.uint32, OffsetsLayoutType, MutAnyOrigin
    ],
    k_cache: KCacheType,
    freqs_cis: TileTensor[freq_dtype, FreqLayoutType, MutAnyOrigin],
    gamma: TileTensor[dtype, GammaLayoutType, MutAnyOrigin],
    epsilon: Scalar[dtype],
    weight_offset: Scalar[dtype],
    total_rows: Int,
):
    comptime assert input_row_offsets.flat_rank == 1
    comptime assert freqs_cis.flat_rank == 2
    comptime assert gamma.flat_rank == 1

    comptime num_heads = Int(KCacheType.kv_params.num_heads)
    comptime head_dim = Int(KCacheType.kv_params.head_size)
    comptime simd_width = simd_width_of[dtype]()
    comptime wide_align = align_of[SIMD[dtype, simd_width]]()
    comptime accum_type = get_accum_type[dtype]()
    comptime assert head_dim == 256, "Only 256-column BF16 key rows are supported"
    comptime assert gamma.static_shape[0] == head_dim
    comptime assert freqs_cis.static_shape[1] == head_dim
    comptime assert head_dim == WARP_SIZE * simd_width

    var shared_norm = stack_allocation[
        warps_per_block * head_dim,
        Scalar[dtype],
        address_space=AddressSpace.SHARED,
    ]()
    var shared_batch_idx = stack_allocation[
        warps_per_block,
        Int32,
        address_space=AddressSpace.SHARED,
    ]()
    var shared_post_seq = stack_allocation[
        warps_per_block,
        Int32,
        address_space=AddressSpace.SHARED,
    ]()

    var tid = thread_idx.x
    var warp_idx = tid // UInt(WARP_SIZE)
    var local_tid = tid % UInt(WARP_SIZE)
    var row = block_idx.x * UInt(warps_per_block) + warp_idx
    var col = local_tid * UInt(simd_width)

    if row < UInt(total_rows):
        var flat_row = Int(row)
        var global_token_idx = flat_row // num_heads
        var head_idx = flat_row % num_heads

        if local_tid == 0:
            var batch_idx = get_batch_from_row_offsets(
                input_row_offsets, global_token_idx
            )
            var token_idx = Int(
                UInt32(global_token_idx) - input_row_offsets[batch_idx]
            )
            shared_batch_idx[Int(warp_idx)] = Int32(batch_idx)
            shared_post_seq[Int(warp_idx)] = Int32(
                token_idx + Int(k_cache.cache_length(batch_idx))
            )
        syncwarp()

        var batch_idx = Int(shared_batch_idx[Int(warp_idx)])
        var cache_token_idx = Int(shared_post_seq[Int(warp_idx)])
        var epsilon_accum = epsilon.cast[accum_type]()
        var weight_offset_accum = weight_offset.cast[accum_type]()
        var vec_data = k_cache.load[width=simd_width](
            batch_idx, head_idx, cache_token_idx, Int(col)
        ).cast[accum_type]()
        var gamma_val = gamma.load[width=simd_width, alignment=wide_align](
            Coord(Idx(Int(col)))
        )

        var thread_m2 = (vec_data**2).reduce_add()
        var row_m2 = warp.sum(thread_m2)
        var norm_factor = rsqrt(
            (row_m2 / Scalar[accum_type](head_dim)) + epsilon_accum
        )
        var norm_val = vec_data * norm_factor * (
            gamma_val.cast[accum_type]() + weight_offset_accum
        )
        var shared_base = Int(warp_idx) * head_dim
        shared_norm.store[width=simd_width, alignment=wide_align](
            shared_base + Int(col), norm_val.cast[dtype]()
        )
        syncwarp()

        if local_tid < UInt(WARP_SIZE // 2):
            var re_offset = Int(local_tid) * simd_width
            var im_offset = re_offset + head_dim // 2
            var rope_re = shared_norm.load[width=simd_width](
                shared_base + re_offset
            ).cast[accum_type]()
            var rope_im = shared_norm.load[width=simd_width](
                shared_base + im_offset
            ).cast[accum_type]()
            var freq = freqs_cis.load[width=simd_width * 2, alignment=1](
                Coord(Idx(cache_token_idx), Idx(re_offset * 2))
            )
            comptime cache_dtype = KCacheType.dtype
            var rope_val = _rope_complex_mul_half[
                accum_type,
                freq_dtype,
                simd_width,
                simd_width * 2,
            ](rope_re, rope_im, freq)
            k_cache.store(
                batch_idx,
                head_idx,
                cache_token_idx,
                re_offset,
                rope_val[0].cast[cache_dtype](),
            )
            k_cache.store(
                batch_idx,
                head_idx,
                cache_token_idx,
                im_offset,
                rope_val[1].cast[cache_dtype](),
            )


def _bench_trial_k_norm_rope_ragged_paged[
    dtype: DType,
    freq_dtype: DType,
    params: KVCacheStaticParams,
    page_size: Int,
    cache_dtype: DType,
](
    kv_collection: PagedKVCacheCollection[cache_dtype, params, page_size],
    gamma: TileTensor[dtype, ...],
    epsilon: Scalar[dtype],
    weight_offset: Scalar[dtype],
    layer_idx: UInt32,
    total_seq_len: UInt32,
    input_row_offsets: TileTensor[DType.uint32, ...],
    freqs_cis: TileTensor[freq_dtype, ...],
    ctx: DeviceContext,
) raises:
    comptime head_dim = Int(params.head_size)
    comptime assert dtype == DType.bfloat16
    comptime assert cache_dtype == DType.bfloat16
    comptime assert head_dim == 128 or head_dim == 256

    var total_rows = Int(total_seq_len) * Int(params.num_heads)
    comptime default_warps_per_block = 2
    comptime default_block_size = default_warps_per_block * WARP_SIZE
    comptime large_row_warps_per_block = 8
    comptime large_row_block_size = large_row_warps_per_block * WARP_SIZE
    comptime min_large_row_blocks_per_sm = 6
    var large_row_grid_dim = ceildiv(total_rows, large_row_warps_per_block)
    var min_large_row_blocks = (
        ctx.default_device_info.sm_count * min_large_row_blocks_per_sm
    )
    var k_cache = kv_collection.get_key_cache(Int(layer_idx))
    comptime if head_dim == 128:
        var batch_size = Int(input_row_offsets.dim(0)) - 1
        var is_decode_uniform = Int(total_seq_len) == batch_size
        var large_row_grid_dim_128 = ceildiv(
            total_rows, large_row_warps_per_block * 2
        )

        if is_decode_uniform:
            if large_row_grid_dim_128 >= min_large_row_blocks:
                comptime kernel = _gemma3_k_norm_rope_decode_kernel[
                    dtype,
                    freq_dtype,
                    PagedKVCacheCollection[
                        cache_dtype, params, page_size
                    ].CacheType,
                    freqs_cis.LayoutType,
                    gamma.LayoutType,
                    large_row_block_size,
                    large_row_warps_per_block,
                ]
                ctx.enqueue_function[kernel, kernel](
                    k_cache,
                    freqs_cis,
                    gamma,
                    epsilon,
                    weight_offset,
                    total_rows,
                    grid_dim=large_row_grid_dim_128,
                    block_dim=large_row_block_size,
                    attributes=pdl_launch_attributes(PDLLevel(1)),
                )
            else:
                comptime kernel = _gemma3_k_norm_rope_decode_kernel[
                    dtype,
                    freq_dtype,
                    PagedKVCacheCollection[
                        cache_dtype, params, page_size
                    ].CacheType,
                    freqs_cis.LayoutType,
                    gamma.LayoutType,
                    default_block_size,
                    default_warps_per_block,
                ]
                ctx.enqueue_function[kernel, kernel](
                    k_cache,
                    freqs_cis,
                    gamma,
                    epsilon,
                    weight_offset,
                    total_rows,
                    grid_dim=ceildiv(total_rows, default_warps_per_block * 2),
                    block_dim=default_block_size,
                    attributes=pdl_launch_attributes(PDLLevel(1)),
                )
        else:
            if large_row_grid_dim_128 >= min_large_row_blocks:
                comptime kernel = _gemma3_k_norm_rope_ragged_kernel[
                    dtype,
                    freq_dtype,
                    PagedKVCacheCollection[
                        cache_dtype, params, page_size
                    ].CacheType,
                    input_row_offsets.LayoutType,
                    freqs_cis.LayoutType,
                    gamma.LayoutType,
                    large_row_block_size,
                    large_row_warps_per_block,
                ]
                ctx.enqueue_function[kernel, kernel](
                    input_row_offsets,
                    k_cache,
                    freqs_cis,
                    gamma,
                    epsilon,
                    weight_offset,
                    total_rows,
                    grid_dim=large_row_grid_dim_128,
                    block_dim=large_row_block_size,
                    attributes=pdl_launch_attributes(PDLLevel(1)),
                )
            else:
                comptime kernel = _gemma3_k_norm_rope_ragged_kernel[
                    dtype,
                    freq_dtype,
                    PagedKVCacheCollection[
                        cache_dtype, params, page_size
                    ].CacheType,
                    input_row_offsets.LayoutType,
                    freqs_cis.LayoutType,
                    gamma.LayoutType,
                    default_block_size,
                    default_warps_per_block,
                ]
                ctx.enqueue_function[kernel, kernel](
                    input_row_offsets,
                    k_cache,
                    freqs_cis,
                    gamma,
                    epsilon,
                    weight_offset,
                    total_rows,
                    grid_dim=ceildiv(total_rows, default_warps_per_block * 2),
                    block_dim=default_block_size,
                    attributes=pdl_launch_attributes(PDLLevel(1)),
                )
    else:
        if large_row_grid_dim >= min_large_row_blocks:
            comptime kernel = _gemma3_k_norm_rope_ragged_wide_kernel[
                dtype,
                freq_dtype,
                PagedKVCacheCollection[
                    cache_dtype, params, page_size
                ].CacheType,
                input_row_offsets.LayoutType,
                freqs_cis.LayoutType,
                gamma.LayoutType,
                large_row_block_size,
                large_row_warps_per_block,
            ]
            ctx.enqueue_function[kernel, kernel](
                input_row_offsets,
                k_cache,
                freqs_cis,
                gamma,
                epsilon,
                weight_offset,
                total_rows,
                grid_dim=large_row_grid_dim,
                block_dim=large_row_block_size,
                attributes=pdl_launch_attributes(PDLLevel(1)),
            )
        else:
            comptime kernel = _gemma3_k_norm_rope_ragged_wide_kernel[
                dtype,
                freq_dtype,
                PagedKVCacheCollection[
                    cache_dtype, params, page_size
                ].CacheType,
                input_row_offsets.LayoutType,
                freqs_cis.LayoutType,
                gamma.LayoutType,
                default_block_size,
                default_warps_per_block,
            ]
            ctx.enqueue_function[kernel, kernel](
                input_row_offsets,
                k_cache,
                freqs_cis,
                gamma,
                epsilon,
                weight_offset,
                total_rows,
                grid_dim=ceildiv(total_rows, default_warps_per_block),
                block_dim=default_block_size,
                attributes=pdl_launch_attributes(PDLLevel(1)),
            )


def bench_gemma3_k_norm_rope_boundary[
    dtype: DType,
    head_dim: Int,
    num_kv_heads: Int,
    page_size: Int,
](
    ctx: DeviceContext,
    mut bench: Bench,
    batch_size: Int,
    seq_len: Int,
    cache_len: Int,
    cache_len_step: Int,
) raises:
    comptime layer_idx = 0
    comptime num_layers = 1
    comptime kv_params = KVCacheStaticParams(
        num_heads=UInt(num_kv_heads), head_size=UInt(head_dim)
    )
    comptime CollectionType = PagedKVCacheCollection[dtype, kv_params, page_size]
    comptime kv_block_layout = Layout.row_major[6]()
    comptime cache_lengths_layout = Layout(UNKNOWN_VALUE)

    var total_seq_len = UInt32(batch_size * seq_len)
    var max_cache_len = cache_len + (batch_size - 1) * cache_len_step
    var max_context_length = seq_len + max_cache_len
    var paged_lut_cols = ceildiv(max_context_length, page_size)
    var num_pages = batch_size * paged_lut_cols

    var input_row_offsets_h = alloc[Scalar[DType.uint32]](batch_size + 1)
    var cache_lengths_h = alloc[Scalar[DType.uint32]](batch_size)
    var gamma_h = alloc[Scalar[dtype]](head_dim)
    var freqs_h = alloc[Scalar[dtype]](max_context_length * head_dim)
    var paged_lut_h = alloc[Scalar[DType.uint32]](batch_size * paged_lut_cols)

    for i in range(batch_size):
        input_row_offsets_h[i] = UInt32(i * seq_len)
        cache_lengths_h[i] = UInt32(cache_len + i * cache_len_step)
    input_row_offsets_h[batch_size] = total_seq_len

    for i in range(head_dim):
        gamma_h[i] = (Float64(i + head_dim) / Float64(head_dim)).cast[dtype]()

    random(
        LayoutTensor[dtype, Layout.row_major[2](), MutAnyOrigin](
            freqs_h,
            RuntimeLayout[Layout.row_major[2]()].row_major(
                IndexList[2](max_context_length, head_dim)
            ),
        )
    )

    var paged_lut_host = LayoutTensor[
        DType.uint32, Layout.row_major[2](), MutAnyOrigin
    ](
        paged_lut_h,
        RuntimeLayout[Layout.row_major[2]()].row_major(
            IndexList[2](batch_size, paged_lut_cols)
        ),
    )
    for bs in range(batch_size):
        for page_idx in range(paged_lut_cols):
            paged_lut_host[bs, page_idx] = UInt32(bs * paged_lut_cols + page_idx)

    var input_row_offsets_d = ctx.enqueue_create_buffer[DType.uint32](
        batch_size + 1
    )
    var cache_lengths_d = ctx.enqueue_create_buffer[DType.uint32](batch_size)
    var gamma_d = ctx.enqueue_create_buffer[dtype](head_dim)
    var freqs_d = ctx.enqueue_create_buffer[dtype](max_context_length * head_dim)
    var paged_lut_d = ctx.enqueue_create_buffer[DType.uint32](
        batch_size * paged_lut_cols
    )

    ctx.enqueue_copy(input_row_offsets_d, input_row_offsets_h)
    ctx.enqueue_copy(cache_lengths_d, cache_lengths_h)
    ctx.enqueue_copy(gamma_d, gamma_h)
    ctx.enqueue_copy(freqs_d, freqs_h)
    ctx.enqueue_copy(paged_lut_d, paged_lut_h)

    var kv_block_shape = IndexList[6](
        num_pages,
        2,
        num_layers,
        page_size,
        num_kv_heads,
        head_dim,
    )
    var kv_block_size = kv_block_shape.flattened_length()
    var initial_kv_block_h = alloc[Scalar[dtype]](kv_block_size)
    var baseline_kv_block_h = alloc[Scalar[dtype]](kv_block_size)
    var trial_kv_block_h = alloc[Scalar[dtype]](kv_block_size)
    random(
        LayoutTensor[dtype, Layout.row_major[6](), MutAnyOrigin](
            initial_kv_block_h,
            RuntimeLayout[Layout.row_major[6]()].row_major(kv_block_shape),
        )
    )

    var baseline_kv_block_d = ctx.enqueue_create_buffer[dtype](kv_block_size)
    var trial_kv_block_d = ctx.enqueue_create_buffer[dtype](kv_block_size)
    ctx.enqueue_copy(baseline_kv_block_d, initial_kv_block_h)
    ctx.enqueue_copy(trial_kv_block_d, initial_kv_block_h)

    var gamma_tensor = TileTensor(
        gamma_d.unsafe_ptr(), row_major(Idx[head_dim]())
    )
    var input_row_offsets_tensor = TileTensor(
        input_row_offsets_d.unsafe_ptr(), row_major(Idx(batch_size + 1))
    )
    var freqs_tensor = TileTensor(
        freqs_d.unsafe_ptr(),
        row_major(Idx(max_context_length), Idx[head_dim]()),
    )

    var baseline_kv_collection = CollectionType(
        LayoutTensor[dtype, kv_block_layout, MutAnyOrigin](
            baseline_kv_block_d.unsafe_ptr(),
            RuntimeLayout[kv_block_layout].row_major(kv_block_shape),
        ),
        LayoutTensor[DType.uint32, cache_lengths_layout, ImmutAnyOrigin](
            cache_lengths_d.unsafe_ptr(),
            RuntimeLayout[cache_lengths_layout].row_major(
                IndexList[1](batch_size)
            ),
        ),
        LayoutTensor[DType.uint32, Layout.row_major[2](), ImmutAnyOrigin](
            paged_lut_d.unsafe_ptr(),
            RuntimeLayout[Layout.row_major[2]()].row_major(
                IndexList[2](batch_size, paged_lut_cols)
            ),
        ),
        UInt32(seq_len),
        UInt32(max_context_length),
    )
    var trial_kv_collection = CollectionType(
        LayoutTensor[dtype, kv_block_layout, MutAnyOrigin](
            trial_kv_block_d.unsafe_ptr(),
            RuntimeLayout[kv_block_layout].row_major(kv_block_shape),
        ),
        LayoutTensor[DType.uint32, cache_lengths_layout, ImmutAnyOrigin](
            cache_lengths_d.unsafe_ptr(),
            RuntimeLayout[cache_lengths_layout].row_major(
                IndexList[1](batch_size)
            ),
        ),
        LayoutTensor[DType.uint32, Layout.row_major[2](), ImmutAnyOrigin](
            paged_lut_d.unsafe_ptr(),
            RuntimeLayout[Layout.row_major[2]()].row_major(
                IndexList[2](batch_size, paged_lut_cols)
            ),
        ),
        UInt32(seq_len),
        UInt32(max_context_length),
    )

    var epsilon = Scalar[dtype](1e-6)
    var weight_offset = Scalar[dtype](1.0)

    @always_inline
    @__copy_capture(
        baseline_kv_collection,
        gamma_tensor,
        epsilon,
        weight_offset,
        input_row_offsets_tensor,
        total_seq_len,
        freqs_tensor,
    )
    @parameter
    def run_baseline(ctx: DeviceContext) raises:
        rms_norm_kv_cache_ragged_paged[
            target="gpu",
            multiply_before_cast=True,
            per_head_norm=True,
        ](
            baseline_kv_collection,
            gamma_tensor,
            epsilon,
            weight_offset,
            UInt32(layer_idx),
            total_seq_len,
            input_row_offsets_tensor,
            ctx,
        )
        _rope_k_cache_ragged[target="gpu", interleaved=False](
            Int(total_seq_len),
            input_row_offsets_tensor,
            baseline_kv_collection,
            freqs_tensor,
            UInt32(layer_idx),
            ctx,
        )

    @always_inline
    @__copy_capture(
        trial_kv_collection,
        gamma_tensor,
        epsilon,
        weight_offset,
        input_row_offsets_tensor,
        freqs_tensor,
        total_seq_len,
    )
    @parameter
    def run_trial(ctx: DeviceContext) raises:
        k_rms_norm_rope_ragged[
            dtype,
            dtype,
            CollectionType,
            interleaved=False,
            target="gpu",
        ](
            Int(total_seq_len),
            input_row_offsets_tensor,
            trial_kv_collection,
            freqs_tensor,
            gamma_tensor,
            epsilon,
            weight_offset,
            UInt32(layer_idx),
            ctx,
        )

    @always_inline
    @parameter
    def baseline_bench(mut bencher: Bencher) raises:
        bencher.iter_custom[run_baseline](ctx)

    @always_inline
    @parameter
    def trial_bench(mut bencher: Bencher) raises:
        bencher.iter_custom[run_trial](ctx)

    bench.bench_function[baseline_bench](
        BenchId(
            "gemma3_k_norm_rope_boundary_baseline",
            input_id=String(
                dtype,
                "/bs=",
                batch_size,
                "/seq=",
                seq_len,
                "/cache=",
                cache_len,
                "/step=",
                cache_len_step,
            ),
        ),
    )
    bench.bench_function[trial_bench](
        BenchId(
            "gemma3_k_norm_rope_boundary_trial",
            input_id=String(
                dtype,
                "/bs=",
                batch_size,
                "/seq=",
                seq_len,
                "/cache=",
                cache_len,
                "/step=",
                cache_len_step,
            ),
        ),
    )

    run_baseline(ctx)
    run_trial(ctx)
    ctx.enqueue_copy(baseline_kv_block_h, baseline_kv_block_d)
    ctx.enqueue_copy(trial_kv_block_h, trial_kv_block_d)
    ctx.synchronize()

    var baseline_kv_collection_host = CollectionType(
        LayoutTensor[dtype, kv_block_layout, MutAnyOrigin](
            baseline_kv_block_h,
            RuntimeLayout[kv_block_layout].row_major(kv_block_shape),
        ),
        LayoutTensor[DType.uint32, cache_lengths_layout, ImmutAnyOrigin](
            cache_lengths_h,
            RuntimeLayout[cache_lengths_layout].row_major(
                IndexList[1](batch_size)
            ),
        ),
        LayoutTensor[DType.uint32, Layout.row_major[2](), ImmutAnyOrigin](
            paged_lut_h,
            RuntimeLayout[Layout.row_major[2]()].row_major(
                IndexList[2](batch_size, paged_lut_cols)
            ),
        ),
        UInt32(seq_len),
        UInt32(max_context_length),
    )
    var trial_kv_collection_host = CollectionType(
        LayoutTensor[dtype, kv_block_layout, MutAnyOrigin](
            trial_kv_block_h,
            RuntimeLayout[kv_block_layout].row_major(kv_block_shape),
        ),
        LayoutTensor[DType.uint32, cache_lengths_layout, ImmutAnyOrigin](
            cache_lengths_h,
            RuntimeLayout[cache_lengths_layout].row_major(
                IndexList[1](batch_size)
            ),
        ),
        LayoutTensor[DType.uint32, Layout.row_major[2](), ImmutAnyOrigin](
            paged_lut_h,
            RuntimeLayout[Layout.row_major[2]()].row_major(
                IndexList[2](batch_size, paged_lut_cols)
            ),
        ),
        UInt32(seq_len),
        UInt32(max_context_length),
    )
    var baseline_k_cache_host = baseline_kv_collection_host.get_key_cache(layer_idx)
    var trial_k_cache_host = trial_kv_collection_host.get_key_cache(layer_idx)

    for bs_idx in range(batch_size):
        for tok_idx in range(seq_len):
            var post_seq_idx = Int(cache_lengths_h[bs_idx]) + tok_idx
            for head_idx in range(num_kv_heads):
                for dim_idx in range(head_dim):
                    assert_almost_equal(
                        baseline_k_cache_host.load[width=1](
                            bs_idx,
                            head_idx,
                            post_seq_idx,
                            dim_idx,
                        ),
                        trial_k_cache_host.load[width=1](
                            bs_idx,
                            head_idx,
                            post_seq_idx,
                            dim_idx,
                        ),
                        rtol=2e-2,
                        atol=2e-2,
                    )

    _ = input_row_offsets_d
    _ = cache_lengths_d
    _ = gamma_d
    _ = freqs_d
    _ = paged_lut_d
    _ = baseline_kv_block_d
    _ = trial_kv_block_d

    input_row_offsets_h.free()
    cache_lengths_h.free()
    gamma_h.free()
    freqs_h.free()
    paged_lut_h.free()
    initial_kv_block_h.free()
    baseline_kv_block_h.free()
    trial_kv_block_h.free()


def main() raises:
    comptime dtype = get_defined_dtype["dtype", DType.bfloat16]()
    comptime head_dim = get_defined_int["head_dim", 128]()
    comptime num_kv_heads = get_defined_int["num_kv_heads", 16]()
    comptime page_size = 128

    var batch_size = arg_parse("batch_size", 64)
    var seq_len = arg_parse("seq_len", 1)
    var cache_len = arg_parse("cache_len", 1024)
    var cache_len_step = arg_parse("cache_len_step", 0)

    seed(0)

    var bench = Bench(BenchConfig(num_repetitions=1))
    with DeviceContext() as ctx:
        bench_gemma3_k_norm_rope_boundary[
            dtype,
            head_dim,
            num_kv_heads,
            page_size,
        ](ctx, bench, batch_size, seq_len, cache_len, cache_len_step)

    bench.dump_report()
