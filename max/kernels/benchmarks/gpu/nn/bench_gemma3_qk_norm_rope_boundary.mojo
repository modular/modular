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
from std.random import random_float64, seed
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
from std.memory import stack_allocation
from std.testing import assert_almost_equal
from internal_utils import arg_parse, update_bench_config_args
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
from nn.fused_qk_rope import _rope_complex_mul_half, fused_qk_rope_ragged
from nn.kv_cache import rms_norm_kv_cache_ragged_paged
from nn.normalization import _rms_norm_warp_tiling_subkernel, rms_norm_gpu
from nn.rope import q_rms_norm_fused_qk_rope_ragged
from std.utils.numerics import get_accum_type
from std.utils.static_tuple import StaticTuple

from std.utils import IndexList


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(block_size))
)
def _gemma3_qk_norm_rope_single_launch_kernel[
    dtype: DType,
    freq_dtype: DType,
    KCacheType: KVCacheT,
    QLayoutType: TensorLayout,
    OutputLayoutType: TensorLayout,
    OffsetsLayoutType: TensorLayout,
    FreqLayoutType: TensorLayout,
    GammaLayoutType: TensorLayout,
    block_size: Int,
    warps_per_block: Int,
    num_q_heads: Int,
    num_kv_heads: Int,
    head_dim: Int,
](
    q: TileTensor[dtype, QLayoutType, MutAnyOrigin],
    output: TileTensor[mut=True, dtype, OutputLayoutType, MutAnyOrigin],
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
    comptime assert (
        head_dim == 128
    ), "Prototype currently targets 128-wide heads"
    comptime assert q.flat_rank == 3
    comptime assert output.flat_rank == 3
    comptime assert input_row_offsets.flat_rank == 1
    comptime assert freqs_cis.flat_rank == 2
    comptime assert gamma.flat_rank == 1

    comptime simd_width = simd_width_of[dtype]()
    comptime align = align_of[SIMD[dtype, simd_width]]()
    comptime accum_type = get_accum_type[dtype]()
    comptime half_warp_size = WARP_SIZE // 2
    comptime total_heads = num_q_heads + num_kv_heads

    var shared_norm = stack_allocation[
        warps_per_block * 2 * head_dim,
        Scalar[dtype],
        address_space=AddressSpace.SHARED,
    ]()

    var tid = thread_idx.x
    var warp_idx = tid // UInt(WARP_SIZE)
    var sub_warp_idx = (tid % UInt(WARP_SIZE)) // UInt(half_warp_size)
    var local_tid = tid % UInt(half_warp_size)
    var row = (
        block_idx.x * UInt(warps_per_block * 2) + warp_idx * 2 + sub_warp_idx
    )
    var col = local_tid * UInt(simd_width)

    var is_active_row = row < UInt(total_rows)
    var flat_row = Int(row)
    var global_token_idx = 0
    var head_slot = 0
    var batch_idx = 0
    var token_idx = 0
    var post_seq_idx = 0

    if is_active_row:
        global_token_idx = flat_row // total_heads
        head_slot = flat_row % total_heads
        batch_idx = get_batch_from_row_offsets(
            input_row_offsets, global_token_idx
        )
        token_idx = Int(UInt32(global_token_idx) - input_row_offsets[batch_idx])
        post_seq_idx = k_cache.cache_length(batch_idx) + token_idx

    var is_q_row = is_active_row and head_slot < num_q_heads
    var k_head_idx = 0
    if is_active_row and not is_q_row:
        k_head_idx = head_slot - num_q_heads
    var vec_data = SIMD[accum_type, simd_width](0)
    var gamma_val = SIMD[dtype, simd_width](0)

    if is_active_row and col < UInt(head_dim):
        if is_q_row:
            vec_data = q.load[width=simd_width](
                Coord(Idx(global_token_idx), Idx(head_slot), Idx(Int(col)))
            ).cast[accum_type]()
        else:
            vec_data = k_cache.load[width=simd_width](
                batch_idx, k_head_idx, post_seq_idx, Int(col)
            ).cast[accum_type]()
        gamma_val = gamma.load[width=simd_width, alignment=align](
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
        epsilon.cast[accum_type](),
        weight_offset.cast[accum_type](),
        head_dim,
    )

    if is_active_row and col < UInt(head_dim):
        var shared_offset = (
            Int(warp_idx) * 2 + Int(sub_warp_idx)
        ) * head_dim + Int(col)
        shared_norm.store[width=simd_width, alignment=align](
            shared_offset, norm_val
        )

    syncwarp()

    if not is_active_row or local_tid >= UInt(half_warp_size // 2):
        return

    var re_offset = Int(local_tid) * simd_width
    var im_offset = re_offset + head_dim // 2
    var freq = freqs_cis.load[width=simd_width * 2, alignment=1](
        Coord(Idx(post_seq_idx), Idx(re_offset * 2))
    )

    var shared_base = (Int(warp_idx) * 2 + Int(sub_warp_idx)) * head_dim
    var rope_re = shared_norm.load[width=simd_width](shared_base + re_offset)
    var rope_im = shared_norm.load[width=simd_width](shared_base + im_offset)

    if is_q_row:
        var q_rope = _rope_complex_mul_half[
            dtype,
            freq_dtype,
            simd_width,
            simd_width * 2,
        ](rope_re, rope_im, freq)
        output.store[alignment=align](
            Coord(Idx(global_token_idx), Idx(head_slot), Idx(re_offset)),
            q_rope[0],
        )
        output.store[alignment=align](
            Coord(Idx(global_token_idx), Idx(head_slot), Idx(im_offset)),
            q_rope[1],
        )
    else:
        comptime cache_dtype = KCacheType.dtype
        var k_rope = _rope_complex_mul_half[
            accum_type,
            freq_dtype,
            simd_width,
            simd_width * 2,
        ](
            rope_re.cast[accum_type](),
            rope_im.cast[accum_type](),
            freq,
        )
        k_cache.store(
            batch_idx,
            k_head_idx,
            post_seq_idx,
            re_offset,
            k_rope[0].cast[cache_dtype](),
        )
        k_cache.store(
            batch_idx,
            k_head_idx,
            post_seq_idx,
            im_offset,
            k_rope[1].cast[cache_dtype](),
        )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(block_size))
)
def _gemma3_qk_norm_rope_decode_ragged_kernel[
    dtype: DType,
    freq_dtype: DType,
    KCacheType: KVCacheT,
    QLayoutType: TensorLayout,
    OutputLayoutType: TensorLayout,
    StartPosLayoutType: TensorLayout,
    FreqLayoutType: TensorLayout,
    QGammaLayoutType: TensorLayout,
    KGammaLayoutType: TensorLayout,
    block_size: Int,
    warps_per_block: Int,
](
    q: TileTensor[dtype, QLayoutType, MutAnyOrigin],
    output: TileTensor[mut=True, dtype, OutputLayoutType, MutAnyOrigin],
    start_pos: TileTensor[DType.uint32, StartPosLayoutType, MutAnyOrigin],
    k_cache: KCacheType,
    freqs_cis: TileTensor[freq_dtype, FreqLayoutType, MutAnyOrigin],
    q_gamma: TileTensor[dtype, QGammaLayoutType, MutAnyOrigin],
    k_gamma: TileTensor[dtype, KGammaLayoutType, MutAnyOrigin],
    epsilon: Scalar[dtype],
    weight_offset: Scalar[dtype],
    total_rows: Int,
):
    comptime assert q.flat_rank == 3
    comptime assert output.flat_rank == 3
    comptime assert start_pos.flat_rank == 1
    comptime assert freqs_cis.flat_rank == 2
    comptime assert q_gamma.flat_rank == 1
    comptime assert k_gamma.flat_rank == 1

    comptime num_q_heads = q.static_shape[1]
    comptime head_dim = q.static_shape[2]
    comptime num_kv_heads = Int(KCacheType.kv_params.num_heads)
    comptime assert head_dim == 128, "Only 128-column BF16 rows are supported"
    comptime assert head_dim == Int(KCacheType.kv_params.head_size)
    comptime assert freqs_cis.static_shape[1] == head_dim

    comptime simd_width = simd_width_of[dtype]()
    comptime vec_width = simd_width // 2
    comptime align = align_of[SIMD[dtype, vec_width]]()
    comptime wide_align = align_of[SIMD[dtype, simd_width]]()
    comptime accum_type = get_accum_type[dtype]()
    comptime half_warp_size = WARP_SIZE // 2
    comptime total_heads = num_q_heads + num_kv_heads
    comptime assert head_dim == half_warp_size * simd_width

    var shared_post_seq = stack_allocation[
        warps_per_block * 2,
        Int32,
        address_space=AddressSpace.SHARED,
    ]()

    var tid = thread_idx.x
    var warp_idx = tid // UInt(WARP_SIZE)
    var sub_warp_idx = (tid % UInt(WARP_SIZE)) // UInt(half_warp_size)
    var local_tid = tid % UInt(half_warp_size)
    var row = (
        block_idx.x * UInt(warps_per_block * 2) + warp_idx * 2 + sub_warp_idx
    )
    var col = local_tid * UInt(simd_width)

    var flat_row = Int(row)
    var global_token_idx = 0
    var head_slot = 0
    var batch_idx = 0
    var post_seq_idx = 0
    var shared_post_idx = 0

    var is_active_row = row < UInt(total_rows)
    if is_active_row:
        global_token_idx = flat_row // total_heads
        head_slot = flat_row % total_heads
        batch_idx = global_token_idx
        shared_post_idx = Int(warp_idx) * 2 + Int(sub_warp_idx)
        if local_tid == 0:
            shared_post_seq[shared_post_idx] = Int32(Int(start_pos[batch_idx]))
        syncwarp()
        post_seq_idx = Int(shared_post_seq[shared_post_idx])

    var is_q_row = is_active_row and head_slot < num_q_heads
    if is_active_row:
        var epsilon_accum = epsilon.cast[accum_type]()
        var weight_offset_accum = weight_offset.cast[accum_type]()

        if is_q_row:
            var re_offset = Int(local_tid) * vec_width
            var im_offset = re_offset + head_dim // 2
            var freq_offset = Int(local_tid) * simd_width
            var q_re = q.load[width=vec_width, alignment=align](
                Coord(Idx(global_token_idx), Idx(head_slot), Idx(re_offset))
            )
            var q_im = q.load[width=vec_width, alignment=align](
                Coord(Idx(global_token_idx), Idx(head_slot), Idx(im_offset))
            )
            var thread_m2 = (q_re.cast[accum_type]() ** 2).reduce_add() + (
                q_im.cast[accum_type]() ** 2
            ).reduce_add()
            var row_m2 = warp.lane_group_sum[num_lanes=half_warp_size](
                thread_m2
            )
            var norm_factor = rsqrt(
                (row_m2 / Scalar[accum_type](head_dim)) + epsilon_accum
            )

            var q_gamma_re = q_gamma.load[width=vec_width, alignment=align](
                Coord(Idx(re_offset))
            )
            var q_gamma_im = q_gamma.load[width=vec_width, alignment=align](
                Coord(Idx(im_offset))
            )
            var q_norm_re = (
                q_re.cast[accum_type]()
                * norm_factor
                * (q_gamma_re.cast[accum_type]() + weight_offset_accum)
            ).cast[dtype]()
            var q_norm_im = (
                q_im.cast[accum_type]()
                * norm_factor
                * (q_gamma_im.cast[accum_type]() + weight_offset_accum)
            ).cast[dtype]()

            var q_freq = freqs_cis.load[width=simd_width, alignment=1](
                Coord(Idx(post_seq_idx), Idx(freq_offset))
            )
            var q_rope = _rope_complex_mul_half[
                dtype,
                freq_dtype,
                vec_width,
                simd_width,
            ](q_norm_re, q_norm_im, q_freq)
            output.store[alignment=align](
                Coord(Idx(global_token_idx), Idx(head_slot), Idx(re_offset)),
                q_rope[0],
            )
            output.store[alignment=align](
                Coord(Idx(global_token_idx), Idx(head_slot), Idx(im_offset)),
                q_rope[1],
            )
        else:
            var k_head_idx = head_slot - num_q_heads
            var vec_data = k_cache.load[width=simd_width](
                batch_idx, k_head_idx, post_seq_idx, Int(col)
            ).cast[accum_type]()
            var gamma_val = k_gamma.load[
                width=simd_width, alignment=wide_align
            ](Coord(Idx(Int(col))))
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
                comptime cache_dtype = KCacheType.dtype
                var k_freq = freqs_cis.load[width=simd_width * 2, alignment=1](
                    Coord(Idx(post_seq_idx), Idx(re_offset * 2))
                )
                var k_rope = _rope_complex_mul_half[
                    accum_type,
                    freq_dtype,
                    simd_width,
                    simd_width * 2,
                ](
                    norm_val.cast[accum_type](),
                    partner_norm.cast[accum_type](),
                    k_freq,
                )
                k_cache.store(
                    batch_idx,
                    k_head_idx,
                    post_seq_idx,
                    re_offset,
                    k_rope[0].cast[cache_dtype](),
                )
                k_cache.store(
                    batch_idx,
                    k_head_idx,
                    post_seq_idx,
                    im_offset,
                    k_rope[1].cast[cache_dtype](),
                )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(block_size))
)
def _gemma3_qk_norm_rope_decode_uniform_kernel[
    dtype: DType,
    freq_dtype: DType,
    KCacheType: KVCacheT,
    QLayoutType: TensorLayout,
    OutputLayoutType: TensorLayout,
    FreqLayoutType: TensorLayout,
    QGammaLayoutType: TensorLayout,
    KGammaLayoutType: TensorLayout,
    block_size: Int,
    warps_per_block: Int,
](
    q: TileTensor[dtype, QLayoutType, MutAnyOrigin],
    output: TileTensor[mut=True, dtype, OutputLayoutType, MutAnyOrigin],
    k_cache: KCacheType,
    freqs_cis: TileTensor[freq_dtype, FreqLayoutType, MutAnyOrigin],
    q_gamma: TileTensor[dtype, QGammaLayoutType, MutAnyOrigin],
    k_gamma: TileTensor[dtype, KGammaLayoutType, MutAnyOrigin],
    epsilon: Scalar[dtype],
    weight_offset: Scalar[dtype],
    total_rows: Int,
    post_seq_idx: Int,
):
    comptime assert q.flat_rank == 3
    comptime assert output.flat_rank == 3
    comptime assert freqs_cis.flat_rank == 2
    comptime assert q_gamma.flat_rank == 1
    comptime assert k_gamma.flat_rank == 1

    comptime num_q_heads = q.static_shape[1]
    comptime head_dim = q.static_shape[2]
    comptime num_kv_heads = Int(KCacheType.kv_params.num_heads)
    comptime assert head_dim == 128, "Only 128-column BF16 rows are supported"
    comptime assert head_dim == Int(KCacheType.kv_params.head_size)
    comptime assert freqs_cis.static_shape[1] == head_dim

    comptime simd_width = simd_width_of[dtype]()
    comptime vec_width = simd_width // 2
    comptime align = align_of[SIMD[dtype, vec_width]]()
    comptime wide_align = align_of[SIMD[dtype, simd_width]]()
    comptime accum_type = get_accum_type[dtype]()
    comptime half_warp_size = WARP_SIZE // 2
    comptime total_heads = num_q_heads + num_kv_heads
    comptime assert head_dim == half_warp_size * simd_width

    var tid = thread_idx.x
    var warp_idx = tid // UInt(WARP_SIZE)
    var sub_warp_idx = (tid % UInt(WARP_SIZE)) // UInt(half_warp_size)
    var local_tid = tid % UInt(half_warp_size)
    var row = (
        block_idx.x * UInt(warps_per_block * 2) + warp_idx * 2 + sub_warp_idx
    )
    var col = local_tid * UInt(simd_width)

    var flat_row = Int(row)
    var global_token_idx = 0
    var head_slot = 0
    var batch_idx = 0

    var is_active_row = row < UInt(total_rows)
    if is_active_row:
        global_token_idx = flat_row // total_heads
        head_slot = flat_row % total_heads
        batch_idx = global_token_idx

    var is_q_row = is_active_row and head_slot < num_q_heads
    if is_active_row:
        var epsilon_accum = epsilon.cast[accum_type]()
        var weight_offset_accum = weight_offset.cast[accum_type]()

        if is_q_row:
            var re_offset = Int(local_tid) * vec_width
            var im_offset = re_offset + head_dim // 2
            var freq_offset = Int(local_tid) * simd_width
            var q_re = q.load[width=vec_width, alignment=align](
                Coord(Idx(global_token_idx), Idx(head_slot), Idx(re_offset))
            )
            var q_im = q.load[width=vec_width, alignment=align](
                Coord(Idx(global_token_idx), Idx(head_slot), Idx(im_offset))
            )
            var thread_m2 = (q_re.cast[accum_type]() ** 2).reduce_add() + (
                q_im.cast[accum_type]() ** 2
            ).reduce_add()
            var row_m2 = warp.lane_group_sum[num_lanes=half_warp_size](
                thread_m2
            )
            var norm_factor = rsqrt(
                (row_m2 / Scalar[accum_type](head_dim)) + epsilon_accum
            )

            var q_gamma_re = q_gamma.load[width=vec_width, alignment=align](
                Coord(Idx(re_offset))
            )
            var q_gamma_im = q_gamma.load[width=vec_width, alignment=align](
                Coord(Idx(im_offset))
            )
            var q_norm_re = (
                q_re.cast[accum_type]()
                * norm_factor
                * (q_gamma_re.cast[accum_type]() + weight_offset_accum)
            ).cast[dtype]()
            var q_norm_im = (
                q_im.cast[accum_type]()
                * norm_factor
                * (q_gamma_im.cast[accum_type]() + weight_offset_accum)
            ).cast[dtype]()

            var q_freq = freqs_cis.load[width=simd_width, alignment=1](
                Coord(Idx(post_seq_idx), Idx(freq_offset))
            )
            var q_rope = _rope_complex_mul_half[
                dtype,
                freq_dtype,
                vec_width,
                simd_width,
            ](q_norm_re, q_norm_im, q_freq)
            output.store[alignment=align](
                Coord(Idx(global_token_idx), Idx(head_slot), Idx(re_offset)),
                q_rope[0],
            )
            output.store[alignment=align](
                Coord(Idx(global_token_idx), Idx(head_slot), Idx(im_offset)),
                q_rope[1],
            )
        else:
            var k_head_idx = head_slot - num_q_heads
            var vec_data = k_cache.load[width=simd_width](
                batch_idx, k_head_idx, post_seq_idx, Int(col)
            ).cast[accum_type]()
            var gamma_val = k_gamma.load[
                width=simd_width, alignment=wide_align
            ](Coord(Idx(Int(col))))
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
                comptime cache_dtype = KCacheType.dtype
                var k_freq = freqs_cis.load[width=simd_width * 2, alignment=1](
                    Coord(Idx(post_seq_idx), Idx(re_offset * 2))
                )
                var k_rope = _rope_complex_mul_half[
                    accum_type,
                    freq_dtype,
                    simd_width,
                    simd_width * 2,
                ](
                    norm_val.cast[accum_type](),
                    partner_norm.cast[accum_type](),
                    k_freq,
                )
                k_cache.store(
                    batch_idx,
                    k_head_idx,
                    post_seq_idx,
                    re_offset,
                    k_rope[0].cast[cache_dtype](),
                )
                k_cache.store(
                    batch_idx,
                    k_head_idx,
                    post_seq_idx,
                    im_offset,
                    k_rope[1].cast[cache_dtype](),
                )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(block_size))
)
def _gemma3_qk_norm_rope_decode_uniform_wide_kernel[
    dtype: DType,
    freq_dtype: DType,
    KCacheType: KVCacheT,
    QLayoutType: TensorLayout,
    OutputLayoutType: TensorLayout,
    FreqLayoutType: TensorLayout,
    QGammaLayoutType: TensorLayout,
    KGammaLayoutType: TensorLayout,
    block_size: Int,
    warps_per_block: Int,
](
    q: TileTensor[dtype, QLayoutType, MutAnyOrigin],
    output: TileTensor[mut=True, dtype, OutputLayoutType, MutAnyOrigin],
    k_cache: KCacheType,
    freqs_cis: TileTensor[freq_dtype, FreqLayoutType, MutAnyOrigin],
    q_gamma: TileTensor[dtype, QGammaLayoutType, MutAnyOrigin],
    k_gamma: TileTensor[dtype, KGammaLayoutType, MutAnyOrigin],
    epsilon: Scalar[dtype],
    weight_offset: Scalar[dtype],
    total_rows: Int,
    post_seq_idx: Int,
):
    comptime assert q.flat_rank == 3
    comptime assert output.flat_rank == 3
    comptime assert freqs_cis.flat_rank == 2
    comptime assert q_gamma.flat_rank == 1
    comptime assert k_gamma.flat_rank == 1

    comptime num_q_heads = q.static_shape[1]
    comptime head_dim = q.static_shape[2]
    comptime num_kv_heads = Int(KCacheType.kv_params.num_heads)
    comptime assert head_dim == 256, "Only 256-column BF16 rows are supported"
    comptime assert head_dim == Int(KCacheType.kv_params.head_size)
    comptime assert freqs_cis.static_shape[1] == head_dim

    comptime simd_width = simd_width_of[dtype]()
    comptime align = align_of[SIMD[dtype, simd_width]]()
    comptime accum_type = get_accum_type[dtype]()
    comptime total_heads = num_q_heads + num_kv_heads
    comptime assert head_dim == WARP_SIZE * simd_width

    var shared_norm = stack_allocation[
        warps_per_block * head_dim,
        Scalar[dtype],
        address_space=AddressSpace.SHARED,
    ]()

    var tid = thread_idx.x
    var warp_idx = tid // UInt(WARP_SIZE)
    var local_tid = tid % UInt(WARP_SIZE)
    var row = block_idx.x * UInt(warps_per_block) + warp_idx
    var col = local_tid * UInt(simd_width)

    var flat_row = Int(row)
    var global_token_idx = 0
    var head_slot = 0
    var batch_idx = 0

    var is_active_row = row < UInt(total_rows)
    if is_active_row:
        global_token_idx = flat_row // total_heads
        head_slot = flat_row % total_heads
        batch_idx = global_token_idx

    var is_q_row = is_active_row and head_slot < num_q_heads
    if is_active_row:
        var epsilon_accum = epsilon.cast[accum_type]()
        var weight_offset_accum = weight_offset.cast[accum_type]()

        if is_q_row:
            var q_vec = q.load[width=simd_width, alignment=align](
                Coord(Idx(global_token_idx), Idx(head_slot), Idx(Int(col)))
            )
            var thread_m2 = (q_vec.cast[accum_type]() ** 2).reduce_add()
            var row_m2 = warp.sum(thread_m2)
            var norm_factor = rsqrt(
                (row_m2 / Scalar[accum_type](head_dim)) + epsilon_accum
            )
            var gamma_val = q_gamma.load[width=simd_width, alignment=align](
                Coord(Idx(Int(col)))
            )
            var norm_val = (
                q_vec.cast[accum_type]()
                * norm_factor
                * (gamma_val.cast[accum_type]() + weight_offset_accum)
            ).cast[dtype]()
            var shared_base = Int(warp_idx) * head_dim
            shared_norm.store[width=simd_width, alignment=align](
                shared_base + Int(col), norm_val
            )
            syncwarp()

            if local_tid < UInt(WARP_SIZE // 2):
                var re_offset = Int(local_tid) * simd_width
                var im_offset = re_offset + head_dim // 2
                var rope_re = shared_norm.load[width=simd_width](
                    shared_base + re_offset
                )
                var rope_im = shared_norm.load[width=simd_width](
                    shared_base + im_offset
                )
                var q_freq = freqs_cis.load[width=simd_width * 2, alignment=1](
                    Coord(Idx(post_seq_idx), Idx(re_offset * 2))
                )
                var q_rope = _rope_complex_mul_half[
                    dtype,
                    freq_dtype,
                    simd_width,
                    simd_width * 2,
                ](rope_re, rope_im, q_freq)
                output.store[alignment=align](
                    Coord(
                        Idx(global_token_idx), Idx(head_slot), Idx(re_offset)
                    ),
                    q_rope[0],
                )
                output.store[alignment=align](
                    Coord(
                        Idx(global_token_idx), Idx(head_slot), Idx(im_offset)
                    ),
                    q_rope[1],
                )
        else:
            var k_head_idx = head_slot - num_q_heads
            var k_vec = k_cache.load[width=simd_width](
                batch_idx, k_head_idx, post_seq_idx, Int(col)
            ).cast[accum_type]()
            var thread_m2 = (k_vec**2).reduce_add()
            var row_m2 = warp.sum(thread_m2)
            var norm_factor = rsqrt(
                (row_m2 / Scalar[accum_type](head_dim)) + epsilon_accum
            )
            var gamma_val = k_gamma.load[width=simd_width, alignment=align](
                Coord(Idx(Int(col)))
            )
            var norm_val = (
                k_vec
                * norm_factor
                * (gamma_val.cast[accum_type]() + weight_offset_accum)
            )
            var shared_base = Int(warp_idx) * head_dim
            shared_norm.store[width=simd_width, alignment=align](
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
                var k_freq = freqs_cis.load[width=simd_width * 2, alignment=1](
                    Coord(Idx(post_seq_idx), Idx(re_offset * 2))
                )
                comptime cache_dtype = KCacheType.dtype
                var k_rope = _rope_complex_mul_half[
                    accum_type,
                    freq_dtype,
                    simd_width,
                    simd_width * 2,
                ](rope_re, rope_im, k_freq)
                k_cache.store(
                    batch_idx,
                    k_head_idx,
                    post_seq_idx,
                    re_offset,
                    k_rope[0].cast[cache_dtype](),
                )
                k_cache.store(
                    batch_idx,
                    k_head_idx,
                    post_seq_idx,
                    im_offset,
                    k_rope[1].cast[cache_dtype](),
                )


def _gemma3_qk_norm_rope_decode_ragged_wide_kernel[
    dtype: DType,
    freq_dtype: DType,
    KCacheType: KVCacheT,
    QLayoutType: TensorLayout,
    OutputLayoutType: TensorLayout,
    StartPosLayoutType: TensorLayout,
    FreqLayoutType: TensorLayout,
    QGammaLayoutType: TensorLayout,
    KGammaLayoutType: TensorLayout,
    block_size: Int,
    warps_per_block: Int,
](
    q: TileTensor[dtype, QLayoutType, MutAnyOrigin],
    output: TileTensor[mut=True, dtype, OutputLayoutType, MutAnyOrigin],
    start_pos: TileTensor[DType.uint32, StartPosLayoutType, MutAnyOrigin],
    k_cache: KCacheType,
    freqs_cis: TileTensor[freq_dtype, FreqLayoutType, MutAnyOrigin],
    q_gamma: TileTensor[dtype, QGammaLayoutType, MutAnyOrigin],
    k_gamma: TileTensor[dtype, KGammaLayoutType, MutAnyOrigin],
    epsilon: Scalar[dtype],
    weight_offset: Scalar[dtype],
    total_rows: Int,
):
    comptime assert q.flat_rank == 3
    comptime assert output.flat_rank == 3
    comptime assert start_pos.flat_rank == 1
    comptime assert freqs_cis.flat_rank == 2
    comptime assert q_gamma.flat_rank == 1
    comptime assert k_gamma.flat_rank == 1

    comptime num_q_heads = q.static_shape[1]
    comptime head_dim = q.static_shape[2]
    comptime num_kv_heads = Int(KCacheType.kv_params.num_heads)
    comptime assert head_dim == 256, "Only 256-column BF16 rows are supported"
    comptime assert head_dim == Int(KCacheType.kv_params.head_size)
    comptime assert freqs_cis.static_shape[1] == head_dim

    comptime simd_width = simd_width_of[dtype]()
    comptime align = align_of[SIMD[dtype, simd_width]]()
    comptime accum_type = get_accum_type[dtype]()
    comptime total_heads = num_q_heads + num_kv_heads
    comptime assert head_dim == WARP_SIZE * simd_width

    var shared_norm = stack_allocation[
        warps_per_block * head_dim,
        Scalar[dtype],
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

    var flat_row = Int(row)
    var global_token_idx = 0
    var head_slot = 0
    var batch_idx = 0
    var post_seq_idx = 0

    var is_active_row = row < UInt(total_rows)
    if is_active_row:
        global_token_idx = flat_row // total_heads
        head_slot = flat_row % total_heads
        batch_idx = global_token_idx
        if local_tid == 0:
            shared_post_seq[Int(warp_idx)] = Int32(Int(start_pos[batch_idx]))
        syncwarp()
        post_seq_idx = Int(shared_post_seq[Int(warp_idx)])

    var is_q_row = is_active_row and head_slot < num_q_heads
    if is_active_row:
        var epsilon_accum = epsilon.cast[accum_type]()
        var weight_offset_accum = weight_offset.cast[accum_type]()
        var shared_base = Int(warp_idx) * head_dim

        if is_q_row:
            var q_vec = q.load[width=simd_width, alignment=align](
                Coord(Idx(global_token_idx), Idx(head_slot), Idx(Int(col)))
            )
            var thread_m2 = (q_vec.cast[accum_type]() ** 2).reduce_add()
            var row_m2 = warp.sum(thread_m2)
            var norm_factor = rsqrt(
                (row_m2 / Scalar[accum_type](head_dim)) + epsilon_accum
            )
            var gamma_val = q_gamma.load[width=simd_width, alignment=align](
                Coord(Idx(Int(col)))
            )
            var norm_val = (
                q_vec.cast[accum_type]()
                * norm_factor
                * (gamma_val.cast[accum_type]() + weight_offset_accum)
            ).cast[dtype]()
            shared_norm.store[width=simd_width, alignment=align](
                shared_base + Int(col), norm_val
            )
            syncwarp()

            if local_tid < UInt(WARP_SIZE // 2):
                var re_offset = Int(local_tid) * simd_width
                var im_offset = re_offset + head_dim // 2
                var rope_re = shared_norm.load[width=simd_width](
                    shared_base + re_offset
                )
                var rope_im = shared_norm.load[width=simd_width](
                    shared_base + im_offset
                )
                var q_freq = freqs_cis.load[width=simd_width * 2, alignment=1](
                    Coord(Idx(post_seq_idx), Idx(re_offset * 2))
                )
                var q_rope = _rope_complex_mul_half[
                    dtype,
                    freq_dtype,
                    simd_width,
                    simd_width * 2,
                ](rope_re, rope_im, q_freq)
                output.store[alignment=align](
                    Coord(
                        Idx(global_token_idx), Idx(head_slot), Idx(re_offset)
                    ),
                    q_rope[0],
                )
                output.store[alignment=align](
                    Coord(
                        Idx(global_token_idx), Idx(head_slot), Idx(im_offset)
                    ),
                    q_rope[1],
                )
        else:
            var k_head_idx = head_slot - num_q_heads
            var k_vec = k_cache.load[width=simd_width](
                batch_idx, k_head_idx, post_seq_idx, Int(col)
            ).cast[accum_type]()
            var thread_m2 = (k_vec**2).reduce_add()
            var row_m2 = warp.sum(thread_m2)
            var norm_factor = rsqrt(
                (row_m2 / Scalar[accum_type](head_dim)) + epsilon_accum
            )
            var gamma_val = k_gamma.load[width=simd_width, alignment=align](
                Coord(Idx(Int(col)))
            )
            var norm_val = (
                k_vec
                * norm_factor
                * (gamma_val.cast[accum_type]() + weight_offset_accum)
            )
            shared_norm.store[width=simd_width, alignment=align](
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
                var k_freq = freqs_cis.load[width=simd_width * 2, alignment=1](
                    Coord(Idx(post_seq_idx), Idx(re_offset * 2))
                )
                comptime cache_dtype = KCacheType.dtype
                var k_rope = _rope_complex_mul_half[
                    accum_type,
                    freq_dtype,
                    simd_width,
                    simd_width * 2,
                ](rope_re, rope_im, k_freq)
                k_cache.store(
                    batch_idx,
                    k_head_idx,
                    post_seq_idx,
                    re_offset,
                    k_rope[0].cast[cache_dtype](),
                )
                k_cache.store(
                    batch_idx,
                    k_head_idx,
                    post_seq_idx,
                    im_offset,
                    k_rope[1].cast[cache_dtype](),
                )


def bench_gemma3_qk_norm_rope_boundary[
    dtype: DType,
    head_dim: Int,
    num_q_heads: Int,
    num_kv_heads: Int,
    page_size: Int,
](
    ctx: DeviceContext,
    mut bench: Bench,
    batch_size: Int,
    seq_len: Int,
    cache_len: Int,
    cache_len_step: Int,
    trial_baseline_disable_decode_fastpath: Bool,
    verify_results: Bool,
) raises:
    comptime max_seq_len = 2048
    comptime num_layers = 1
    comptime layer_idx = 0
    comptime kv_params = KVCacheStaticParams(
        num_heads=UInt(num_kv_heads), head_size=UInt(head_dim)
    )
    comptime CollectionType = PagedKVCacheCollection[
        dtype, kv_params, page_size
    ]

    var total_seq_len = UInt32(batch_size * seq_len)
    var max_cache_len = cache_len + (batch_size - 1) * cache_len_step
    var max_context_length = seq_len + max_cache_len
    var paged_lut_cols = ceildiv(max_context_length, page_size)
    var num_pages = batch_size * paged_lut_cols
    var q_elems = Int(total_seq_len) * num_q_heads * head_dim

    var input_row_offsets_h = alloc[Scalar[DType.uint32]](batch_size + 1)
    var cache_lengths_h = alloc[Scalar[DType.uint32]](batch_size)
    var q_gamma_h = alloc[Scalar[dtype]](head_dim)
    var k_gamma_h = alloc[Scalar[dtype]](head_dim)
    var freqs_h = alloc[Scalar[dtype]](max_seq_len * head_dim)
    var q_h = alloc[Scalar[dtype]](q_elems)
    var paged_lut_h = alloc[Scalar[DType.uint32]](batch_size * paged_lut_cols)
    var baseline_output_h = alloc[Scalar[dtype]](q_elems)
    var fused_output_h = alloc[Scalar[dtype]](q_elems)
    var trial_output_h = alloc[Scalar[dtype]](q_elems)

    var running_offset: UInt32 = 0
    for i in range(batch_size):
        input_row_offsets_h[i] = running_offset
        cache_lengths_h[i] = UInt32(cache_len + i * cache_len_step)
        running_offset += UInt32(seq_len)
    input_row_offsets_h[batch_size] = running_offset

    # Gemma keeps distinct learned Q/K RMSNorm weights, so benchmark the exact
    # live seam with separate deterministic gamma vectors.
    for i in range(head_dim):
        q_gamma_h[i] = (Float64(i + head_dim) / Float64(head_dim)).cast[dtype]()
        k_gamma_h[i] = (Float64(i + (head_dim // 2)) / Float64(head_dim)).cast[
            dtype
        ]()

    for i in range(max_seq_len * head_dim):
        freqs_h[i] = Scalar[dtype](random_float64(-1, 1).cast[dtype]())

    for i in range(q_elems):
        q_h[i] = Scalar[dtype](random_float64(0, 100).cast[dtype]())

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
            paged_lut_host[bs, page_idx] = UInt32(
                bs * paged_lut_cols + page_idx
            )

    var input_row_offsets_d = ctx.enqueue_create_buffer[DType.uint32](
        batch_size + 1
    )
    var cache_lengths_d = ctx.enqueue_create_buffer[DType.uint32](batch_size)
    var q_gamma_d = ctx.enqueue_create_buffer[dtype](head_dim)
    var k_gamma_d = ctx.enqueue_create_buffer[dtype](head_dim)
    var freqs_d = ctx.enqueue_create_buffer[dtype](max_seq_len * head_dim)
    var q_d = ctx.enqueue_create_buffer[dtype](q_elems)
    var baseline_q_norm_d = ctx.enqueue_create_buffer[dtype](q_elems)
    var baseline_output_d = ctx.enqueue_create_buffer[dtype](q_elems)
    var fused_output_d = ctx.enqueue_create_buffer[dtype](q_elems)
    var trial_output_d = ctx.enqueue_create_buffer[dtype](q_elems)
    var paged_lut_d = ctx.enqueue_create_buffer[DType.uint32](
        batch_size * paged_lut_cols
    )

    ctx.enqueue_copy(input_row_offsets_d, input_row_offsets_h)
    ctx.enqueue_copy(cache_lengths_d, cache_lengths_h)
    ctx.enqueue_copy(q_gamma_d, q_gamma_h)
    ctx.enqueue_copy(k_gamma_d, k_gamma_h)
    ctx.enqueue_copy(freqs_d, freqs_h)
    ctx.enqueue_copy(q_d, q_h)
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
    var baseline_kv_block_h = alloc[Scalar[dtype]](kv_block_size)
    var fused_kv_block_h = alloc[Scalar[dtype]](kv_block_size)
    var trial_kv_block_h = alloc[Scalar[dtype]](kv_block_size)
    random(
        LayoutTensor[dtype, Layout.row_major[6](), MutAnyOrigin](
            baseline_kv_block_h,
            RuntimeLayout[Layout.row_major[6]()].row_major(kv_block_shape),
        )
    )

    var baseline_kv_block_d = ctx.enqueue_create_buffer[dtype](kv_block_size)
    var fused_kv_block_d = ctx.enqueue_create_buffer[dtype](kv_block_size)
    var trial_kv_block_d = ctx.enqueue_create_buffer[dtype](kv_block_size)
    ctx.enqueue_copy(baseline_kv_block_d, baseline_kv_block_h)
    ctx.enqueue_copy(fused_kv_block_d, baseline_kv_block_h)
    ctx.enqueue_copy(trial_kv_block_d, baseline_kv_block_h)

    var q_shape = IndexList[3](Int(total_seq_len), num_q_heads, head_dim)
    var q_layout = row_major(
        (Idx(total_seq_len), Idx[num_q_heads](), Idx[head_dim]())
    )
    var q_tensor = TileTensor(q_d.unsafe_ptr(), q_layout)
    var baseline_q_norm_tensor = TileTensor(
        baseline_q_norm_d.unsafe_ptr(), q_layout
    )
    var baseline_output_tensor = TileTensor(
        baseline_output_d.unsafe_ptr(), q_layout
    )
    var fused_output_tensor = TileTensor(fused_output_d.unsafe_ptr(), q_layout)
    var trial_output_tensor = TileTensor(trial_output_d.unsafe_ptr(), q_layout)
    var q_gamma_tensor = TileTensor(
        q_gamma_d.unsafe_ptr(), row_major(Idx[head_dim]())
    )
    var k_gamma_tensor = TileTensor(
        k_gamma_d.unsafe_ptr(), row_major(Idx[head_dim]())
    )
    var input_row_offsets_tensor = TileTensor(
        input_row_offsets_d.unsafe_ptr(), row_major(Idx(batch_size + 1))
    )
    var start_pos_tensor = TileTensor(
        cache_lengths_d.unsafe_ptr(), row_major(Idx(batch_size))
    )
    comptime freqs_layout = row_major[max_seq_len, head_dim]()
    var freqs_tensor = TileTensor(freqs_d.unsafe_ptr(), freqs_layout)

    comptime kv_block_layout = Layout.row_major[6]()
    comptime cache_lengths_layout = Layout(UNKNOWN_VALUE)
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
    var fused_kv_collection = CollectionType(
        LayoutTensor[dtype, kv_block_layout, MutAnyOrigin](
            fused_kv_block_d.unsafe_ptr(),
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
    @__copy_capture(q_tensor)
    @parameter
    def input_fn[
        width: Int, rank: Int
    ](coords: IndexList[rank]) -> SIMD[dtype, width]:
        return q_tensor.load[width=width](Coord(coords))

    @always_inline
    @__copy_capture(baseline_q_norm_tensor)
    @parameter
    def baseline_q_norm_output_fn[
        width: Int, alignment: Int
    ](coords: IndexList[3], val: SIMD[dtype, width]) -> None:
        baseline_q_norm_tensor.store[alignment=alignment](Coord(coords), val)

    @always_inline
    @__copy_capture(
        q_shape,
        q_gamma_tensor,
        k_gamma_tensor,
        epsilon,
        weight_offset,
        baseline_kv_collection,
        input_row_offsets_tensor,
        total_seq_len,
        freqs_tensor,
        baseline_q_norm_tensor,
        baseline_output_tensor,
    )
    @parameter
    def run_baseline(ctx: DeviceContext) raises:
        rms_norm_gpu[
            input_fn,
            baseline_q_norm_output_fn,
            multiply_before_cast=True,
        ](q_shape, q_gamma_tensor, epsilon, weight_offset, ctx)
        rms_norm_kv_cache_ragged_paged[
            target="gpu",
            multiply_before_cast=True,
            per_head_norm=True,
        ](
            baseline_kv_collection,
            k_gamma_tensor,
            epsilon,
            weight_offset,
            UInt32(layer_idx),
            total_seq_len,
            input_row_offsets_tensor,
            ctx,
        )
        fused_qk_rope_ragged[
            CollectionType.CacheType, interleaved=False, target="gpu"
        ](
            baseline_q_norm_tensor,
            input_row_offsets_tensor,
            baseline_kv_collection,
            freqs_tensor,
            None,
            UInt32(layer_idx),
            baseline_output_tensor,
            Optional[DeviceContext](ctx),
        )

    @always_inline
    @__copy_capture(
        q_tensor,
        input_row_offsets_tensor,
        start_pos_tensor,
        fused_kv_collection,
        freqs_tensor,
        q_gamma_tensor,
        k_gamma_tensor,
        epsilon,
        weight_offset,
        fused_output_tensor,
    )
    @parameter
    def run_fused(ctx: DeviceContext) raises:
        q_rms_norm_fused_qk_rope_ragged[
            dtype,
            dtype,
            CollectionType,
            interleaved=False,
            target="gpu",
        ](
            q_tensor,
            input_row_offsets_tensor,
            start_pos_tensor,
            fused_kv_collection,
            freqs_tensor,
            q_gamma_tensor,
            k_gamma_tensor,
            epsilon,
            weight_offset,
            UInt32(layer_idx),
            fused_output_tensor,
            ctx,
        )

    @always_inline
    @__copy_capture(
        q_tensor,
        input_row_offsets_tensor,
        start_pos_tensor,
        trial_kv_collection,
        baseline_q_norm_tensor,
        freqs_tensor,
        q_gamma_tensor,
        k_gamma_tensor,
        epsilon,
        weight_offset,
        trial_output_tensor,
    )
    @parameter
    def run_trial_fused(ctx: DeviceContext) raises:
        q_rms_norm_fused_qk_rope_ragged[
            dtype,
            dtype,
            CollectionType,
            interleaved=False,
            target="gpu",
        ](
            q_tensor,
            input_row_offsets_tensor,
            start_pos_tensor,
            trial_kv_collection,
            freqs_tensor,
            q_gamma_tensor,
            k_gamma_tensor,
            epsilon,
            weight_offset,
            UInt32(layer_idx),
            trial_output_tensor,
            ctx,
        )

    @always_inline
    @__copy_capture(
        q_tensor,
        input_row_offsets_tensor,
        start_pos_tensor,
        trial_kv_collection,
        freqs_tensor,
        q_gamma_tensor,
        k_gamma_tensor,
        epsilon,
        weight_offset,
        trial_output_tensor,
        total_seq_len,
    )
    @parameter
    def run_trial(ctx: DeviceContext) raises:
        if trial_baseline_disable_decode_fastpath:
            rms_norm_gpu[
                input_fn,
                baseline_q_norm_output_fn,
                multiply_before_cast=True,
            ](q_shape, q_gamma_tensor, epsilon, weight_offset, ctx)
            rms_norm_kv_cache_ragged_paged[
                target="gpu",
                multiply_before_cast=True,
                per_head_norm=True,
            ](
                trial_kv_collection,
                k_gamma_tensor,
                epsilon,
                weight_offset,
                UInt32(layer_idx),
                total_seq_len,
                input_row_offsets_tensor,
                ctx,
            )
            fused_qk_rope_ragged[
                CollectionType.CacheType,
                interleaved=False,
                target="gpu",
                allow_decode_fastpath=False,
            ](
                baseline_q_norm_tensor,
                input_row_offsets_tensor,
                trial_kv_collection,
                freqs_tensor,
                None,
                UInt32(layer_idx),
                trial_output_tensor,
                Optional[DeviceContext](ctx),
            )
            return

        comptime if dtype == DType.bfloat16:
            comptime if head_dim == 128:
                if seq_len == 1:
                    comptime block_size = 64
                    comptime warps_per_block = 2
                    comptime total_heads = num_q_heads + num_kv_heads

                    var total_rows = Int(total_seq_len) * total_heads
                    var k_cache = trial_kv_collection.get_key_cache(layer_idx)

                    if cache_len_step == 0:
                        comptime kernel = _gemma3_qk_norm_rope_decode_uniform_kernel[
                            dtype,
                            dtype,
                            CollectionType.CacheType,
                            type_of(q_tensor).LayoutType,
                            type_of(trial_output_tensor).LayoutType,
                            type_of(freqs_tensor).LayoutType,
                            type_of(q_gamma_tensor).LayoutType,
                            type_of(k_gamma_tensor).LayoutType,
                            block_size,
                            warps_per_block,
                        ]
                        ctx.enqueue_function[kernel, kernel](
                            q_tensor,
                            trial_output_tensor,
                            k_cache,
                            freqs_tensor,
                            q_gamma_tensor,
                            k_gamma_tensor,
                            epsilon,
                            weight_offset,
                            total_rows,
                            cache_len,
                            grid_dim=ceildiv(total_rows, warps_per_block * 2),
                            block_dim=block_size,
                        )
                    else:
                        comptime kernel = _gemma3_qk_norm_rope_decode_ragged_kernel[
                            dtype,
                            dtype,
                            CollectionType.CacheType,
                            type_of(q_tensor).LayoutType,
                            type_of(trial_output_tensor).LayoutType,
                            type_of(start_pos_tensor).LayoutType,
                            type_of(freqs_tensor).LayoutType,
                            type_of(q_gamma_tensor).LayoutType,
                            type_of(k_gamma_tensor).LayoutType,
                            block_size,
                            warps_per_block,
                        ]
                        ctx.enqueue_function[kernel, kernel](
                            q_tensor,
                            trial_output_tensor,
                            start_pos_tensor,
                            k_cache,
                            freqs_tensor,
                            q_gamma_tensor,
                            k_gamma_tensor,
                            epsilon,
                            weight_offset,
                            total_rows,
                            grid_dim=ceildiv(total_rows, warps_per_block * 2),
                            block_dim=block_size,
                        )
                else:
                    run_trial_fused(ctx)
            else:
                comptime if head_dim == 256:
                    if seq_len == 1:
                        comptime block_size = 64
                        comptime warps_per_block = 2
                        comptime total_heads = num_q_heads + num_kv_heads

                        var total_rows = Int(total_seq_len) * total_heads
                        var k_cache = trial_kv_collection.get_key_cache(
                            layer_idx
                        )

                        if cache_len_step == 0:
                            comptime kernel = _gemma3_qk_norm_rope_decode_uniform_wide_kernel[
                                dtype,
                                dtype,
                                CollectionType.CacheType,
                                type_of(q_tensor).LayoutType,
                                type_of(trial_output_tensor).LayoutType,
                                type_of(freqs_tensor).LayoutType,
                                type_of(q_gamma_tensor).LayoutType,
                                type_of(k_gamma_tensor).LayoutType,
                                block_size,
                                warps_per_block,
                            ]
                            ctx.enqueue_function[kernel, kernel](
                                q_tensor,
                                trial_output_tensor,
                                k_cache,
                                freqs_tensor,
                                q_gamma_tensor,
                                k_gamma_tensor,
                                epsilon,
                                weight_offset,
                                total_rows,
                                cache_len,
                                grid_dim=ceildiv(total_rows, warps_per_block),
                                block_dim=block_size,
                            )
                        else:
                            comptime kernel = _gemma3_qk_norm_rope_decode_ragged_wide_kernel[
                                dtype,
                                dtype,
                                CollectionType.CacheType,
                                type_of(q_tensor).LayoutType,
                                type_of(trial_output_tensor).LayoutType,
                                type_of(start_pos_tensor).LayoutType,
                                type_of(freqs_tensor).LayoutType,
                                type_of(q_gamma_tensor).LayoutType,
                                type_of(k_gamma_tensor).LayoutType,
                                block_size,
                                warps_per_block,
                            ]
                            ctx.enqueue_function[kernel, kernel](
                                q_tensor,
                                trial_output_tensor,
                                start_pos_tensor,
                                k_cache,
                                freqs_tensor,
                                q_gamma_tensor,
                                k_gamma_tensor,
                                epsilon,
                                weight_offset,
                                total_rows,
                                grid_dim=ceildiv(total_rows, warps_per_block),
                                block_dim=block_size,
                            )
                    else:
                        run_trial_fused(ctx)
                else:
                    run_trial_fused(ctx)
        else:
            run_trial_fused(ctx)

    @always_inline
    @parameter
    def baseline_bench(mut bencher: Bencher) raises:
        bencher.iter_custom[run_baseline](ctx)

    @always_inline
    @parameter
    def fused_bench(mut bencher: Bencher) raises:
        bencher.iter_custom[run_fused](ctx)

    @always_inline
    @parameter
    def trial_bench(mut bencher: Bencher) raises:
        bencher.iter_custom[run_trial](ctx)

    bench.bench_function[baseline_bench](
        BenchId(
            "gemma3_qk_norm_rope_boundary_baseline",
            input_id=String(
                dtype,
                "/bs=",
                batch_size,
                "/seq=",
                seq_len,
                "/cache=",
                cache_len,
                "/cache_step=",
                cache_len_step,
            ),
        ),
    )
    bench.bench_function[fused_bench](
        BenchId(
            "gemma3_qk_norm_rope_boundary_fused",
            input_id=String(
                dtype,
                "/bs=",
                batch_size,
                "/seq=",
                seq_len,
                "/cache=",
                cache_len,
                "/cache_step=",
                cache_len_step,
            ),
        ),
    )
    comptime if dtype == DType.bfloat16 and (
        head_dim == 128 or head_dim == 256
    ):
        bench.bench_function[trial_bench](
            BenchId(
                "gemma3_qk_norm_rope_boundary_trial",
                input_id=String(
                    dtype,
                    "/bs=",
                    batch_size,
                    "/seq=",
                    seq_len,
                    "/cache=",
                    cache_len,
                    "/cache_step=",
                    cache_len_step,
                ),
            ),
        )

    if verify_results:
        ctx.enqueue_copy(baseline_kv_block_d, baseline_kv_block_h)
        ctx.enqueue_copy(fused_kv_block_d, baseline_kv_block_h)
        comptime if dtype == DType.bfloat16 and (
            head_dim == 128 or head_dim == 256
        ):
            ctx.enqueue_copy(trial_kv_block_d, baseline_kv_block_h)
        run_baseline(ctx)
        run_fused(ctx)
        comptime if dtype == DType.bfloat16 and (
            head_dim == 128 or head_dim == 256
        ):
            run_trial(ctx)
        ctx.enqueue_copy(baseline_output_h, baseline_output_d)
        ctx.enqueue_copy(fused_output_h, fused_output_d)
        comptime if dtype == DType.bfloat16 and (
            head_dim == 128 or head_dim == 256
        ):
            ctx.enqueue_copy(trial_output_h, trial_output_d)
        ctx.enqueue_copy(baseline_kv_block_h, baseline_kv_block_d)
        ctx.enqueue_copy(fused_kv_block_h, fused_kv_block_d)
        comptime if dtype == DType.bfloat16 and (
            head_dim == 128 or head_dim == 256
        ):
            ctx.enqueue_copy(trial_kv_block_h, trial_kv_block_d)
        ctx.synchronize()

        for i in range(q_elems):
            assert_almost_equal(
                baseline_output_h[i], fused_output_h[i], rtol=2e-2, atol=2e-2
            )
            comptime if dtype == DType.bfloat16 and (
                head_dim == 128 or head_dim == 256
            ):
                assert_almost_equal(
                    baseline_output_h[i],
                    trial_output_h[i],
                    rtol=2e-2,
                    atol=2e-2,
                )

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
        var fused_kv_collection_host = CollectionType(
            LayoutTensor[dtype, kv_block_layout, MutAnyOrigin](
                fused_kv_block_h,
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
        var baseline_k_cache_host = baseline_kv_collection_host.get_key_cache(
            layer_idx
        )
        var fused_k_cache_host = fused_kv_collection_host.get_key_cache(
            layer_idx
        )

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
                            fused_k_cache_host.load[width=1](
                                bs_idx,
                                head_idx,
                                post_seq_idx,
                                dim_idx,
                            ),
                            rtol=2e-2,
                            atol=2e-2,
                        )
        comptime if dtype == DType.bfloat16 and (
            head_dim == 128 or head_dim == 256
        ):
            var trial_kv_collection_host = CollectionType(
                LayoutTensor[dtype, kv_block_layout, MutAnyOrigin](
                    trial_kv_block_h,
                    RuntimeLayout[kv_block_layout].row_major(kv_block_shape),
                ),
                LayoutTensor[
                    DType.uint32, cache_lengths_layout, ImmutAnyOrigin
                ](
                    cache_lengths_h,
                    RuntimeLayout[cache_lengths_layout].row_major(
                        IndexList[1](batch_size)
                    ),
                ),
                LayoutTensor[
                    DType.uint32, Layout.row_major[2](), ImmutAnyOrigin
                ](
                    paged_lut_h,
                    RuntimeLayout[Layout.row_major[2]()].row_major(
                        IndexList[2](batch_size, paged_lut_cols)
                    ),
                ),
                UInt32(seq_len),
                UInt32(max_context_length),
            )
            var trial_k_cache_host = trial_kv_collection_host.get_key_cache(
                layer_idx
            )
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
    _ = q_gamma_d
    _ = k_gamma_d
    _ = freqs_d
    _ = q_d
    _ = baseline_q_norm_d
    _ = baseline_output_d
    _ = fused_output_d
    _ = trial_output_d
    _ = paged_lut_d
    _ = baseline_kv_block_d
    _ = fused_kv_block_d
    _ = trial_kv_block_d

    input_row_offsets_h.free()
    cache_lengths_h.free()
    q_gamma_h.free()
    k_gamma_h.free()
    freqs_h.free()
    q_h.free()
    paged_lut_h.free()
    baseline_output_h.free()
    fused_output_h.free()
    trial_output_h.free()
    baseline_kv_block_h.free()
    fused_kv_block_h.free()
    trial_kv_block_h.free()


def bench_gemma3_qk_norm_rope_boundary_pair[
    dtype: DType,
    head_dim: Int,
    num_q_heads: Int,
    num_kv_heads: Int,
    page_size: Int,
](
    ctx: DeviceContext,
    mut bench: Bench,
    batch_size: Int,
    seq_len: Int,
    cache_len: Int,
    cache_len_step: Int,
    trial_baseline_disable_decode_fastpath: Bool,
    verify_results: Bool,
) raises:
    comptime assert (
        dtype == DType.bfloat16
    ), "pair_only only supports the BF16 boundary trial"
    comptime assert (
        head_dim == 128 or head_dim == 256
    ), "pair_only only supports head_dim 128 or 256"
    if verify_results:
        raise Error("pair_only requires --verify=False")
    if trial_baseline_disable_decode_fastpath:
        raise Error(
            "pair_only requires trial_baseline_disable_decode_fastpath=False"
        )

    comptime max_seq_len = 2048
    comptime num_layers = 1
    comptime layer_idx = 0
    comptime kv_params = KVCacheStaticParams(
        num_heads=UInt(num_kv_heads), head_size=UInt(head_dim)
    )
    comptime CollectionType = PagedKVCacheCollection[
        dtype, kv_params, page_size
    ]

    var total_seq_len = UInt32(batch_size * seq_len)
    var max_cache_len = cache_len + (batch_size - 1) * cache_len_step
    var max_context_length = seq_len + max_cache_len
    var paged_lut_cols = ceildiv(max_context_length, page_size)
    var num_pages = batch_size * paged_lut_cols
    var q_elems = Int(total_seq_len) * num_q_heads * head_dim

    var input_row_offsets_h = alloc[Scalar[DType.uint32]](batch_size + 1)
    var cache_lengths_h = alloc[Scalar[DType.uint32]](batch_size)
    var q_gamma_h = alloc[Scalar[dtype]](head_dim)
    var k_gamma_h = alloc[Scalar[dtype]](head_dim)
    var freqs_h = alloc[Scalar[dtype]](max_seq_len * head_dim)
    var q_h = alloc[Scalar[dtype]](q_elems)
    var paged_lut_h = alloc[Scalar[DType.uint32]](batch_size * paged_lut_cols)

    var running_offset: UInt32 = 0
    for i in range(batch_size):
        input_row_offsets_h[i] = running_offset
        cache_lengths_h[i] = UInt32(cache_len + i * cache_len_step)
        running_offset += UInt32(seq_len)
    input_row_offsets_h[batch_size] = running_offset

    for i in range(head_dim):
        q_gamma_h[i] = (Float64(i + head_dim) / Float64(head_dim)).cast[dtype]()
        k_gamma_h[i] = (Float64(i + (head_dim // 2)) / Float64(head_dim)).cast[
            dtype
        ]()

    for i in range(max_seq_len * head_dim):
        freqs_h[i] = Scalar[dtype](random_float64(-1, 1).cast[dtype]())

    for i in range(q_elems):
        q_h[i] = Scalar[dtype](random_float64(0, 100).cast[dtype]())

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
            paged_lut_host[bs, page_idx] = UInt32(
                bs * paged_lut_cols + page_idx
            )

    var input_row_offsets_d = ctx.enqueue_create_buffer[DType.uint32](
        batch_size + 1
    )
    var cache_lengths_d = ctx.enqueue_create_buffer[DType.uint32](batch_size)
    var q_gamma_d = ctx.enqueue_create_buffer[dtype](head_dim)
    var k_gamma_d = ctx.enqueue_create_buffer[dtype](head_dim)
    var freqs_d = ctx.enqueue_create_buffer[dtype](max_seq_len * head_dim)
    var q_d = ctx.enqueue_create_buffer[dtype](q_elems)
    var fused_output_d = ctx.enqueue_create_buffer[dtype](q_elems)
    var trial_output_d = ctx.enqueue_create_buffer[dtype](q_elems)
    var paged_lut_d = ctx.enqueue_create_buffer[DType.uint32](
        batch_size * paged_lut_cols
    )

    ctx.enqueue_copy(input_row_offsets_d, input_row_offsets_h)
    ctx.enqueue_copy(cache_lengths_d, cache_lengths_h)
    ctx.enqueue_copy(q_gamma_d, q_gamma_h)
    ctx.enqueue_copy(k_gamma_d, k_gamma_h)
    ctx.enqueue_copy(freqs_d, freqs_h)
    ctx.enqueue_copy(q_d, q_h)
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
    var kv_block_seed_h = alloc[Scalar[dtype]](kv_block_size)
    random(
        LayoutTensor[dtype, Layout.row_major[6](), MutAnyOrigin](
            kv_block_seed_h,
            RuntimeLayout[Layout.row_major[6]()].row_major(kv_block_shape),
        )
    )

    var fused_kv_block_d = ctx.enqueue_create_buffer[dtype](kv_block_size)
    var trial_kv_block_d = ctx.enqueue_create_buffer[dtype](kv_block_size)
    ctx.enqueue_copy(fused_kv_block_d, kv_block_seed_h)
    ctx.enqueue_copy(trial_kv_block_d, kv_block_seed_h)

    var q_layout = row_major(
        (Idx(total_seq_len), Idx[num_q_heads](), Idx[head_dim]())
    )
    var q_tensor = TileTensor(q_d.unsafe_ptr(), q_layout)
    var fused_output_tensor = TileTensor(fused_output_d.unsafe_ptr(), q_layout)
    var trial_output_tensor = TileTensor(trial_output_d.unsafe_ptr(), q_layout)
    var q_gamma_tensor = TileTensor(
        q_gamma_d.unsafe_ptr(), row_major(Idx[head_dim]())
    )
    var k_gamma_tensor = TileTensor(
        k_gamma_d.unsafe_ptr(), row_major(Idx[head_dim]())
    )
    var input_row_offsets_tensor = TileTensor(
        input_row_offsets_d.unsafe_ptr(), row_major(Idx(batch_size + 1))
    )
    var start_pos_tensor = TileTensor(
        cache_lengths_d.unsafe_ptr(), row_major(Idx(batch_size))
    )
    comptime freqs_layout = row_major[max_seq_len, head_dim]()
    var freqs_tensor = TileTensor(freqs_d.unsafe_ptr(), freqs_layout)

    comptime kv_block_layout = Layout.row_major[6]()
    comptime cache_lengths_layout = Layout(UNKNOWN_VALUE)
    var fused_kv_collection = CollectionType(
        LayoutTensor[dtype, kv_block_layout, MutAnyOrigin](
            fused_kv_block_d.unsafe_ptr(),
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
        q_tensor,
        input_row_offsets_tensor,
        start_pos_tensor,
        fused_kv_collection,
        freqs_tensor,
        q_gamma_tensor,
        k_gamma_tensor,
        epsilon,
        weight_offset,
        fused_output_tensor,
    )
    @parameter
    def run_fused(ctx: DeviceContext) raises:
        q_rms_norm_fused_qk_rope_ragged[
            dtype,
            dtype,
            CollectionType,
            interleaved=False,
            target="gpu",
        ](
            q_tensor,
            input_row_offsets_tensor,
            start_pos_tensor,
            fused_kv_collection,
            freqs_tensor,
            q_gamma_tensor,
            k_gamma_tensor,
            epsilon,
            weight_offset,
            UInt32(layer_idx),
            fused_output_tensor,
            ctx,
        )

    @always_inline
    @__copy_capture(
        q_tensor,
        input_row_offsets_tensor,
        start_pos_tensor,
        trial_kv_collection,
        freqs_tensor,
        q_gamma_tensor,
        k_gamma_tensor,
        epsilon,
        weight_offset,
        trial_output_tensor,
        total_seq_len,
    )
    @parameter
    def run_trial_fused(ctx: DeviceContext) raises:
        q_rms_norm_fused_qk_rope_ragged[
            dtype,
            dtype,
            CollectionType,
            interleaved=False,
            target="gpu",
        ](
            q_tensor,
            input_row_offsets_tensor,
            start_pos_tensor,
            trial_kv_collection,
            freqs_tensor,
            q_gamma_tensor,
            k_gamma_tensor,
            epsilon,
            weight_offset,
            UInt32(layer_idx),
            trial_output_tensor,
            ctx,
        )

    @always_inline
    @__copy_capture(
        q_tensor,
        start_pos_tensor,
        trial_kv_collection,
        freqs_tensor,
        q_gamma_tensor,
        k_gamma_tensor,
        epsilon,
        weight_offset,
        trial_output_tensor,
        total_seq_len,
    )
    @parameter
    def run_trial(ctx: DeviceContext) raises:
        comptime if dtype == DType.bfloat16:
            comptime if head_dim == 128:
                if seq_len == 1:
                    comptime block_size = 64
                    comptime warps_per_block = 2
                    comptime total_heads = num_q_heads + num_kv_heads

                    var total_rows = Int(total_seq_len) * total_heads
                    var k_cache = trial_kv_collection.get_key_cache(layer_idx)

                    if cache_len_step == 0:
                        comptime kernel = _gemma3_qk_norm_rope_decode_uniform_kernel[
                            dtype,
                            dtype,
                            CollectionType.CacheType,
                            type_of(q_tensor).LayoutType,
                            type_of(trial_output_tensor).LayoutType,
                            type_of(freqs_tensor).LayoutType,
                            type_of(q_gamma_tensor).LayoutType,
                            type_of(k_gamma_tensor).LayoutType,
                            block_size,
                            warps_per_block,
                        ]
                        ctx.enqueue_function[kernel, kernel](
                            q_tensor,
                            trial_output_tensor,
                            k_cache,
                            freqs_tensor,
                            q_gamma_tensor,
                            k_gamma_tensor,
                            epsilon,
                            weight_offset,
                            total_rows,
                            cache_len,
                            grid_dim=ceildiv(total_rows, warps_per_block * 2),
                            block_dim=block_size,
                        )
                    else:
                        comptime kernel = _gemma3_qk_norm_rope_decode_ragged_kernel[
                            dtype,
                            dtype,
                            CollectionType.CacheType,
                            type_of(q_tensor).LayoutType,
                            type_of(trial_output_tensor).LayoutType,
                            type_of(start_pos_tensor).LayoutType,
                            type_of(freqs_tensor).LayoutType,
                            type_of(q_gamma_tensor).LayoutType,
                            type_of(k_gamma_tensor).LayoutType,
                            block_size,
                            warps_per_block,
                        ]
                        ctx.enqueue_function[kernel, kernel](
                            q_tensor,
                            trial_output_tensor,
                            start_pos_tensor,
                            k_cache,
                            freqs_tensor,
                            q_gamma_tensor,
                            k_gamma_tensor,
                            epsilon,
                            weight_offset,
                            total_rows,
                            grid_dim=ceildiv(total_rows, warps_per_block * 2),
                            block_dim=block_size,
                        )
                else:
                    run_trial_fused(ctx)
            else:
                comptime if head_dim == 256:
                    if seq_len == 1:
                        comptime block_size = 64
                        comptime warps_per_block = 2
                        comptime total_heads = num_q_heads + num_kv_heads

                        var total_rows = Int(total_seq_len) * total_heads
                        var k_cache = trial_kv_collection.get_key_cache(
                            layer_idx
                        )

                        if cache_len_step == 0:
                            comptime kernel = _gemma3_qk_norm_rope_decode_uniform_wide_kernel[
                                dtype,
                                dtype,
                                CollectionType.CacheType,
                                type_of(q_tensor).LayoutType,
                                type_of(trial_output_tensor).LayoutType,
                                type_of(freqs_tensor).LayoutType,
                                type_of(q_gamma_tensor).LayoutType,
                                type_of(k_gamma_tensor).LayoutType,
                                block_size,
                                warps_per_block,
                            ]
                            ctx.enqueue_function[kernel, kernel](
                                q_tensor,
                                trial_output_tensor,
                                k_cache,
                                freqs_tensor,
                                q_gamma_tensor,
                                k_gamma_tensor,
                                epsilon,
                                weight_offset,
                                total_rows,
                                cache_len,
                                grid_dim=ceildiv(total_rows, warps_per_block),
                                block_dim=block_size,
                            )
                        else:
                            comptime kernel = _gemma3_qk_norm_rope_decode_ragged_wide_kernel[
                                dtype,
                                dtype,
                                CollectionType.CacheType,
                                type_of(q_tensor).LayoutType,
                                type_of(trial_output_tensor).LayoutType,
                                type_of(start_pos_tensor).LayoutType,
                                type_of(freqs_tensor).LayoutType,
                                type_of(q_gamma_tensor).LayoutType,
                                type_of(k_gamma_tensor).LayoutType,
                                block_size,
                                warps_per_block,
                            ]
                            ctx.enqueue_function[kernel, kernel](
                                q_tensor,
                                trial_output_tensor,
                                start_pos_tensor,
                                k_cache,
                                freqs_tensor,
                                q_gamma_tensor,
                                k_gamma_tensor,
                                epsilon,
                                weight_offset,
                                total_rows,
                                grid_dim=ceildiv(total_rows, warps_per_block),
                                block_dim=block_size,
                            )
                    else:
                        run_trial_fused(ctx)
                else:
                    run_trial_fused(ctx)
        else:
            run_trial_fused(ctx)

    @always_inline
    @parameter
    def fused_bench(mut bencher: Bencher) raises:
        bencher.iter_custom[run_fused](ctx)

    @always_inline
    @parameter
    def trial_bench(mut bencher: Bencher) raises:
        bencher.iter_custom[run_trial](ctx)

    bench.bench_function[fused_bench](
        BenchId(
            "gemma3_qk_norm_rope_boundary_pair_fused",
            input_id=String(
                dtype,
                "/bs=",
                batch_size,
                "/seq=",
                seq_len,
                "/cache=",
                cache_len,
                "/cache_step=",
                cache_len_step,
                "/pair_only",
            ),
        ),
    )
    bench.bench_function[trial_bench](
        BenchId(
            "gemma3_qk_norm_rope_boundary_pair_trial",
            input_id=String(
                dtype,
                "/bs=",
                batch_size,
                "/seq=",
                seq_len,
                "/cache=",
                cache_len,
                "/cache_step=",
                cache_len_step,
                "/pair_only",
            ),
        ),
    )

    _ = input_row_offsets_d
    _ = cache_lengths_d
    _ = q_gamma_d
    _ = k_gamma_d
    _ = freqs_d
    _ = q_d
    _ = fused_output_d
    _ = trial_output_d
    _ = paged_lut_d
    _ = fused_kv_block_d
    _ = trial_kv_block_d

    input_row_offsets_h.free()
    cache_lengths_h.free()
    q_gamma_h.free()
    k_gamma_h.free()
    freqs_h.free()
    q_h.free()
    paged_lut_h.free()
    kv_block_seed_h.free()


def bench_gemma3_qk_norm_rope_boundary_pair_single_variant[
    dtype: DType,
    head_dim: Int,
    num_q_heads: Int,
    num_kv_heads: Int,
    page_size: Int,
    run_trial_variant: Bool,
](
    ctx: DeviceContext,
    mut bench: Bench,
    batch_size: Int,
    seq_len: Int,
    cache_len: Int,
    cache_len_step: Int,
    trial_baseline_disable_decode_fastpath: Bool,
    verify_results: Bool,
) raises:
    comptime assert (
        dtype == DType.bfloat16
    ), "single pair variant only supports the BF16 boundary trial"
    comptime assert (
        head_dim == 128 or head_dim == 256
    ), "single pair variant only supports head_dim 128 or 256"
    if verify_results:
        raise Error("single pair variant requires --verify=False")
    if trial_baseline_disable_decode_fastpath:
        raise Error(
            "single pair variant requires"
            " trial_baseline_disable_decode_fastpath=False"
        )

    comptime max_seq_len = 2048
    comptime num_layers = 1
    comptime layer_idx = 0
    comptime kv_params = KVCacheStaticParams(
        num_heads=UInt(num_kv_heads), head_size=UInt(head_dim)
    )
    comptime CollectionType = PagedKVCacheCollection[
        dtype, kv_params, page_size
    ]

    var total_seq_len = UInt32(batch_size * seq_len)
    var max_cache_len = cache_len + (batch_size - 1) * cache_len_step
    var max_context_length = seq_len + max_cache_len
    var paged_lut_cols = ceildiv(max_context_length, page_size)
    var num_pages = batch_size * paged_lut_cols
    var q_elems = Int(total_seq_len) * num_q_heads * head_dim

    var input_row_offsets_h = alloc[Scalar[DType.uint32]](batch_size + 1)
    var cache_lengths_h = alloc[Scalar[DType.uint32]](batch_size)
    var q_gamma_h = alloc[Scalar[dtype]](head_dim)
    var k_gamma_h = alloc[Scalar[dtype]](head_dim)
    var freqs_h = alloc[Scalar[dtype]](max_seq_len * head_dim)
    var q_h = alloc[Scalar[dtype]](q_elems)
    var paged_lut_h = alloc[Scalar[DType.uint32]](batch_size * paged_lut_cols)

    var running_offset: UInt32 = 0
    for i in range(batch_size):
        input_row_offsets_h[i] = running_offset
        cache_lengths_h[i] = UInt32(cache_len + i * cache_len_step)
        running_offset += UInt32(seq_len)
    input_row_offsets_h[batch_size] = running_offset

    for i in range(head_dim):
        q_gamma_h[i] = (Float64(i + head_dim) / Float64(head_dim)).cast[dtype]()
        k_gamma_h[i] = (Float64(i + (head_dim // 2)) / Float64(head_dim)).cast[
            dtype
        ]()

    for i in range(max_seq_len * head_dim):
        freqs_h[i] = Scalar[dtype](random_float64(-1, 1).cast[dtype]())

    for i in range(q_elems):
        q_h[i] = Scalar[dtype](random_float64(0, 100).cast[dtype]())

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
            paged_lut_host[bs, page_idx] = UInt32(
                bs * paged_lut_cols + page_idx
            )

    var input_row_offsets_d = ctx.enqueue_create_buffer[DType.uint32](
        batch_size + 1
    )
    var cache_lengths_d = ctx.enqueue_create_buffer[DType.uint32](batch_size)
    var q_gamma_d = ctx.enqueue_create_buffer[dtype](head_dim)
    var k_gamma_d = ctx.enqueue_create_buffer[dtype](head_dim)
    var freqs_d = ctx.enqueue_create_buffer[dtype](max_seq_len * head_dim)
    var q_d = ctx.enqueue_create_buffer[dtype](q_elems)
    var output_d = ctx.enqueue_create_buffer[dtype](q_elems)
    var paged_lut_d = ctx.enqueue_create_buffer[DType.uint32](
        batch_size * paged_lut_cols
    )

    ctx.enqueue_copy(input_row_offsets_d, input_row_offsets_h)
    ctx.enqueue_copy(cache_lengths_d, cache_lengths_h)
    ctx.enqueue_copy(q_gamma_d, q_gamma_h)
    ctx.enqueue_copy(k_gamma_d, k_gamma_h)
    ctx.enqueue_copy(freqs_d, freqs_h)
    ctx.enqueue_copy(q_d, q_h)
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
    var kv_block_seed_h = alloc[Scalar[dtype]](kv_block_size)
    random(
        LayoutTensor[dtype, Layout.row_major[6](), MutAnyOrigin](
            kv_block_seed_h,
            RuntimeLayout[Layout.row_major[6]()].row_major(kv_block_shape),
        )
    )

    var kv_block_d = ctx.enqueue_create_buffer[dtype](kv_block_size)
    ctx.enqueue_copy(kv_block_d, kv_block_seed_h)

    var q_layout = row_major(
        (Idx(total_seq_len), Idx[num_q_heads](), Idx[head_dim]())
    )
    var q_tensor = TileTensor(q_d.unsafe_ptr(), q_layout)
    var output_tensor = TileTensor(output_d.unsafe_ptr(), q_layout)
    var q_gamma_tensor = TileTensor(
        q_gamma_d.unsafe_ptr(), row_major(Idx[head_dim]())
    )
    var k_gamma_tensor = TileTensor(
        k_gamma_d.unsafe_ptr(), row_major(Idx[head_dim]())
    )
    var input_row_offsets_tensor = TileTensor(
        input_row_offsets_d.unsafe_ptr(), row_major(Idx(batch_size + 1))
    )
    var start_pos_tensor = TileTensor(
        cache_lengths_d.unsafe_ptr(), row_major(Idx(batch_size))
    )
    comptime freqs_layout = row_major[max_seq_len, head_dim]()
    var freqs_tensor = TileTensor(freqs_d.unsafe_ptr(), freqs_layout)

    comptime kv_block_layout = Layout.row_major[6]()
    comptime cache_lengths_layout = Layout(UNKNOWN_VALUE)
    var kv_collection = CollectionType(
        LayoutTensor[dtype, kv_block_layout, MutAnyOrigin](
            kv_block_d.unsafe_ptr(),
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

    comptime if run_trial_variant:

        @always_inline
        @__copy_capture(
            q_tensor,
            input_row_offsets_tensor,
            start_pos_tensor,
            kv_collection,
            freqs_tensor,
            q_gamma_tensor,
            k_gamma_tensor,
            epsilon,
            weight_offset,
            output_tensor,
            total_seq_len,
        )
        @parameter
        def run_trial_fused(ctx: DeviceContext) raises:
            q_rms_norm_fused_qk_rope_ragged[
                dtype,
                dtype,
                CollectionType,
                interleaved=False,
                target="gpu",
            ](
                q_tensor,
                input_row_offsets_tensor,
                start_pos_tensor,
                kv_collection,
                freqs_tensor,
                q_gamma_tensor,
                k_gamma_tensor,
                epsilon,
                weight_offset,
                UInt32(layer_idx),
                output_tensor,
                ctx,
            )

        @always_inline
        @__copy_capture(
            q_tensor,
            start_pos_tensor,
            kv_collection,
            freqs_tensor,
            q_gamma_tensor,
            k_gamma_tensor,
            epsilon,
            weight_offset,
            output_tensor,
            total_seq_len,
        )
        @parameter
        def run_trial(ctx: DeviceContext) raises:
            comptime if dtype == DType.bfloat16:
                comptime if head_dim == 128:
                    if seq_len == 1:
                        comptime block_size = 64
                        comptime warps_per_block = 2
                        comptime total_heads = num_q_heads + num_kv_heads

                        var total_rows = Int(total_seq_len) * total_heads
                        var k_cache = kv_collection.get_key_cache(layer_idx)

                        if cache_len_step == 0:
                            comptime kernel = _gemma3_qk_norm_rope_decode_uniform_kernel[
                                dtype,
                                dtype,
                                CollectionType.CacheType,
                                type_of(q_tensor).LayoutType,
                                type_of(output_tensor).LayoutType,
                                type_of(freqs_tensor).LayoutType,
                                type_of(q_gamma_tensor).LayoutType,
                                type_of(k_gamma_tensor).LayoutType,
                                block_size,
                                warps_per_block,
                            ]
                            ctx.enqueue_function[kernel, kernel](
                                q_tensor,
                                output_tensor,
                                k_cache,
                                freqs_tensor,
                                q_gamma_tensor,
                                k_gamma_tensor,
                                epsilon,
                                weight_offset,
                                total_rows,
                                cache_len,
                                grid_dim=ceildiv(
                                    total_rows, warps_per_block * 2
                                ),
                                block_dim=block_size,
                            )
                        else:
                            comptime kernel = _gemma3_qk_norm_rope_decode_ragged_kernel[
                                dtype,
                                dtype,
                                CollectionType.CacheType,
                                type_of(q_tensor).LayoutType,
                                type_of(output_tensor).LayoutType,
                                type_of(start_pos_tensor).LayoutType,
                                type_of(freqs_tensor).LayoutType,
                                type_of(q_gamma_tensor).LayoutType,
                                type_of(k_gamma_tensor).LayoutType,
                                block_size,
                                warps_per_block,
                            ]
                            ctx.enqueue_function[kernel, kernel](
                                q_tensor,
                                output_tensor,
                                start_pos_tensor,
                                k_cache,
                                freqs_tensor,
                                q_gamma_tensor,
                                k_gamma_tensor,
                                epsilon,
                                weight_offset,
                                total_rows,
                                grid_dim=ceildiv(
                                    total_rows, warps_per_block * 2
                                ),
                                block_dim=block_size,
                            )
                    else:
                        run_trial_fused(ctx)
                else:
                    comptime if head_dim == 256:
                        if seq_len == 1:
                            comptime block_size = 64
                            comptime warps_per_block = 2
                            comptime total_heads = num_q_heads + num_kv_heads

                            var total_rows = Int(total_seq_len) * total_heads
                            var k_cache = kv_collection.get_key_cache(layer_idx)

                            if cache_len_step == 0:
                                comptime kernel = _gemma3_qk_norm_rope_decode_uniform_wide_kernel[
                                    dtype,
                                    dtype,
                                    CollectionType.CacheType,
                                    type_of(q_tensor).LayoutType,
                                    type_of(output_tensor).LayoutType,
                                    type_of(freqs_tensor).LayoutType,
                                    type_of(q_gamma_tensor).LayoutType,
                                    type_of(k_gamma_tensor).LayoutType,
                                    block_size,
                                    warps_per_block,
                                ]
                                ctx.enqueue_function[kernel, kernel](
                                    q_tensor,
                                    output_tensor,
                                    k_cache,
                                    freqs_tensor,
                                    q_gamma_tensor,
                                    k_gamma_tensor,
                                    epsilon,
                                    weight_offset,
                                    total_rows,
                                    cache_len,
                                    grid_dim=ceildiv(
                                        total_rows, warps_per_block
                                    ),
                                    block_dim=block_size,
                                )
                            else:
                                comptime kernel = _gemma3_qk_norm_rope_decode_ragged_wide_kernel[
                                    dtype,
                                    dtype,
                                    CollectionType.CacheType,
                                    type_of(q_tensor).LayoutType,
                                    type_of(output_tensor).LayoutType,
                                    type_of(start_pos_tensor).LayoutType,
                                    type_of(freqs_tensor).LayoutType,
                                    type_of(q_gamma_tensor).LayoutType,
                                    type_of(k_gamma_tensor).LayoutType,
                                    block_size,
                                    warps_per_block,
                                ]
                                ctx.enqueue_function[kernel, kernel](
                                    q_tensor,
                                    output_tensor,
                                    start_pos_tensor,
                                    k_cache,
                                    freqs_tensor,
                                    q_gamma_tensor,
                                    k_gamma_tensor,
                                    epsilon,
                                    weight_offset,
                                    total_rows,
                                    grid_dim=ceildiv(
                                        total_rows, warps_per_block
                                    ),
                                    block_dim=block_size,
                                )
                        else:
                            run_trial_fused(ctx)
                    else:
                        run_trial_fused(ctx)
            else:
                run_trial_fused(ctx)

        @always_inline
        @parameter
        def trial_variant_bench(mut bencher: Bencher) raises:
            bencher.iter_custom[run_trial](ctx)

        bench.bench_function[trial_variant_bench](
            BenchId(
                "gemma3_qk_norm_rope_boundary_pair_trial",
                input_id=String(
                    dtype,
                    "/bs=",
                    batch_size,
                    "/seq=",
                    seq_len,
                    "/cache=",
                    cache_len,
                    "/cache_step=",
                    cache_len_step,
                    "/pair_only/trial_only",
                ),
            ),
        )
    else:

        @always_inline
        @__copy_capture(
            q_tensor,
            input_row_offsets_tensor,
            start_pos_tensor,
            kv_collection,
            freqs_tensor,
            q_gamma_tensor,
            k_gamma_tensor,
            epsilon,
            weight_offset,
            output_tensor,
        )
        @parameter
        def run_fused(ctx: DeviceContext) raises:
            q_rms_norm_fused_qk_rope_ragged[
                dtype,
                dtype,
                CollectionType,
                interleaved=False,
                target="gpu",
            ](
                q_tensor,
                input_row_offsets_tensor,
                start_pos_tensor,
                kv_collection,
                freqs_tensor,
                q_gamma_tensor,
                k_gamma_tensor,
                epsilon,
                weight_offset,
                UInt32(layer_idx),
                output_tensor,
                ctx,
            )

        @always_inline
        @parameter
        def fused_variant_bench(mut bencher: Bencher) raises:
            bencher.iter_custom[run_fused](ctx)

        bench.bench_function[fused_variant_bench](
            BenchId(
                "gemma3_qk_norm_rope_boundary_pair_fused",
                input_id=String(
                    dtype,
                    "/bs=",
                    batch_size,
                    "/seq=",
                    seq_len,
                    "/cache=",
                    cache_len,
                    "/cache_step=",
                    cache_len_step,
                    "/pair_only/fused_only",
                ),
            ),
        )

    _ = input_row_offsets_d
    _ = cache_lengths_d
    _ = q_gamma_d
    _ = k_gamma_d
    _ = freqs_d
    _ = q_d
    _ = output_d
    _ = paged_lut_d
    _ = kv_block_d

    input_row_offsets_h.free()
    cache_lengths_h.free()
    q_gamma_h.free()
    k_gamma_h.free()
    freqs_h.free()
    q_h.free()
    paged_lut_h.free()
    kv_block_seed_h.free()


def main() raises:
    comptime dtype = get_defined_dtype["dtype", DType.bfloat16]()
    comptime head_dim = get_defined_int["head_dim", 128]()
    comptime num_q_heads = get_defined_int["num_q_heads", 32]()
    comptime num_kv_heads = get_defined_int["num_kv_heads", 16]()
    comptime page_size = 128

    var batch_size = arg_parse("batch_size", 64)
    var seq_len = arg_parse("seq_len", 1)
    var cache_len = arg_parse("cache_len", 1024)
    var cache_len_step = arg_parse("cache_len_step", 0)
    var trial_baseline_disable_decode_fastpath = arg_parse(
        "trial_baseline_disable_decode_fastpath", False
    )
    var pair_only = arg_parse("pair_only", False)
    var pair_variant = String(arg_parse("pair_variant", "both"))
    var verify_results = arg_parse("verify", True)

    seed(0)

    var bench = Bench(BenchConfig(num_repetitions=1))
    update_bench_config_args(bench)
    with DeviceContext() as ctx:
        if pair_only:
            if pair_variant == "both":
                bench_gemma3_qk_norm_rope_boundary_pair[
                    dtype,
                    head_dim,
                    num_q_heads,
                    num_kv_heads,
                    page_size,
                ](
                    ctx,
                    bench,
                    batch_size,
                    seq_len,
                    cache_len,
                    cache_len_step,
                    trial_baseline_disable_decode_fastpath,
                    verify_results,
                )
            elif pair_variant == "fused":
                bench_gemma3_qk_norm_rope_boundary_pair_single_variant[
                    dtype,
                    head_dim,
                    num_q_heads,
                    num_kv_heads,
                    page_size,
                    run_trial_variant=False,
                ](
                    ctx,
                    bench,
                    batch_size,
                    seq_len,
                    cache_len,
                    cache_len_step,
                    trial_baseline_disable_decode_fastpath,
                    verify_results,
                )
            elif pair_variant == "trial":
                bench_gemma3_qk_norm_rope_boundary_pair_single_variant[
                    dtype,
                    head_dim,
                    num_q_heads,
                    num_kv_heads,
                    page_size,
                    run_trial_variant=True,
                ](
                    ctx,
                    bench,
                    batch_size,
                    seq_len,
                    cache_len,
                    cache_len_step,
                    trial_baseline_disable_decode_fastpath,
                    verify_results,
                )
            else:
                raise Error("pair_variant must be one of: both, fused, trial")
        else:
            if pair_variant != "both":
                raise Error("pair_variant requires --pair_only=True")
            bench_gemma3_qk_norm_rope_boundary[
                dtype,
                head_dim,
                num_q_heads,
                num_kv_heads,
                page_size,
            ](
                ctx,
                bench,
                batch_size,
                seq_len,
                cache_len,
                cache_len_step,
                trial_baseline_disable_decode_fastpath,
                verify_results,
            )

    bench.dump_report()
