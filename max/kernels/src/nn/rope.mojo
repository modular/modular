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

from std.collections import OptionalReg
from std.math import ceildiv, gcd, rsqrt
from std.sys.info import _current_target, align_of, simd_width_of

from std.algorithm.functional import elementwise
from std.complex import ComplexSIMD
from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_idx_uint as block_idx,
    syncwarp,
    thread_idx_uint as thread_idx,
)
from std.gpu.host import DeviceContext, get_gpu_target
from std.gpu.host.info import is_cpu
from std.gpu.memory import AddressSpace
from std.gpu.primitives.grid_controls import PDLLevel, pdl_launch_attributes
import std.gpu.primitives.warp as warp
from std.memory import stack_allocation
from kv_cache.types import KVCacheT, KVCollectionT
from layout import (
    Coord,
    CoordLike,
    Idx,
    RowMajorLayout,
    RuntimeInt,
    TensorLayout,
    TileTensor,
    coord,
)
from nn._ragged_utils import get_batch_from_row_offsets
from nn.fused_qk_rope import (
    _rope_complex_mul_half,
    fused_qk_rope_ragged,
    rope_k_cache,
)
from nn.normalization import _rms_norm_warp_tiling_subkernel, rms_norm_gpu

from std.utils import IndexList
from std.utils.numerics import get_accum_type
from std.utils.static_tuple import StaticTuple


@always_inline
def _rope[
    dtype: DType,
    freq_dtype: DType,
    width: Int,
](val: SIMD[dtype, width], freq: SIMD[freq_dtype, width]) -> SIMD[dtype, width]:
    x_re, x_im = val.cast[freq_dtype]().deinterleave()
    f_re, f_im = freq.deinterleave()
    var r = ComplexSIMD(x_re, x_im) * ComplexSIMD(f_re, f_im)
    return rebind[SIMD[dtype, width]](r.re.interleave(r.im).cast[dtype]())


# In GGUF, weights are organized as real, imag, real, imag, real, imag, …,
# while in safetensors, the data is stored as real, …, real, imag, …, imag.
# This function return the indices for the real and imaginary part.
@always_inline
def get_safetensors_idx(head_dim_idx: Int, head_size: Int) -> Tuple[Int, Int]:
    return (head_dim_idx // 2, head_dim_idx // 2 + head_size // 2)


@always_inline
def get_identity_rope_coeff[width: Int, dtype: DType]() -> SIMD[dtype, width]:
    # Creates a SIMD vector with real parts set to 1 and imaginary parts to
    # 0, effectively making the RoPE transformation an identity operation.
    return rebind[SIMD[dtype, width]](
        SIMD[dtype, width // 2](1).interleave(SIMD[dtype, width // 2](0))
    )


@always_inline
def apply_rope[
    dtype: DType,
    freq_dtype: DType,
    rank: Int,
    width: Int,
    //,
    *,
    interleaved: Bool,
    alignment: Int,
    output_fn: def[width: Int, alignment: Int](
        idx: IndexList[rank], val: SIMD[dtype, width]
    ) capturing -> None,
](
    x: TileTensor[dtype, ...],
    idx: IndexList[rank],
    freq_val: SIMD[freq_dtype, width],
):
    comptime assert rank - 1 >= 0
    var indices = get_safetensors_idx(idx[rank - 1], x.static_shape[rank - 1])
    var pos_re = idx
    var pos_im = idx
    pos_re[rank - 1] = indices[0]
    pos_im[rank - 1] = indices[1]
    comptime width_2 = width // 2

    var val: SIMD[dtype, width]

    comptime if interleaved:
        var coord = Coord(idx)
        val = x.load[width=width, alignment=1](coord)
    else:
        var re_coord = Coord(pos_re)
        var im_coord = Coord(pos_im)
        val = rebind[SIMD[dtype, width]](
            x.load[width=width_2, alignment=1](re_coord).interleave(
                x.load[width=width_2, alignment=1](im_coord)
            )
        )

    var res = _rope(val, freq_val)

    comptime if interleaved:
        output_fn[alignment=alignment](idx, res)
    else:
        output_re, output_im = res.deinterleave()
        output_fn[alignment=alignment](pos_re, output_re)
        output_fn[alignment=alignment](pos_im, output_im)


@always_inline
def rope_ragged[
    dtype: DType,
    freq_dtype: DType,
    *,
    interleaved: Bool,
    target: StaticString,
    output_fn: def[width: Int, alignment: Int](
        idx: IndexList[3], val: SIMD[dtype, width]
    ) capturing -> None,
    mrope_types: Variadic.TypesOfTrait[CoordLike] = Variadic.empty_of_trait[
        CoordLike
    ],
    mrope_section: Optional[Coord[*mrope_types]] = None,
    PositionIdsLayoutType: TensorLayout = RowMajorLayout[
        RuntimeInt[DType.int64], RuntimeInt[DType.int64]
    ],
](
    x: TileTensor[dtype, ...],
    input_row_offsets: TileTensor[DType.uint32, ...],
    start_pos: TileTensor[DType.uint32, ...],
    freqs_cis: TileTensor[freq_dtype, ...],
    context: Optional[DeviceContext],
    position_ids: OptionalReg[
        TileTensor[DType.uint32, PositionIdsLayoutType, ImmutAnyOrigin]
    ] = None,
) raises where (
    input_row_offsets.flat_rank == 1
    and start_pos.flat_rank == 1
    and freqs_cis.flat_rank == 2
):
    comptime assert freqs_cis.LayoutType._shape_types[
        1
    ].is_static_value, "Need static rope_dim for freqs_cis"
    comptime head_size = Int(x.static_shape[2])
    comptime rope_dim = Int(freqs_cis.static_shape[1])
    comptime unroped_dim = head_size - rope_dim
    comptime has_nope = unroped_dim > 0

    @always_inline
    @parameter
    @__copy_capture(x, input_row_offsets, start_pos, freqs_cis)
    def rope_fn[
        width: Int, rank: Int, alignment: Int = 1
    ](idx_arg: IndexList[rank]):
        comptime assert rank == 3, "Invalid rank passed to rope kernel"
        comptime assert freqs_cis.flat_rank >= 2

        comptime if width == 1:
            assert False, (
                "RoPE kernel called with simd width = 1, We should never be"
                " here. This is indicative of an uneven last dimension of"
                " the rope tensor. Ensure the model's head_size is"
                " divisible by the simd width of your target hardware."
            )
            return
        else:
            var idx = rebind[IndexList[3]](idx_arg)

            var global_token_idx = idx[0]

            var batch_idx: Int = get_batch_from_row_offsets(
                input_row_offsets, global_token_idx
            )
            var token_idx = Int(
                UInt32(global_token_idx) - input_row_offsets[batch_idx]
            )
            var head_dim_idx = idx[2]

            # Use position_ids if provided, otherwise fall back to cache calculation
            var post_seq_idx = start_pos[batch_idx] + UInt32(token_idx)

            var position_ids_idx = Int(post_seq_idx)
            if position_ids:
                comptime PIdTensor = type_of(position_ids.value())
                comptime assert PIdTensor.flat_rank == 2
                comptime if mrope_section:
                    var section_idx = 0

                    comptime for i in range(len(mrope_section.value())):
                        comptime val = mrope_section.value()[i].value()
                        if head_dim_idx < val:
                            section_idx = i
                            break
                    position_ids_idx = Int(
                        position_ids.value()[section_idx, global_token_idx]
                    )
                else:
                    position_ids_idx = Int(
                        position_ids.value()[0, global_token_idx]
                    )

            # WARN assumes head_size % simd_width == 0
            # guarded by constrained statement below
            var is_unroped_region = head_dim_idx < unroped_dim

            var f_c_temp: SIMD[freq_dtype, width]

            comptime if has_nope:
                if is_unroped_region:
                    f_c_temp = get_identity_rope_coeff[width, freq_dtype]()
                else:
                    f_c_temp = freqs_cis.load[width=width, alignment=1](
                        coord[freqs_cis.linear_idx_type](
                            (position_ids_idx, head_dim_idx - unroped_dim)
                        )
                    )
            else:
                f_c_temp = freqs_cis.load[width=width, alignment=1](
                    coord[freqs_cis.linear_idx_type](
                        (position_ids_idx, head_dim_idx)
                    )
                )
            apply_rope[
                interleaved=interleaved,
                alignment=alignment,
                output_fn=output_fn,
            ](x, idx, f_c_temp)

    var launch_shape_index_list = IndexList[x.rank]()

    comptime for i in range(x.rank):
        launch_shape_index_list[i] = Int(x.dim(i))

    comptime compile_target = _current_target() if is_cpu[
        target
    ]() else get_gpu_target()
    comptime target_simd_width = simd_width_of[dtype, target=compile_target]()
    comptime kernel_simd_width = gcd(target_simd_width, rope_dim)

    comptime if mrope_section:
        comptime for i in range(len(mrope_section.value())):
            comptime assert (
                mrope_section.value()[i].static_value % kernel_simd_width == 0
            ), "mrope_section must be divisible by rope kernel simd_width"

    comptime assert (
        kernel_simd_width >= 2 and rope_dim % kernel_simd_width == 0
    ), (
        "Rope kernel simd width must be between 2 and rope_dim and divisible by"
        " rope_dim. Ensure the model's head_size is divisible by the simd width"
        " of your target hardware."
    )

    comptime if is_cpu[target]():
        elementwise[func=rope_fn, simd_width=kernel_simd_width, target=target](
            launch_shape_index_list
        )
    else:
        elementwise[func=rope_fn, simd_width=kernel_simd_width, target=target](
            launch_shape_index_list, context.value()
        )


def _rope_k_cache_ragged[
    freq_dtype: DType,
    collection_t: KVCollectionT,
    *,
    interleaved: Bool,
    target: StaticString,
](
    total_seq_len: Int,
    input_row_offsets: TileTensor[DType.uint32, ...],
    kv_collection: collection_t,
    freqs_cis: TileTensor[freq_dtype, ...],
    layer_idx: UInt32,
    context: DeviceContext,
) raises:
    comptime assert not is_cpu[target](), "Only the GPU path is implemented"
    comptime assert (
        input_row_offsets.flat_rank == 1
    ), "input_row_offsets must be rank 1"
    comptime assert freqs_cis.flat_rank == 2, "freqs_cis must be rank 2"
    comptime kv_params = collection_t.CacheType.kv_params
    comptime head_size = Int(kv_params.head_size)
    comptime rope_dim = Int(freqs_cis.static_shape[1])
    comptime assert rope_dim == head_size, (
        "_rope_k_cache_ragged currently expects freqs_cis width to match the "
        "key head size"
    )

    var k_cache = kv_collection.get_key_cache(Int(layer_idx))

    comptime if (
        not interleaved
        and collection_t.CacheType.dtype == DType.bfloat16
        and head_size == 128
    ):
        var total_rows = total_seq_len * Int(kv_params.num_heads)
        var batch_size = Int(input_row_offsets.dim(0)) - 1
        var is_decode_uniform = total_seq_len == batch_size
        comptime default_warps_per_block = 2
        comptime default_block_size = default_warps_per_block * WARP_SIZE
        comptime large_row_warps_per_block = 8
        comptime large_row_block_size = large_row_warps_per_block * WARP_SIZE
        comptime min_large_row_blocks_per_sm = 6
        var large_row_grid_dim = ceildiv(
            total_rows, large_row_warps_per_block * 2
        )
        var min_large_row_blocks = (
            context.default_device_info.sm_count * min_large_row_blocks_per_sm
        )

        if is_decode_uniform:
            if large_row_grid_dim >= min_large_row_blocks:
                comptime kernel = _rope_k_cache_decode_ragged_kernel[
                    freq_dtype,
                    collection_t.CacheType,
                    freqs_cis.LayoutType,
                    large_row_block_size,
                    large_row_warps_per_block,
                ]
                context.enqueue_function[kernel, kernel](
                    k_cache,
                    freqs_cis,
                    total_rows,
                    grid_dim=large_row_grid_dim,
                    block_dim=large_row_block_size,
                    attributes=pdl_launch_attributes(PDLLevel(1)),
                )
            else:
                comptime kernel = _rope_k_cache_decode_ragged_kernel[
                    freq_dtype,
                    collection_t.CacheType,
                    freqs_cis.LayoutType,
                    default_block_size,
                    default_warps_per_block,
                ]
                context.enqueue_function[kernel, kernel](
                    k_cache,
                    freqs_cis,
                    total_rows,
                    grid_dim=ceildiv(total_rows, default_warps_per_block * 2),
                    block_dim=default_block_size,
                    attributes=pdl_launch_attributes(PDLLevel(1)),
                )
        else:
            if large_row_grid_dim >= min_large_row_blocks:
                comptime kernel = _rope_k_cache_ragged_kernel[
                    freq_dtype,
                    collection_t.CacheType,
                    input_row_offsets.LayoutType,
                    freqs_cis.LayoutType,
                    large_row_block_size,
                    large_row_warps_per_block,
                ]
                context.enqueue_function[kernel, kernel](
                    input_row_offsets,
                    k_cache,
                    freqs_cis,
                    total_rows,
                    grid_dim=large_row_grid_dim,
                    block_dim=large_row_block_size,
                    attributes=pdl_launch_attributes(PDLLevel(1)),
                )
            else:
                comptime kernel = _rope_k_cache_ragged_kernel[
                    freq_dtype,
                    collection_t.CacheType,
                    input_row_offsets.LayoutType,
                    freqs_cis.LayoutType,
                    default_block_size,
                    default_warps_per_block,
                ]
                context.enqueue_function[kernel, kernel](
                    input_row_offsets,
                    k_cache,
                    freqs_cis,
                    total_rows,
                    grid_dim=ceildiv(total_rows, default_warps_per_block * 2),
                    block_dim=default_block_size,
                    attributes=pdl_launch_attributes(PDLLevel(1)),
                )
        return

    @always_inline
    @parameter
    @__copy_capture(k_cache, input_row_offsets)
    def rope_fn[
        width: Int, rank: Int, alignment: Int = 1
    ](idx_arg: IndexList[rank]):
        comptime assert rank == 3, "Invalid rank passed to key rope kernel"
        comptime if width == 1:
            return
        else:
            var idx = rebind[IndexList[3]](idx_arg)
            var global_token_idx = idx[0]
            var batch_idx = get_batch_from_row_offsets(
                input_row_offsets, global_token_idx
            )
            var token_idx = Int(
                UInt32(global_token_idx) - input_row_offsets[batch_idx]
            )
            var head_idx = idx[1]
            var head_dim_idx = idx[2]
            var post_seq_idx = k_cache.cache_length(batch_idx) + token_idx
            var freq = freqs_cis.load[width=width, alignment=1](
                Coord(Idx(post_seq_idx), Idx(head_dim_idx))
            )

            rope_k_cache[interleaved=interleaved](
                k_cache,
                batch_idx,
                head_idx,
                post_seq_idx,
                head_dim_idx,
                freq,
                head_size,
            )

    var launch_shape = IndexList[3](
        total_seq_len, Int(kv_params.num_heads), head_size
    )
    comptime kernel_simd_width = gcd(
        simd_width_of[collection_t.CacheType.dtype, target=get_gpu_target()](),
        head_size,
    )
    comptime assert kernel_simd_width >= 2, "invalid simd_width and head size"

    elementwise[func=rope_fn, simd_width=kernel_simd_width, target=target](
        launch_shape, context
    )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(block_size))
)
def _rope_k_cache_ragged_kernel[
    freq_dtype: DType,
    KCacheType: KVCacheT,
    OffsetsLayoutType: TensorLayout,
    FreqLayoutType: TensorLayout,
    block_size: Int,
    warps_per_block: Int,
](
    input_row_offsets: TileTensor[
        DType.uint32, OffsetsLayoutType, MutAnyOrigin
    ],
    k_cache: KCacheType,
    freqs_cis: TileTensor[freq_dtype, FreqLayoutType, MutAnyOrigin],
    total_rows: Int,
):
    comptime assert input_row_offsets.flat_rank == 1
    comptime assert freqs_cis.flat_rank == 2

    comptime cache_dtype = KCacheType.dtype
    comptime num_heads = Int(KCacheType.kv_params.num_heads)
    comptime head_dim = Int(KCacheType.kv_params.head_size)
    comptime simd_width = simd_width_of[cache_dtype]()
    comptime vec_width = simd_width // 2
    comptime accum_type = get_accum_type[cache_dtype]()
    comptime half_warp_size = WARP_SIZE // 2
    comptime assert head_dim == 128, "Only 128-column BF16 key rows are supported"
    comptime assert head_dim == half_warp_size * simd_width
    comptime assert freqs_cis.static_shape[1] == head_dim

    var tid = thread_idx.x
    var warp_idx = tid // UInt(WARP_SIZE)
    var sub_warp_idx = (tid % UInt(WARP_SIZE)) // UInt(half_warp_size)
    var local_tid = tid % UInt(half_warp_size)
    var row = block_idx.x * UInt(warps_per_block * 2) + warp_idx * 2 + sub_warp_idx
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
        var post_seq_idx = k_cache.cache_length(batch_idx) + token_idx
        var re_offset = Int(local_tid) * vec_width
        var im_offset = re_offset + head_dim // 2
        var freq_offset = Int(local_tid) * simd_width
        var k_re = k_cache.load[width=vec_width](
            batch_idx, head_idx, post_seq_idx, re_offset
        ).cast[accum_type]()
        var k_im = k_cache.load[width=vec_width](
            batch_idx, head_idx, post_seq_idx, im_offset
        ).cast[accum_type]()
        var freq = freqs_cis.load[width=simd_width, alignment=1](
            Coord(Idx(post_seq_idx), Idx(freq_offset))
        )
        var k_rope = _rope_complex_mul_half[
            accum_type,
            freq_dtype,
            vec_width,
            simd_width,
        ](k_re, k_im, freq)
        k_cache.store(
            batch_idx,
            head_idx,
            post_seq_idx,
            re_offset,
            k_rope[0].cast[cache_dtype](),
        )
        k_cache.store(
            batch_idx,
            head_idx,
            post_seq_idx,
            im_offset,
            k_rope[1].cast[cache_dtype](),
        )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(block_size))
)
def _rope_k_cache_decode_ragged_kernel[
    freq_dtype: DType,
    KCacheType: KVCacheT,
    FreqLayoutType: TensorLayout,
    block_size: Int,
    warps_per_block: Int,
](
    k_cache: KCacheType,
    freqs_cis: TileTensor[freq_dtype, FreqLayoutType, MutAnyOrigin],
    total_rows: Int,
):
    comptime assert freqs_cis.flat_rank == 2

    comptime cache_dtype = KCacheType.dtype
    comptime num_heads = Int(KCacheType.kv_params.num_heads)
    comptime head_dim = Int(KCacheType.kv_params.head_size)
    comptime simd_width = simd_width_of[cache_dtype]()
    comptime vec_width = simd_width // 2
    comptime accum_type = get_accum_type[cache_dtype]()
    comptime half_warp_size = WARP_SIZE // 2
    comptime assert head_dim == 128, "Only 128-column BF16 key rows are supported"
    comptime assert head_dim == half_warp_size * simd_width
    comptime assert freqs_cis.static_shape[1] == head_dim

    var tid = thread_idx.x
    var warp_idx = tid // UInt(WARP_SIZE)
    var sub_warp_idx = (tid % UInt(WARP_SIZE)) // UInt(half_warp_size)
    var local_tid = tid % UInt(half_warp_size)
    var row = block_idx.x * UInt(warps_per_block * 2) + warp_idx * 2 + sub_warp_idx
    if row < UInt(total_rows):
        var flat_row = Int(row)
        var global_token_idx = flat_row // num_heads
        var head_idx = flat_row % num_heads
        var batch_idx = global_token_idx
        var post_seq_idx = k_cache.cache_length(batch_idx)
        var re_offset = Int(local_tid) * vec_width
        var im_offset = re_offset + head_dim // 2
        var freq_offset = Int(local_tid) * simd_width
        var k_re = k_cache.load[width=vec_width](
            batch_idx, head_idx, post_seq_idx, re_offset
        ).cast[accum_type]()
        var k_im = k_cache.load[width=vec_width](
            batch_idx, head_idx, post_seq_idx, im_offset
        ).cast[accum_type]()
        var freq = freqs_cis.load[width=simd_width, alignment=1](
            Coord(Idx(post_seq_idx), Idx(freq_offset))
        )
        var k_rope = _rope_complex_mul_half[
            accum_type,
            freq_dtype,
            vec_width,
            simd_width,
        ](k_re, k_im, freq)
        k_cache.store(
            batch_idx,
            head_idx,
            post_seq_idx,
            re_offset,
            k_rope[0].cast[cache_dtype](),
        )
        k_cache.store(
            batch_idx,
            head_idx,
            post_seq_idx,
            im_offset,
            k_rope[1].cast[cache_dtype](),
        )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(block_size))
)
def _k_rms_norm_rope_ragged_kernel[
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
def _k_rms_norm_rope_decode_ragged_kernel[
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


def _k_rms_norm_rope_ragged_wide_kernel[
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


def k_rms_norm_rope_ragged[
    dtype: DType,
    freq_dtype: DType,
    collection_t: KVCollectionT,
    *,
    interleaved: Bool,
    target: StaticString,
](
    total_seq_len: Int,
    input_row_offsets: TileTensor[DType.uint32, ...],
    kv_collection: collection_t,
    freqs_cis: TileTensor[freq_dtype, ...],
    gamma: TileTensor[dtype, ...],
    epsilon: Scalar[dtype],
    weight_offset: Scalar[dtype],
    layer_idx: UInt32,
    context: DeviceContext,
) raises:
    comptime assert not is_cpu[target](), "Only the GPU path is implemented"
    comptime assert (
        not interleaved
    ), "k_rms_norm_rope_ragged currently supports non-interleaved RoPE only"
    comptime assert (
        input_row_offsets.flat_rank == 1
    ), "input_row_offsets must be rank 1"
    comptime assert freqs_cis.flat_rank == 2, "freqs_cis must be rank 2"
    comptime assert gamma.flat_rank == 1, "gamma must be rank 1"
    comptime head_dim = Int(collection_t.CacheType.kv_params.head_size)
    comptime assert dtype == DType.bfloat16, (
        "k_rms_norm_rope_ragged currently supports BF16 inputs only"
    )
    comptime assert collection_t.CacheType.dtype == DType.bfloat16, (
        "k_rms_norm_rope_ragged currently supports BF16 key cache only"
    )
    comptime assert head_dim == 128 or head_dim == 256, (
        "k_rms_norm_rope_ragged currently supports 128- and 256-column keys only"
    )
    comptime assert gamma.static_shape[0] == head_dim
    comptime assert freqs_cis.static_shape[1] == head_dim

    if total_seq_len == 0:
        return

    var total_rows = total_seq_len * Int(collection_t.CacheType.kv_params.num_heads)
    var batch_size = Int(input_row_offsets.dim(0)) - 1
    var is_decode_uniform = total_seq_len == batch_size
    comptime default_warps_per_block = 2
    comptime default_block_size = default_warps_per_block * WARP_SIZE
    comptime large_row_warps_per_block = 8
    comptime large_row_block_size = large_row_warps_per_block * WARP_SIZE
    comptime min_large_row_blocks_per_sm = 5
    var large_row_grid_dim = ceildiv(total_rows, large_row_warps_per_block * 2)
    var min_large_row_blocks = (
        context.default_device_info.sm_count * min_large_row_blocks_per_sm
    )
    var k_cache = kv_collection.get_key_cache(Int(layer_idx))

    comptime if head_dim == 128:
        if is_decode_uniform:
            if large_row_grid_dim >= min_large_row_blocks:
                comptime kernel = _k_rms_norm_rope_decode_ragged_kernel[
                    dtype,
                    freq_dtype,
                    collection_t.CacheType,
                    freqs_cis.LayoutType,
                    gamma.LayoutType,
                    large_row_block_size,
                    large_row_warps_per_block,
                ]
                context.enqueue_function[kernel, kernel](
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
                comptime kernel = _k_rms_norm_rope_decode_ragged_kernel[
                    dtype,
                    freq_dtype,
                    collection_t.CacheType,
                    freqs_cis.LayoutType,
                    gamma.LayoutType,
                    default_block_size,
                    default_warps_per_block,
                ]
                context.enqueue_function[kernel, kernel](
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
                comptime kernel = _k_rms_norm_rope_ragged_kernel[
                    dtype,
                    freq_dtype,
                    collection_t.CacheType,
                    input_row_offsets.LayoutType,
                    freqs_cis.LayoutType,
                    gamma.LayoutType,
                    large_row_block_size,
                    large_row_warps_per_block,
                ]
                context.enqueue_function[kernel, kernel](
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
                comptime kernel = _k_rms_norm_rope_ragged_kernel[
                    dtype,
                    freq_dtype,
                    collection_t.CacheType,
                    input_row_offsets.LayoutType,
                    freqs_cis.LayoutType,
                    gamma.LayoutType,
                    default_block_size,
                    default_warps_per_block,
                ]
                context.enqueue_function[kernel, kernel](
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
        large_row_grid_dim = ceildiv(total_rows, large_row_warps_per_block)
        if large_row_grid_dim >= min_large_row_blocks:
            comptime kernel = _k_rms_norm_rope_ragged_wide_kernel[
                dtype,
                freq_dtype,
                collection_t.CacheType,
                input_row_offsets.LayoutType,
                freqs_cis.LayoutType,
                gamma.LayoutType,
                large_row_block_size,
                large_row_warps_per_block,
            ]
            context.enqueue_function[kernel, kernel](
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
            comptime kernel = _k_rms_norm_rope_ragged_wide_kernel[
                dtype,
                freq_dtype,
                collection_t.CacheType,
                input_row_offsets.LayoutType,
                freqs_cis.LayoutType,
                gamma.LayoutType,
                default_block_size,
                default_warps_per_block,
            ]
            context.enqueue_function[kernel, kernel](
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


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(block_size))
)
def _q_rms_norm_rope_ragged_kernel[
    dtype: DType,
    freq_dtype: DType,
    QLayoutType: TensorLayout,
    OutputLayoutType: TensorLayout,
    OffsetsLayoutType: TensorLayout,
    StartPosLayoutType: TensorLayout,
    FreqLayoutType: TensorLayout,
    GammaLayoutType: TensorLayout,
    block_size: Int,
    warps_per_block: Int,
](
    x: TileTensor[dtype, QLayoutType, MutAnyOrigin],
    output: TileTensor[mut=True, dtype, OutputLayoutType, MutAnyOrigin],
    input_row_offsets: TileTensor[
        DType.uint32, OffsetsLayoutType, MutAnyOrigin
    ],
    start_pos: TileTensor[DType.uint32, StartPosLayoutType, MutAnyOrigin],
    freqs_cis: TileTensor[freq_dtype, FreqLayoutType, MutAnyOrigin],
    gamma: TileTensor[dtype, GammaLayoutType, MutAnyOrigin],
    epsilon: Scalar[dtype],
    weight_offset: Scalar[dtype],
    num_rows: Int,
):
    comptime assert x.flat_rank == 3
    comptime assert output.flat_rank == 3
    comptime assert input_row_offsets.flat_rank == 1
    comptime assert start_pos.flat_rank == 1
    comptime assert freqs_cis.flat_rank == 2
    comptime assert gamma.flat_rank == 1

    comptime simd_width = simd_width_of[dtype]()
    comptime vec_width = simd_width // 2
    comptime align = align_of[SIMD[dtype, vec_width]]()
    comptime accum_type = get_accum_type[dtype]()
    comptime half_warp_size = WARP_SIZE // 2
    comptime num_q_heads = x.static_shape[1]
    comptime head_dim = x.static_shape[2]

    comptime assert head_dim == 128, "Only 128-column BF16 query rows are supported"
    comptime assert head_dim == half_warp_size * simd_width
    comptime assert freqs_cis.static_shape[1] == head_dim

    var tid = thread_idx.x
    var warp_idx = tid // UInt(WARP_SIZE)
    var sub_warp_idx = (tid % UInt(WARP_SIZE)) // UInt(half_warp_size)
    var local_tid = tid % UInt(half_warp_size)
    var row = block_idx.x * UInt(warps_per_block * 2) + warp_idx * 2 + sub_warp_idx
    if row < UInt(num_rows):
        var row_int = Int(row)
        var global_token_idx = row_int // num_q_heads
        var head_idx = row_int % num_q_heads
        var batch_idx = get_batch_from_row_offsets(
            input_row_offsets, global_token_idx
        )
        var token_idx = Int(
            UInt32(global_token_idx) - input_row_offsets[batch_idx]
        )
        var position = Int(start_pos[batch_idx] + UInt32(token_idx))

        var re_offset = Int(local_tid) * vec_width
        var im_offset = re_offset + head_dim // 2
        var freq_offset = Int(local_tid) * simd_width

        var q_re = x.load[width=vec_width, alignment=align](
            Coord(Idx(global_token_idx), Idx(head_idx), Idx(re_offset))
        )
        var q_im = x.load[width=vec_width, alignment=align](
            Coord(Idx(global_token_idx), Idx(head_idx), Idx(im_offset))
        )
        var thread_m2 = (q_re.cast[accum_type]() ** 2).reduce_add() + (
            q_im.cast[accum_type]() ** 2
        ).reduce_add()
        var row_m2 = warp.lane_group_sum[num_lanes=half_warp_size](thread_m2)
        var norm_factor = rsqrt(
            (row_m2 / Scalar[accum_type](head_dim)) + epsilon.cast[accum_type]()
        )

        var gamma_re = gamma.load[width=vec_width, alignment=align](
            Coord(Idx(re_offset))
        )
        var gamma_im = gamma.load[width=vec_width, alignment=align](
            Coord(Idx(im_offset))
        )
        var weight_offset_accum = weight_offset.cast[accum_type]()
        var norm_re = (
            q_re.cast[accum_type]()
            * norm_factor
            * (gamma_re.cast[accum_type]() + weight_offset_accum)
        ).cast[dtype]()
        var norm_im = (
            q_im.cast[accum_type]()
            * norm_factor
            * (gamma_im.cast[accum_type]() + weight_offset_accum)
        ).cast[dtype]()

        var freq = freqs_cis.load[width=simd_width, alignment=1](
            Coord(Idx(position), Idx(freq_offset))
        )
        var rope_val = _rope_complex_mul_half[
            dtype,
            freq_dtype,
            vec_width,
            simd_width,
        ](norm_re, norm_im, freq)

        output.store[alignment=align](
            Coord(Idx(global_token_idx), Idx(head_idx), Idx(re_offset)),
            rope_val[0],
        )
        output.store[alignment=align](
            Coord(Idx(global_token_idx), Idx(head_idx), Idx(im_offset)),
            rope_val[1],
        )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(block_size))
)
def _q_rms_norm_rope_decode_ragged_kernel[
    dtype: DType,
    freq_dtype: DType,
    QLayoutType: TensorLayout,
    OutputLayoutType: TensorLayout,
    StartPosLayoutType: TensorLayout,
    FreqLayoutType: TensorLayout,
    GammaLayoutType: TensorLayout,
    block_size: Int,
    warps_per_block: Int,
](
    x: TileTensor[dtype, QLayoutType, MutAnyOrigin],
    output: TileTensor[mut=True, dtype, OutputLayoutType, MutAnyOrigin],
    start_pos: TileTensor[DType.uint32, StartPosLayoutType, MutAnyOrigin],
    freqs_cis: TileTensor[freq_dtype, FreqLayoutType, MutAnyOrigin],
    gamma: TileTensor[dtype, GammaLayoutType, MutAnyOrigin],
    epsilon: Scalar[dtype],
    weight_offset: Scalar[dtype],
    num_rows: Int,
):
    comptime assert x.flat_rank == 3
    comptime assert output.flat_rank == 3
    comptime assert start_pos.flat_rank == 1
    comptime assert freqs_cis.flat_rank == 2
    comptime assert gamma.flat_rank == 1

    comptime simd_width = simd_width_of[dtype]()
    comptime vec_width = simd_width // 2
    comptime align = align_of[SIMD[dtype, vec_width]]()
    comptime accum_type = get_accum_type[dtype]()
    comptime half_warp_size = WARP_SIZE // 2
    comptime num_q_heads = x.static_shape[1]
    comptime head_dim = x.static_shape[2]

    comptime assert head_dim == 128, "Only 128-column BF16 query rows are supported"
    comptime assert head_dim == half_warp_size * simd_width
    comptime assert freqs_cis.static_shape[1] == head_dim

    var tid = thread_idx.x
    var warp_idx = tid // UInt(WARP_SIZE)
    var sub_warp_idx = (tid % UInt(WARP_SIZE)) // UInt(half_warp_size)
    var local_tid = tid % UInt(half_warp_size)
    var row = block_idx.x * UInt(warps_per_block * 2) + warp_idx * 2 + sub_warp_idx
    if row < UInt(num_rows):
        var row_int = Int(row)
        var global_token_idx = row_int // num_q_heads
        var head_idx = row_int % num_q_heads
        var batch_idx = global_token_idx
        var position = Int(start_pos[batch_idx])

        var re_offset = Int(local_tid) * vec_width
        var im_offset = re_offset + head_dim // 2
        var freq_offset = Int(local_tid) * simd_width

        var q_re = x.load[width=vec_width, alignment=align](
            Coord(Idx(global_token_idx), Idx(head_idx), Idx(re_offset))
        )
        var q_im = x.load[width=vec_width, alignment=align](
            Coord(Idx(global_token_idx), Idx(head_idx), Idx(im_offset))
        )
        var thread_m2 = (q_re.cast[accum_type]() ** 2).reduce_add() + (
            q_im.cast[accum_type]() ** 2
        ).reduce_add()
        var row_m2 = warp.lane_group_sum[num_lanes=half_warp_size](thread_m2)
        var norm_factor = rsqrt(
            (row_m2 / Scalar[accum_type](head_dim)) + epsilon.cast[accum_type]()
        )

        var gamma_re = gamma.load[width=vec_width, alignment=align](
            Coord(Idx(re_offset))
        )
        var gamma_im = gamma.load[width=vec_width, alignment=align](
            Coord(Idx(im_offset))
        )
        var weight_offset_accum = weight_offset.cast[accum_type]()
        var norm_re = (
            q_re.cast[accum_type]()
            * norm_factor
            * (gamma_re.cast[accum_type]() + weight_offset_accum)
        ).cast[dtype]()
        var norm_im = (
            q_im.cast[accum_type]()
            * norm_factor
            * (gamma_im.cast[accum_type]() + weight_offset_accum)
        ).cast[dtype]()

        var freq = freqs_cis.load[width=simd_width, alignment=1](
            Coord(Idx(position), Idx(freq_offset))
        )
        var rope_val = _rope_complex_mul_half[
            dtype,
            freq_dtype,
            vec_width,
            simd_width,
        ](norm_re, norm_im, freq)

        output.store[alignment=align](
            Coord(Idx(global_token_idx), Idx(head_idx), Idx(re_offset)),
            rope_val[0],
        )
        output.store[alignment=align](
            Coord(Idx(global_token_idx), Idx(head_idx), Idx(im_offset)),
            rope_val[1],
        )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(block_size))
)
def _q_rms_norm_rope_ragged_wide_kernel[
    dtype: DType,
    freq_dtype: DType,
    QLayoutType: TensorLayout,
    OutputLayoutType: TensorLayout,
    OffsetsLayoutType: TensorLayout,
    StartPosLayoutType: TensorLayout,
    FreqLayoutType: TensorLayout,
    GammaLayoutType: TensorLayout,
    block_size: Int,
    warps_per_block: Int,
](
    x: TileTensor[dtype, QLayoutType, MutAnyOrigin],
    output: TileTensor[mut=True, dtype, OutputLayoutType, MutAnyOrigin],
    input_row_offsets: TileTensor[
        DType.uint32, OffsetsLayoutType, MutAnyOrigin
    ],
    start_pos: TileTensor[DType.uint32, StartPosLayoutType, MutAnyOrigin],
    freqs_cis: TileTensor[freq_dtype, FreqLayoutType, MutAnyOrigin],
    gamma: TileTensor[dtype, GammaLayoutType, MutAnyOrigin],
    epsilon: Scalar[dtype],
    weight_offset: Scalar[dtype],
    num_rows: Int,
):
    comptime assert x.flat_rank == 3
    comptime assert output.flat_rank == 3
    comptime assert input_row_offsets.flat_rank == 1
    comptime assert start_pos.flat_rank == 1
    comptime assert freqs_cis.flat_rank == 2
    comptime assert gamma.flat_rank == 1

    comptime simd_width = simd_width_of[dtype]()
    comptime wide_align = align_of[SIMD[dtype, simd_width]]()
    comptime accum_type = get_accum_type[dtype]()
    comptime num_q_heads = x.static_shape[1]
    comptime head_dim = x.static_shape[2]

    comptime assert head_dim == 256, "Only 256-column BF16 query rows are supported"
    comptime assert freqs_cis.static_shape[1] == head_dim
    comptime assert gamma.static_shape[0] == head_dim
    comptime assert head_dim == WARP_SIZE * simd_width

    var shared_norm = stack_allocation[
        warps_per_block * head_dim,
        Scalar[dtype],
        address_space=AddressSpace.SHARED,
    ]()
    var shared_position = stack_allocation[
        warps_per_block,
        Int32,
        address_space=AddressSpace.SHARED,
    ]()

    var tid = thread_idx.x
    var warp_idx = tid // UInt(WARP_SIZE)
    var local_tid = tid % UInt(WARP_SIZE)
    var row = block_idx.x * UInt(warps_per_block) + warp_idx
    var col = local_tid * UInt(simd_width)

    if row < UInt(num_rows):
        var flat_row = Int(row)
        var global_token_idx = flat_row // num_q_heads
        var head_idx = flat_row % num_q_heads

        if local_tid == 0:
            var batch_idx = get_batch_from_row_offsets(
                input_row_offsets, global_token_idx
            )
            var token_idx = Int(
                UInt32(global_token_idx) - input_row_offsets[batch_idx]
            )
            shared_position[Int(warp_idx)] = Int32(
                Int(start_pos[batch_idx] + UInt32(token_idx))
            )
        syncwarp()

        var position = Int(shared_position[Int(warp_idx)])
        var epsilon_accum = epsilon.cast[accum_type]()
        var weight_offset_accum = weight_offset.cast[accum_type]()
        var q_vec = x.load[width=simd_width, alignment=wide_align](
            Coord(Idx(global_token_idx), Idx(head_idx), Idx(Int(col)))
        )
        var thread_m2 = (q_vec.cast[accum_type]() ** 2).reduce_add()
        var row_m2 = warp.sum(thread_m2)
        var norm_factor = rsqrt(
            (row_m2 / Scalar[accum_type](head_dim)) + epsilon_accum
        )
        var gamma_val = gamma.load[width=simd_width, alignment=wide_align](
            Coord(Idx(Int(col)))
        )
        var norm_val = (
            q_vec.cast[accum_type]()
            * norm_factor
            * (gamma_val.cast[accum_type]() + weight_offset_accum)
        ).cast[dtype]()
        var shared_base = Int(warp_idx) * head_dim
        shared_norm.store[width=simd_width, alignment=wide_align](
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
            var freq = freqs_cis.load[width=simd_width * 2, alignment=1](
                Coord(Idx(position), Idx(re_offset * 2))
            )
            var rope_val = _rope_complex_mul_half[
                dtype,
                freq_dtype,
                simd_width,
                simd_width * 2,
            ](rope_re, rope_im, freq)

            output.store[alignment=wide_align](
                Coord(Idx(global_token_idx), Idx(head_idx), Idx(re_offset)),
                rope_val[0],
            )
            output.store[alignment=wide_align](
                Coord(Idx(global_token_idx), Idx(head_idx), Idx(im_offset)),
                rope_val[1],
            )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(block_size))
)
def _q_rms_norm_rope_decode_ragged_wide_kernel[
    dtype: DType,
    freq_dtype: DType,
    QLayoutType: TensorLayout,
    OutputLayoutType: TensorLayout,
    StartPosLayoutType: TensorLayout,
    FreqLayoutType: TensorLayout,
    GammaLayoutType: TensorLayout,
    block_size: Int,
    warps_per_block: Int,
](
    x: TileTensor[dtype, QLayoutType, MutAnyOrigin],
    output: TileTensor[mut=True, dtype, OutputLayoutType, MutAnyOrigin],
    start_pos: TileTensor[DType.uint32, StartPosLayoutType, MutAnyOrigin],
    freqs_cis: TileTensor[freq_dtype, FreqLayoutType, MutAnyOrigin],
    gamma: TileTensor[dtype, GammaLayoutType, MutAnyOrigin],
    epsilon: Scalar[dtype],
    weight_offset: Scalar[dtype],
    num_rows: Int,
):
    comptime assert x.flat_rank == 3
    comptime assert output.flat_rank == 3
    comptime assert start_pos.flat_rank == 1
    comptime assert freqs_cis.flat_rank == 2
    comptime assert gamma.flat_rank == 1

    comptime simd_width = simd_width_of[dtype]()
    comptime wide_align = align_of[SIMD[dtype, simd_width]]()
    comptime accum_type = get_accum_type[dtype]()
    comptime num_q_heads = x.static_shape[1]
    comptime head_dim = x.static_shape[2]

    comptime assert head_dim == 256, "Only 256-column BF16 query rows are supported"
    comptime assert freqs_cis.static_shape[1] == head_dim
    comptime assert gamma.static_shape[0] == head_dim
    comptime assert head_dim == WARP_SIZE * simd_width

    var shared_norm = stack_allocation[
        warps_per_block * head_dim,
        Scalar[dtype],
        address_space=AddressSpace.SHARED,
    ]()
    var shared_position = stack_allocation[
        warps_per_block,
        Int32,
        address_space=AddressSpace.SHARED,
    ]()

    var tid = thread_idx.x
    var warp_idx = tid // UInt(WARP_SIZE)
    var local_tid = tid % UInt(WARP_SIZE)
    var row = block_idx.x * UInt(warps_per_block) + warp_idx
    var col = local_tid * UInt(simd_width)

    if row < UInt(num_rows):
        var flat_row = Int(row)
        var global_token_idx = flat_row // num_q_heads
        var head_idx = flat_row % num_q_heads

        if local_tid == 0:
            shared_position[Int(warp_idx)] = Int32(
                Int(start_pos[global_token_idx])
            )
        syncwarp()

        var position = Int(shared_position[Int(warp_idx)])
        var epsilon_accum = epsilon.cast[accum_type]()
        var weight_offset_accum = weight_offset.cast[accum_type]()
        var q_vec = x.load[width=simd_width, alignment=wide_align](
            Coord(Idx(global_token_idx), Idx(head_idx), Idx(Int(col)))
        )
        var thread_m2 = (q_vec.cast[accum_type]() ** 2).reduce_add()
        var row_m2 = warp.sum(thread_m2)
        var norm_factor = rsqrt(
            (row_m2 / Scalar[accum_type](head_dim)) + epsilon_accum
        )
        var gamma_val = gamma.load[width=simd_width, alignment=wide_align](
            Coord(Idx(Int(col)))
        )
        var norm_val = (
            q_vec.cast[accum_type]()
            * norm_factor
            * (gamma_val.cast[accum_type]() + weight_offset_accum)
        ).cast[dtype]()
        var shared_base = Int(warp_idx) * head_dim
        shared_norm.store[width=simd_width, alignment=wide_align](
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
            var freq = freqs_cis.load[width=simd_width * 2, alignment=1](
                Coord(Idx(position), Idx(re_offset * 2))
            )
            var rope_val = _rope_complex_mul_half[
                dtype,
                freq_dtype,
                simd_width,
                simd_width * 2,
            ](rope_re, rope_im, freq)

            output.store[alignment=wide_align](
                Coord(Idx(global_token_idx), Idx(head_idx), Idx(re_offset)),
                rope_val[0],
            )
            output.store[alignment=wide_align](
                Coord(Idx(global_token_idx), Idx(head_idx), Idx(im_offset)),
                rope_val[1],
            )


def q_rms_norm_rope_ragged[
    dtype: DType,
    freq_dtype: DType,
    *,
    target: StaticString,
](
    x: TileTensor[dtype, ...],
    input_row_offsets: TileTensor[DType.uint32, ...],
    start_pos: TileTensor[DType.uint32, ...],
    freqs_cis: TileTensor[freq_dtype, ...],
    gamma: TileTensor[dtype, ...],
    epsilon: Scalar[dtype],
    weight_offset: Scalar[dtype],
    output: TileTensor[mut=True, dtype, ...],
    context: DeviceContext,
) raises:
    comptime assert not is_cpu[target](), "Only the GPU path is implemented"
    comptime assert x.flat_rank == 3
    comptime assert output.flat_rank == 3
    comptime assert input_row_offsets.flat_rank == 1
    comptime assert start_pos.flat_rank == 1
    comptime assert freqs_cis.flat_rank == 2
    comptime assert gamma.flat_rank == 1
    comptime head_dim = Int(x.static_shape[2])

    comptime if head_dim == 128 and dtype == DType.bfloat16:
        comptime num_q_heads = x.static_shape[1]
        var total_tokens = Int(x.dim(0))
        var is_decode_uniform = total_tokens == Int(start_pos.dim(0))
        var num_rows = total_tokens * num_q_heads
        comptime sm_count = context.default_device_info.sm_count
        comptime default_warps_per_block = 2
        comptime default_block_size = default_warps_per_block * WARP_SIZE
        comptime large_row_warps_per_block = 8
        comptime large_row_block_size = large_row_warps_per_block * WARP_SIZE
        comptime min_large_row_blocks_per_sm = 5

        var large_row_grid_dim = ceildiv(num_rows, large_row_warps_per_block * 2)
        if is_decode_uniform:
            if large_row_grid_dim >= sm_count * min_large_row_blocks_per_sm:
                comptime kernel = _q_rms_norm_rope_decode_ragged_kernel[
                    dtype,
                    freq_dtype,
                    x.LayoutType,
                    output.LayoutType,
                    start_pos.LayoutType,
                    freqs_cis.LayoutType,
                    gamma.LayoutType,
                    large_row_block_size,
                    large_row_warps_per_block,
                ]
                context.enqueue_function[kernel, kernel](
                    x,
                    output,
                    start_pos,
                    freqs_cis,
                    gamma,
                    epsilon,
                    weight_offset,
                    num_rows,
                    grid_dim=large_row_grid_dim,
                    block_dim=large_row_block_size,
                )
            else:
                comptime kernel = _q_rms_norm_rope_decode_ragged_kernel[
                    dtype,
                    freq_dtype,
                    x.LayoutType,
                    output.LayoutType,
                    start_pos.LayoutType,
                    freqs_cis.LayoutType,
                    gamma.LayoutType,
                    default_block_size,
                    default_warps_per_block,
                ]
                context.enqueue_function[kernel, kernel](
                    x,
                    output,
                    start_pos,
                    freqs_cis,
                    gamma,
                    epsilon,
                    weight_offset,
                    num_rows,
                    grid_dim=ceildiv(num_rows, default_warps_per_block * 2),
                    block_dim=default_block_size,
                )
        else:
            if large_row_grid_dim >= sm_count * min_large_row_blocks_per_sm:
                comptime kernel = _q_rms_norm_rope_ragged_kernel[
                    dtype,
                    freq_dtype,
                    x.LayoutType,
                    output.LayoutType,
                    input_row_offsets.LayoutType,
                    start_pos.LayoutType,
                    freqs_cis.LayoutType,
                    gamma.LayoutType,
                    large_row_block_size,
                    large_row_warps_per_block,
                ]
                context.enqueue_function[kernel, kernel](
                    x,
                    output,
                    input_row_offsets,
                    start_pos,
                    freqs_cis,
                    gamma,
                    epsilon,
                    weight_offset,
                    num_rows,
                    grid_dim=large_row_grid_dim,
                    block_dim=large_row_block_size,
                )
            else:
                comptime kernel = _q_rms_norm_rope_ragged_kernel[
                    dtype,
                    freq_dtype,
                    x.LayoutType,
                    output.LayoutType,
                    input_row_offsets.LayoutType,
                    start_pos.LayoutType,
                    freqs_cis.LayoutType,
                    gamma.LayoutType,
                    default_block_size,
                    default_warps_per_block,
                ]
                context.enqueue_function[kernel, kernel](
                    x,
                    output,
                    input_row_offsets,
                    start_pos,
                    freqs_cis,
                    gamma,
                    epsilon,
                    weight_offset,
                    num_rows,
                    grid_dim=ceildiv(num_rows, default_warps_per_block * 2),
                    block_dim=default_block_size,
                )
    elif head_dim == 256 and dtype == DType.bfloat16:
        comptime num_q_heads = x.static_shape[1]
        var total_tokens = Int(x.dim(0))
        var is_decode_uniform = total_tokens == Int(start_pos.dim(0))
        var num_rows = total_tokens * num_q_heads
        comptime sm_count = context.default_device_info.sm_count
        comptime default_warps_per_block = 2
        comptime default_block_size = default_warps_per_block * WARP_SIZE
        comptime large_row_warps_per_block = 8
        comptime large_row_block_size = large_row_warps_per_block * WARP_SIZE
        comptime min_large_row_blocks_per_sm = 6
        var large_row_grid_dim = ceildiv(num_rows, large_row_warps_per_block)

        if is_decode_uniform:
            if large_row_grid_dim >= sm_count * min_large_row_blocks_per_sm:
                comptime kernel = _q_rms_norm_rope_decode_ragged_wide_kernel[
                    dtype,
                    freq_dtype,
                    x.LayoutType,
                    output.LayoutType,
                    start_pos.LayoutType,
                    freqs_cis.LayoutType,
                    gamma.LayoutType,
                    large_row_block_size,
                    large_row_warps_per_block,
                ]
                context.enqueue_function[kernel, kernel](
                    x,
                    output,
                    start_pos,
                    freqs_cis,
                    gamma,
                    epsilon,
                    weight_offset,
                    num_rows,
                    grid_dim=large_row_grid_dim,
                    block_dim=large_row_block_size,
                )
            else:
                comptime kernel = _q_rms_norm_rope_decode_ragged_wide_kernel[
                    dtype,
                    freq_dtype,
                    x.LayoutType,
                    output.LayoutType,
                    start_pos.LayoutType,
                    freqs_cis.LayoutType,
                    gamma.LayoutType,
                    default_block_size,
                    default_warps_per_block,
                ]
                context.enqueue_function[kernel, kernel](
                    x,
                    output,
                    start_pos,
                    freqs_cis,
                    gamma,
                    epsilon,
                    weight_offset,
                    num_rows,
                    grid_dim=ceildiv(num_rows, default_warps_per_block),
                    block_dim=default_block_size,
                )
        else:
            if large_row_grid_dim >= sm_count * min_large_row_blocks_per_sm:
                comptime kernel = _q_rms_norm_rope_ragged_wide_kernel[
                    dtype,
                    freq_dtype,
                    x.LayoutType,
                    output.LayoutType,
                    input_row_offsets.LayoutType,
                    start_pos.LayoutType,
                    freqs_cis.LayoutType,
                    gamma.LayoutType,
                    large_row_block_size,
                    large_row_warps_per_block,
                ]
                context.enqueue_function[kernel, kernel](
                    x,
                    output,
                    input_row_offsets,
                    start_pos,
                    freqs_cis,
                    gamma,
                    epsilon,
                    weight_offset,
                    num_rows,
                    grid_dim=large_row_grid_dim,
                    block_dim=large_row_block_size,
                )
            else:
                comptime kernel = _q_rms_norm_rope_ragged_wide_kernel[
                    dtype,
                    freq_dtype,
                    x.LayoutType,
                    output.LayoutType,
                    input_row_offsets.LayoutType,
                    start_pos.LayoutType,
                    freqs_cis.LayoutType,
                    gamma.LayoutType,
                    default_block_size,
                    default_warps_per_block,
                ]
                context.enqueue_function[kernel, kernel](
                    x,
                    output,
                    input_row_offsets,
                    start_pos,
                    freqs_cis,
                    gamma,
                    epsilon,
                    weight_offset,
                    num_rows,
                    grid_dim=ceildiv(num_rows, default_warps_per_block),
                    block_dim=default_block_size,
                )
    else:
        @always_inline
        @__copy_capture(x)
        @parameter
        def input_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank]) -> SIMD[dtype, width]:
            return x.load[width=width](Coord(coords))

        @always_inline
        @__copy_capture(output)
        @parameter
        def norm_output_fn[
            width: Int, alignment: Int
        ](coords: IndexList[3], val: SIMD[dtype, width]) -> None:
            output.store[alignment=alignment](Coord(coords), val)

        @always_inline
        @__copy_capture(output)
        def rope_output_fn[
            width: Int, alignment: Int
        ](idx: IndexList[3], val: SIMD[dtype, width]) capturing -> None:
            output.store[alignment=alignment](Coord(idx), val)

        var shape = IndexList[3](Int(x.dim(0)), Int(x.dim(1)), Int(x.dim(2)))
        rms_norm_gpu[
            input_fn,
            norm_output_fn,
            multiply_before_cast=True,
        ](shape, gamma, epsilon, weight_offset, context)
        rope_ragged[
            dtype,
            freq_dtype,
            interleaved=False,
            target=target,
            output_fn=rope_output_fn,
        ](
            output,
            input_row_offsets,
            start_pos,
            freqs_cis,
            Optional[DeviceContext](context),
        )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(block_size))
)
def _q_rms_norm_fused_qk_rope_ragged_kernel[
    dtype: DType,
    freq_dtype: DType,
    KCacheType: KVCacheT,
    QLayoutType: TensorLayout,
    OutputLayoutType: TensorLayout,
    OffsetsLayoutType: TensorLayout,
    StartPosLayoutType: TensorLayout,
    FreqLayoutType: TensorLayout,
    QGammaLayoutType: TensorLayout,
    KGammaLayoutType: TensorLayout,
    block_size: Int,
    warps_per_block: Int,
](
    x: TileTensor[dtype, QLayoutType, MutAnyOrigin],
    output: TileTensor[mut=True, dtype, OutputLayoutType, MutAnyOrigin],
    input_row_offsets: TileTensor[
        DType.uint32, OffsetsLayoutType, MutAnyOrigin
    ],
    start_pos: TileTensor[DType.uint32, StartPosLayoutType, MutAnyOrigin],
    k_cache: KCacheType,
    freqs_cis: TileTensor[freq_dtype, FreqLayoutType, MutAnyOrigin],
    q_gamma: TileTensor[dtype, QGammaLayoutType, MutAnyOrigin],
    k_gamma: TileTensor[dtype, KGammaLayoutType, MutAnyOrigin],
    epsilon: Scalar[dtype],
    weight_offset: Scalar[dtype],
    total_rows: Int,
):
    comptime assert x.flat_rank == 3
    comptime assert output.flat_rank == 3
    comptime assert input_row_offsets.flat_rank == 1
    comptime assert start_pos.flat_rank == 1
    comptime assert freqs_cis.flat_rank == 2
    comptime assert q_gamma.flat_rank == 1
    comptime assert k_gamma.flat_rank == 1

    comptime num_q_heads = x.static_shape[1]
    comptime head_dim = x.static_shape[2]
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
    var row = block_idx.x * UInt(warps_per_block * 2) + warp_idx * 2 + sub_warp_idx
    var col = local_tid * UInt(simd_width)

    var flat_row = Int(row)
    var global_token_idx = 0
    var head_slot = 0
    var batch_idx = 0
    var token_idx = 0
    var q_post_seq_idx = 0
    var k_post_seq_idx = 0

    var is_active_row = row < UInt(total_rows)
    if is_active_row:
        global_token_idx = flat_row // total_heads
        head_slot = flat_row % total_heads
        batch_idx = get_batch_from_row_offsets(input_row_offsets, global_token_idx)
        token_idx = Int(
            UInt32(global_token_idx) - input_row_offsets[batch_idx]
        )
        q_post_seq_idx = Int(start_pos[batch_idx] + UInt32(token_idx))
        k_post_seq_idx = k_cache.cache_length(batch_idx) + token_idx

    var is_q_row = is_active_row and head_slot < num_q_heads
    if is_active_row:
        var epsilon_accum = epsilon.cast[accum_type]()
        var weight_offset_accum = weight_offset.cast[accum_type]()

        if is_q_row:
            var re_offset = Int(local_tid) * vec_width
            var im_offset = re_offset + head_dim // 2
            var freq_offset = Int(local_tid) * simd_width
            var q_re = x.load[width=vec_width, alignment=align](
                Coord(Idx(global_token_idx), Idx(head_slot), Idx(re_offset))
            )
            var q_im = x.load[width=vec_width, alignment=align](
                Coord(Idx(global_token_idx), Idx(head_slot), Idx(im_offset))
            )
            var thread_m2 = (q_re.cast[accum_type]() ** 2).reduce_add() + (
                q_im.cast[accum_type]() ** 2
            ).reduce_add()
            var row_m2 = warp.lane_group_sum[num_lanes=half_warp_size](thread_m2)
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
                Coord(Idx(q_post_seq_idx), Idx(freq_offset))
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
                batch_idx, k_head_idx, k_post_seq_idx, Int(col)
            ).cast[accum_type]()
            var gamma_val = k_gamma.load[width=simd_width, alignment=wide_align](
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
                comptime cache_dtype = KCacheType.dtype
                var k_freq = freqs_cis.load[width=simd_width * 2, alignment=1](
                    Coord(Idx(k_post_seq_idx), Idx(re_offset * 2))
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
                    k_post_seq_idx,
                    re_offset,
                    k_rope[0].cast[cache_dtype](),
                )
                k_cache.store(
                    batch_idx,
                    k_head_idx,
                    k_post_seq_idx,
                    im_offset,
                    k_rope[1].cast[cache_dtype](),
                )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(block_size))
)
def _q_rms_norm_fused_qk_rope_decode_ragged_kernel[
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
    x: TileTensor[dtype, QLayoutType, MutAnyOrigin],
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
    comptime assert x.flat_rank == 3
    comptime assert output.flat_rank == 3
    comptime assert start_pos.flat_rank == 1
    comptime assert freqs_cis.flat_rank == 2
    comptime assert q_gamma.flat_rank == 1
    comptime assert k_gamma.flat_rank == 1

    comptime num_q_heads = x.static_shape[1]
    comptime head_dim = x.static_shape[2]
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
    var row = block_idx.x * UInt(warps_per_block * 2) + warp_idx * 2 + sub_warp_idx
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
        post_seq_idx = Int(start_pos[batch_idx])

    var is_q_row = is_active_row and head_slot < num_q_heads
    if is_active_row:
        var epsilon_accum = epsilon.cast[accum_type]()
        var weight_offset_accum = weight_offset.cast[accum_type]()

        if is_q_row:
            var re_offset = Int(local_tid) * vec_width
            var im_offset = re_offset + head_dim // 2
            var freq_offset = Int(local_tid) * simd_width
            var q_re = x.load[width=vec_width, alignment=align](
                Coord(Idx(global_token_idx), Idx(head_slot), Idx(re_offset))
            )
            var q_im = x.load[width=vec_width, alignment=align](
                Coord(Idx(global_token_idx), Idx(head_slot), Idx(im_offset))
            )
            var thread_m2 = (q_re.cast[accum_type]() ** 2).reduce_add() + (
                q_im.cast[accum_type]() ** 2
            ).reduce_add()
            var row_m2 = warp.lane_group_sum[num_lanes=half_warp_size](thread_m2)
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
            var gamma_val = k_gamma.load[width=simd_width, alignment=wide_align](
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
def _q_rms_norm_fused_qk_rope_decode_ragged_wide_kernel[
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
    x: TileTensor[dtype, QLayoutType, MutAnyOrigin],
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
    comptime assert x.flat_rank == 3
    comptime assert output.flat_rank == 3
    comptime assert start_pos.flat_rank == 1
    comptime assert freqs_cis.flat_rank == 2
    comptime assert q_gamma.flat_rank == 1
    comptime assert k_gamma.flat_rank == 1

    comptime num_q_heads = x.static_shape[1]
    comptime head_dim = x.static_shape[2]
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
            var q_vec = x.load[width=simd_width, alignment=align](
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
                    Coord(Idx(global_token_idx), Idx(head_slot), Idx(re_offset)),
                    q_rope[0],
                )
                output.store[alignment=align](
                    Coord(Idx(global_token_idx), Idx(head_slot), Idx(im_offset)),
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
            var norm_val = k_vec * norm_factor * (
                gamma_val.cast[accum_type]() + weight_offset_accum
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


def _q_rms_norm_fused_qk_rope_generic_fallback[
    dtype: DType,
    freq_dtype: DType,
    collection_t: KVCollectionT,
    *,
    interleaved: Bool,
    target: StaticString,
](
    x: TileTensor[dtype, ...],
    input_row_offsets: TileTensor[DType.uint32, ...],
    start_pos: TileTensor[DType.uint32, ...],
    kv_collection: collection_t,
    freqs_cis: TileTensor[freq_dtype, ...],
    q_gamma: TileTensor[dtype, ...],
    k_gamma: TileTensor[dtype, ...],
    epsilon: Scalar[dtype],
    weight_offset: Scalar[dtype],
    layer_idx: UInt32,
    output: TileTensor[mut=True, dtype, ...],
    context: DeviceContext,
) raises:
    comptime assert input_row_offsets.flat_rank == 1
    comptime assert start_pos.flat_rank == 1
    comptime assert freqs_cis.flat_rank == 2
    comptime head_dim = Int(x.static_shape[2])

    comptime if dtype == DType.bfloat16 and head_dim == 256:
        q_rms_norm_rope_ragged[
            dtype,
            freq_dtype,
            target=target,
        ](
            x,
            input_row_offsets,
            start_pos,
            freqs_cis,
            q_gamma,
            epsilon,
            weight_offset,
            output,
            context,
        )
        k_rms_norm_rope_ragged[
            dtype,
            freq_dtype,
            collection_t,
            interleaved=interleaved,
            target=target,
        ](
            Int(x.dim(0)),
            input_row_offsets,
            kv_collection,
            freqs_cis,
            k_gamma,
            epsilon,
            weight_offset,
            layer_idx,
            context,
        )
    else:
        @always_inline
        @parameter
        @__copy_capture(x)
        def q_input_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank]) -> SIMD[dtype, width]:
            return x.load[width=width](Coord(coords))

        @always_inline
        @parameter
        @__copy_capture(output)
        def q_output_fn[
            width: Int, alignment: Int
        ](coords: IndexList[3], val: SIMD[dtype, width]) -> None:
            output.store[alignment=alignment](Coord(coords), val)

        var q_shape = IndexList[3](
            Int(x.dim(0)),
            Int(x.dim(1)),
            Int(x.dim(2)),
        )
        rms_norm_gpu[
            q_input_fn,
            q_output_fn,
            multiply_before_cast=True,
        ](q_shape, q_gamma, epsilon, weight_offset, context)

        var k_cache = kv_collection.get_key_cache(Int(layer_idx))

        @always_inline
        @parameter
        @__copy_capture(k_cache, input_row_offsets)
        def k_input_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank]) -> SIMD[dtype, width]:
            var global_token_idx = coords[0]
            var batch_idx = get_batch_from_row_offsets(
                input_row_offsets, global_token_idx
            )
            var batch_start = rebind[Scalar[DType.uint32]](
                input_row_offsets.load[width=1](Coord(Idx(batch_idx)))
            )
            var token_idx = Int(UInt32(global_token_idx) - batch_start)
            var cache_token_idx = token_idx + Int(k_cache.cache_length(batch_idx))
            return k_cache.load[width=width](
                bs=batch_idx,
                tok_idx=cache_token_idx,
                head_idx=coords[1],
                head_dim_idx=coords[2],
            ).cast[dtype]()

        @always_inline
        @parameter
        @__copy_capture(k_cache, input_row_offsets)
        def k_output_fn[
            width: Int, alignment: Int
        ](coords: IndexList[3], val: SIMD[dtype, width]) -> None:
            var global_token_idx = coords[0]
            var batch_idx = get_batch_from_row_offsets(
                input_row_offsets, global_token_idx
            )
            var batch_start = rebind[Scalar[DType.uint32]](
                input_row_offsets.load[width=1](Coord(Idx(batch_idx)))
            )
            var token_idx = Int(UInt32(global_token_idx) - batch_start)
            var cache_token_idx = token_idx + Int(k_cache.cache_length(batch_idx))
            k_cache.store(
                bs=batch_idx,
                tok_idx=cache_token_idx,
                head_idx=coords[1],
                head_dim_idx=coords[2],
                val=val.cast[collection_t.CacheType.dtype](),
            )

        var k_shape = IndexList[3](
            Int(x.dim(0)),
            Int(collection_t.CacheType.kv_params.num_heads),
            Int(k_gamma.dim(0)),
        )
        rms_norm_gpu[
            k_input_fn,
            k_output_fn,
            multiply_before_cast=True,
        ](k_shape, k_gamma, epsilon, weight_offset, context)

        rope_ragged[
            dtype,
            freq_dtype,
            interleaved=interleaved,
            target=target,
            output_fn=q_output_fn,
        ](
            output,
            input_row_offsets,
            start_pos,
            freqs_cis,
            Optional[DeviceContext](context),
        )
        _rope_k_cache_ragged[
            freq_dtype,
            collection_t,
            interleaved=interleaved,
            target=target,
        ](
            Int(x.dim(0)),
            input_row_offsets,
            kv_collection,
            freqs_cis,
            layer_idx,
            context,
        )


def q_rms_norm_fused_qk_rope_ragged[
    dtype: DType,
    freq_dtype: DType,
    collection_t: KVCollectionT,
    *,
    interleaved: Bool,
    target: StaticString,
](
    x: TileTensor[dtype, ...],
    input_row_offsets: TileTensor[DType.uint32, ...],
    start_pos: TileTensor[DType.uint32, ...],
    kv_collection: collection_t,
    freqs_cis: TileTensor[freq_dtype, ...],
    q_gamma: TileTensor[dtype, ...],
    k_gamma: TileTensor[dtype, ...],
    epsilon: Scalar[dtype],
    weight_offset: Scalar[dtype],
    layer_idx: UInt32,
    output: TileTensor[mut=True, dtype, ...],
    context: DeviceContext,
) raises:
    comptime assert (
        not interleaved
    ), "q_rms_norm_fused_qk_rope_ragged currently supports non-interleaved RoPE only"

    comptime head_dim = Int(x.static_shape[2])
    comptime num_q_heads = x.static_shape[1]
    comptime num_kv_heads = Int(collection_t.CacheType.kv_params.num_heads)
    var total_tokens = Int(x.dim(0))
    var is_decode_uniform = total_tokens == Int(start_pos.dim(0))
    var total_rows = total_tokens * (num_q_heads + num_kv_heads)

    comptime if head_dim == 128 and dtype == DType.bfloat16:
        comptime block_size = 64
        comptime warps_per_block = 2
        var k_cache = kv_collection.get_key_cache(Int(layer_idx))
        if is_decode_uniform:
            # Decode submits one token per active request, so batch_idx matches
            # global_token_idx and start_pos already carries the RoPE position.
            comptime kernel = _q_rms_norm_fused_qk_rope_decode_ragged_kernel[
                dtype,
                freq_dtype,
                collection_t.CacheType,
                x.LayoutType,
                output.LayoutType,
                start_pos.LayoutType,
                freqs_cis.LayoutType,
                q_gamma.LayoutType,
                k_gamma.LayoutType,
                block_size,
                warps_per_block,
            ]
            context.enqueue_function[kernel, kernel](
                x,
                output,
                start_pos,
                k_cache,
                freqs_cis,
                q_gamma,
                k_gamma,
                epsilon,
                weight_offset,
                total_rows,
                grid_dim=ceildiv(total_rows, warps_per_block * 2),
                block_dim=block_size,
            )
        else:
            comptime kernel = _q_rms_norm_fused_qk_rope_ragged_kernel[
                dtype,
                freq_dtype,
                collection_t.CacheType,
                x.LayoutType,
                output.LayoutType,
                input_row_offsets.LayoutType,
                start_pos.LayoutType,
                freqs_cis.LayoutType,
                q_gamma.LayoutType,
                k_gamma.LayoutType,
                block_size,
                warps_per_block,
            ]
            context.enqueue_function[kernel, kernel](
                x,
                output,
                input_row_offsets,
                start_pos,
                k_cache,
                freqs_cis,
                q_gamma,
                k_gamma,
                epsilon,
                weight_offset,
                total_rows,
                grid_dim=ceildiv(total_rows, warps_per_block * 2),
                block_dim=block_size,
            )
    elif dtype == DType.bfloat16 and head_dim == 256:
        if is_decode_uniform:
            comptime block_size = 64
            comptime warps_per_block = 2
            var k_cache = kv_collection.get_key_cache(Int(layer_idx))
            comptime kernel = _q_rms_norm_fused_qk_rope_decode_ragged_wide_kernel[
                dtype,
                freq_dtype,
                collection_t.CacheType,
                x.LayoutType,
                output.LayoutType,
                start_pos.LayoutType,
                freqs_cis.LayoutType,
                q_gamma.LayoutType,
                k_gamma.LayoutType,
                block_size,
                warps_per_block,
            ]
            context.enqueue_function[kernel, kernel](
                x,
                output,
                start_pos,
                k_cache,
                freqs_cis,
                q_gamma,
                k_gamma,
                epsilon,
                weight_offset,
                total_rows,
                grid_dim=ceildiv(total_rows, warps_per_block),
                block_dim=block_size,
            )
        else:
            _q_rms_norm_fused_qk_rope_generic_fallback[
                dtype,
                freq_dtype,
                collection_t,
                interleaved=interleaved,
                target=target,
            ](
                x,
                input_row_offsets,
                start_pos,
                kv_collection,
                freqs_cis,
                q_gamma,
                k_gamma,
                epsilon,
                weight_offset,
                layer_idx,
                output,
                context,
            )
    else:
        _q_rms_norm_fused_qk_rope_generic_fallback[
            dtype,
            freq_dtype,
            collection_t,
            interleaved=interleaved,
            target=target,
        ](
            x,
            input_row_offsets,
            start_pos,
            kv_collection,
            freqs_cis,
            q_gamma,
            k_gamma,
            epsilon,
            weight_offset,
            layer_idx,
            output,
            context,
        )
