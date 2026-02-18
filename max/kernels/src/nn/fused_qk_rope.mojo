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

from collections import OptionalReg
from math import gcd
from sys.info import _current_target, simd_width_of

from algorithm.functional import elementwise
from utils.numerics import get_accum_type
from complex import ComplexSIMD
from gpu.host import DeviceContext, get_gpu_target
from gpu.host.info import is_cpu
from kv_cache.types import KVCacheT, KVCollectionT
from layout._coord import (
    Coord,
    CoordLike,
    Idx,
    RuntimeInt,
    ComptimeInt,
    coord_to_index_list,
)
from layout._layout import TensorLayout, RowMajorLayout, Layout, row_major
from layout._tile_tensor import TileTensor
from nn._ragged_utils import get_batch_from_row_offsets

from utils import IndexList


@always_inline
fn _rope[
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
fn get_safetensors_idx(head_dim_idx: Int, head_size: Int) -> Tuple[Int, Int]:
    return (head_dim_idx // 2, head_dim_idx // 2 + head_size // 2)


@always_inline
fn get_identity_rope_coeff[width: Int, dtype: DType]() -> SIMD[dtype, width]:
    # Creates a SIMD vector with real parts set to 1 and imaginary parts to
    # 0, effectively making the RoPE transformation an identity operation.
    return rebind[SIMD[dtype, width]](
        SIMD[dtype, width // 2](1).interleave(SIMD[dtype, width // 2](0))
    )


@always_inline
fn rope_q_proj[
    dtype: DType,
    freq_dtype: DType,
    rank: Int,
    width: Int,
    //,
    *,
    interleaved: Bool,
](
    q_proj: TileTensor[dtype, ...],
    output: TileTensor[mut=True, dtype, ...],
    idx: IndexList[rank],
    freq_val: SIMD[freq_dtype, width],
    head_size: Int,
):
    comptime assert q_proj.flat_rank == rank
    comptime assert output.flat_rank == rank
    var coord = Coord(idx)
    comptime assert q_proj.flat_rank == coord.flat_rank
    comptime assert output.flat_rank == coord.flat_rank

    var indices = get_safetensors_idx(idx[rank - 1], head_size)
    var pos_re = idx
    var pos_im = idx
    pos_re[rank - 1] = indices[0]
    pos_im[rank - 1] = indices[1]
    comptime width_2 = width // 2

    var coord_re = Coord(pos_re)
    var coord_im = Coord(pos_im)
    comptime assert q_proj.flat_rank == coord_re.flat_rank
    comptime assert q_proj.flat_rank == coord_im.flat_rank
    comptime assert output.flat_rank == coord_re.flat_rank
    comptime assert output.flat_rank == coord_im.flat_rank

    var val: SIMD[dtype, width]

    @parameter
    if interleaved:
        val = q_proj.load[width=width](coord)
    else:
        val = rebind[SIMD[dtype, width]](
            q_proj.load[width=width_2](coord_re).interleave(
                q_proj.load[width=width_2](coord_im)
            )
        )

    var res = _rope(val, freq_val)

    @parameter
    if interleaved:
        output.store(coord, res)
    else:
        output_re, output_im = res.deinterleave()
        output.store(coord_re, output_re)
        output.store(coord_im, output_im)


@always_inline
fn rope_k_cache[
    freq_dtype: DType, cache_t: KVCacheT, width: Int, //, *, interleaved: Bool
](
    k_cache: cache_t,
    b_idx: Int,
    h_idx: Int,
    s_idx: Int,
    d_idx: Int,
    freq_val: SIMD[freq_dtype, width],
    head_size: Int,
):
    h_re, h_im = get_safetensors_idx(d_idx, head_size)
    comptime width_2 = width // 2
    comptime cache_type = cache_t.dtype
    # TODO: Remove this once FP8 KVCache is supported (KERN-2394).
    comptime accum_type = get_accum_type[cache_type]()

    var val: SIMD[accum_type, width]

    @parameter
    if interleaved:
        val = k_cache.load[width=width](b_idx, h_idx, s_idx, d_idx).cast[
            accum_type
        ]()
    else:
        val = rebind[SIMD[accum_type, width]](
            k_cache.load[width=width_2](b_idx, h_idx, s_idx, h_re)
            .cast[accum_type]()
            .interleave(
                k_cache.load[width=width_2](b_idx, h_idx, s_idx, h_im).cast[
                    accum_type
                ]()
            )
        )

    var res = _rope(val, freq_val).cast[cache_type]()

    @parameter
    if interleaved:
        k_cache.store(b_idx, h_idx, s_idx, d_idx, res)
    else:
        output_re, output_im = res.deinterleave()
        k_cache.store(b_idx, h_idx, s_idx, h_re, output_re)
        k_cache.store(b_idx, h_idx, s_idx, h_im, output_im)


@always_inline
fn fused_qk_rope[
    dtype: DType,
    collection_t: KVCollectionT,
    //,
    cache_t: KVCacheT,
    *,
    interleaved: Bool,
    target: StaticString,
](
    q_proj: TileTensor[dtype, ...],
    kv_collection: collection_t,
    freqs_cis: TileTensor[dtype, ...],
    layer_idx: UInt32,
    valid_lengths: TileTensor[DType.uint32, ...],
    output: TileTensor[mut=True, dtype, ...],
    context: Optional[DeviceContext],
) raises:
    """Applies RoPE to query and key tensors.

    Args:
        q_proj: Query projection tensor of shape [batch, seq_len, n_heads, head_dim].
        kv_collection: The KV cache collection containing the key cache.
        freqs_cis: Frequency tensor for RoPE of shape [max_seq_len, head_dim].
        layer_idx: The layer index for accessing the correct cache.
        valid_lengths: Tensor of shape [batch] containing the valid length for each
            sequence. RoPE is only applied to positions within these lengths.
        output: Output tensor for Q with RoPE applied, same shape as q_proj.
        context: Optional device context for GPU execution.
    """
    comptime assert q_proj.flat_rank == 4
    comptime assert freqs_cis.flat_rank == 2
    comptime assert output.flat_rank == 4
    comptime assert valid_lengths.flat_rank == 1

    comptime kv_params = cache_t.kv_params

    var batch_size = Int(q_proj.dim[0]())
    var new_seq_len = Int(q_proj.dim[1]())
    comptime num_q_heads = Int(q_proj.static_shape[2])
    comptime num_k_heads = kv_params.num_heads
    comptime head_size = Int(q_proj.static_shape[3])

    var k_cache = kv_collection.get_key_cache(Int(layer_idx))

    @always_inline
    @parameter
    @__copy_capture(k_cache, valid_lengths)
    fn rope_fn[
        width: Int, rank: Int, alignment: Int = 1
    ](idx_arg: IndexList[rank]):
        comptime assert rank == 4, "Invalid rank passed to rope kernel"

        @parameter
        if width == 1:
            return
        else:
            var idx = rebind[IndexList[4]](idx_arg)
            var bs_idx = idx[0]
            var seq_idx = idx[1]

            # Check if this position is within the valid length for this batch
            var valid_len = Int(valid_lengths[bs_idx])
            if seq_idx >= valid_len:
                return

            # post_seq_idx: sum of start_pos (cache_lengths[batch_idx]) and
            # seq_idx (idx[1]).
            var post_seq_idx = k_cache.cache_length(bs_idx) + seq_idx
            var head_idx = idx[2]
            var head_dim_idx = idx[3]

            # WARN assumes head_size % simd_width == 0
            # guarded by constrained statement below
            var is_q_proj = head_idx < num_q_heads
            var f_c_temp = freqs_cis.load[width=width](
                (Idx(post_seq_idx), Idx(head_dim_idx))
            )

            if is_q_proj:
                rope_q_proj[interleaved=interleaved](
                    q_proj, output, idx, f_c_temp, head_size
                )
            else:
                head_idx -= num_q_heads
                rope_k_cache[interleaved=interleaved](
                    k_cache,
                    bs_idx,
                    head_idx,
                    post_seq_idx,
                    head_dim_idx,
                    f_c_temp,
                    head_size,
                )

    var launch_shape = IndexList[4](
        batch_size,
        new_seq_len,
        num_q_heads + Int(num_k_heads),  # concat q and k along head dim
        head_size,
    )
    comptime compile_target = _current_target() if is_cpu[
        target
    ]() else get_gpu_target()
    comptime target_simd_width = simd_width_of[dtype, target=compile_target]()
    comptime kernel_simd_width = gcd(
        target_simd_width, Int(kv_params.head_size)
    )
    comptime assert kernel_simd_width >= 2, "invalid simd_width and head size"

    @parameter
    if is_cpu[target]():
        elementwise[func=rope_fn, simd_width=kernel_simd_width, target=target](
            launch_shape
        )
    else:
        elementwise[func=rope_fn, simd_width=kernel_simd_width, target=target](
            launch_shape, context.value()
        )


@always_inline
fn fused_qk_rope_ragged[
    dtype: DType,
    freq_dtype: DType,
    collection_t: KVCollectionT,
    //,
    cache_t: KVCacheT,
    *,
    interleaved: Bool,
    target: StaticString,
    mrope_types: Variadic.TypesOfTrait[CoordLike] = Variadic.empty_of_trait[
        CoordLike
    ],
    mrope_section: Optional[Coord[*mrope_types]] = None,
    PositionIdsLayoutType: TensorLayout = RowMajorLayout[
        RuntimeInt[DType.int64], RuntimeInt[DType.int64]
    ],
](
    q_proj: TileTensor[dtype, ...],
    input_row_offsets: TileTensor[DType.uint32, ...],
    kv_collection: collection_t,
    freqs_cis: TileTensor[freq_dtype, ...],
    position_ids: OptionalReg[
        TileTensor[DType.uint32, PositionIdsLayoutType, ImmutAnyOrigin]
    ],
    layer_idx: UInt32,
    output: TileTensor[mut=True, dtype, ...],
    context: Optional[DeviceContext],
) raises:
    """Applies RoPE (Rotary Position Embedding) to query and key tensors.

    This function can applies RoPE only to the last `rope_dim` elements of each
    head, leaving the first `unroped_dim` elements unchanged. This is required
    for DeepSeek models where only part of each head undergoes rotary
    transformation.
    """
    comptime assert q_proj.flat_rank == 3, "q_proj must be rank 3"
    comptime assert freqs_cis.flat_rank == 2, "freqs_cis must be rank 2"
    comptime assert output.flat_rank == 3, "output must be rank 3"
    comptime assert PositionIdsLayoutType.rank == 2
    comptime assert (
        input_row_offsets.flat_rank == 1
    ), "input_row_offsets must be rank 1"
    comptime kv_params = cache_t.kv_params
    comptime num_q_heads = Int(q_proj.static_shape[1])
    comptime num_k_heads = kv_params.num_heads
    comptime q_head_size = Int(q_proj.static_shape[2])
    comptime k_head_size = kv_params.head_size
    var batch_size = input_row_offsets.dim[0]() - 1

    # Add rope dimension parameters
    comptime rope_dim = Int(freqs_cis.static_shape[1])

    # Check if shape of freqs_cis matches head_size.
    # If not, we only rope the last `rope_dim` dimensions of each head.
    comptime unroped_dim = q_head_size - rope_dim
    comptime has_nope = unroped_dim > 0

    comptime assert freqs_cis.LayoutType._shape_types[
        1
    ].is_static_value, "Need static shape for freqs_cis"
    comptime assert rope_dim <= q_head_size and rope_dim <= Int(k_head_size), (
        "rope_dim must be smaller or equal to head size, but got rope_dim = "
        + String(rope_dim)
        + " and head_size = "
        + String(k_head_size)
    )
    comptime assert (
        rope_dim == q_head_size and rope_dim == Int(k_head_size)
    ) or interleaved, (
        "Partial RoPE operation only supported for interleaved pattern"
    )

    var k_cache = kv_collection.get_key_cache(Int(layer_idx))

    @always_inline
    @parameter
    @__copy_capture(k_cache, batch_size, input_row_offsets, position_ids)
    fn rope_fn[
        width: Int, rank: Int, alignment: Int = 1
    ](idx_arg: IndexList[rank]):
        comptime assert rank == 3, "Invalid rank passed to rope kernel"

        @parameter
        if width == 1:
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
            var head_idx = idx[1]
            var head_dim_idx = idx[2]

            # Use position_ids if provided, otherwise fall back to cache calculation
            var post_seq_idx = k_cache.cache_length(batch_idx) + token_idx

            var position_ids_idx = post_seq_idx
            if position_ids:
                comptime PIdTensor = type_of(position_ids.value())
                comptime assert PIdTensor.flat_rank == 2

                @parameter
                if mrope_section:
                    var section_idx = 0

                    @parameter
                    for i in range(len(mrope_section.value())):
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
            var is_q_proj = head_idx < num_q_heads
            var is_unroped_region = head_dim_idx < unroped_dim

            var f_c_temp: SIMD[freq_dtype, width]

            @parameter
            if has_nope:
                if is_unroped_region:
                    f_c_temp = get_identity_rope_coeff[width, freq_dtype]()
                else:
                    f_c_temp = freqs_cis.load[width=width](
                        (Idx(position_ids_idx), Idx(head_dim_idx - unroped_dim))
                    )
            else:
                f_c_temp = freqs_cis.load[width=width](
                    (Idx(position_ids_idx), Idx(head_dim_idx))
                )

            if is_q_proj:
                rope_q_proj[interleaved=interleaved](
                    q_proj, output, idx, f_c_temp, q_head_size
                )
            else:

                @parameter
                if has_nope:
                    if is_unroped_region:
                        return

                head_idx -= num_q_heads
                # in case k_head_size != q_head_size
                head_dim_idx += Int(k_head_size - UInt(q_head_size))
                rope_k_cache[interleaved=interleaved](
                    k_cache,
                    batch_idx,
                    head_idx,
                    post_seq_idx,
                    head_dim_idx,
                    f_c_temp,
                    Int(k_head_size),
                )

    var launch_shape = IndexList[3](
        Int(q_proj.dim[0]()),
        num_q_heads + Int(num_k_heads),  # concat q and k along head dim
        q_head_size,
    )
    comptime compile_target = _current_target() if is_cpu[
        target
    ]() else get_gpu_target()
    comptime target_simd_width = simd_width_of[dtype, target=compile_target]()
    comptime kernel_simd_width = gcd(target_simd_width, rope_dim)

    @parameter
    if mrope_section:

        @parameter
        for i in range(len(mrope_section.value())):
            comptime assert (
                Int(mrope_section.value()[i].value()) % kernel_simd_width == 0
            ), "mrope_section must be divisible by rope kernel simd_width"

    comptime assert kernel_simd_width >= 2, "invalid simd_width and head size"

    @parameter
    if is_cpu[target]():
        elementwise[func=rope_fn, simd_width=kernel_simd_width, target=target](
            launch_shape
        )
    else:
        elementwise[func=rope_fn, simd_width=kernel_simd_width, target=target](
            launch_shape, context.value()
        )
