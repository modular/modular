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
from complex import ComplexSIMD
from gpu.host import DeviceContext, get_gpu_target
from gpu.host.info import is_cpu
from layout._coord import Coord, CoordLike, ComptimeInt, RuntimeInt, Idx, coord
from layout._layout import Layout, _RowMajor
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
fn apply_rope[
    dtype: DType,
    freq_dtype: DType,
    rank: Int,
    width: Int,
    //,
    *,
    interleaved: Bool,
    alignment: Int,
    output_fn: fn[width: Int, alignment: Int](
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

    @parameter
    if interleaved:
        var coord = Coord(idx)
        comptime assert coord.flat_rank == x.flat_rank
        val = x.load[width=width](coord)
    else:
        var re_coord = Coord(pos_re)
        comptime assert re_coord.flat_rank == x.flat_rank
        var im_coord = Coord(pos_im)
        comptime assert im_coord.flat_rank == x.flat_rank
        val = rebind[SIMD[dtype, width]](
            x.load[width=width_2](re_coord).interleave(
                x.load[width=width_2](im_coord)
            )
        )

    var res = _rope(val, freq_val)

    @parameter
    if interleaved:
        output_fn[alignment=alignment](idx, res)
    else:
        output_re, output_im = res.deinterleave()
        output_fn[alignment=alignment](pos_re, output_re)
        output_fn[alignment=alignment](pos_im, output_im)


@always_inline
fn rope_ragged[
    dtype: DType,
    freq_dtype: DType,
    *,
    interleaved: Bool,
    target: StaticString,
    output_fn: fn[width: Int, alignment: Int](
        idx: IndexList[3], val: SIMD[dtype, width]
    ) capturing -> None,
    mrope_types: Variadic.TypesOfTrait[CoordLike] = Variadic.empty_of_trait[
        CoordLike
    ],
    mrope_section: Optional[Coord[*mrope_types]] = None,
](
    x: TileTensor[dtype, ...],
    input_row_offsets: TileTensor[DType.uint32, ...],
    start_pos: TileTensor[DType.uint32, ...],
    freqs_cis: TileTensor[freq_dtype, ...],
    context: Optional[DeviceContext],
    position_ids: OptionalReg[
        TileTensor[
            DType.uint32,
            Layout[
                Variadic.types[
                    RuntimeInt[DType.int64], RuntimeInt[DType.int64]
                ],
                _RowMajor[
                    *Variadic.types[
                        RuntimeInt[DType.int64], RuntimeInt[DType.int64]
                    ]
                ],
            ],
            MutAnyOrigin,
        ]
    ] = None,
) raises where (
    input_row_offsets.flat_rank == 1
    and start_pos.flat_rank == 1
    and position_ids.T.flat_rank == 2
    and freqs_cis.flat_rank == 2
):
    comptime assert (
        freqs_cis.all_dims_known
    ), "freqs_cis shape must be statically shaped"
    comptime head_size = x.static_shape[2]
    comptime rope_dim = freqs_cis.static_shape[1]
    comptime unroped_dim = head_size - rope_dim
    comptime has_nope = unroped_dim > 0

    @always_inline
    @parameter
    @__copy_capture(x, input_row_offsets, start_pos, freqs_cis)
    fn rope_fn[
        width: Int, rank: Int, alignment: Int = 1
    ](idx_arg: IndexList[rank]):
        comptime assert rank == 3, "Invalid rank passed to rope kernel"

        @parameter
        if width == 1:
            debug_assert(
                False,
                (
                    "RoPE kernel called with simd width = 1, We should never be"
                    " here. This is indicative of an uneven last dimension of"
                    " the rope tensor. Ensure the model's head_size is"
                    " divisible by the simd width of your target hardware."
                ),
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
            var is_unroped_region = head_dim_idx < unroped_dim

            var f_c_temp: SIMD[freq_dtype, width]

            @parameter
            if has_nope:
                if is_unroped_region:
                    f_c_temp = get_identity_rope_coeff[width, freq_dtype]()
                else:
                    f_c_temp = freqs_cis.load[width=width](
                        coord[freqs_cis.linear_idx_type](
                            (position_ids_idx, head_dim_idx - unroped_dim)
                        )
                    )
            else:
                f_c_temp = freqs_cis.load[width=width](
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

    @parameter
    for i in range(x.rank):
        launch_shape_index_list[i] = Int(x.dim(i))

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
                mrope_section.value()[i].static_value % kernel_simd_width == 0
            ), "mrope_section must be divisible by rope kernel simd_width"

    comptime assert (
        kernel_simd_width >= 2 and rope_dim % kernel_simd_width == 0
    ), (
        "Rope kernel simd width must be between 2 and rope_dim and divisible by"
        " rope_dim. Ensure the model's head_size is divisible by the simd width"
        " of your target hardware."
    )

    @parameter
    if is_cpu[target]():
        elementwise[func=rope_fn, simd_width=kernel_simd_width, target=target](
            launch_shape_index_list
        )
    else:
        elementwise[func=rope_fn, simd_width=kernel_simd_width, target=target](
            launch_shape_index_list, context.value()
        )
