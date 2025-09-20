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

from buffer import NDBuffer, DimList
from complex import ComplexSIMD
from gpu.host import DeviceContext
from nn._ragged_utils import get_batch_from_row_offsets
from collections import OptionalReg
from utils import IndexList
from sys.info import alignof
from layout import IntTuple, Layout, LayoutTensor, UNKNOWN_VALUE
from algorithm.functional import elementwise
from math import gcd
from sys.info import _current_target, simd_width_of
from gpu.host import get_gpu_target
from gpu.host.info import is_cpu


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
fn get_safetensors_idx(head_dim_idx: Int, head_size: Int) -> (Int, Int):
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
    x_layout: Layout,
    rank: Int,
    width: Int, //,
    *,
    interleaved: Bool,
    alignment: Int,
    output_fn: fn[width: Int, alignment: Int] (
        idx: IndexList[rank], val: SIMD[dtype, width]
    ) capturing -> None,
](
    x: LayoutTensor[dtype, x_layout, MutableAnyOrigin],
    idx: IndexList[rank],
    freq_val: SIMD[freq_dtype, width],
):
    var indices = get_safetensors_idx(idx[rank - 1], x.shape[rank - 1]())
    var pos_re = idx
    var pos_im = idx
    pos_re[rank - 1] = indices[0]
    pos_im[rank - 1] = indices[1]
    alias width_2 = width // 2

    var val: SIMD[dtype, width]

    @parameter
    if interleaved:
        val = x.load[width=width](idx)
    else:
        val = rebind[SIMD[dtype, width]](
            x.load[width=width_2](pos_re).interleave(
                x.load[width=width_2](pos_im)
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
    x_layout: Layout,
    freq_dtype: DType,
    input_row_offsets_layout: Layout,
    start_pos_layout: Layout,
    freqs_cis_layout: Layout,
    *,
    interleaved: Bool,
    target: StaticString,
    output_fn: fn[width: Int, alignment: Int] (
        idx: IndexList[3], val: SIMD[dtype, width]
    ) capturing -> None,
    mrope_section: Optional[IntTuple] = None,
](
    x: LayoutTensor[dtype, x_layout, MutableAnyOrigin],
    input_row_offsets: LayoutTensor[
        DType.uint32, input_row_offsets_layout, MutableAnyOrigin
    ],
    start_pos: LayoutTensor[DType.uint32, start_pos_layout, MutableAnyOrigin],
    freqs_cis: LayoutTensor[freq_dtype, freqs_cis_layout, MutableAnyOrigin],
    context: Optional[DeviceContext],
    position_ids: OptionalReg[
        LayoutTensor[
            DType.uint32,
            Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE),
            MutableAnyOrigin,
        ]
    ] = None,
) raises:
    @parameter
    for i in range(len(x.layout.shape)):
        constrained[
            x.layout.shape[i].is_value(),
            "x.layout.shape["
            + String(i)
            + "] must be a scalar, was "
            + String(x.layout.shape[i]),
        ]()
    constrained[
        x.layout.shape[1].all_known() and x.layout.shape[2].all_known(),
        "x.shape[1] (num_heads) and x.shape[2] (head_dim) must be static, was "
        + String(x.layout.shape),
    ]()
    constrained[
        input_row_offsets.layout.all_dims_known(),
        "input_row_offsets shape must be statically shaped",
    ]()
    constrained[
        freqs_cis.layout.all_dims_known(),
        "freqs_cis shape must be statically shaped",
    ]()
    debug_assert(
        input_row_offsets.shape[0]() - 1 == start_pos.shape[0](),
        (
            "input_row_offsets shape must be batch_size + 1 and start_pos must"
            " be batch_size"
        ),
    )
    alias head_size = x.shape[2]()
    alias rope_dim = freqs_cis.shape[1]()
    alias unroped_dim = head_size - rope_dim
    alias has_nope = unroped_dim > 0

    @always_inline
    @parameter
    @__copy_capture(x, input_row_offsets, start_pos, freqs_cis)
    fn rope_fn[
        width: Int, rank: Int, alignment: Int = 1
    ](idx_arg: IndexList[rank]):
        constrained[rank == 3, "Invalid rank passed to rope kernel"]()

        @parameter
        if width == 1:
            # constrained[False, "ROPE SIMD_WIDTH=1, We should never be here"]()
            return
        else:
            var idx = rebind[IndexList[3]](idx_arg)

            var global_token_idx = idx[0]

            var batch_idx: Int = get_batch_from_row_offsets(
                input_row_offsets, global_token_idx
            )
            var token_idx = Int(global_token_idx - input_row_offsets[batch_idx])
            var head_idx = idx[1]
            var head_dim_idx = idx[2]

            # Use position_ids if provided, otherwise fall back to cache calculation
            var post_seq_idx = start_pos[batch_idx] + token_idx

            var position_ids_idx = Int(post_seq_idx)
            if position_ids:

                @parameter
                if mrope_section:
                    var section_idx = 0

                    @parameter
                    for i in range(len(mrope_section.value())):
                        alias val = mrope_section.value().value(i)
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
                        position_ids_idx, head_dim_idx - unroped_dim
                    )
            else:
                f_c_temp = freqs_cis.load[width=width](
                    position_ids_idx, head_dim_idx
                )
            apply_rope[
                interleaved=interleaved,
                alignment=alignment,
                output_fn=output_fn,
            ](x, idx, f_c_temp)

    var launch_shape_int_tuple = x.runtime_layout
    var launch_shape_index_list = IndexList[x.layout.rank()]()

    @parameter
    for i in range(x.layout.rank()):
        launch_shape_index_list[i] = launch_shape_int_tuple.dim(i)

    alias compile_target = _current_target() if is_cpu[
        target
    ]() else get_gpu_target()
    alias target_simd_width = simd_width_of[dtype, target=compile_target]()
    alias kernel_simd_width = gcd(target_simd_width, rope_dim)

    @parameter
    if mrope_section:

        @parameter
        for i in range(len(mrope_section.value())):
            constrained[
                Int(mrope_section.value()[i]) % kernel_simd_width == 0,
                "mrope_section must be divisible by rope kernel simd_width",
            ]()

    constrained[kernel_simd_width >= 2, "invalid simd_width and head size"]()

    @parameter
    if is_cpu[target]():
        elementwise[func=rope_fn, simd_width=kernel_simd_width, target=target](
            launch_shape_index_list
        )
    else:
        elementwise[func=rope_fn, simd_width=kernel_simd_width, target=target](
            launch_shape_index_list, context.value()
        )
