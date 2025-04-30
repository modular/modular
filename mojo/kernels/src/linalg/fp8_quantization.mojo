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


from sys import simdwidthof
from utils.numerics import FPUtils, max_finite, min_finite
import gpu.warp as warp
from algorithm.functional import (
    _elementwise_impl_gpu,
)
from buffer import NDBuffer
from buffer.dimlist import DimList
from gpu.host import DeviceContext
from gpu.host._compile import _get_gpu_target
from utils.index import Index, IndexList
from sys.info import _is_sm_9x_or_newer


@always_inline
fn quantize_static_scaled_fp8[
    out_dtype: DType,
    in_dtype: DType,
    is_scale_inverted: Bool = True,
](
    out_buffer: NDBuffer[mut=True, out_dtype, 2, *_],
    in_buffer: NDBuffer[in_dtype, 2, *_],
    scale: Float32,
    context: DeviceContext,
) raises:
    constrained[
        in_dtype in (DType.float32, DType.float16, DType.bfloat16),
        "input type should be float16, bfloat16 or float32",
    ]()
    constrained[
        out_dtype is DType.float8_e4m3fn, "output type should be float8_e4m3fn"
    ]()

    @always_inline
    @parameter
    @__copy_capture(out_buffer, in_buffer, scale)
    fn scaled_fp8_quant[width: Int, rank: Int](idx_arg: IndexList[rank]):
        constrained[
            _is_sm_9x_or_newer(),
            "this kernel is only supported on sm90 or newer",
        ]()
        constrained[rank == 2, "rank should be equal to 2"]()

        var idx = rebind[IndexList[2]](idx_arg)
        var in_vec_f32 = in_buffer.load[width=width](idx).cast[DType.float32]()

        var inversed_scale: Float32 = 1.0 / scale

        @parameter
        for i in range(width):
            var scaled_input_f32: Float32

            scaled_input_f32 = in_vec_f32[i] * inversed_scale
            in_vec_f32[i] = max(
                Float32(min_finite[out_dtype]()),
                min(Float32(max_finite[out_dtype]()), scaled_input_f32),
            )

        var scaled_in_vec = in_vec_f32.cast[out_dtype]()
        out_buffer.store(idx, rebind[SIMD[out_dtype, width]](scaled_in_vec))

    alias compile_target = _get_gpu_target()
    alias target_simd_width = simdwidthof[in_dtype, target=compile_target]()

    _elementwise_impl_gpu[func=scaled_fp8_quant, simd_width=target_simd_width](
        IndexList[2](in_buffer.dim[0](), in_buffer.dim[1]()), context
    )
