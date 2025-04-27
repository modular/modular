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

from sys import external_call

from builtin.simd import _simd_apply


@always_inline
fn libm_call[
    dtype: DType,
    simd_width: Int,
    fn_fp32: StaticString,
    fn_fp64: StaticString,
](arg: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
    @always_inline("nodebug")
    @parameter
    fn _float32_dispatch[
        input_type: DType, result_type: DType
    ](arg: SIMD[input_type, 1]) -> SIMD[result_type, 1]:
        return external_call[fn_fp32, SIMD[result_type, 1]](arg)

    @always_inline("nodebug")
    @parameter
    fn _float64_dispatch[
        input_type: DType, result_type: DType
    ](arg: SIMD[input_type, 1]) -> SIMD[result_type, 1]:
        return external_call[fn_fp64, SIMD[result_type, 1]](arg)

    constrained[
        dtype.is_floating_point(), "input dtype must be floating point"
    ]()

    @parameter
    if dtype is DType.float64:
        return _simd_apply[_float64_dispatch, dtype, simd_width](arg)
    return _simd_apply[_float32_dispatch, DType.float32, simd_width](
        arg.cast[DType.float32]()
    ).cast[dtype]()
