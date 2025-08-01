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
    width: Int, //,
    fn_fp32: StaticString,
    fn_fp64: StaticString,
](arg: SIMD[dtype, width]) -> SIMD[dtype, width]:
    @always_inline("nodebug")
    @parameter
    fn _float32_dispatch[
        input_type: DType, result_type: DType
    ](arg: Scalar[input_type]) -> Scalar[result_type]:
        return external_call[fn_fp32, Scalar[result_type]](arg)

    @always_inline("nodebug")
    @parameter
    fn _float64_dispatch[
        input_type: DType, result_type: DType
    ](arg: Scalar[input_type]) -> Scalar[result_type]:
        return external_call[fn_fp64, Scalar[result_type]](arg)

    constrained[
        dtype in [DType.float32, DType.float64],
        "input dtype must be float32 or float64",
    ]()

    @parameter
    if dtype is DType.float32:
        return _simd_apply[_float32_dispatch, dtype, width](arg)
    else:
        return _simd_apply[_float64_dispatch, dtype, width](arg)
