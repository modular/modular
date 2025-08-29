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

# ===----------------------------------------------------------------------=== #
# Op implementation traits
# ===----------------------------------------------------------------------=== #


trait ElementwiseUnaryOp:
    @staticmethod
    fn elementwise[
        dtype: DType,
        width: Int,
    ](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
        ...


trait ElementwiseUnaryMixedOp:
    @staticmethod
    fn elementwise[
        dtype: DType,
        out_dtype: DType,
        width: Int,
    ](x: SIMD[dtype, width]) -> SIMD[out_dtype, width]:
        ...


trait ElementwiseBinaryOp:
    @staticmethod
    fn elementwise[
        dtype: DType,
        width: Int,
    ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[dtype, width]:
        ...


trait ElementwiseBinaryComparisonOp:
    @staticmethod
    fn elementwise[
        dtype: DType,
        width: Int,
    ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[
        DType.bool, width
    ]:
        ...
