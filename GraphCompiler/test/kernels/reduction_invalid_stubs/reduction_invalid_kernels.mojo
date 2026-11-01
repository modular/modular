# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the MAX License:
# https://modular.com/legal/max-license
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Registered `ReduceOp` monoids that each break one graph-compiler rule."""

import extensibility
from algorithm.reduce_op import ReduceOp


@extensibility.register("invalid_no_outputs")
struct InvalidNoOutputs[dtype: DType, W: Int](ReduceOp):
    """No `SIMD` field of width `W`."""

    comptime Single = InvalidNoOutputs[Self.dtype, 1]
    comptime width = Self.W

    var best: Scalar[Self.dtype]
    var pair: SIMD[Self.dtype, 2]
    var count: Int

    def __init__(out self):
        self.best = 0
        self.pair = 0
        self.count = 0

    def accumulate[
        val_dtype: DType, w: Int
    ](
        mut self,
        val: SIMD[val_dtype, w],
        idx: SIMD[.int64, w] = SIMD[.int64, w](0),
    ):
        pass

    def join(mut self, other: Self):
        self.count += other.count

    def __getitem__(self, j: Int) -> Self.Single:
        return Self.Single()

    def __setitem__(mut self, j: Int, s: Self.Single):
        pass


@extensibility.register("invalid_width_literal")
struct InvalidWidthLiteral[dtype: DType, W: Int](ReduceOp):
    """`width` is a literal, not a parameter."""

    comptime Single = InvalidWidthLiteral[Self.dtype, Self.W]
    comptime width = 4

    var acc: SIMD[Self.dtype, Self.W]

    def __init__(out self):
        self.acc = 0

    def accumulate[
        val_dtype: DType, w: Int
    ](
        mut self,
        val: SIMD[val_dtype, w],
        idx: SIMD[.int64, w] = SIMD[.int64, w](0),
    ):
        pass

    def join(mut self, other: Self):
        self.acc += other.acc

    def __getitem__(self, j: Int) -> Self.Single:
        return Self.Single()

    def __setitem__(mut self, j: Int, s: Self.Single):
        pass


@extensibility.register("invalid_width_expr")
struct InvalidWidthExpr[dtype: DType, W: Int](ReduceOp):
    """`width` is an expression of a parameter."""

    comptime Single = InvalidWidthExpr[Self.dtype, Self.W]
    comptime width = Self.W * 2

    var acc: SIMD[Self.dtype, Self.W]

    def __init__(out self):
        self.acc = 0

    def accumulate[
        val_dtype: DType, w: Int
    ](
        mut self,
        val: SIMD[val_dtype, w],
        idx: SIMD[.int64, w] = SIMD[.int64, w](0),
    ):
        pass

    def join(mut self, other: Self):
        self.acc += other.acc

    def __getitem__(self, j: Int) -> Self.Single:
        return Self.Single()

    def __setitem__(mut self, j: Int, s: Self.Single):
        pass


@extensibility.register("invalid_uninferable_param")
struct InvalidUninferableParam[dtype: DType, W: Int, K: Int](ReduceOp):
    """`K` is not inferable and has no default."""

    comptime Single = InvalidUninferableParam[Self.dtype, 1, Self.K]
    comptime width = Self.W

    var acc: SIMD[Self.dtype, Self.W]

    def __init__(out self):
        self.acc = 0

    def accumulate[
        val_dtype: DType, w: Int
    ](
        mut self,
        val: SIMD[val_dtype, w],
        idx: SIMD[.int64, w] = SIMD[.int64, w](0),
    ):
        pass

    def join(mut self, other: Self):
        self.acc += other.acc

    def __getitem__(self, j: Int) -> Self.Single:
        return Self.Single()

    def __setitem__(mut self, j: Int, s: Self.Single):
        pass


@extensibility.register("invalid_entry_point")
struct InvalidEntryPoint[dtype: DType, W: Int](ReduceOp):
    """Also defines an `elementwise` entry point."""

    comptime Single = InvalidEntryPoint[Self.dtype, 1]
    comptime width = Self.W

    var acc: SIMD[Self.dtype, Self.W]

    def __init__(out self):
        self.acc = 0

    def accumulate[
        val_dtype: DType, w: Int
    ](
        mut self,
        val: SIMD[val_dtype, w],
        idx: SIMD[.int64, w] = SIMD[.int64, w](0),
    ):
        pass

    def join(mut self, other: Self):
        self.acc += other.acc

    def __getitem__(self, j: Int) -> Self.Single:
        return Self.Single()

    def __setitem__(mut self, j: Int, s: Self.Single):
        pass

    @staticmethod
    def elementwise[
        dt: DType, width: SIMDLength
    ](x: SIMD[dt, width]) -> SIMD[dt, width]:
        return x
