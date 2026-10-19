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
"""AMDGCN lowerings for the `std.math` functions `HIPPlugin` overrides."""

from std.collections import OptionalReg
from std.collections.string.string_span import get_static_string
from std.math.math import _UnaryMathOp, _UnaryMathPluginHookFnType
from std.sys import llvm_intrinsic


comptime _MathFn = OptionalReg[_UnaryMathPluginHookFnType]


def _amdgcn_unary_math_fn[op: _UnaryMathOp, dtype: DType]() -> _MathFn:
    """Returns the AMDGCN lowering of the `std.math` function `op` for
    `dtype`, or `None` to keep the stdlib's own lowering."""
    comptime match op:
        case .rsqrt:
            return comptime (_MathFn(_amdgcn_unary_math["llvm.amdgcn.rsq"]))
        case .recip:
            return comptime (_MathFn(_amdgcn_unary_math["llvm.amdgcn.rcp"]))
        case .exp2:
            return comptime (
                _lowering[
                    _amdgcn_unary_math["llvm.amdgcn.exp2"],
                    dtype in (DType.float16, DType.float32),
                ]
            )
        case .log2:
            return comptime (
                _lowering[
                    _amdgcn_unary_math["llvm.amdgcn.log"],
                    dtype in (DType.float16, DType.float32),
                ]
            )
        case .exp | .log | .sqrt | .sin | .cos | .tanh:
            return None


comptime _lowering[
    lowering: _UnaryMathPluginHookFnType, supported: Bool
]: _MathFn = _MathFn(lowering) if supported else _MathFn(None)


def _amdgcn_suffix[dtype: DType]() -> StaticString:
    comptime if dtype == .float16:
        return "f16"
    elif dtype == .float32:
        return "f32"
    else:
        comptime assert dtype == .float64, "no AMDGCN overload for this dtype"
        return "f64"


@inline(.always)
def _amdgcn_unary_math[
    dtype: DType, width: SIMDLength, //, intrinsic: StaticString
](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
    """Applies the `intrinsic` overload for `dtype`, such as
    `llvm.amdgcn.rsq.f32`, one element at a time.

    The intrinsics have only f16, f32, and f64 overloads, so other dtypes
    round-trip through f32.
    """
    comptime if dtype not in (DType.float16, DType.float32, DType.float64):
        return _amdgcn_unary_math[intrinsic](x.cast[.float32]()).cast[dtype]()
    else:
        var res = SIMD[dtype, width]()

        comptime for i in range(width):
            res[i] = llvm_intrinsic[
                get_static_string[intrinsic, ".", _amdgcn_suffix[dtype]()](),
                Scalar[dtype],
                has_side_effect=False,
            ](x[i])
        return res
