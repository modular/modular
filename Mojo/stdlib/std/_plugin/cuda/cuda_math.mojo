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
"""NVIDIA PTX lowerings of `std.math` functions, for `CUDAPluginHooks`."""

from std.collections import OptionalReg
from std.math.math import (
    _UnaryMathOp,
    _UnaryMathPluginHookFnType,
    _llvm_unary_fn,
)
from std.sys._assembly import inlined_assembly
from std.sys.info import CompilationTarget, _is_sm_9x_or_newer


comptime _MathFn = OptionalReg[_UnaryMathPluginHookFnType]


def _call_ptx_intrinsic_scalar[
    dtype: DType,
    //,
    *,
    instruction: StaticString,
    constraints: StaticString,
](arg: Scalar[dtype]) -> Scalar[dtype]:
    return inlined_assembly[
        instruction + " $0, $1;",
        Scalar[dtype],
        constraints=constraints,
        has_side_effect=False,
    ](arg)


def _call_ptx_intrinsic[
    dtype: DType,
    width: SIMDLength,
    //,
    *,
    instruction: StaticString,
    constraints: StaticString,
](arg: SIMD[dtype, width]) -> SIMD[dtype, width]:
    comptime if width == 1:
        return _call_ptx_intrinsic_scalar[
            instruction=instruction, constraints=constraints
        ](arg[0])

    var res = SIMD[dtype, width]()

    comptime for i in range(width):
        res[i] = _call_ptx_intrinsic_scalar[
            instruction=instruction, constraints=constraints
        ](arg[i])
    return res


def _call_ptx_intrinsic[
    dtype: DType,
    width: SIMDLength,
    //,
    *,
    scalar_instruction: StaticString,
    vector2_instruction: StaticString,
    scalar_constraints: StaticString,
    vector_constraints: StaticString,
](arg: SIMD[dtype, width]) -> SIMD[dtype, width]:
    comptime if width == 1:
        return _call_ptx_intrinsic_scalar[
            instruction=scalar_instruction, constraints=scalar_constraints
        ](arg[0])

    var res = SIMD[dtype, width]()

    comptime for i in range(0, width, 2):
        res = res.insert[offset=i](
            inlined_assembly[
                vector2_instruction + " $0, $1;",
                SIMD[dtype, 2],
                constraints=vector_constraints,
                has_side_effect=False,
            ](arg.slice[2, offset=i]())
        )

    return res


def _nvptx_unary_math_fn[
    op: _UnaryMathOp, dtype: DType, target: CompilationTarget
]() -> _MathFn:
    """Returns the NVIDIA lowering of the `std.math` function `op` for
    `dtype` on `target`, or `None` to keep the stdlib's own lowering.

    Narrower dtypes that get `None` here reach the float32 lowering through
    the stdlib, which converts them to float32 and calls `op` again.
    """
    comptime match op:
        case .sqrt:
            # NVIDIA has no approximate f64 sqrt (`sqrt.approx.d` does not
            # exist), so f64 keeps the stdlib's `llvm.sqrt`, which lowers to
            # the IEEE `sqrt.rn.f64`.
            return comptime (_lowering[_sqrt, dtype != DType.float64])
        case .rsqrt:
            return comptime (_MathFn(_rsqrt))
        case .recip:
            return comptime (_MathFn(_recip))
        case .exp2:
            return comptime (
                _lowering[
                    _exp2,
                    dtype in (DType.float16, DType.float32)
                    or (
                        dtype == DType.bfloat16 and target._is_sm_9x_or_newer()
                    ),
                ]
            )
        case .tanh:
            return comptime (
                _lowering[
                    _tanh,
                    dtype in (DType.float16, DType.bfloat16, DType.float32),
                ]
            )
        case .log:
            return comptime (_lowering[_log, dtype == DType.float32])
        case .log2:
            return comptime (_lowering[_log2, dtype == DType.float32])
        case .sin:
            # float64 is claimed only so `_sin` rejects it, rather than
            # letting the stdlib fall back to `llvm.sin`.
            return comptime (
                _lowering[_sin, dtype in (DType.float32, DType.float64)]
            )
        case .cos:
            # As for `sin`, float64 is claimed only so `_cos` rejects it.
            return comptime (
                _lowering[_cos, dtype in (DType.float32, DType.float64)]
            )
        case .exp:
            return None


comptime _lowering[
    lowering: _UnaryMathPluginHookFnType, supported: Bool
]: _MathFn = _MathFn(lowering) if supported else _MathFn(None)


@inline(.always)
def _sqrt_nvvm(x: SIMD, out res: type_of(x)):
    comptime assert x.dtype == .float32, "must be DType.float32"
    res = {}

    comptime for i in range(x.length):
        res[i] = _llvm_unary_fn["llvm.nvvm.sqrt.approx.ftz.f"](x[i])


@inline(.always)
def _sqrt(x: SIMD) -> type_of(x):
    comptime if x.dtype in (DType.float16, DType.bfloat16):
        return _sqrt_nvvm(x.cast[.float32]()).cast[x.dtype]()
    return _sqrt_nvvm(x)


@inline(.always)
def _rsqrt_nvvm(x: SIMD, out res: type_of(x)):
    comptime assert x.dtype in (
        DType.float32,
        DType.float64,
    ), "must be f32 or f64 type"

    comptime instruction = "llvm.nvvm.rsqrt.approx.ftz.f" if x.dtype == .float32 else "llvm.nvvm.rsqrt.approx.d"
    res = {}

    comptime for i in range(x.length):
        res[i] = _llvm_unary_fn[instruction](x[i])


@inline(.always)
def _rsqrt(x: SIMD) -> type_of(x):
    comptime if x.dtype in (DType.float16, DType.bfloat16):
        return _rsqrt_nvvm(x.cast[.float32]()).cast[x.dtype]()
    return _rsqrt_nvvm(x)


@inline(.always)
def _recip_nvvm(x: SIMD, out res: type_of(x)):
    comptime assert x.dtype in (
        DType.float32,
        DType.float64,
    ), "must be f32 or f64 type"

    comptime instruction = "llvm.nvvm.rcp.approx.ftz.f" if x.dtype == .float32 else "llvm.nvvm.rcp.approx.ftz.d"
    res = {}

    comptime for i in range(x.length):
        res[i] = _llvm_unary_fn[instruction](x[i])


@inline(.always)
def _recip(x: SIMD) -> type_of(x):
    comptime if x.dtype in (DType.float16, DType.bfloat16):
        return _recip_nvvm(x.cast[.float32]()).cast[x.dtype]()
    return _recip_nvvm(x)


@inline(.always)
def _exp2(x: SIMD) -> type_of(x):
    comptime if x.dtype == .float16:
        comptime if _is_sm_9x_or_newer():
            return _call_ptx_intrinsic[
                scalar_instruction="ex2.approx.f16",
                vector2_instruction="ex2.approx.f16x2",
                scalar_constraints="=h,h",
                vector_constraints="=r,r",
            ](x)
        else:
            return _call_ptx_intrinsic[
                instruction="ex2.approx.f16", constraints="=h,h"
            ](x)
    elif x.dtype == .bfloat16:
        comptime assert _is_sm_9x_or_newer(), "ex2.approx.bf16 needs sm_90"
        return _call_ptx_intrinsic[
            scalar_instruction="ex2.approx.ftz.bf16",
            vector2_instruction="ex2.approx.ftz.bf16x2",
            scalar_constraints="=h,h",
            vector_constraints="=r,r",
        ](x)
    return _call_ptx_intrinsic[
        instruction="ex2.approx.ftz.f32", constraints="=f,f"
    ](x)


@inline(.always)
def _log2(x: SIMD) -> type_of(x):
    return _call_ptx_intrinsic[
        instruction="lg2.approx.ftz.f32", constraints="=f,f"
    ](x)


@inline(.always)
def _log(x: SIMD) -> type_of(x):
    comptime ln2 = 0.69314718055966295651160180568695068359375
    return ln2 * _log2(x)


@inline(.always)
def _sin(x: SIMD) -> type_of(x):
    comptime assert (
        x.dtype != DType.float64
    ), "DType.float64 is not supported for sin on NVIDIA GPU"
    return _call_ptx_intrinsic[
        instruction="sin.approx.ftz.f32", constraints="=f,f"
    ](x)


@inline(.always)
def _cos(x: SIMD) -> type_of(x):
    comptime assert (
        x.dtype != DType.float64
    ), "DType.float64 is not supported for cos on NVIDIA GPU"
    return _call_ptx_intrinsic[
        instruction="cos.approx.ftz.f32", constraints="=f,f"
    ](x)


@inline(.always)
def _tanh(x: SIMD) -> type_of(x):
    comptime if x.dtype == .float16:
        return _call_ptx_intrinsic[
            scalar_instruction="tanh.approx.f16",
            vector2_instruction="tanh.approx.f16x2",
            scalar_constraints="=h,h",
            vector_constraints="=r,r",
        ](x)
    elif x.dtype == .bfloat16:
        comptime if _is_sm_9x_or_newer():
            return _call_ptx_intrinsic[
                scalar_instruction="tanh.approx.bf16",
                vector2_instruction="tanh.approx.bf16x2",
                scalar_constraints="=h,h",
                vector_constraints="=r,r",
            ](x)
        else:
            return _call_ptx_intrinsic[
                instruction="tanh.approx.f32", constraints="=f,f"
            ](x.cast[.float32]()).cast[x.dtype]()
    return _call_ptx_intrinsic[
        instruction="tanh.approx.f32", constraints="=f,f"
    ](x)
