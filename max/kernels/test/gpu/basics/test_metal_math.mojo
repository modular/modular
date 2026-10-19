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

from std.math import (
    cos,
    exp2,
    hypot,
    j1,
    log,
    remainder,
    rsqrt,
    scalb,
    sin,
    sqrt,
)
from std.testing import assert_almost_equal
from std.utils.numerics import FPUtils

from max.gpu import thread_idx
from max.gpu.host import DeviceContext

comptime _N = 8
comptime _UnaryFn = def[dtype: DType, width: SIMDLength, //](
    SIMD[dtype, width]
) thin -> SIMD[dtype, width] where dtype.is_floating_point()
comptime _InputFn = def[dtype: DType](Int) thin -> Scalar[dtype]


def _input[dtype: DType](i: Int) -> Scalar[dtype]:
    return Scalar[dtype](i + 1) * 0.5


def _subnormal_input[dtype: DType](i: Int) -> Scalar[dtype]:
    # The `i + 1`-th smallest subnormal, built from bits because arithmetic
    # would flush it. These exercise the paths `std.math` takes when the
    # plugin's `flushes_subnormals` hook is set.
    return Scalar[dtype](from_bits=Scalar[FPUtils[dtype].integral_type](i + 1))


def _hypot_self[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> SIMD[dtype, width] where dtype.is_floating_point():
    return hypot(x, x)


def _remainder_tiny[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> SIMD[dtype, width] where dtype.is_floating_point():
    return remainder(x, SIMD[dtype, width](_subnormal_input[dtype](2)))


def _pow_self[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> SIMD[dtype, width] where dtype.is_floating_point():
    return x**x


def _scalb_double[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> SIMD[dtype, width] where dtype.is_floating_point():
    return scalb(x, SIMD[dtype, width](1))


def _kernel[
    func: _UnaryFn, input: _InputFn, dtype: DType
](
    output: MutPointer[Scalar[dtype], MutAnyOrigin]
) where dtype.is_floating_point():
    output[thread_idx.x] = func(input[dtype](thread_idx.x))


def _check[
    func: _UnaryFn, name: StaticString, dtype: DType, input: _InputFn
](ctx: DeviceContext) raises where dtype.is_floating_point():
    var out_device = ctx.enqueue_create_buffer[dtype](_N)
    ctx.enqueue_function[_kernel[func, input, dtype]](
        out_device, grid_dim=1, block_dim=_N
    )
    with out_device.map_to_host() as out_host:
        for i in range(_N):
            assert_almost_equal(
                out_host[i].cast[.float32](),
                func(input[dtype](i)).cast[.float32](),
                msg=String(name, "(", input[dtype](i), ") for ", dtype),
                atol=0,
                rtol=1e-5 if dtype == .float32 else 1e-2,
            )


def _check_all[
    func: _UnaryFn, name: StaticString, input: _InputFn = _input
](ctx: DeviceContext) raises:
    _check[func, name, .float32, input](ctx)
    _check[func, name, .float16, input](ctx)
    _check[func, name, .bfloat16, input](ctx)


def _comptime_kernel[
    func: _UnaryFn, dtype: DType
](
    output: MutPointer[Scalar[dtype], MutAnyOrigin]
) where dtype.is_floating_point():
    # The compile-time interpreter evaluates `func`, and it cannot run the
    # `air.*` intrinsics that the Metal hooks lower to.
    comptime result = func(Scalar[dtype](2))
    output[0] = result


def _check_comptime[
    func: _UnaryFn, name: StaticString, dtype: DType
](ctx: DeviceContext) raises where dtype.is_floating_point():
    var out_device = ctx.enqueue_create_buffer[dtype](1)
    ctx.enqueue_function[_comptime_kernel[func, dtype]](
        out_device, grid_dim=1, block_dim=1
    )
    with out_device.map_to_host() as out_host:
        assert_almost_equal(
            out_host[0].cast[.float32](),
            func(Scalar[dtype](2)).cast[.float32](),
            msg=String("comptime ", name, " for ", dtype),
            atol=0,
            rtol=1e-6 if dtype == .float32 else 1e-2,
        )


def _check_comptime_all[
    func: _UnaryFn, name: StaticString
](ctx: DeviceContext) raises:
    _check_comptime[func, name, .float32](ctx)
    _check_comptime[func, name, .float16](ctx)
    _check_comptime[func, name, .bfloat16](ctx)


def main() raises:
    with DeviceContext() as ctx:
        _check_comptime_all[sqrt, "sqrt"](ctx)
        _check_comptime_all[rsqrt, "rsqrt"](ctx)
        _check_comptime_all[exp2, "exp2"](ctx)
        _check_comptime_all[sin, "sin"](ctx)
        _check_comptime_all[cos, "cos"](ctx)
        _check_comptime_all[_pow_self, "pow"](ctx)
        _check_all[sqrt, "sqrt"](ctx)
        _check_all[rsqrt, "rsqrt"](ctx)
        _check_all[exp2, "exp2"](ctx)
        _check_all[sin, "sin"](ctx)
        _check_all[cos, "cos"](ctx)
        _check_all[_pow_self, "pow"](ctx)
        _check_all[log, "log", _subnormal_input](ctx)
        _check_all[j1, "j1", _subnormal_input](ctx)
        _check_all[_hypot_self, "hypot", _subnormal_input](ctx)
        _check_all[_remainder_tiny, "remainder", _subnormal_input](ctx)
        _check_all[_scalb_double, "scalb", _subnormal_input](ctx)
