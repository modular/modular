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

from math import cosh, exp, exp2, log, sinh
from sys import simd_width_of

from algorithm.functional import elementwise
from gpu import *
from gpu.host import DeviceBuffer, DeviceContext, get_gpu_target
from testing import assert_almost_equal, assert_equal, TestSuite

from utils import Index, IndexList

comptime length = 8192


def run_elementwise[
    dtype: DType, math_fn: fn (x: SIMD) -> type_of(x)
](ctx: DeviceContext, in_device: DeviceBuffer[dtype],):
    comptime pack_size = simd_width_of[dtype, target = get_gpu_target()]()

    var out_device = ctx.enqueue_create_buffer[dtype](length)

    var in_buffer = Span[Scalar[dtype]](
        ptr=in_device.unsafe_ptr(), length=length
    )
    var out_buffer = Span[Scalar[dtype]](
        ptr=out_device.unsafe_ptr(), length=length
    )

    @always_inline
    @__copy_capture(out_buffer, in_buffer)
    @parameter
    fn func[
        simd_width: Int, rank: Int, alignment: Int = 1
    ](idx0: IndexList[rank]):
        var idx = rebind[IndexList[1]](idx0)
        var val = in_buffer.unsafe_ptr().load[width=simd_width](idx[0])
        var result = math_fn(val)
        out_buffer.unsafe_ptr().store[width=simd_width](idx[0], result)

    elementwise[func, pack_size, target="gpu"](IndexList[1](length), ctx)

    with in_device.map_to_host() as in_host, out_device.map_to_host() as out_host:
        for i in range(length):
            var expected_value = math_fn(in_host[i])

            comptime atol = 1e-05 if dtype is DType.float32 else 1e-4
            comptime rtol = 2e-05 if dtype is DType.float32 else 2e-2
            assert_almost_equal(
                out_host[i],
                expected_value,
                msg=String("values did not match at position ", i),
                atol=atol,
                rtol=rtol,
            )


def _test_exp[dtype: DType](ctx: DeviceContext):
    var input = ctx.enqueue_create_buffer[dtype](length)
    comptime epsilon = 0.001
    with input.map_to_host() as in_host:
        for i in range(length):
            in_host[i] = log(Scalar[dtype](i) + epsilon)
    run_elementwise[dtype, exp](ctx, input)


def _test_exp2[dtype: DType](ctx: DeviceContext):
    var input = ctx.enqueue_create_buffer[dtype](length)
    comptime epsilon = 0.001
    with input.map_to_host() as in_host:
        for i in range(length):
            in_host[i] = log(Scalar[dtype](i) + epsilon)
    run_elementwise[dtype, exp2](ctx, input)


def _test_cosh[dtype: DType](ctx: DeviceContext):
    var input = ctx.enqueue_create_buffer[dtype](length)
    with input.map_to_host() as in_host:
        for i in range(length):
            in_host[i] = (Scalar[dtype](i) / length * 10) - 5
    run_elementwise[dtype, cosh](ctx, input)


def _test_sinh[dtype: DType](ctx: DeviceContext):
    var input = ctx.enqueue_create_buffer[dtype](length)
    with input.map_to_host() as in_host:
        for i in range(length):
            in_host[i] = (Scalar[dtype](i) / length * 10) - 5
    run_elementwise[dtype, sinh](ctx, input)


def test_math_accuracy():
    with DeviceContext() as ctx:
        _test_exp[DType.float32](ctx)
        _test_exp[DType.float16](ctx)
        _test_exp[DType.bfloat16](ctx)
        _test_exp2[DType.float32](ctx)
        _test_exp2[DType.float16](ctx)
        _test_exp2[DType.bfloat16](ctx)
        _test_cosh[DType.float32](ctx)
        _test_cosh[DType.float16](ctx)
        _test_cosh[DType.bfloat16](ctx)
        _test_sinh[DType.float32](ctx)
        _test_sinh[DType.float16](ctx)
        _test_sinh[DType.bfloat16](ctx)


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
