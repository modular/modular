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

from math import exp, exp2, log
from sys import simdwidthof

from algorithm.functional import elementwise
from buffer import DimList, NDBuffer
from gpu import *
from gpu.host import DeviceBuffer, DeviceContext
from gpu.host import get_gpu_target
from testing import assert_almost_equal

from utils import Index, IndexList

alias length = 8192


def run_elementwise[
    type: DType, math_fn: fn (x: SIMD) -> __type_of(x)
](ctx: DeviceContext, in_device: DeviceBuffer[type],):
    alias pack_size = simdwidthof[type, target = get_gpu_target()]()

    var out_device = ctx.enqueue_create_buffer[type](length)

    var in_buffer = NDBuffer[type, 1](in_device._unsafe_ptr(), Index(length))
    var out_buffer = NDBuffer[type, 1](out_device._unsafe_ptr(), Index(length))

    @always_inline
    @__copy_capture(out_buffer, in_buffer)
    @parameter
    fn func[simd_width: Int, rank: Int](idx0: IndexList[rank]):
        var idx = rebind[IndexList[1]](idx0)
        var val = in_buffer.load[width=simd_width](idx)
        var result = math_fn(val)
        out_buffer.store[width=simd_width](idx, result)

    elementwise[func, pack_size, target="gpu"](IndexList[1](length), ctx)

    with in_device.map_to_host() as in_host, out_device.map_to_host() as out_host:
        for i in range(length):
            var expected_value = math_fn(in_host[i])

            alias atol = 1e-05 if type is DType.float32 else 1e-4
            alias rtol = 2e-05 if type is DType.float32 else 2e-2
            assert_almost_equal(
                out_host[i],
                expected_value,
                msg=String("values did not match at position ", i),
                atol=atol,
                rtol=rtol,
            )


def test_exp[type: DType](ctx: DeviceContext):
    var input = ctx.enqueue_create_buffer[type](length)
    alias epsilon = 0.001
    with input.map_to_host() as in_host:
        for i in range(length):
            in_host[i] = log(Scalar[type](i) + epsilon)
    run_elementwise[type, exp](ctx, input)


def test_exp2[type: DType](ctx: DeviceContext):
    var input = ctx.enqueue_create_buffer[type](length)
    alias epsilon = 0.001
    with input.map_to_host() as in_host:
        for i in range(length):
            in_host[i] = log(Scalar[type](i) + epsilon)
    run_elementwise[type, exp2](ctx, input)


def main():
    with DeviceContext() as ctx:
        test_exp[DType.float32](ctx)
        test_exp[DType.float16](ctx)
        test_exp[DType.bfloat16](ctx)
        test_exp2[DType.float32](ctx)
        test_exp2[DType.float16](ctx)
        test_exp2[DType.bfloat16](ctx)
