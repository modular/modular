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

from sys import simd_width_of

from algorithm.functional import elementwise
from asyncrt_test_utils import create_test_device_context, expect_eq
from buffer import NDBuffer
from gpu import *
from gpu.host import DeviceContext, get_gpu_target
from testing import TestSuite

from utils import IndexList
from utils.index import Index


fn run_elementwise[dtype: DType](ctx: DeviceContext) raises:
    print("-")
    print("run_elementwise[", dtype, "]:")

    alias pack_size = simd_width_of[dtype, target = get_gpu_target()]()

    alias rank = 2
    alias dim_x = 2
    alias dim_y = 8
    alias length = dim_x * dim_y

    var in0 = ctx.enqueue_create_buffer[dtype](length)
    var out = ctx.enqueue_create_buffer[dtype](length)

    # Initialize the input and outputs with known values.
    with in0.map_to_host() as in_host, out.map_to_host() as out_host:
        for i in range(length):
            in_host[i] = i
            out_host[i] = length + i

    var in_buffer = NDBuffer[dtype, 2](in0.unsafe_ptr(), Index(dim_x, dim_y))
    var out_buffer = NDBuffer[dtype, 2](out.unsafe_ptr(), Index(dim_x, dim_y))

    @always_inline
    @__copy_capture(in_buffer, out_buffer)
    @parameter
    fn func[
        simd_width: Int, rank: Int, alignment: Int = 1
    ](idx0: IndexList[rank]):
        var idx = rebind[IndexList[2]](idx0)
        out_buffer.store(
            idx,
            in_buffer.load[width=simd_width](idx) + 42,
        )

    elementwise[func, pack_size, target="gpu"](
        IndexList[2](2, 8),
        ctx,
    )

    with out.map_to_host() as out_host:
        for i in range(length):
            if i < 10:
                print("at index", i, "the value is", out_host[i])
            expect_eq(
                out_host[i],
                i + 42,
                "at index ",
                i,
                " the value is ",
                out_host[i],
            )


def test_elementwise_float32():
    var ctx = create_test_device_context()
    run_elementwise[DType.float32](ctx)


def test_elementwise_bfloat16():
    var ctx = create_test_device_context()
    run_elementwise[DType.bfloat16](ctx)


def test_elementwise_float16():
    var ctx = create_test_device_context()
    run_elementwise[DType.float16](ctx)


def test_elementwise_int8():
    var ctx = create_test_device_context()
    run_elementwise[DType.int8](ctx)


def main():
    var suite = TestSuite()

    suite.test[test_elementwise_float32]()
    suite.test[test_elementwise_bfloat16]()
    suite.test[test_elementwise_float16]()
    suite.test[test_elementwise_int8]()

    suite^.run()
