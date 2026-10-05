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

from asyncrt_test_utils import create_test_device_context
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.testing import TestSuite, assert_equal


def vec_func[
    OpType: ImplicitlyCopyable
    & RegisterPassable
    & def(Float32, Float32) -> Float32
](
    in0: Pointer[Float32, MutAnyOrigin],
    in1: Pointer[Float32, MutAnyOrigin],
    output: Pointer[Float32, MutAnyOrigin],
    len_dev: Int32,
    op: OpType,
):
    # `Int` is not device-passable; widen the fixed-width arg.
    var len = Int(len_dev)
    var tid = global_idx.x
    if tid >= len:
        return
    output[unsafe_offset=tid] = op(
        in0[unsafe_offset=tid], in1[unsafe_offset=tid]
    )


def vec_func5[
    Op0: ImplicitlyCopyable
    & RegisterPassable
    & def(Float32, Float32) -> Float32,
    Op1: ImplicitlyCopyable
    & RegisterPassable
    & def(Float32, Float32) -> Float32,
    Op2: ImplicitlyCopyable
    & RegisterPassable
    & def(Float32, Float32) -> Float32,
    Op3: ImplicitlyCopyable
    & RegisterPassable
    & def(Float32, Float32) -> Float32,
    Op4: ImplicitlyCopyable
    & RegisterPassable
    & def(Float32, Float32) -> Float32,
](
    in0: Pointer[Float32, MutAnyOrigin],
    in1: Pointer[Float32, MutAnyOrigin],
    output: Pointer[Float32, MutAnyOrigin],
    len_dev: Int32,
    op0: Op0,
    op1: Op1,
    op2: Op2,
    op3: Op3,
    op4: Op4,
):
    var len = Int(len_dev)
    var tid = global_idx.x
    if tid >= len:
        return
    var a = in0[unsafe_offset=tid]
    var b = in1[unsafe_offset=tid]
    output[unsafe_offset=tid] = (
        op0(a, b) + op1(a, b) + op2(a, b) + op3(a, b) + op4(a, b)
    )


def test_capture_2_5() raises:
    var ctx = create_test_device_context()
    run_captured_func(ctx, 2.5)


def test_capture_neg_1_5() raises:
    var ctx = create_test_device_context()
    run_captured_func(ctx, -1.5)


@inline(.never)
def run_captured_func(ctx: DeviceContext, captured: Float32) raises:
    print("-")
    print("run_captured_func(", captured, "):")

    comptime length = 1024

    var in0 = ctx.enqueue_create_buffer[.float32](length)
    var in1 = ctx.enqueue_create_buffer[.float32](length)
    in1.enqueue_fill(2)
    var out = ctx.enqueue_create_buffer[.float32](length)

    # Initialize the input and outputs with known values.
    with in0.map_to_host() as in0_host, out.map_to_host() as out_host:
        for i in range(length):
            in0_host[i] = Float32(i)
            out_host[i] = Float32(length + i)

    def add_with_captured(
        left: Float32, right: Float32
    ) {var captured} -> Float32:
        return left + right + captured

    var block_dim = 32

    comptime kernel = vec_func[OpType=type_of(add_with_captured)]
    ctx.enqueue_function[kernel](
        in0,
        in1,
        out,
        Int32(length),
        host_arg=add_with_captured,
        grid_dim=(length // block_dim),
        block_dim=(block_dim),
    )

    with out.map_to_host() as out_host:
        for i in range(length):
            if i < 10:
                print("at index", i, "the value is", out_host[i])
            assert_equal(
                out_host[i],
                Float32(Float32(i + 2) + captured),
                String("at index ", i, " the value is ", out_host[i]),
            )


def test_capture_five_host_args() raises:
    """Each of five closures lands in its own kernel parameter slot."""
    var ctx = create_test_device_context()
    comptime length = 1024

    var in0 = ctx.enqueue_create_buffer[.float32](length)
    var in1 = ctx.enqueue_create_buffer[.float32](length)
    in1.enqueue_fill(2)
    var out = ctx.enqueue_create_buffer[.float32](length)
    with in0.map_to_host() as in0_host:
        for i in range(length):
            in0_host[i] = Float32(i)

    var c0: Float32 = 1
    var c1: Float32 = 10
    var c2: Float32 = 100
    var c3: Float32 = 1000
    var c4: Float32 = 10000

    def op0(left: Float32, right: Float32) {var c0} -> Float32:
        return left + c0

    def op1(left: Float32, right: Float32) {var c1} -> Float32:
        return right * c1

    def op2(left: Float32, right: Float32) {var c2} -> Float32:
        return left - right + c2

    def op3(left: Float32, right: Float32) {var c3} -> Float32:
        return c3

    def op4(left: Float32, right: Float32) {var c4} -> Float32:
        return c4 - left

    var block_dim = 32
    comptime kernel = vec_func5[
        type_of(op0), type_of(op1), type_of(op2), type_of(op3), type_of(op4)
    ]
    ctx.enqueue_function[kernel](
        in0,
        in1,
        out,
        Int32(length),
        host_arg=op0,
        host_arg2=op1,
        host_arg3=op2,
        host_arg4=op3,
        host_arg5=op4,
        grid_dim=(length // block_dim),
        block_dim=(block_dim),
    )

    with out.map_to_host() as out_host:
        for i in range(length):
            var a = Float32(i)
            var expected = (a + c0) + (2 * c1) + (a - 2 + c2) + c3 + (c4 - a)
            assert_equal(
                out_host[i],
                expected,
                String("at index ", i, " the value is ", out_host[i]),
            )


def main() raises:
    # TODO(MOCO-2556): Use automatic discovery when it can handle global_idx.
    # TestSuite.discover_tests[__functions_in_module()]().run()
    var suite = TestSuite()

    suite.test[test_capture_2_5]()
    suite.test[test_capture_neg_1_5]()
    suite.test[test_capture_five_host_args]()

    suite^.run()
