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

from asyncrt_test_utils import create_test_device_context, expect_eq
from builtin.device_passable import DevicePassable
from gpu import *
from gpu.host import DeviceContext
from testing import TestSuite

alias T = DType.float64
alias S = Scalar[T]


@register_passable("trivial")
struct TwoS:
    var s0: S
    var s1: S

    fn __init__(out self, s: S):
        self.s0 = 1
        self.s1 = s


struct OneS(DevicePassable):
    alias device_type: AnyTrivialRegType = TwoS

    fn _to_device_type(self, target: OpaquePointer):
        target.bitcast[Self.device_type]()[] = TwoS(self.s)

    @staticmethod
    fn get_type_name() -> String:
        return "OneS"

    @staticmethod
    fn get_device_type_name() -> String:
        return "TwoS"

    var s: S

    fn __init__(out self, s: S):
        self.s = s


fn vec_func(
    in0: UnsafePointer[S],
    in1: UnsafePointer[S],
    output: UnsafePointer[S],
    s: TwoS,
    len: Int,
):
    var tid = global_idx.x
    if tid >= UInt(len):
        return
    output[tid] = in0[tid] + in1[tid] + s.s1 + s.s0


def test_function_unchecked():
    var ctx = create_test_device_context()
    _run_test_function_unchecked(ctx)


fn _run_test_function_unchecked(ctx: DeviceContext) raises:
    alias length = 1024
    alias block_dim = 32

    var scalar: S = 2
    var twos = TwoS(scalar)

    # Initialize the input and outputs with known values.
    var in0 = ctx.enqueue_create_buffer[T](length)
    var out = ctx.enqueue_create_buffer[T](length)
    with in0.map_to_host() as in0_host, out.map_to_host() as out_host:
        for i in range(length):
            in0_host[i] = i
            out_host[i] = length + i
    var in1 = ctx.enqueue_create_buffer[T](length).enqueue_fill(scalar)

    ctx.enqueue_function_unchecked[vec_func](
        in0,
        in1,
        out,
        twos,
        length,
        grid_dim=(length // block_dim),
        block_dim=(block_dim),
    )

    with out.map_to_host() as out_host:
        for i in range(length):
            if i < 10:
                print("at index", i, "the value is", out_host[i])
            expect_eq(
                out_host[i],
                i + 5,
                "at index ",
                i,
                " the value is ",
                out_host[i],
            )


def test_function_checked():
    var ctx = create_test_device_context()
    _run_test_function_checked(ctx)


fn _run_test_function_checked(ctx: DeviceContext) raises:
    alias length = 1024
    alias block_dim = 32

    var scalar: S = 2
    var ones = OneS(scalar)

    # Initialize the input and outputs with known values.
    var in0 = ctx.enqueue_create_buffer[T](length)
    var out = ctx.enqueue_create_buffer[T](length)
    with in0.map_to_host() as in0_host, out.map_to_host() as out_host:
        for i in range(length):
            in0_host[i] = i
            out_host[i] = length + i
    var in1 = ctx.enqueue_create_buffer[T](length).enqueue_fill(scalar)

    var compiled_vec_func = ctx.compile_function_checked[vec_func, vec_func]()
    ctx.enqueue_function_checked(
        compiled_vec_func,
        in0,
        in1,
        out,
        ones,
        length,
        grid_dim=(length // block_dim),
        block_dim=(block_dim),
    )

    with out.map_to_host() as out_host:
        for i in range(length):
            if i < 10:
                print("at index", i, "the value is", out_host[i])
            expect_eq(
                out_host[i],
                i + 5,
                "at index ",
                i,
                " the value is ",
                out_host[i],
            )


def test_function_experimental():
    var ctx = create_test_device_context()
    _run_test_function_experimental(ctx)


fn _run_test_function_experimental(ctx: DeviceContext) raises:
    alias length = 1024
    alias block_dim = 32

    var scalar: S = 2
    var ones = OneS(scalar)

    # Initialize the input and outputs with known values.
    var in0 = ctx.enqueue_create_buffer[T](length)
    var out = ctx.enqueue_create_buffer[T](length)
    with in0.map_to_host() as in0_host, out.map_to_host() as out_host:
        for i in range(length):
            in0_host[i] = i
            out_host[i] = length + i
    var in1 = ctx.enqueue_create_buffer[T](length).enqueue_fill(scalar)

    var compiled_vec_func = ctx.compile_function_experimental[vec_func]()
    ctx.enqueue_function_experimental(
        compiled_vec_func,
        in0,
        in1,
        out,
        ones,
        length,
        grid_dim=(length // block_dim),
        block_dim=(block_dim),
    )

    with out.map_to_host() as out_host:
        for i in range(length):
            if i < 10:
                print("at index", i, "the value is", out_host[i])
            expect_eq(
                out_host[i],
                i + 5,
                "at index ",
                i,
                " the value is ",
                out_host[i],
            )


def main():
    # TODO(MOCO-2556): Use automatic discovery when it can handle global_idx.
    # TestSuite.discover_tests[__functions_in_module()]().run()
    var suite = TestSuite()

    suite.test[test_function_unchecked]()
    suite.test[test_function_checked]()
    suite.test[test_function_experimental]()

    suite^.run()
