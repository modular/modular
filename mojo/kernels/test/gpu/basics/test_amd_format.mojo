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
# FIXME: KERN-1377
# UNSUPPORTED: AMD-GPU
# REQUIRES: AMD-GPU
# RUN: %mojo-no-debug %s | FileCheck %s

from collections import InlineArray
from collections.string import StaticString
from os import abort

from builtin._format_float import _write_float
from builtin.simd import Float8_e4m3fn, Float8_e5m2
from gpu.host import DeviceContext
from memory import Span, memcmp, memcpy
from testing import assert_true

from utils.write import _WriteBufferStack


struct Buffer[capacity: Int](Writer):
    var data: InlineArray[UInt8, capacity]
    var pos: Int

    fn __init__(out self):
        self.data = InlineArray[UInt8, capacity](fill=0)
        self.pos = 0

    fn write_bytes(mut self, bytes: Span[Byte, _]):
        len_bytes = len(bytes)
        # If empty then return
        if len_bytes == 0:
            return
        # Continue writing to buffer
        memcpy(self.data.unsafe_ptr() + self.pos, bytes.unsafe_ptr(), len_bytes)
        self.pos += len_bytes

    fn write[*Ts: Writable](mut self, *args: *Ts):
        @parameter
        fn write_arg[T: Writable](arg: T):
            arg.write_to(self)

        args.each[write_arg]()


fn check_float[type: DType, //, expected: StaticString](f8: Scalar[type]):
    var f8_str = Buffer[len(expected)]()
    _write_float(f8_str, f8)
    var res = memcmp(
        expected.unsafe_ptr(), f8_str.data.unsafe_ptr(), len(expected)
    )
    if res != 0:
        abort()


fn check_8e5m2[expected: StaticString](f8: Float8_e5m2):
    check_float[expected](f8)


fn check_8e4m3[expected: StaticString](f8: Float8_e4m3fn):
    check_float[expected](f8)


fn test_format_float8_e5m2():
    check_8e5m2["0.0"](0)
    check_8e5m2["0.125"](0.125)
    check_8e5m2["1.25"](1.25)
    check_8e5m2["1.52587890625e-05"](1.52587890625e-05)
    check_8e5m2["-57344.0"](-57344)
    check_8e5m2["-0.0001068115234375"](-0.0001068115234375)
    check_8e5m2["nan"](FloatLiteral.nan)
    check_8e5m2["inf"](FloatLiteral.infinity)
    check_8e5m2["-inf"](FloatLiteral.negative_infinity)
    check_8e5m2["-0.0"](FloatLiteral.negative_zero)


fn test_format_float8_e4m3fn():
    check_8e4m3["0.0"](0)
    check_8e4m3["0.001953125"](0.001953125)
    check_8e4m3["-0.01953125"](-0.01953125)
    check_8e4m3["0.001953125"](0.001953125)
    check_8e4m3["0.02734375"](0.02734375)
    check_8e4m3["0.029296875"](0.029296875)
    check_8e4m3["0.03125"](0.03125)
    check_8e4m3["0.25"](0.25)
    check_8e4m3["208.0"](208)
    check_8e4m3["-12.0"](-12)
    check_8e4m3["-104.0"](-104)


def main():
    # TODO(KERN-1259): Add tests for fnuz types when they're working
    with DeviceContext() as ctx:
        # CHECK-LABEL: == test_format_float8_e5m2
        print("== test_format_float8_e5m2")
        ctx.enqueue_function[test_format_float8_e5m2](grid_dim=1, block_dim=1)

        # CHECK-LABEL: == test_format_float8_e4m3fn
        print("== test_format_float8_e4m3fn")
        ctx.enqueue_function[test_format_float8_e4m3fn](grid_dim=1, block_dim=1)
