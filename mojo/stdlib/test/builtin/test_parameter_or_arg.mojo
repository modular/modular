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
# RUN: %mojo %s

from builtin.parameter_or_arg import ParameterOrArg, Parameter
from testing import assert_equal, assert_true


def some_func(x: ParameterOrArg[T=String]):
    @parameter
    if x.is_parameter:
        print("Is comptime!! Value is:")
        # I can use the alias here
        alias y = x.comptime_value
        print(y)
        assert_equal(y, "COMPTIME")
    else:
        print("Is runtime!! Value is:")
        print(x.runtime_value())
        assert_equal(x.runtime_value(), "RUNTIME")

    assert_true(x.value().endswith("TIME"))


def test_maybe_parameter():
    some_func(Parameter["COMPTIME"]())
    some_func("RUNTIME")


struct TestBuffer[_SizeParam: Optional[Int], //]:
    var data: UnsafePointer[Int]
    var size: ParameterOrArg[_SizeParam]

    fn __init__(out self, size: ParameterOrArg[_SizeParam]):
        self.size = size
        self.data = UnsafePointer[Int].alloc(self.size.value())

    fn __del__(owned self):
        self.data.free()

    fn get_info(self) -> String:
        @parameter
        if __type_of(self.size).is_parameter:
            return "Fixed size buffer: " + String(self.size.comptime_value)
        else:
            return "Dynamic size buffer: " + String(self.size.runtime_value())

    fn get_size(self) -> Int:
        return self.size.value()


def test_struct_with_parameter_or_arg():
    # Test with compile-time known size
    var fixed_buffer = TestBuffer(Parameter[42]())
    assert_equal(fixed_buffer.get_info(), "Fixed size buffer: 42")
    assert_equal(fixed_buffer.get_size(), 42)

    # Test with runtime size
    var dynamic_size = 24
    var dynamic_buffer = TestBuffer(dynamic_size)
    assert_equal(dynamic_buffer.get_info(), "Dynamic size buffer: 24")
    assert_equal(dynamic_buffer.get_size(), 24)

    # Test with literal value (becomes runtime)
    var literal_buffer = TestBuffer(16)
    assert_equal(literal_buffer.get_info(), "Dynamic size buffer: 16")
    assert_equal(literal_buffer.get_size(), 16)


fn func_with_default[
    _param_x: Optional[Int] = 42
](x: ParameterOrArg[_param_x] = ParameterOrArg[_param_x]()) -> String:
    @parameter
    if x.is_parameter:
        return "Comptime: " + String(x.comptime_value)
    else:
        return "Runtime: " + String(x.runtime_value())


def test_func_with_default():
    assert_equal(func_with_default(), "Comptime: 42")
    assert_equal(func_with_default(100), "Runtime: 100")
    assert_equal(func_with_default(Parameter[84]()), "Comptime: 84")


def main():
    test_maybe_parameter()
    test_struct_with_parameter_or_arg()
    test_func_with_default()
