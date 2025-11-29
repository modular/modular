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

from testing import (
    TestSuite,
    assert_equal,
    assert_raises,
    assert_true,
    assert_false,
)
from sys.ffi import CStringSlice


def test_default_slice():
    var cslice = CStringSlice[MutOrigin.external]()
    assert_equal(len(cslice), 0)
    assert_equal(String(cslice), "")
    assert_false(cslice.unsafe_ptr())


def test_init_from_invalid_string():
    with assert_raises(contains="not nul-terminated"):
        _ = CStringSlice(String(""))

    with assert_raises(contains="not nul-terminated"):
        _ = CStringSlice(String("mojo!"))

    with assert_raises(contains="interior nul byte"):
        _ = CStringSlice(String("mojo\0mojo"))


def test_init_from_invalid_byte_span():
    with assert_raises(contains="not nul-terminated"):
        _ = CStringSlice(Span[Byte, MutOrigin.external]())

    with assert_raises(contains="not nul-terminated"):
        _ = CStringSlice(Span[Byte]([Byte(1), Byte(2)]))

    with assert_raises(contains="interior nul byte"):
        _ = CStringSlice(Span[Byte]([Byte(1), Byte(0), Byte(2)]))


def test_c_string_slice_from_ptr():
    var string = String("mojo!\0")
    var ptr = string.unsafe_ptr().bitcast[Int8]()
    var cslice = CStringSlice(unsafe_from_ptr=ptr)
    assert_equal(len(cslice), 5)
    assert_equal(String(cslice), "mojo!")
    assert_equal(Int(cslice.unsafe_ptr()), Int(ptr))


def test_c_string_slice_from_nul_ptr():
    var cslice = CStringSlice[ImmutOrigin.external](unsafe_from_ptr={})
    assert_equal(len(cslice), 0)
    assert_equal(String(cslice), "")
    assert_false(cslice.unsafe_ptr())
    assert_equal(len(cslice.as_bytes()), 0)
    assert_equal(len(cslice.as_bytes_with_nul()), 0)


def test_c_string_slice_from_nul_string():
    var string = String("\0")
    var cslice = CStringSlice(string)
    assert_equal(len(cslice), 0)
    assert_equal(String(cslice), "")
    assert_equal(Int(cslice.unsafe_ptr()), Int(string.unsafe_ptr()))


def test_c_string_slice_from_nul_span():
    var span: List[Byte] = [Byte(0)]
    var cslice = CStringSlice(span)
    assert_equal(len(cslice), 0)
    assert_equal(String(cslice), "")
    assert_equal(Int(cslice.unsafe_ptr()), Int(span.unsafe_ptr()))


def test_c_string_slice_from_string():
    var string = String("mojo!\0")
    var cslice = CStringSlice(string)
    assert_equal(len(cslice), 5)
    assert_equal(String(cslice), "mojo!")
    assert_equal(Int(cslice.unsafe_ptr()), Int(string.unsafe_ptr()))


def test_c_string_slice_from_span():
    var string: List[Byte] = [
        Byte(109),
        Byte(111),
        Byte(106),
        Byte(111),
        Byte(33),
        Byte(0),
    ]
    var cslice = CStringSlice(string)
    assert_equal(len(cslice), 5)
    assert_equal(String(cslice), "mojo!")
    assert_equal(Int(cslice.unsafe_ptr()), Int(string.unsafe_ptr()))


def test_c_string_copy():
    var string = String("mojo!\0")
    var cslice = CStringSlice(string)

    var copy = cslice
    assert_true(copy == cslice)
    assert_equal(Int(copy.unsafe_ptr()), Int(cslice.unsafe_ptr()))


def test_c_string_eq():
    var first = CStringSlice(String("mojo!\0"))
    var second = CStringSlice(String("mojo!\0"))
    var third = CStringSlice(String("not mojo\0"))
    var fourth = CStringSlice[ImmutOrigin.external]()

    assert_true(first == second)
    assert_true(first != third)
    assert_true(second != third)
    assert_true(first != fourth)


def test_c_string_as_bytes():
    var string = String("mojo!\0")
    var cslice = CStringSlice(string)

    assert_equal(len(cslice.as_bytes()), 5)
    assert_equal(len(cslice.as_bytes_with_nul()), 6)

    assert_true(string.as_bytes() != cslice.as_bytes())
    assert_true(string.as_bytes() == cslice.as_bytes_with_nul())


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
