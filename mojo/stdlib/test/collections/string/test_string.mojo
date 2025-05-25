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


from collections.string.string import (
    _calc_initial_buffer_size_int32,
    _calc_initial_buffer_size_int64,
    _StringCapacityField,
)
from math import isinf, isnan

from memory import UnsafePointer, memcpy
from python import Python, PythonObject
from testing import (
    assert_equal,
    assert_false,
    assert_not_equal,
    # assert_raises,
    assert_true,
)


struct assert_raises:
    var message_contains: String
    var triggered: Bool

    @always_inline
    fn __init__(out self, *, contains: String = ""):
        self.message_contains = contains
        self.triggered = False

    fn __enter__(self):
        pass

    fn __exit__(self) -> Bool:
        if not self.triggered:
            print("FAIL: Expected exception, but none was raised")
        return self.triggered

    fn __exit__(mut self, error: Error) -> Bool:
        self.triggered = True
        err_str = String(error)
        if not (self.message_contains in err_str):
            print("FAIL: Expected message containing:", self.message_contains)
            print("      But got error:", err_str)
            return False
        return True


def test_format_args_2():
    with assert_raises(contains="Index -1 not in *args"):
        _ = String("{-1} {0}").format("First")

    with assert_raises(contains="Index 1 not in *args"):
        _ = String("A {0} B {1}").format("First")

    with assert_raises(contains="Index 1 not in *args"):
        _ = String("A {1} B {0}").format("First")

    with assert_raises(contains="Index 1 not in *args"):
        _ = String("A {1} B {0}").format()

    with assert_raises(
        contains="Automatic indexing require more args in *args"
    ):
        _ = String("A {} B {}").format("First")

    with assert_raises(
        contains="Cannot both use manual and automatic indexing"
    ):
        _ = String("A {} B {1}").format("First", "Second")

    with assert_raises(contains="Index first not in kwargs"):
        _ = String("A {first} B {second}").format(1, 2)

    var s = String(" {} , {} {} !").format("Hello", "Beautiful", "World")
    assert_equal(s, " Hello , Beautiful World !")


def test_format_args():
    fn curly(c: StaticString) -> String:
        return "there is a single curly " + c + " left unclosed or unescaped"

    with assert_raises(contains=curly("{")):
        _ = String("{ {}").format(1)

    with assert_raises(contains=curly("{")):
        _ = String("{ {0}").format(1)

    with assert_raises(contains=curly("{")):
        _ = String("{}{").format(1)

    with assert_raises(contains=curly("}")):
        _ = String("{}}").format(1)

    with assert_raises(contains=curly("{")):
        _ = String("{} {").format(1)

    with assert_raises(contains=curly("{")):
        _ = String("{").format(1)

    with assert_raises(contains=curly("}")):
        _ = String("}").format(1)

    with assert_raises(contains=""):
        _ = String("{}").format()

    assert_equal(String("}}").format(), "}")
    assert_equal(String("{{").format(), "{")

    assert_equal(String("{{}}{}{{}}").format("foo"), "{}foo{}")

    assert_equal(String("{{ {0}").format("foo"), "{ foo")
    assert_equal(String("{{{0}").format("foo"), "{foo")
    assert_equal(String("{{0}}").format("foo"), "{0}")
    assert_equal(String("{{}}").format("foo"), "{}")
    assert_equal(String("{{0}}").format("foo"), "{0}")
    assert_equal(String("{{{0}}}").format("foo"), "{foo}")

    var vinput = "{} {}"
    var output = String(vinput).format("123", 456)
    assert_equal(len(output), 7)


def main():
    test_format_args()
    test_format_args_2()
