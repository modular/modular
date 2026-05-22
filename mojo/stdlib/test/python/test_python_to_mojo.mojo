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

from std.python import Python, PythonObject
from std.testing import (
    assert_equal,
    assert_equal_pyobj,
    assert_false,
    assert_raises,
    assert_true,
    TestSuite,
)


def test_string() raises:
    var py_string = PythonObject("mojo")
    var py_string_capitalized = py_string.capitalize()

    var cap_mojo_string = String(py_string_capitalized)
    assert_equal(cap_mojo_string, "Mojo")
    assert_equal_pyobj(cap_mojo_string, PythonObject("Mojo"))

    var os = Python.import_module("os")
    assert_true(String(os.environ).startswith("environ({"))


def test_string_construction_ascii_fast_path() raises:
    # All-ASCII input goes through PyUnicode_FromKindAndData(kind=1).
    assert_equal(String(py=PythonObject("hello world")), "hello world")
    # Empty string is the scalar-only edge (length < SIMD block width).
    assert_equal(String(py=PythonObject("")), "")
    # Length exceeding typical SIMD block width exercises the vectorized loop.
    assert_equal(
        String(py=PythonObject("0123456789abcdef0123456789abcdef")),
        "0123456789abcdef0123456789abcdef",
    )
    # 0x7F is the highest ASCII byte and must take the fast path.
    assert_equal(String(py=PythonObject("\x7F")), "\x7F")
    # Embedded NUL bytes are within `[0, 128)` so they also take the fast
    # path; verify the explicit length is honored rather than C-string-style
    # truncation.
    assert_equal(String(py=PythonObject("foo\0bar")).byte_length(), 7)


def test_string_construction_non_ascii_fallback() raises:
    # Inputs with any byte >= 128 fall through to PyUnicode_DecodeUTF8 so
    # multi-byte UTF-8 sequences decode to their proper code points.
    # 2-byte sequence (`é` = U+00E9 = 0xC3 0xA9).
    assert_equal(String(py=PythonObject("héllo")), "héllo")
    # 4-byte sequence (fire emoji).
    assert_equal(String(py=PythonObject("\U0001F525 fire")), "\U0001F525 fire")
    # U+0080 = 0xC2 0x80, the smallest non-ASCII code point. 0x80 alone is
    # not valid UTF-8, so the 2-byte sequence is the boundary case.
    assert_equal(String(py=PythonObject("\u0080")), "\u0080")
    # Long ASCII prefix followed by a high byte. Exercises the
    # non-first-iteration miss case in whichever loop the SIMD-mask width
    # makes active on the host (vectorized body or scalar tail).
    var long_then_nonascii = "0123456789abcdef0123456789abcdef01234567" + "é"
    assert_equal(
        String(py=PythonObject(long_then_nonascii)), long_then_nonascii
    )


def test_int() raises:
    assert_equal(Int(py=PythonObject(5)), 5)
    assert_equal(Int(py=PythonObject(-1)), -1)

    # Test error trying conversion from Python '"str"'
    with assert_raises(contains="invalid literal for int()"):
        _ = Int(py=PythonObject("str"))

    assert_equal_pyobj(Int(5), PythonObject(5))
    assert_equal_pyobj(Int(-1), PythonObject(-1))


def test_float() raises:
    var py_float = PythonObject(1.0)
    var mojo_float = Float64(1.0)
    assert_equal(Float64(py=py_float), mojo_float)


def test_bool() raises:
    assert_true(Bool(PythonObject(True)))
    assert_false(Bool(PythonObject(False)))

    assert_equal_pyobj(Bool(True), PythonObject(True))
    assert_equal_pyobj(Bool(False), PythonObject(False))


def test_numpy_int() raises:
    var np = Python.import_module("numpy")
    var py_numpy_int = np.int64(1)
    var mojo_int = Int(1)
    assert_equal(Int(py=py_numpy_int), mojo_int)


def test_int_subclass_override_takes_fallback() raises:
    # An `int` subclass with overridden `__int__` must observe the
    # override, matching CPython's `int(x)` semantics. `PyLong_CheckExact`
    # rejects subclasses so the fast path is skipped and `__int__()` runs.
    var mod = Python.evaluate(
        "class _MyInt(int):\n    def __int__(self):\n        return 99\n",
        file=True,
        name="_int_subclass_test_mod",
    )
    var my_int = mod._MyInt(1)
    assert_equal(Int(py=my_int), 99)


def test_numpy_float() raises:
    var np = Python.import_module("numpy")
    var py_numpy_float = np.float64(1.0)
    var mojo_float = Float64(1.0)
    assert_equal(Float64(py=py_numpy_float), mojo_float)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
