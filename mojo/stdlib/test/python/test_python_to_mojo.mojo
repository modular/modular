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
from std.python._cpython import PyObjectPtr
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


def test_string_subclass_override_takes_fallback() raises:
    # A `str` subclass with overridden `__str__` must observe the override.
    # `PyUnicode_CheckExact` rejects subclasses, so the fast path is skipped.
    var mod = Python.evaluate(
        (
            "class _MyStr(str):\n    def __str__(self):\n        return"
            " 'override'\n"
        ),
        file=True,
        name="_str_subclass_test_mod",
    )
    var my_str = mod._MyStr("original")
    assert_equal(String(py=my_str), "override")


def test_string_from_non_str_object() raises:
    # Non-str objects fall through to `py.__str__()`. Exercises the slow path
    # with an `int` input (Python's `str(5)` returns "5").
    assert_equal(String(py=PythonObject(5)), "5")


def test_string_empty_and_unicode() raises:
    # Edge cases on the fast path: empty, 2-byte UTF-8, 4-byte UTF-8 (emoji),
    # and embedded NUL (verifies the byte length from `PyUnicode_AsUTF8AndSize`
    # is honored, not strlen).
    assert_equal(String(py=PythonObject("")), "")
    assert_equal(String(py=PythonObject("héllo")), "héllo")
    assert_equal(String(py=PythonObject("\U0001F525")), "\U0001F525")
    assert_equal(String(py=PythonObject("foo\0bar")).byte_length(), 7)


def test_string_from_null_pythonobject() raises:
    # `PythonObject` constructed without arguments via `from_borrowed=PyObjectPtr()`
    # has a null `_obj_ptr`. The fast path must skip `PyUnicode_CheckExact`
    # on null (`Py_TYPE(NULL)` aborts) and fall through to `py.__str__()`,
    # which CPython renders as "<NULL>". Regression for an Illegal-instruction
    # abort surfaced by the `non-trivial-init` integration test.
    var null_obj = PythonObject(from_borrowed=PyObjectPtr())
    assert_equal(String(py=null_obj), "<NULL>")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
