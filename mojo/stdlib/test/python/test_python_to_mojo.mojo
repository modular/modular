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


def test_int_from_bool() raises:
    # `bool` is a subclass of `int` in CPython, so `PyLong_CheckExact` rejects
    # it and we take the slow path through `py.__int__()`. Verify the standard
    # int(True) == 1, int(False) == 0 contract is preserved.
    assert_equal(Int(py=PythonObject(True)), 1)
    assert_equal(Int(py=PythonObject(False)), 0)


def test_int_overflow_from_big_pyint() raises:
    # Python ints larger than `Py_ssize_t` must raise on conversion. This
    # exercises the `self == -1 and PyErr_Occurred()` disambiguation after
    # `PyLong_AsSsize_t` on the fast path.
    with assert_raises(contains="too large to convert"):
        _ = Int(py=Python.evaluate("1 << 100"))


def test_numpy_float() raises:
    var np = Python.import_module("numpy")
    var py_numpy_float = np.float64(1.0)
    var mojo_float = Float64(1.0)
    assert_equal(Float64(py=py_numpy_float), mojo_float)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
