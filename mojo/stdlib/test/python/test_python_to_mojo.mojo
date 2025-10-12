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

from python import Python, PythonObject
from testing import (
    assert_equal,
    assert_equal_pyobj,
    assert_false,
    assert_raises,
    assert_true,
    TestSuite,
)


def test_string():
    var py_string = PythonObject("mojo")
    var py_string_capitalized = py_string.capitalize()

    var cap_mojo_string = String(py_string_capitalized)
    assert_equal(cap_mojo_string, "Mojo")
    assert_equal_pyobj(cap_mojo_string.to_python_object(), PythonObject("Mojo"))

    var os = Python.import_module("os")
    assert_true(String(os.environ).startswith("environ({"))


def test_range():
    var array_size: PythonObject = 2

    # we check that the numbers appear in order
    # and that there are not less iterations than expected
    # by ensuring the list is empty at the end.
    var expected = [0, 1]
    for i in range(array_size):
        assert_equal(i, expected.pop(0))
    assert_false(expected)

    var start: PythonObject = 0
    var end: PythonObject = 4
    expected = [0, 1, 2, 3]
    for i in range(start, end):
        assert_equal(i, expected.pop(0))
    assert_false(expected)

    var start2: PythonObject = 5
    var end2: PythonObject = 10
    var step: PythonObject = 2
    expected = [5, 7, 9]
    for i in range(start2, end2, step):
        assert_equal(i, expected.pop(0))
    assert_false(expected)


def test_int():
    assert_equal(Int(PythonObject(5)), 5)
    assert_equal(Int(PythonObject(-1)), -1)

    # Test error trying conversion from Python '"str"'
    with assert_raises(contains="invalid literal for int()"):
        _ = Int(PythonObject("str"))

    assert_equal_pyobj(Int(5).to_python_object(), PythonObject(5))
    assert_equal_pyobj(Int(-1).to_python_object(), PythonObject(-1))


def test_float():
    var py_float = PythonObject(1.0)
    var mojo_float = Float64(1.0)
    assert_equal(Float64(py_float), mojo_float)


def test_bool():
    assert_true(Bool(PythonObject(True)))
    assert_false(Bool(PythonObject(False)))

    assert_equal_pyobj(Bool(True).to_python_object(), PythonObject(True))
    assert_equal_pyobj(Bool(False).to_python_object(), PythonObject(False))


def test_numpy_int():
    var np = Python.import_module("numpy")
    var py_numpy_int = np.int64(1)
    var mojo_int = Int(1)
    assert_equal(Int(py_numpy_int), mojo_int)


def test_numpy_float():
    var np = Python.import_module("numpy")
    var py_numpy_float = np.float64(1.0)
    var mojo_float = Float64(1.0)
    assert_equal(Float64(py_numpy_float), mojo_float)


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
