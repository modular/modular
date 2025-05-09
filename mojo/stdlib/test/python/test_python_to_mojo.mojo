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
from testing import assert_equal, assert_false, assert_true


fn test_string_to_python_to_mojo(mut python: Python) raises:
    var py_string = PythonObject("mojo")
    var py_string_capitalized = py_string.capitalize()

    var cap_mojo_string = String(py_string_capitalized)
    assert_equal(cap_mojo_string, "Mojo")


fn test_range() raises:
    var array_size: PythonObject = 2

    # we check that the numbers appear in order
    # and that there are not less iterations than expected
    # by ensuring the list is empty at the end.
    var expected = List[Int](0, 1)
    for i in range(array_size):
        assert_equal(i, expected.pop(0))
    assert_false(expected)

    var start: PythonObject = 0
    var end: PythonObject = 4
    expected = List[Int](0, 1, 2, 3)
    for i in range(start, end):
        assert_equal(i, expected.pop(0))
    assert_false(expected)

    var start2: PythonObject = 5
    var end2: PythonObject = 10
    var step: PythonObject = 2
    expected = List[Int](5, 7, 9)
    for i in range(start2, end2, step):
        assert_equal(i, expected.pop(0))
    assert_false(expected)


fn test_python_to_string() raises:
    var os = Python.import_module("os")
    assert_true(String(os.environ).startswith("environ({"))


fn test_python_to_int() raises:
    var py_int = PythonObject(1)
    var mojo_int = Int(1)
    assert_equal(Int(py_int), mojo_int)


fn test_python_to_float() raises:
    var py_float = PythonObject(1.0)
    var mojo_float = Float64(1.0)
    assert_equal(Float64(py_float), mojo_float)


fn test_python_to_bool() raises:
    var py_bool = PythonObject(True)
    var mojo_bool = Bool(True)
    assert_equal(Bool(py_bool), mojo_bool)


fn test_python_numpy_int_to_mojo_int() raises:
    var np = Python.import_module("numpy")
    var py_numpy_int = np.int64(1)
    var mojo_int = Int(1)
    assert_equal(Int(py_numpy_int), mojo_int)


fn test_python_numpy_float_to_mojo_float() raises:
    var np = Python.import_module("numpy")
    var py_numpy_float = np.float64(1.0)
    var mojo_float = Float64(1.0)
    assert_equal(Float64(py_numpy_float), mojo_float)


def main():
    var python = Python()
    test_string_to_python_to_mojo(python)
    test_range()
    test_python_to_string()
    test_python_to_int()
    test_python_to_float()
    test_python_to_bool()
    test_python_numpy_int_to_mojo_int()
    test_python_numpy_float_to_mojo_float()
