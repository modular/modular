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

from std.python import ConvertibleToPython, Python, PythonObject
from std.testing import assert_equal, assert_true, TestSuite


def _assert_py_list(obj: PythonObject, expected_len: Int) raises:
    var builtins = Python.import_module("builtins")
    assert_true(builtins.type(obj) is builtins.list)
    assert_equal(Int(py=builtins.len(obj)), expected_len)


def _refcount(obj: PythonObject) raises -> Int:
    var sys = Python.import_module("sys")
    return Int(py=sys.getrefcount(obj))


# Converting containers of non-raising elements must not need `try`.
def _list_int_to_python(var values: List[Int]) -> PythonObject:
    return PythonObject(values^)


def _nested_list_to_python(var values: List[List[Int]]) -> PythonObject:
    return PythonObject(values^)


def _array_of_lists_to_python(var values: Array[List[Int], 3]) -> PythonObject:
    return PythonObject(values^)


def test_empty() raises:
    _assert_py_list(PythonObject(List[Int]()), 0)
    _assert_py_list(PythonObject(Array[Int, 0]()), 0)


def test_list_int() raises:
    var values: List = [1, 2, 3]
    values[0] = 0
    var py_values = _list_int_to_python(values^)
    _assert_py_list(py_values, 3)
    assert_equal(Int(py=py_values[0]), 0)
    assert_equal(Int(py=py_values[1]), 2)
    assert_equal(Int(py=py_values[2]), 3)
    assert_equal(String(py_values), "[0, 2, 3]")


def test_list_string() raises:
    var values: List[String] = ["a", "bc", ""]
    var py_values = PythonObject(values^)
    _assert_py_list(py_values, 3)
    var builtins = Python.import_module("builtins")
    assert_true(builtins.type(py_values[0]) is builtins.str)
    assert_equal(String(py=py_values[1]), "bc")
    assert_equal(String(py_values), "['a', 'bc', '']")


def test_list_bool() raises:
    var values: List = [True, False]
    var py_values = PythonObject(values^)
    _assert_py_list(py_values, 2)
    var builtins = Python.import_module("builtins")
    assert_true(builtins.type(py_values[0]) is builtins.bool)
    assert_equal(String(py_values), "[True, False]")


def test_list_simd_float() raises:
    var values: List = [Float64(1.5), Float64(-0.25)]
    var py_values = PythonObject(values^)
    _assert_py_list(py_values, 2)
    var builtins = Python.import_module("builtins")
    assert_true(builtins.type(py_values[0]) is builtins.float)
    assert_equal(Float64(py=py_values[0]), 1.5)
    assert_equal(Float64(py=py_values[1]), -0.25)


def test_array_int() raises:
    var values = [1, 2, 3]
    values[0] = 0
    var py_values = PythonObject(values^)
    _assert_py_list(py_values, 3)
    assert_equal(String(py_values), "[0, 2, 3]")


def test_nested_list() raises:
    var values: List[List[Int]] = [[1, 2], [], [3]]
    var py_values = _nested_list_to_python(values^)
    _assert_py_list(py_values, 3)
    _assert_py_list(py_values[0], 2)
    _assert_py_list(py_values[1], 0)
    _assert_py_list(py_values[2], 1)
    assert_equal(Int(py=py_values[0][1]), 2)
    assert_equal(String(py_values), "[[1, 2], [], [3]]")


def test_array_of_lists() raises:
    var values: Array[List[Int], 3] = [[1], [2, 3], []]
    var py_values = _array_of_lists_to_python(values^)
    _assert_py_list(py_values, 3)
    _assert_py_list(py_values[1], 2)
    assert_equal(Int(py=py_values[1][0]), 2)
    assert_equal(String(py_values), "[[1], [2, 3], []]")


def test_convertible_to_python_bound() raises:
    def convert[
        T: ConvertibleToPython
    ](var value: T) raises (T.ConversionToPythonErrorType) -> PythonObject:
        return value^.to_python_object()

    var ints: List = [1, 2]
    assert_equal(String(convert(ints^)), "[1, 2]")
    var strings: List[String] = ["x"]
    assert_equal(String(convert(strings^)), "['x']")
    var array: Array[Int, 2] = [3, 4]
    assert_equal(String(convert(array^)), "[3, 4]")


def test_pass_to_python_function() raises:
    var sum_list = Python.evaluate("lambda xs: type(xs) is list and sum(xs)")
    var values: List = [1, 2, 3]
    assert_equal(Int(py=sum_list(values^)), 6)
    var nested: List[List[Int]] = [[1, 2], [3]]
    var flatten = Python.evaluate("lambda xss: [x for xs in xss for x in xs]")
    assert_equal(String(flatten(nested^)), "[1, 2, 3]")


# ===----------------------------------------------------------------------=== #
# Partial failure
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct _ConversionFailed(Movable):
    var index: Int


struct _Element(ConvertibleToPython, Movable):
    """Converts to a shared sentinel object, or fails with its own error type.

    Every live element and every converted item holds one reference to the
    sentinel, so the sentinel's refcount shows whether anything leaked.
    """

    comptime ConversionToPythonErrorType = _ConversionFailed

    var index: Int
    var fail: Bool
    var sentinel: PythonObject

    def __init__(out self, index: Int, fail: Bool, sentinel: PythonObject):
        self.index = index
        self.fail = fail
        self.sentinel = sentinel.copy()

    def to_python_object(var self) raises _ConversionFailed -> PythonObject:
        if self.fail:
            raise _ConversionFailed(self.index)
        return self.sentinel.copy()


def test_list_partial_failure_cleans_up() raises:
    var sentinel = Python.evaluate("object()")
    var baseline = _refcount(sentinel)

    var values = List[_Element]()
    for i in range(5):
        values.append(_Element(i, i == 2, sentinel))
    assert_equal(_refcount(sentinel), baseline + 5)

    var raised = False
    try:
        _ = values^.to_python_object()
    except e:
        # The element's own error type surfaces, not a generic `Error`.
        assert_equal(e.index, 2)
        raised = True
    assert_true(raised)

    # The converted items, the failing element, and the elements not yet
    # converted are all released, along with the partially filled list.
    assert_equal(_refcount(sentinel), baseline)


def test_array_partial_failure_cleans_up() raises:
    var sentinel = Python.evaluate("object()")
    var baseline = _refcount(sentinel)

    var values: Array[_Element, 3] = [
        _Element(0, False, sentinel),
        _Element(1, True, sentinel),
        _Element(2, False, sentinel),
    ]
    assert_equal(_refcount(sentinel), baseline + 3)

    var raised = False
    try:
        _ = PythonObject(values^)
    except e:
        assert_equal(e.index, 1)
        raised = True
    assert_true(raised)

    assert_equal(_refcount(sentinel), baseline)


def test_list_conversion_does_not_leak() raises:
    var sentinel = Python.evaluate("object()")
    var baseline = _refcount(sentinel)

    var values = List[_Element]()
    for i in range(4):
        values.append(_Element(i, False, sentinel))

    var py_values: PythonObject
    try:
        py_values = PythonObject(values^)
    except e:
        raise Error("unexpected conversion failure at index ", e.index)
    # The elements are consumed, so only the Python list's items hold
    # references to the sentinel now.
    assert_equal(_refcount(sentinel), baseline + 4)
    _ = py_values^
    assert_equal(_refcount(sentinel), baseline)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
