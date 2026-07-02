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
from std.python.numpy import from_numpy_array, to_numpy_array
from std.testing import (
    assert_almost_equal,
    assert_equal,
    assert_raises,
    TestSuite,
)


def test_to_numpy_array_float64() raises:
    var values: List[Float64] = [1.0, 2.5, -3.0, 4.25]
    var arr = to_numpy_array(values)

    assert_equal(Int(py=arr.size), 4)
    assert_equal(String(py=arr.dtype), "float64")
    for i in range(len(values)):
        assert_almost_equal(Float64(py=arr[i]), values[i])


def test_to_numpy_array_is_independent() raises:
    # A copy must not observe later mutations of the Mojo data.
    var values: List[Float64] = [1.0, 2.0, 3.0]
    var arr = to_numpy_array(values)
    values[0] = 99.0
    assert_almost_equal(Float64(py=arr[0]), 1.0)


def test_to_numpy_array_from_span() raises:
    var values: List[Float32] = [1.5, 2.5, 3.5]
    var arr = to_numpy_array(Span(values))
    assert_equal(String(py=arr.dtype), "float32")
    assert_almost_equal(Float64(py=arr[1]), 2.5)


def test_to_numpy_array_empty() raises:
    var values = List[Float64]()
    var arr = to_numpy_array(values)
    assert_equal(Int(py=arr.size), 0)
    assert_equal(String(py=arr.dtype), "float64")


def test_to_numpy_array_float16() raises:
    # `float16` is the dtype `ctypes` cannot express directly, so exercise it
    # explicitly. Values chosen to be exactly representable in float16.
    var values: List[Float16] = [1.0, 2.5, -0.5]
    var arr = to_numpy_array(values)
    assert_equal(Int(py=arr.size), 3)
    assert_equal(String(py=arr.dtype), "float16")
    assert_almost_equal(Float64(py=arr[1]), 2.5)
    # The copy is independent of later mutations.
    values[0] = 7.0
    assert_almost_equal(Float64(py=arr[0]), 1.0)


def test_from_numpy_array_float64() raises:
    var np = Python.import_module("numpy")
    var array = np.arange(5, dtype="float64")
    var span = from_numpy_array[DType.float64](array)

    assert_equal(len(span), 5)
    var total = Float64(0)
    for value in span:
        total += value
    assert_almost_equal(total, 10.0)
    _ = array


def test_from_numpy_array_aliases() raises:
    # A borrow must observe NumPy-side writes, and vice versa.
    var np = Python.import_module("numpy")
    var array = np.zeros(3, dtype="int64")
    var span = from_numpy_array[DType.int64](array)
    array[1] = 7
    assert_equal(Int(span[1]), 7)
    span[2] = 9
    assert_equal(Int(py=array[2]), 9)
    _ = array


def test_from_numpy_array_dtype_mismatch_raises() raises:
    var np = Python.import_module("numpy")
    var array = np.arange(4, dtype="float64")
    with assert_raises(contains="dtype mismatch"):
        _ = from_numpy_array[DType.int32](array)
    _ = array


def test_from_numpy_array_non_contiguous_raises() raises:
    var np = Python.import_module("numpy")
    # A strided slice is not C-contiguous.
    var array = np.arange(10, dtype="float64")[::2]
    with assert_raises(contains="C-contiguous"):
        _ = from_numpy_array[DType.float64](array)
    _ = array


def test_from_numpy_array_wrong_ndim_raises() raises:
    var np = Python.import_module("numpy")
    var array = np.ones(Python.tuple(2, 3), dtype="float64")
    with assert_raises(contains="1-D"):
        _ = from_numpy_array[DType.float64](array)
    _ = array


def test_roundtrip_int64_signed() raises:
    var values: List[Int64] = [
        -9223372036854775807,
        -1,
        0,
        5000000000,
        9223372036854775807,
    ]
    var arr = to_numpy_array(values)
    var span = from_numpy_array[DType.int64](arr)
    assert_equal(len(span), len(values))
    for i in range(len(values)):
        assert_equal(span[i], values[i])
    _ = arr


def test_from_numpy_array_empty() raises:
    var np = Python.import_module("numpy")
    var array = np.empty(0, dtype="float64")
    var span = from_numpy_array[DType.float64](array)
    assert_equal(len(span), 0)
    _ = array


def test_from_numpy_array_read_only_raises() raises:
    var np = Python.import_module("numpy")
    var array = np.arange(4, dtype="float64")
    _ = array.setflags(write=False)
    with assert_raises(contains="writable"):
        _ = from_numpy_array[DType.float64](array)
    _ = array


def _assert_dtype_name[dtype: DType](expected: StaticString) raises:
    var values: List[Scalar[dtype]] = [Scalar[dtype](1), Scalar[dtype](2)]
    var arr = to_numpy_array(values)
    assert_equal(String(py=arr.dtype), String(expected))


def test_to_numpy_array_dtype_names() raises:
    _assert_dtype_name[DType.int8]("int8")
    _assert_dtype_name[DType.int16]("int16")
    _assert_dtype_name[DType.int32]("int32")
    _assert_dtype_name[DType.int64]("int64")
    _assert_dtype_name[DType.uint8]("uint8")
    _assert_dtype_name[DType.uint16]("uint16")
    _assert_dtype_name[DType.uint32]("uint32")
    _assert_dtype_name[DType.uint64]("uint64")
    _assert_dtype_name[DType.float16]("float16")
    _assert_dtype_name[DType.float32]("float32")
    _assert_dtype_name[DType.float64]("float64")


def _first_via_read_borrow[
    dtype: DType
](array: PythonObject) raises -> Scalar[dtype]:
    var span = from_numpy_array[dtype](array)
    return span[0]


def test_from_numpy_array_read_only_borrow() raises:
    var np = Python.import_module("numpy")
    var array = np.arange(4, dtype="float64")
    _ = array.setflags(write=False)
    var first = _first_via_read_borrow[DType.float64](array)
    assert_almost_equal(first, 0.0)
    _ = array


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
