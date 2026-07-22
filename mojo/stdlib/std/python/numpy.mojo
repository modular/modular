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
"""NumPy interoperability helpers for Mojo collections.

These functions move flat numeric data between Mojo `Span` and NumPy
arrays when a Mojo program drives CPython (via `Python.import_module`), without
hand-written `ctypes` plumbing:

- `copy_to_numpy_array` builds a NumPy array from a Mojo `Span` by copying
  the data into a new, independent array.
- `from_numpy_array` borrows a NumPy array's buffer as a Mojo `Span`
  (zero-copy).

Only 1-D, C-contiguous arrays of the fixed-width numeric dtypes (`int8` through
`int64`, `uint8` through `uint64`, `float16`, `float32`, `float64`) are
supported. This targets the common case of handing computed numeric data to a
library such as `matplotlib`.
"""

from std.memory import unsafe_memcpy
from std.collections import Span

from .python import Python
from .python_object import PythonObject


def _numpy_dtype_name[dtype: DType]() -> Optional[StaticString]:
    """Returns the NumPy dtype string for `dtype`, or `None` if unsupported."""
    if dtype == DType.int8:
        return StaticString("int8")
    elif dtype == DType.int16:
        return StaticString("int16")
    elif dtype == DType.int32:
        return StaticString("int32")
    elif dtype == DType.int64:
        return StaticString("int64")
    elif dtype == DType.uint8:
        return StaticString("uint8")
    elif dtype == DType.uint16:
        return StaticString("uint16")
    elif dtype == DType.uint32:
        return StaticString("uint32")
    elif dtype == DType.uint64:
        return StaticString("uint64")
    elif dtype == DType.float16:
        return StaticString("float16")
    elif dtype == DType.float32:
        return StaticString("float32")
    elif dtype == DType.float64:
        return StaticString("float64")
    else:
        return None


def _is_numpy_dtype[dtype: DType]() -> Bool:
    """Reports whether `dtype` maps to a supported NumPy dtype."""
    return _numpy_dtype_name[dtype]() is not None


# ===----------------------------------------------------------------------=== #
# copy_to_numpy_array
# ===----------------------------------------------------------------------=== #


def copy_to_numpy_array[
    dtype: DType, origin: Origin
](data: Span[Scalar[dtype], origin]) raises -> PythonObject:
    """Builds a 1-D NumPy array from a Mojo `Span` of scalars.

    The data is copied into a new, independent NumPy array, so the result
    remains valid after `data` is later mutated or freed. Unlike
    `from_numpy_array`, which returns a zero-copy view, this function does not
    alias `data`: mutating `data` after this call is not reflected in the
    returned array, and writes to the array are not reflected in `data`.

    Example:

    ```mojo
    from std.python.numpy import copy_to_numpy_array
    from std.math import sin

    var values = List[Float64](capacity=1024)
    for i in range(1024):
        var x = Float64(i) * 0.01
        values.append(sin(x) * sin(x))

    var arr = copy_to_numpy_array(values)  # an independent NumPy float64 array
    ```

    Parameters:
        dtype: The element dtype of the span (inferred).
        origin: The origin of the span (inferred).

    Args:
        data: The scalars to copy into a NumPy array.

    Constraints:
        `dtype` must be one of the fixed-width numeric dtypes supported by
        NumPy: `int8`-`int64`, `uint8`-`uint64`, `float16`, `float32`, or
        `float64`.

    Returns:
        A 1-D NumPy `ndarray` of dtype `dtype` and length `len(data)`.

    Raises:
        If NumPy is unavailable, or if the underlying NumPy calls fail.
    """
    comptime is_supported = _is_numpy_dtype[dtype]()
    comptime assert is_supported, String(
        "copy_to_numpy_array: unsupported dtype '",
        dtype,
        "'; expected a fixed-width numeric dtype (int8-int64, uint8-uint64,",
        " float16, float32, or float64). Note: `Int` is a machine-word integer",
        " and is not supported here — use a fixed-width scalar such as",
        " `Int64` (for example, `[Int64(i) for i in range(n)]`).",
    )

    var np = Python.import_module("numpy")
    var n = len(data)
    var dtype_str = String(_numpy_dtype_name[dtype]().value())

    if n == 0:
        return np.empty(0, dtype=dtype_str)

    var arr = np.empty(n, dtype=dtype_str)
    var dst = arr.ctypes.data.unsafe_get_as_pointer[dtype]()
    unsafe_memcpy(dest=dst, src=data.unsafe_ptr(), count=n)
    return arr


# ===----------------------------------------------------------------------=== #
# from_numpy_array
# ===----------------------------------------------------------------------=== #


def from_numpy_array[
    mut: Bool,
    //,
    dtype: DType,
    origin: Origin[mut=mut],
](ref[origin] array: PythonObject) raises -> Span[Scalar[dtype], origin]:
    """Borrows a 1-D C-contiguous NumPy array as a Mojo `Span`.

    The returned span aliases the NumPy array's buffer; no bytes are copied. Its
    origin is tied to `array`, so the compiler keeps `array` alive for as long as
    the span is used; you must still not resize or reallocate `array` while the
    span is in use, or the span will dangle. Only pass arrays whose buffer is
    owned by NumPy (or another Python object).

    The borrow follows the mutability of the `array` reference: an immutable
    (`read`) reference yields a read-only span. A mutable reference
    yields a mutable span, so writes are visible to NumPy and vice versa.
    Creating a mutable span fails if the underlying NumPy array is not
    writable. Pass `array` as an immutable reference to avoid this error.

    Example:

    ```mojo
    from std.python import Python
    from std.python.numpy import from_numpy_array

    var np = Python.import_module("numpy")
    var array = np.arange(8, dtype="float64")
    var span = from_numpy_array[DType.float64](array)
    var total = Float64(0)
    for value in span:
        total += value
    ```

    Parameters:
        mut: The mutability of the borrow, inferred from `array`.
        dtype: The expected element dtype of the array.
        origin: The origin of the borrow, inferred from `array`.

    Args:
        array: A 1-D, C-contiguous NumPy `ndarray` whose dtype matches `dtype`.

    Constraints:
        `dtype` must be one of the fixed-width numeric dtypes supported by
        NumPy.

    Returns:
        A `Span` of length `array.size` viewing the array's buffer, with the same
        mutability and origin as the `array` binding.

    Raises:
        If `array` is not 1-D, is not C-contiguous, has a dtype that does not
        match `dtype`, or is not writable when borrowed mutably.
    """
    comptime is_supported = _is_numpy_dtype[dtype]()
    comptime assert is_supported, String(
        "from_numpy_array: unsupported dtype '",
        dtype,
        "'; expected a fixed-width numeric dtype (int8-int64, uint8-uint64,",
        " float16, float32, or float64).",
    )

    var ndim = Int(py=array.ndim)
    if ndim != 1:
        raise Error(
            String(t"from_numpy_array: expected a 1-D array, got ndim={ndim}")
        )

    var actual_dtype = String(py=array.dtype)
    var expected_dtype = String(_numpy_dtype_name[dtype]().value())
    if actual_dtype != expected_dtype:
        raise Error(
            String(
                t"from_numpy_array: dtype mismatch: array is '{actual_dtype}'"
                t" but '{expected_dtype}' was requested"
            )
        )

    if not Bool(py=array.flags["C_CONTIGUOUS"]):
        raise Error("from_numpy_array: array must be C-contiguous")

    comptime if mut:
        if not Bool(py=array.flags["WRITEABLE"]):
            raise Error(
                "from_numpy_array: a mutable borrow requires a writable array"
                " (bind `array` as a `read` argument to borrow a read-only"
                " array)"
            )

    var n = Int(py=array.size)
    # `unsafe_get_as_pointer` yields a mutable `MutAnyOrigin` pointer (the
    # buffer is Python-owned). Cast its mutability and origin to match the
    # inferred `origin` so the span's lifetime and mutability are checked
    # against the `array` binding.
    var ptr = array.ctypes.data.unsafe_get_as_pointer[dtype]()
    return Span[Scalar[dtype], origin](
        ptr=ptr.unsafe_mut_cast[mut]().unsafe_origin_cast[origin](), length=n
    )
