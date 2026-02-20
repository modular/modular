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


from collections import Optional
from math import exp2

import testing
from reflection import call_location, SourceLocation
from testing.testing import _assert_cmp_error

from utils.numerics import FPUtils


# ===----------------------------------------------------------------------=== #
# Index formatting helpers
# ===----------------------------------------------------------------------=== #


fn _flat_to_nd_index(flat_idx: Int, shape: List[Int]) -> String:
    """Convert a flat index to an N-dimensional index string.

    Args:
        flat_idx: The flat (linear) index into the buffer.
        shape: The shape of the N-dimensional buffer.

    Returns:
        A string representation of the N-dimensional index, e.g., "[2, 3, 4]".
    """
    if len(shape) == 0:
        return String("i=", flat_idx)

    # Compute N-dimensional indices from flat index (row-major order)
    var indices = List[Int](capacity=len(shape))
    var remaining = flat_idx
    for dim_idx in range(len(shape) - 1, -1, -1):
        var dim_size = shape[dim_idx]
        indices.append(remaining % dim_size)
        remaining //= dim_size

    # Build string in correct order (indices were computed in reverse)
    var result = String("[")
    for idx in range(len(indices) - 1, -1, -1):
        if idx < len(indices) - 1:
            result += ", "
        result += String(indices[idx])
    result += "]"
    return result


fn _format_index(i: Int, shape: List[Int]) -> String:
    """Format an index for error messages.

    Args:
        i: The flat index.
        shape: Optional shape for N-dimensional index display. If empty,
               displays just the flat index.

    Returns:
        A formatted string like "at [2, 3, 4]" or "at i=52".
    """
    if len(shape) > 0:
        return _flat_to_nd_index(i, shape)
    else:
        return String("i=", i)


# ===----------------------------------------------------------------------=== #
# assert_almost_equal
# ===----------------------------------------------------------------------=== #


@always_inline
fn assert_almost_equal[
    dtype: DType,
    //,
](
    x: UnsafePointer[Scalar[dtype]],
    y: UnsafePointer[Scalar[dtype]],
    num_elements: Int,
    msg: String = "",
    *,
    shape: List[Int] = List[Int](),
    location: Optional[SourceLocation] = None,
    atol: Float64 = 1e-08,
    rtol: Float64 = 1e-05,
    equal_nan: Bool = False,
) raises:
    """Assert that two buffers are element-wise almost equal.

    Compares each element of `x` and `y` using the formula:
    `|x - y| <= atol + rtol * |y|`

    Args:
        x: Pointer to the first buffer.
        y: Pointer to the second buffer.
        num_elements: Number of elements to compare.
        msg: Optional message to include in assertion errors.
        shape: Optional shape for N-dimensional index display in error messages.
               If provided, error messages will show indices like "[2, 3, 4]"
               instead of flat indices like "i=52".
        location: Optional source location for error reporting.
        atol: Absolute tolerance (default: 1e-08).
        rtol: Relative tolerance (default: 1e-05).
        equal_nan: If True, NaN values in the same position are considered equal.

    Raises:
        Error: If any elements differ by more than the specified tolerances.

    Example:
        ```mojo
        # Basic usage with flat index in errors:
        assert_almost_equal(a.data, b.data, a.num_elements())

        # With shape for better error messages:
        assert_almost_equal(
            a.data, b.data, a.num_elements(),
            shape=List[Int](2, 3, 4)
        )
        ```
    """
    for i in range(num_elements):
        testing.assert_almost_equal(
            x[i],
            y[i],
            msg=String(msg, " at ", _format_index(i, shape)),
            atol=atol,
            rtol=rtol,
            equal_nan=equal_nan,
            location=location.or_else(call_location()),
        )


# ===----------------------------------------------------------------------=== #
# assert_equal
# ===----------------------------------------------------------------------=== #


@always_inline
fn assert_equal[
    dtype: DType,
    //,
](
    x: UnsafePointer[Scalar[dtype]],
    y: UnsafePointer[Scalar[dtype]],
    num_elements: Int,
    msg: String = "",
    *,
    shape: List[Int] = List[Int](),
    location: Optional[SourceLocation] = None,
) raises:
    """Assert that two buffers are element-wise exactly equal.

    Args:
        x: Pointer to the first buffer.
        y: Pointer to the second buffer.
        num_elements: Number of elements to compare.
        msg: Optional message to include in assertion errors.
        shape: Optional shape for N-dimensional index display in error messages.
               If provided, error messages will show indices like "[2, 3, 4]"
               instead of flat indices like "i=52".
        location: Optional source location for error reporting.

    Raises:
        Error: If any elements are not exactly equal.

    Example:
        ```mojo
        # Basic usage:
        assert_equal(a.data, b.data, a.num_elements())

        # With shape for better error messages:
        assert_equal(
            a.data, b.data, a.num_elements(),
            shape=List[Int](2, 3, 4)
        )
        ```
    """
    for i in range(num_elements):
        testing.assert_equal(
            x[i],
            y[i],
            msg=String(msg, " at ", _format_index(i, shape)),
            location=location.or_else(call_location()),
        )


# ===----------------------------------------------------------------------=== #
# assert_with_measure
# ===----------------------------------------------------------------------=== #


@always_inline
fn assert_with_measure[
    dtype: DType,
    //,
    measure: fn[dtype: DType](
        UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
        UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
        Int,
    ) -> Float64,
](
    x: UnsafePointer[Scalar[dtype]],
    y: UnsafePointer[Scalar[dtype]],
    num_elements: Int,
    msg: String = "",
    *,
    location: Optional[SourceLocation] = None,
    threshold: Optional[Float64] = None,
) raises:
    """Assert that a custom measure between two buffers is below a threshold.

    Computes a measure (e.g., correlation, KL divergence) between `x` and `y`,
    and asserts that it does not exceed the specified threshold.

    Args:
        x: Pointer to the first buffer.
        y: Pointer to the second buffer.
        num_elements: Number of elements in each buffer.
        msg: Optional message to include in assertion errors.
        location: Optional source location for error reporting.
        threshold: Maximum allowed value for the measure. If not specified,
                   defaults to sqrt(machine epsilon) for the dtype.

    Parameters:
        dtype: The data type of the buffer elements.
        measure: A function that computes a scalar measure between two buffers.
                 Signature: `fn[dtype](ptr1, ptr2, n) -> Float64`

    Raises:
        Error: If the computed measure exceeds the threshold.

    Example:
        ```mojo
        from internal_utils._measure import relative_difference

        assert_with_measure[relative_difference](
            a.data, b.data, a.num_elements(),
            threshold=0.001
        )
        ```
    """
    comptime sqrt_eps = exp2(
        -0.5 * Float64(FPUtils[dtype].mantissa_width())
    ).cast[DType.float64]()
    var m = measure(
        x.address_space_cast[AddressSpace.GENERIC](),
        y.address_space_cast[AddressSpace.GENERIC](),
        num_elements,
    )
    var t = threshold.or_else(sqrt_eps)
    if m > t:
        raise _assert_cmp_error["`left > right`, left = measure"](
            String(m),
            String(t),
            msg=msg,
            loc=location.or_else(call_location()),
        )


# ===----------------------------------------------------------------------=== #
# pytorch_like_tolerances_for
# ===----------------------------------------------------------------------=== #


@always_inline
fn pytorch_like_tolerances_for[dtype: DType]() -> Tuple[Float64, Float64]:
    """Get PyTorch-like default tolerances for a given dtype.

    Returns tolerance values modeled after PyTorch's default tolerances
    for floating-point comparisons.

    Parameters:
        dtype: The data type to get tolerances for.

    Returns:
        A tuple of (rtol, atol) - relative and absolute tolerances.

    Example:
        ```mojo
        rtol, atol = pytorch_like_tolerances_for[DType.float16]()
        assert_almost_equal(x, y, n, rtol=rtol, atol=atol)
        ```
    """

    comptime if dtype == DType.float16:
        return (1e-3, 1e-5)
    elif dtype == DType.bfloat16:
        return (1.6e-2, 1e-5)
    elif dtype == DType.float32:
        return (1.3e-6, 1e-5)
    elif dtype == DType.float64:
        return (1e-7, 1e-7)
    else:
        return (0.0, 0.0)
