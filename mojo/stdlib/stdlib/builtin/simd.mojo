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
"""Implements SIMD primitives and abstractions.

Provides high-performance SIMD primitives and abstractions for
vectorized computation in Mojo. It enables efficient data-parallel operations
by leveraging hardware vector processing units across different architectures.

Key Features:
1. Architecture-agnostic SIMD abstractions with automatic hardware detection
2. Optimized vector operations for common numerical computations
3. Explicit control over vectorization strategies and memory layouts
4. Zero-cost abstractions that compile to efficient machine code
5. Support for different vector widths and element types

Primary Components:
- Vector types: Strongly-typed vector containers with element-wise operations
- SIMD intrinsics: Low-level access to hardware SIMD instructions
- Vectorized algorithms: Common algorithms optimized for SIMD execution
- Memory utilities: Aligned memory allocation and vector load/store operations

Performance Considerations:
- Vector width selection should match target hardware capabilities
- Memory alignment affects load/store performance
- Data layout transformations may be necessary for optimal vectorization

Integration:
This module is designed to work seamlessly with other Mojo numerical computing
components, including tensor operations, linear algebra routines, and
domain-specific libraries for machine learning and scientific computing.
"""

import math
from collections import InlineArray
from collections.string.string import (
    _calc_format_buffer_size,
    _calc_initial_buffer_size,
)
from hashlib._hasher import _HashableWithHasher, _Hasher
from hashlib.hash import _hash_simd
from math import Ceilable, CeilDivable, Floorable, Truncable
from math.math import _call_ptx_intrinsic
from os import abort
from sys import (
    CompilationTarget,
    PrefetchOptions,
    _RegisterPackType,
    alignof,
    bitwidthof,
    has_neon,
    is_amd_gpu,
    is_big_endian,
    is_gpu,
    is_nvidia_gpu,
    llvm_intrinsic,
    prefetch,
    simdwidthof,
    sizeof,
)
from sys._assembly import inlined_assembly
from sys.info import _is_sm_9x_or_newer

from bit import byte_swap, pop_count
from builtin._format_float import _write_float
from builtin.device_passable import DevicePassable
from builtin.dtype import _uint_type_of_width
from builtin.format_int import _try_write_int
from builtin.io import _snprintf
from builtin.math import Powable
from documentation import doc_private
from memory import Span, UnsafePointer, bitcast, memcpy
from python import PythonConvertible, PythonObject, Python

from utils import IndexList, StaticTuple
from utils._visualizers import lldb_formatter_wrapping_type
from utils.numerics import FPUtils
from utils.numerics import isinf as _isinf
from utils.numerics import isnan as _isnan
from utils.numerics import max_finite as _max_finite
from utils.numerics import max_or_inf as _max_or_inf
from utils.numerics import min_finite as _min_finite
from utils.numerics import min_or_neg_inf as _min_or_neg_inf
from utils.numerics import nan as _nan

from .dtype import (
    _get_dtype_printf_format,
    _integral_type_of,
    _scientific_notation_digits,
    _unsigned_integral_type_of,
)

# ===----------------------------------------------------------------------=== #
# Type Aliases
# ===----------------------------------------------------------------------=== #

alias Scalar = SIMD[_, size=1]
"""Represents a scalar dtype."""

alias Int8 = Scalar[DType.int8]
"""Represents an 8-bit signed scalar integer."""
alias UInt8 = Scalar[DType.uint8]
"""Represents an 8-bit unsigned scalar integer."""
alias Int16 = Scalar[DType.int16]
"""Represents a 16-bit signed scalar integer."""
alias UInt16 = Scalar[DType.uint16]
"""Represents a 16-bit unsigned scalar integer."""
alias Int32 = Scalar[DType.int32]
"""Represents a 32-bit signed scalar integer."""
alias UInt32 = Scalar[DType.uint32]
"""Represents a 32-bit unsigned scalar integer."""
alias Int64 = Scalar[DType.int64]
"""Represents a 64-bit signed scalar integer."""
alias UInt64 = Scalar[DType.uint64]
"""Represents a 64-bit unsigned scalar integer."""
alias Int128 = Scalar[DType.int128]
"""Represents a 128-bit signed scalar integer."""
alias UInt128 = Scalar[DType.uint128]
"""Represents a 128-bit unsigned scalar integer."""
alias Int256 = Scalar[DType.int256]
"""Represents a 256-bit signed scalar integer."""
alias UInt256 = Scalar[DType.uint256]
"""Represents a 256-bit unsigned scalar integer."""

alias Float8_e5m2 = Scalar[DType.float8_e5m2]
"""Represents the 8-bit E5M2 floating point format from the [OFP8
standard](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-12-01-pdf-1),
encoded as `seeeeemm`:
- (s)ign: 1 bit
- (e)xponent: 5 bits
- (m)antissa: 2 bits
- exponent bias: 15
- nan: {0,1}11111{01,10,11}
- inf: 01111100
- -inf: 11111100
- -0: 10000000
"""
alias Float8_e5m2fnuz = Scalar[DType.float8_e5m2fnuz]
"""Represents an 8-bit floating point format, encoded as `seeeeemm`:
- (s)ign: 1 bit
- (e)xponent: 5 bits
- (m)antissa: 2 bits
- exponent bias: 16
- nan: 10000000
- fn: finite (no inf or -inf encodings)
- uz: unsigned zero (no -0 encoding)
"""
alias Float8_e4m3fn = Scalar[DType.float8_e4m3fn]
"""Represents the E4M3 floating point format defined in the [OFP8
standard](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-12-01-pdf-1).

This type is named differently across libraries and vendors, for example:
- Mojo, PyTorch, JAX, and LLVM refer to it as `e4m3fn`.
- OCP, NVIDIA CUDA, and AMD ROCm refer to it as `e4m3`.

In these contexts, they are all referring to the same finite type specified
in the OFP8 standard above, encoded as `seeeemmm`:
- (s)ign: 1 bit
- (e)xponent: 4 bits
- (m)antissa: 3 bits
- exponent bias: 7
- nan: 01111111, 11111111
- -0: 10000000
- fn: finite (no inf or -inf encodings)
"""
alias Float8_e4m3fnuz = Scalar[DType.float8_e4m3fnuz]
"""Represents an 8-bit e4m3fnuz floating point format, encoded as
`seeeemmm`:
- (s)ign: 1 bit
- (e)xponent: 4 bits
- (m)antissa: 3 bits
- exponent bias: 8
- nan: 10000000
- fn: finite (no inf or -inf encodings)
- uz: unsigned zero (no -0 encoding)
"""
alias BFloat16 = Scalar[DType.bfloat16]
"""Represents a 16-bit brain floating point value."""
alias Float16 = Scalar[DType.float16]
"""Represents a 16-bit floating point value."""
alias Float32 = Scalar[DType.float32]
"""Represents a 32-bit floating point value."""
alias Float64 = Scalar[DType.float64]
"""Represents a 64-bit floating point value."""

alias Byte = UInt8
"""Represents a byte (backed by an 8-bit unsigned integer)."""

# ===----------------------------------------------------------------------=== #
# Utilities
# ===----------------------------------------------------------------------=== #


@always_inline("nodebug")
fn _simd_construction_checks[dtype: DType, size: Int]():
    """Checks if the SIMD size is valid.

    The SIMD size is valid if it is a power of two and is positive.

    Parameters:
      dtype: The data type of SIMD vector elements.
      size: The number of elements in the SIMD vector. The size must not be greater than 2**15.
    """
    constrained[
        dtype is not DType.invalid, "simd type cannot be DType.invalid"
    ]()
    constrained[size.is_power_of_two(), "simd width must be power of 2"]()
    constrained[
        not (dtype is DType.bfloat16 and has_neon()),
        "bf16 is not supported for ARM architectures",
    ]()
    # MOCO-1388: Until LLVM's issue #122571 is fixed, LLVM's SelectionDAG has
    # a limit of 2^15 for the number of operands of the instruction.
    # NOTE: Even after the limit increases in LLVM, compile time might be 3x
    # slower than with GCC, therefore until we have a real use case for large
    # SIMD, we better to keep limit at 2^15.
    # NOTE: Might need to revisit the limit for targets that use GlobalISel
    # as it does have smaller limit now.
    constrained[
        size <= 2**15, "simd size is too large and must be less than 2^15"
    ]()


@always_inline("nodebug")
fn _unchecked_zero[dtype: DType, size: Int]() -> SIMD[dtype, size]:
    var zero = __mlir_op.`pop.cast`[_type = Scalar[dtype]._mlir_type](
        __mlir_attr.`#pop.simd<0> : !pop.scalar<index>`
    )
    return Scalar[dtype](zero)


@always_inline("nodebug")
fn _has_native_bf16_support() -> Bool:
    return is_gpu()


@always_inline("nodebug")
fn _has_native_f8_support() -> Bool:
    return _is_sm_9x_or_newer() or is_nvidia_gpu["sm_89"]() or is_amd_gpu()


# ===----------------------------------------------------------------------=== #
# SIMD
# ===----------------------------------------------------------------------=== #


@lldb_formatter_wrapping_type
@register_passable("trivial")
struct SIMD[dtype: DType, size: Int](
    Absable,
    Boolable,
    Defaultable,
    Ceilable,
    CeilDivable,
    Copyable,
    Movable,
    DevicePassable,
    ExplicitlyCopyable,
    Floatable,
    Floorable,
    Hashable,
    _HashableWithHasher,
    Indexer,
    Powable,
    PythonConvertible,
    Representable,
    Roundable,
    Sized,
    Stringable,
    Truncable,
    Writable,
):
    """Represents a small vector that is backed by a hardware vector element.

    SIMD allows a single instruction to be executed across the multiple data
    elements of the vector.

    Constraints:
        The size of the SIMD vector to be positive and a power of 2.

    Parameters:
        dtype: The data type of SIMD vector elements.
        size: The size of the SIMD vector.
    """

    alias device_type: AnyTrivialRegType = Self
    """SIMD types are remapped to the same type when passed to accelerator devices."""

    fn _to_device_type(self, target: UnsafePointer[NoneType]):
        """Device type mapping is the identity function."""
        target.bitcast[Self.device_type]()[] = self

    @staticmethod
    fn get_type_name() -> String:
        """
        Gets this type's name, for use in error messages when handing arguments
        to kernels.
        TODO: This will go away soon, when we get better error messages for
        kernel calls.

        Returns:
            This type's name.
        """
        return "SIMD[" + repr(dtype) + ", " + repr(size) + "]"

    @staticmethod
    fn get_device_type_name() -> String:
        """
        Gets device_type's name, for use in error messages when handing
        arguments to kernels.
        TODO: This will go away soon, when we get better error messages for
        kernel calls.

        Returns:
            This type's name.
        """
        return Self.get_type_name()

    # Fields
    alias _Mask = SIMD[DType.bool, size]

    alias element_type = dtype
    alias _mlir_type = __mlir_type[
        `!pop.simd<`, size.value, `, `, dtype.value, `>`
    ]

    var value: Self._mlir_type
    """The underlying storage for the vector."""

    alias MAX = Self(_max_or_inf[dtype]())
    """Gets the maximum value for the SIMD value, potentially +inf."""

    alias MIN = Self(_min_or_neg_inf[dtype]())
    """Gets the minimum value for the SIMD value, potentially -inf."""

    alias MAX_FINITE = Self(_max_finite[dtype]())
    """Returns the maximum finite value of SIMD value."""

    alias MIN_FINITE = Self(_min_finite[dtype]())
    """Returns the minimum (lowest) finite value of SIMD value."""

    alias _default_alignment = alignof[Scalar[dtype]]() if is_gpu() else 1

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @always_inline("nodebug")
    fn __init__(out self):
        """Default initializer of the SIMD vector.

        By default the SIMD vectors are initialized to all zeros.
        """
        _simd_construction_checks[dtype, size]()
        self = _unchecked_zero[dtype, size]()

    @always_inline("nodebug")
    fn __init__[
        other_dtype: DType, //
    ](out self, value: SIMD[other_dtype, size], /):
        """Initialize from another SIMD of the same size. If the value
        passed is a scalar, you can initialize a SIMD vector with more elements.

        Parameters:
            other_dtype: The type of the value that is being cast from.

        Args:
            value: The value to cast from.

        Example:

        ```mojo
        print(UInt64(UInt8(42))) # 42
        print(SIMD[DType.uint64, 4](UInt8(42))) # [42, 42, 42, 42]
        ```

        Casting behavior:

        ```mojo
        # Basic casting preserves value within range
        Int8(UInt8(127)) == Int8(127)

        # Numbers above signed max wrap to negative using two's complement
        Int8(UInt8(128)) == Int8(-128)
        Int8(UInt8(129)) == Int8(-127)
        Int8(UInt8(256)) == Int8(0)

        # Negative signed cast to unsigned using two's complement
        UInt8(Int8(-128)) == UInt8(128)
        UInt8(Int8(-127)) == UInt8(129)
        UInt8(Int8(-1)) == UInt8(255)

        # Truncate precision after downcast and upcast
        Float64(Float32(Float64(123456789.123456789))) == Float64(123456792.0)

        # Rightmost bits of significand become 0's on upcast
        Float64(Float32(0.3)) == Float64(0.30000001192092896)

        # Numbers equal after truncation of float literal and cast truncation
        Float32(Float64(123456789.123456789)) == Float32(123456789.123456789)

        # Float to int/uint floors
        Int64(Float64(42.2)) == Int64(42)
        ```
        .
        """
        self = value.cast[dtype]()

    @always_inline
    fn copy(self) -> Self:
        """Explicitly construct a copy of self.

        Returns:
            A copy of this value.
        """
        return self

    @always_inline("nodebug")
    @implicit
    fn __init__(out self, value: UInt, /):
        """Initializes the SIMD vector with an unsigned integer.

        The unsigned integer value is splatted across all the elements of the SIMD
        vector.

        Args:
            value: The input value.
        """

        @parameter
        if bitwidthof[dtype]() > bitwidthof[DType.index]():
            alias dt = _unsigned_integral_type_of[DType.index]()
            self = Self(bitcast[dt](Scalar[DType.index](value)))
        else:
            self = Self(value.value)

    @always_inline("nodebug")
    @implicit
    fn __init__(out self, value: Int, /):
        """Initializes the SIMD vector with a signed integer.

        The signed integer value is splatted across all the elements of the SIMD
        vector.

        Args:
            value: The input value.
        """
        self = Self(value.value)

    @doc_private
    @always_inline("nodebug")
    @implicit
    fn __init__(out self, value: __mlir_type.index, /):
        _simd_construction_checks[dtype, size]()
        var t0 = __mlir_op.`pop.cast_from_builtin`[
            _type = __mlir_type.`!pop.scalar<index>`
        ](value)
        var casted = __mlir_op.`pop.cast`[_type = Scalar[dtype]._mlir_type](t0)
        self = Scalar[dtype](casted)

    @always_inline
    fn __init__[T: Floatable, //](out self: Float64, value: T, /):
        """Initialize a Float64 from a type conforming to Floatable.

        Parameters:
            T: The Floatable type.

        Args:
            value: The object to get the float point representation of.
        """
        self = value.__float__()

    @always_inline
    fn __init__[T: FloatableRaising, //](out self: Float64, value: T, /) raises:
        """Initialize a Float64 from a type conforming to FloatableRaising.

        Parameters:
            T: The FloatableRaising type.

        Args:
            value: The object to get the float point representation of.

        Raises:
            If the type does not have a float point representation.
        """
        self = value.__float__()

    @always_inline
    fn __init__[
        *, `_`: Int = 0
    ](out self: Float64, value: PythonObject, /) raises:
        """Initialize a Float64 from a PythonObject.

        Parameters:
            _: A dummy parameter to ensure this overload has lower priority than
                the others. Its value is ignored.

        Args:
            value: The PythonObject to convert.

        Raises:
            If the conversion to double fails.
        """
        # TODO(MSTDL-1587): Remove the dummy parameter.
        var float_obj = value.__float__()
        var cpython = Python().cpython()
        self = Float64(cpython.PyFloat_AsDouble(float_obj.py_object))
        if self == -1.0 and cpython.PyErr_Occurred():
            # Note that -1.0 does not guarantee an error, it just means we need
            # to check if there was an exception. This is also very unlikely,
            # since the __float__ call above will throw if the underlying Python
            # method fails. Therefore this can only happen if a custom __float__
            # implementation is incorrect, and returns a non-double value.
            raise cpython.get_error()

        _ = float_obj

    @always_inline("nodebug")
    @implicit
    fn __init__(out self, value: IntLiteral, /):
        """Initializes the SIMD vector with an integer.

        The integer value is splatted across all the elements of the SIMD
        vector.

        Args:
            value: The input value.
        """
        _simd_construction_checks[dtype, size]()

        var tn1 = __mlir_attr[
            `#pop<int_literal_convert<`, value.value, `, 0>> : si128`
        ]
        var t0 = __mlir_op.`pop.cast_from_builtin`[
            _type = __mlir_type.`!pop.scalar<si128>`
        ](tn1)
        var casted = __mlir_op.`pop.cast`[_type = Scalar[dtype]._mlir_type](t0)
        self = Scalar[dtype](casted)

    @always_inline("nodebug")
    @implicit
    fn __init__(out self: SIMD[DType.bool, size], value: Bool, /):
        """Initializes the SIMD vector with a bool value.

        The bool value is splatted across all elements of the SIMD vector.

        Args:
            value: The bool value.
        """
        _simd_construction_checks[dtype, size]()

        var casted = __mlir_op.`pop.cast_from_builtin`[
            _type = __mlir_type.`!pop.scalar<bool>`
        ](value.value)
        self = Scalar[DType.bool](casted)

    @always_inline("nodebug")
    @implicit
    fn __init__(out self, value: Self._mlir_type, /):
        """Initializes the SIMD vector with the underlying mlir value.

        Args:
            value: The input value.
        """
        _simd_construction_checks[dtype, size]()
        self.value = value

    @always_inline("nodebug")
    @implicit
    fn __init__(out self, value: Scalar[dtype], /):
        """Constructs a SIMD vector by splatting a scalar value.

        The input value is splatted across all elements of the SIMD vector.

        Args:
            value: The value to splat to the elements of the vector.
        """
        _simd_construction_checks[dtype, size]()

        # Construct by broadcasting a scalar.
        self.value = __mlir_op.`pop.simd.splat`[_type = Self._mlir_type](
            value.value
        )

    @always_inline("nodebug")
    fn __init__(out self, *elems: Scalar[dtype], __list_literal__: () = ()):
        """Constructs a SIMD vector via a variadic list of elements.

        The input values are assigned to the corresponding elements of the SIMD
        vector.

        Constraints:
            The number of input values is equal to size of the SIMD vector.

        Args:
            elems: The variadic list of elements from which the SIMD vector is
                   constructed.
            __list_literal__: Tell Mojo to use this method for list literals.
        """
        _simd_construction_checks[dtype, size]()

        # TODO: Make this a compile-time check when possible.
        debug_assert(
            size == len(elems),
            (
                "mismatch in the number of elements in the SIMD variadic"
                " constructor"
            ),
        )

        __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(self))

        @parameter
        for i in range(size):
            self[i] = elems[i]

    # TODO: should be "builtin" when constrained is replaced with 'requires'.
    @always_inline("nodebug")
    @implicit
    fn __init__(out self, value: FloatLiteral, /):
        """Initializes the SIMD vector with a float.

        The value is splatted across all the elements of the SIMD
        vector.

        Args:
            value: The input value.
        """
        _simd_construction_checks[dtype, size]()
        constrained[
            dtype.is_floating_point(), "the SIMD type must be floating point"
        ]()

        # float_literal_convert implicitly splats to !pop.simd as needed.
        return __mlir_attr[
            `#pop<float_literal_convert<`,
            value.value,
            `>> : `,
            Self._mlir_type,
        ]

    @staticmethod
    fn from_bits[
        int_dtype: DType, //
    ](value: SIMD[int_dtype, size]) -> SIMD[dtype, size]:
        """Initializes the SIMD vector from the bits of an integral SIMD vector.

        Parameters:
            int_dtype: The integral type of the input SIMD vector.

        Args:
            value: The SIMD vector to copy the bits from.

        Returns:
            The bitcast SIMD vector.
        """
        constrained[int_dtype.is_integral(), "the SIMD type must be integral"]()
        return bitcast[dtype, size](value)

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    @always_inline("nodebug")
    fn __getitem__(self, idx: Int) -> Scalar[dtype]:
        """Gets an element from the vector.

        Args:
            idx: The element index.

        Returns:
            The value at position `idx`.
        """
        return __mlir_op.`pop.simd.extractelement`(self.value, idx.value)

    @always_inline("nodebug")
    fn __setitem__(mut self, idx: Int, val: Scalar[dtype]):
        """Sets an element in the vector.

        Args:
            idx: The index to set.
            val: The value to set.
        """
        self.value = __mlir_op.`pop.simd.insertelement`(
            self.value, val.value, idx.value
        )

    fn __contains__(self, value: Scalar[dtype]) -> Bool:
        """Whether the vector contains the value.

        Args:
            value: The value.

        Returns:
            Whether the vector contains the value.
        """
        return (self == value).reduce_or()

    @always_inline("nodebug")
    fn __add__(self, rhs: Self) -> Self:
        """Computes `self + rhs`.

        Args:
            rhs: The rhs value.

        Returns:
            A new vector whose element at position `i` is computed as
            `self[i] + rhs[i]`.
        """
        constrained[dtype.is_numeric(), "the SIMD type must be numeric"]()
        return __mlir_op.`pop.add`(self.value, rhs.value)

    @always_inline("nodebug")
    fn __sub__(self, rhs: Self) -> Self:
        """Computes `self - rhs`.

        Args:
            rhs: The rhs value.

        Returns:
            A new vector whose element at position `i` is computed as
            `self[i] - rhs[i]`.
        """
        constrained[dtype.is_numeric(), "the SIMD type must be numeric"]()
        return __mlir_op.`pop.sub`(self.value, rhs.value)

    @always_inline("nodebug")
    fn __mul__(self, rhs: Self) -> Self:
        """Computes `self * rhs`.

        Args:
            rhs: The rhs value.

        Returns:
            A new vector whose element at position `i` is computed as
            `self[i] * rhs[i]`.
        """

        @parameter
        if dtype is DType.bool:
            return (rebind[Self._Mask](self) & rebind[Self._Mask](rhs)).cast[
                dtype
            ]()
        return __mlir_op.`pop.mul`(self.value, rhs.value)

    @always_inline("nodebug")
    fn __truediv__(self, rhs: Self) -> Self:
        """Computes `self / rhs`.

        Args:
            rhs: The rhs value.

        Returns:
            A new vector whose element at position `i` is computed as
            `self[i] / rhs[i]`.
        """
        constrained[dtype.is_numeric(), "the SIMD type must be numeric"]()
        return __mlir_op.`pop.div`(self.value, rhs.value)

    @always_inline("nodebug")
    fn __floordiv__(self, rhs: Self) -> Self:
        """Returns the division of self and rhs rounded down to the nearest
        integer.

        Constraints:
            The element type of the SIMD vector must be numeric.

        Args:
            rhs: The value to divide with.

        Returns:
            `floor(self / rhs)` value.
        """
        constrained[dtype.is_numeric(), "the type must be numeric"]()

        if not any(rhs):
            # this should raise an exception.
            return 0

        var div = self / rhs

        @parameter
        if dtype.is_floating_point():
            return div.__floor__()
        elif dtype.is_unsigned():
            return div
        else:
            if all((self > 0) & (rhs > 0)):
                return div

            var mod = self - div * rhs
            var mask = ((rhs < 0) ^ (self < 0)) & (mod != 0)
            return div - mask.cast[dtype]()

    @always_inline("nodebug")
    fn __mod__(self, rhs: Self) -> Self:
        """Returns the remainder of self divided by rhs.

        Args:
            rhs: The value to divide on.

        Returns:
            The remainder of dividing self by rhs.
        """
        constrained[dtype.is_numeric(), "the type must be numeric"]()

        if not any(rhs):
            # this should raise an exception.
            return 0

        @parameter
        if dtype.is_unsigned():
            return __mlir_op.`pop.rem`(self.value, rhs.value)
        else:
            var div = self / rhs

            @parameter
            if dtype.is_floating_point():
                div = llvm_intrinsic["llvm.trunc", Self, has_side_effect=False](
                    div
                )

            var mod = self - div * rhs
            var mask = ((rhs < 0) ^ (self < 0)) & (mod != 0)
            return mod + mask.select(rhs, Self(0))

    @always_inline("nodebug")
    fn __pow__(self, exp: Int) -> Self:
        """Computes the vector raised to the power of the input integer value.

        Args:
            exp: The exponent value.

        Returns:
            A SIMD vector where each element is raised to the power of the
            specified exponent value.
        """
        constrained[dtype.is_numeric(), "the SIMD type must be numeric"]()
        return _pow(self, SIMD[DType.index, size](exp))

    # TODO(#22771): remove this overload.
    @always_inline("nodebug")
    fn __pow__(self, exp: Self) -> Self:
        """Computes the vector raised elementwise to the right hand side power.

        Args:
            exp: The exponent value.

        Returns:
            A SIMD vector where each element is raised to the power of the
            specified exponent value.
        """
        constrained[dtype.is_numeric(), "the SIMD type must be numeric"]()
        return _pow(self, exp)

    @always_inline("nodebug")
    fn __lt__(self, rhs: Self) -> Self._Mask:
        """Compares two SIMD vectors using less-than comparison.

        Args:
            rhs: The rhs of the operation.

        Returns:
            A new bool SIMD vector of the same size whose element at position
            `i` is True or False depending on the expression
            `self[i] < rhs[i]`.
        """

        return __mlir_op.`pop.cmp`[pred = __mlir_attr.`#pop<cmp_pred lt>`](
            self.value, rhs.value
        )

    @always_inline("nodebug")
    fn __le__(self, rhs: Self) -> Self._Mask:
        """Compares two SIMD vectors using less-than-or-equal comparison.

        Args:
            rhs: The rhs of the operation.

        Returns:
            A new bool SIMD vector of the same size whose element at position
            `i` is True or False depending on the expression
            `self[i] <= rhs[i]`.
        """

        return __mlir_op.`pop.cmp`[pred = __mlir_attr.`#pop<cmp_pred le>`](
            self.value, rhs.value
        )

    @always_inline("nodebug")
    fn __eq__(self, rhs: Self) -> Self._Mask:
        """Compares two SIMD vectors using equal-to comparison.

        Args:
            rhs: The rhs of the operation.

        Returns:
            A new bool SIMD vector of the same size whose element at position
            `i` is True or False depending on the expression
            `self[i] == rhs[i]`.
        """

        # TODO(KERN-228): support BF16 on neon systems.
        # As a workaround, we roll our own implementation
        @parameter
        if has_neon() and dtype is DType.bfloat16:
            return self.to_bits() == rhs.to_bits()
        else:
            return __mlir_op.`pop.cmp`[pred = __mlir_attr.`#pop<cmp_pred eq>`](
                self.value, rhs.value
            )

    @always_inline("nodebug")
    fn __ne__(self, rhs: Self) -> Self._Mask:
        """Compares two SIMD vectors using not-equal comparison.

        Args:
            rhs: The rhs of the operation.

        Returns:
            A new bool SIMD vector of the same size whose element at position
            `i` is True or False depending on the expression
            `self[i] != rhs[i]`.
        """

        # TODO(KERN-228): support BF16 on neon systems.
        # As a workaround, we roll our own implementation.
        @parameter
        if has_neon() and dtype is DType.bfloat16:
            return self.to_bits() != rhs.to_bits()
        else:
            return __mlir_op.`pop.cmp`[pred = __mlir_attr.`#pop<cmp_pred ne>`](
                self.value, rhs.value
            )

    @always_inline("nodebug")
    fn __gt__(self, rhs: Self) -> Self._Mask:
        """Compares two SIMD vectors using greater-than comparison.

        Args:
            rhs: The rhs of the operation.

        Returns:
            A new bool SIMD vector of the same size whose element at position
            `i` is True or False depending on the expression
            `self[i] > rhs[i]`.
        """

        return __mlir_op.`pop.cmp`[pred = __mlir_attr.`#pop<cmp_pred gt>`](
            self.value, rhs.value
        )

    @always_inline("nodebug")
    fn __ge__(self, rhs: Self) -> Self._Mask:
        """Compares two SIMD vectors using greater-than-or-equal comparison.

        Args:
            rhs: The rhs of the operation.

        Returns:
            A new bool SIMD vector of the same size whose element at position
            `i` is True or False depending on the expression
            `self[i] >= rhs[i]`.
        """

        return __mlir_op.`pop.cmp`[pred = __mlir_attr.`#pop<cmp_pred ge>`](
            self.value, rhs.value
        )

    @always_inline("nodebug")
    fn __pos__(self) -> Self:
        """Defines the unary `+` operation.

        Returns:
            This SIMD vector.
        """
        constrained[dtype.is_numeric(), "the SIMD type must be numeric"]()
        return self

    @always_inline("nodebug")
    fn __neg__(self) -> Self:
        """Defines the unary `-` operation.

        Returns:
            The negation of this SIMD vector.
        """
        constrained[dtype.is_numeric(), "the SIMD type must be numeric"]()
        return __mlir_op.`pop.neg`(self.value)

    @always_inline("nodebug")
    fn __and__(self, rhs: Self) -> Self:
        """Returns `self & rhs`.

        Constraints:
            The element type of the SIMD vector must be bool or integral.

        Args:
            rhs: The RHS value.

        Returns:
            `self & rhs`.
        """
        constrained[
            dtype.is_integral() or dtype is DType.bool,
            "must be an integral or bool type",
        ]()
        return __mlir_op.`pop.simd.and`(self.value, rhs.value)

    @always_inline("nodebug")
    fn __xor__(self, rhs: Self) -> Self:
        """Returns `self ^ rhs`.

        Constraints:
            The element type of the SIMD vector must be bool or integral.

        Args:
            rhs: The RHS value.

        Returns:
            `self ^ rhs`.
        """
        constrained[
            dtype.is_integral() or dtype is DType.bool,
            "must be an integral or bool type",
        ]()
        return __mlir_op.`pop.simd.xor`(self.value, rhs.value)

    @always_inline("nodebug")
    fn __or__(self, rhs: Self) -> Self:
        """Returns `self | rhs`.

        Constraints:
            The element type of the SIMD vector must be bool or integral.

        Args:
            rhs: The RHS value.

        Returns:
            `self | rhs`.
        """
        constrained[
            dtype.is_integral() or dtype is DType.bool,
            "must be an integral or bool type",
        ]()
        return __mlir_op.`pop.simd.or`(self.value, rhs.value)

    @always_inline("nodebug")
    fn __lshift__(self, rhs: Self) -> Self:
        """Returns `self << rhs`.

        Constraints:
            The element type of the SIMD vector must be integral.

        Args:
            rhs: The RHS value.

        Returns:
            `self << rhs`.
        """
        constrained[dtype.is_integral(), "must be an integral type"]()
        debug_assert(all(rhs >= 0), "unhandled negative value")
        debug_assert(
            all(rhs < bitwidthof[dtype]()), "unhandled value greater than size"
        )
        return __mlir_op.`pop.shl`(self.value, rhs.value)

    @always_inline("nodebug")
    fn __rshift__(self, rhs: Self) -> Self:
        """Returns `self >> rhs`.

        Constraints:
            The element type of the SIMD vector must be integral.

        Args:
            rhs: The RHS value.

        Returns:
            `self >> rhs`.
        """
        constrained[dtype.is_integral(), "must be an integral type"]()
        debug_assert(all(rhs >= 0), "unhandled negative value")
        debug_assert(
            all(rhs < bitwidthof[dtype]()), "unhandled value greater than size"
        )
        return __mlir_op.`pop.shr`(self.value, rhs.value)

    @always_inline("nodebug")
    fn __invert__(self) -> Self:
        """Returns `~self`.

        Constraints:
            The element type of the SIMD vector must be boolean or integral.

        Returns:
            The `~self` value.
        """
        constrained[
            dtype.is_integral() or dtype is DType.bool,
            "must be an integral or bool type",
        ]()

        @parameter
        if dtype is DType.bool:
            return rebind[Self](self.select(Self(False), Self(True)))
        else:
            return self ^ -1

    # ===------------------------------------------------------------------=== #
    # In place operations.
    # ===------------------------------------------------------------------=== #

    @always_inline("nodebug")
    fn __iadd__(mut self, rhs: Self):
        """Performs in-place addition.

        The vector is mutated where each element at position `i` is computed as
        `self[i] + rhs[i]`.

        Args:
            rhs: The rhs of the addition operation.
        """
        constrained[dtype.is_numeric(), "the SIMD type must be numeric"]()
        self = self + rhs

    @always_inline("nodebug")
    fn __isub__(mut self, rhs: Self):
        """Performs in-place subtraction.

        The vector is mutated where each element at position `i` is computed as
        `self[i] - rhs[i]`.

        Args:
            rhs: The rhs of the operation.
        """
        constrained[dtype.is_numeric(), "the SIMD type must be numeric"]()
        self = self - rhs

    @always_inline("nodebug")
    fn __imul__(mut self, rhs: Self):
        """Performs in-place multiplication.

        The vector is mutated where each element at position `i` is computed as
        `self[i] * rhs[i]`.

        Args:
            rhs: The rhs of the operation.
        """
        constrained[dtype.is_numeric(), "the SIMD type must be numeric"]()
        self = self * rhs

    @always_inline("nodebug")
    fn __itruediv__(mut self, rhs: Self):
        """In-place true divide operator.

        The vector is mutated where each element at position `i` is computed as
        `self[i] / rhs[i]`.

        Args:
            rhs: The rhs of the operation.
        """
        constrained[dtype.is_numeric(), "the SIMD type must be numeric"]()
        self = self / rhs

    @always_inline("nodebug")
    fn __ifloordiv__(mut self, rhs: Self):
        """In-place flood div operator.

        The vector is mutated where each element at position `i` is computed as
        `self[i] // rhs[i]`.

        Args:
            rhs: The rhs of the operation.
        """
        constrained[dtype.is_numeric(), "the SIMD type must be numeric"]()
        self = self // rhs

    @always_inline("nodebug")
    fn __imod__(mut self, rhs: Self):
        """In-place mod operator.

        The vector is mutated where each element at position `i` is computed as
        `self[i] % rhs[i]`.

        Args:
            rhs: The rhs of the operation.
        """
        constrained[dtype.is_numeric(), "the SIMD type must be numeric"]()
        self = self.__mod__(rhs)

    @always_inline("nodebug")
    fn __ipow__(mut self, rhs: Int):
        """In-place pow operator.

        The vector is mutated where each element at position `i` is computed as
        `pow(self[i], rhs)`.

        Args:
            rhs: The rhs of the operation.
        """
        constrained[dtype.is_numeric(), "the SIMD type must be numeric"]()
        self = self.__pow__(rhs)

    @always_inline("nodebug")
    fn __iand__(mut self, rhs: Self):
        """Computes `self & rhs` and save the result in `self`.

        Constraints:
            The element type of the SIMD vector must be bool or integral.

        Args:
            rhs: The RHS value.
        """
        constrained[
            dtype.is_integral() or dtype is DType.bool,
            "must be an integral or bool type",
        ]()
        self = self & rhs

    @always_inline("nodebug")
    fn __ixor__(mut self, rhs: Self):
        """Computes `self ^ rhs` and save the result in `self`.

        Constraints:
            The element type of the SIMD vector must be bool or integral.

        Args:
            rhs: The RHS value.
        """
        constrained[
            dtype.is_integral() or dtype is DType.bool,
            "must be an integral or bool type",
        ]()
        self = self ^ rhs

    @always_inline("nodebug")
    fn __ior__(mut self, rhs: Self):
        """Computes `self | rhs` and save the result in `self`.

        Constraints:
            The element type of the SIMD vector must be bool or integral.

        Args:
            rhs: The RHS value.
        """
        constrained[
            dtype.is_integral() or dtype is DType.bool,
            "must be an integral or bool type",
        ]()
        self = self | rhs

    @always_inline("nodebug")
    fn __ilshift__(mut self, rhs: Self):
        """Computes `self << rhs` and save the result in `self`.

        Constraints:
            The element type of the SIMD vector must be integral.

        Args:
            rhs: The RHS value.
        """
        constrained[dtype.is_integral(), "must be an integral type"]()
        self = self << rhs

    @always_inline("nodebug")
    fn __irshift__(mut self, rhs: Self):
        """Computes `self >> rhs` and save the result in `self`.

        Constraints:
            The element type of the SIMD vector must be integral.

        Args:
            rhs: The RHS value.
        """
        constrained[dtype.is_integral(), "must be an integral type"]()
        self = self >> rhs

    # ===------------------------------------------------------------------=== #
    # Reversed operations
    # ===------------------------------------------------------------------=== #

    @always_inline("nodebug")
    fn __radd__(self, value: Self) -> Self:
        """Returns `value + self`.

        Args:
            value: The other value.

        Returns:
            `value + self`.
        """
        constrained[dtype.is_numeric(), "the SIMD type must be numeric"]()
        return value + self

    @always_inline("nodebug")
    fn __rsub__(self, value: Self) -> Self:
        """Returns `value - self`.

        Args:
            value: The other value.

        Returns:
            `value - self`.
        """
        constrained[dtype.is_numeric(), "the SIMD type must be numeric"]()
        return value - self

    @always_inline("nodebug")
    fn __rmul__(self, value: Self) -> Self:
        """Returns `value * self`.

        Args:
            value: The other value.

        Returns:
            `value * self`.
        """
        constrained[dtype.is_numeric(), "the SIMD type must be numeric"]()
        return value * self

    @always_inline("nodebug")
    fn __rfloordiv__(self, rhs: Self) -> Self:
        """Returns the division of rhs and self rounded down to the nearest
        integer.

        Constraints:
            The element type of the SIMD vector must be numeric.

        Args:
            rhs: The value to divide by self.

        Returns:
            `floor(rhs / self)` value.
        """
        constrained[dtype.is_numeric(), "the type must be numeric"]()
        return rhs // self

    @always_inline("nodebug")
    fn __rtruediv__(self, value: Self) -> Self:
        """Returns `value / self`.

        Args:
            value: The other value.

        Returns:
            `value / self`.
        """
        constrained[dtype.is_numeric(), "the SIMD type must be numeric"]()
        return value / self

    @always_inline("nodebug")
    fn __rmod__(self, value: Self) -> Self:
        """Returns `value mod self`.

        Args:
            value: The other value.

        Returns:
            `value mod self`.
        """
        constrained[dtype.is_numeric(), "the type must be numeric"]()
        return value % self

    @always_inline("nodebug")
    fn __rpow__(self, base: Self) -> Self:
        """Returns `base ** self`.

        Args:
            base: The base value.

        Returns:
            `base ** self`.
        """
        constrained[dtype.is_numeric(), "the type must be numeric"]()
        return base**self

    @always_inline("nodebug")
    fn __rand__(self, value: Self) -> Self:
        """Returns `value & self`.

        Constraints:
            The element type of the SIMD vector must be bool or integral.

        Args:
            value: The other value.

        Returns:
            `value & self`.
        """
        constrained[
            dtype.is_integral() or dtype is DType.bool,
            "must be an integral or bool type",
        ]()
        return value & self

    @always_inline("nodebug")
    fn __rxor__(self, value: Self) -> Self:
        """Returns `value ^ self`.

        Constraints:
            The element type of the SIMD vector must be bool or integral.

        Args:
            value: The other value.

        Returns:
            `value ^ self`.
        """
        constrained[
            dtype.is_integral() or dtype is DType.bool,
            "must be an integral or bool type",
        ]()
        return value ^ self

    @always_inline("nodebug")
    fn __ror__(self, value: Self) -> Self:
        """Returns `value | self`.

        Constraints:
            The element type of the SIMD vector must be bool or integral.

        Args:
            value: The other value.

        Returns:
            `value | self`.
        """
        constrained[
            dtype.is_integral() or dtype is DType.bool,
            "must be an integral or bool type",
        ]()
        return value | self

    @always_inline("nodebug")
    fn __rlshift__(self, value: Self) -> Self:
        """Returns `value << self`.

        Constraints:
            The element type of the SIMD vector must be integral.

        Args:
            value: The other value.

        Returns:
            `value << self`.
        """
        constrained[dtype.is_integral(), "must be an integral type"]()
        return value << self

    @always_inline("nodebug")
    fn __rrshift__(self, value: Self) -> Self:
        """Returns `value >> self`.

        Constraints:
            The element type of the SIMD vector must be integral.

        Args:
            value: The other value.

        Returns:
            `value >> self`.
        """
        constrained[dtype.is_integral(), "must be an integral type"]()
        return value >> self

    # ===------------------------------------------------------------------=== #
    # Trait implementations
    # ===------------------------------------------------------------------=== #

    fn to_python_object(owned self) raises -> PythonObject:
        """Convert this value to a PythonObject.

        Returns:
            A PythonObject representing the value.
        """
        constrained[size == 1, "only works with scalar values"]()
        return PythonObject(rebind[Scalar[dtype]](self))

    @always_inline("nodebug")
    fn __len__(self) -> Int:
        """Gets the length of the SIMD vector.

        Returns:
            The length of the SIMD vector.
        """

        return self.size

    @always_inline("nodebug")
    fn __bool__(self) -> Bool:
        """Converts the SIMD scalar into a boolean value.

        Constraints:
            The size of the SIMD vector must be 1.

        Returns:
            True if the SIMD scalar is non-zero and False otherwise.
        """
        constrained[
            size == 1,
            (
                "The truth value of a SIMD vector with more than one element is"
                " ambiguous. Use the builtin `any()` or `all()` functions"
                " instead."
            ),
        ]()
        return rebind[Scalar[DType.bool]](self.cast[DType.bool]()).value

    @always_inline("nodebug")
    fn __int__(self) -> Int:
        """Casts to the value to an Int. If there is a fractional component,
        then the fractional part is truncated.

        Constraints:
            The size of the SIMD vector must be 1.

        Returns:
            The value as an integer.
        """
        constrained[size == 1, "expected a scalar type"]()

        alias int_width = bitwidthof[Int]()
        alias type_width = bitwidthof[dtype]()

        @parameter
        if dtype.is_unsigned() and int_width > type_width:
            # If we are casting up, prevent sign extension by first casting to
            # a large unsigned
            return self.cast[_uint_type_of_width[int_width]()]().__int__()
        else:
            return rebind[Scalar[DType.index]](self.cast[DType.index]()).value

    @always_inline("nodebug")
    fn __index__(self) -> __mlir_type.index:
        """Convert to index.

        Returns:
            The corresponding __mlir_type.index value.
        """
        constrained[
            dtype.is_integral(), "cannot index using a floating point type"
        ]()
        return Int(self).value

    @always_inline("nodebug")
    fn __float__(self) -> Float64:
        """Casts the value to a float.

        Constraints:
            The size of the SIMD vector must be 1.

        Returns:
            The value as a float.
        """
        constrained[size == 1, "expected a scalar type"]()
        return rebind[Scalar[dtype]](self).cast[DType.float64]()

    @no_inline
    fn __str__(self) -> String:
        """Get the SIMD as a string.

        Returns:
            A string representation.
        """

        return String.write(self)

    @no_inline
    fn __repr__(self) -> String:
        """Get the representation of the SIMD value e.g. "SIMD[DType.int8, 2](1, 2)".

        Returns:
            The representation of the SIMD value.
        """
        var output = String()
        output.write("SIMD[" + dtype.__repr__() + ", ", size, "](")
        # Write each element.
        for i in range(size):
            var element = self[i]
            # Write separators between each element.
            if i != 0:
                output.write(", ")
            _write_scalar(output, element)
        output.write(")")
        return output^

    @always_inline("nodebug")
    fn __floor__(self) -> Self:
        """Performs elementwise floor on the elements of a SIMD vector.

        Returns:
            The elementwise floor of this SIMD vector.
        """
        return self._floor_ceil_trunc_impl["llvm.floor"]()

    @always_inline("nodebug")
    fn __ceil__(self) -> Self:
        """Performs elementwise ceiling on the elements of a SIMD vector.

        Returns:
            The elementwise ceiling of this SIMD vector.
        """
        return self._floor_ceil_trunc_impl["llvm.ceil"]()

    @always_inline("nodebug")
    fn __trunc__(self) -> Self:
        """Performs elementwise truncation on the elements of a SIMD vector.

        Returns:
            The elementwise truncated values of this SIMD vector.
        """
        return self._floor_ceil_trunc_impl["llvm.trunc"]()

    @always_inline
    fn __abs__(self) -> Self:
        """Defines the absolute value operation.

        Returns:
            The absolute value of this SIMD vector.
        """

        @parameter
        if dtype.is_unsigned() or dtype is DType.bool:
            return self
        elif dtype.is_integral():
            return (self < 0).select(-self, self)
        else:

            @parameter
            if is_nvidia_gpu():

                @parameter
                if dtype.is_half_float():
                    alias prefix = "abs.bf16" if dtype is DType.bfloat16 else "abs.f16"
                    return _call_ptx_intrinsic[
                        scalar_instruction=prefix,
                        vector2_instruction = prefix + "x2",
                        scalar_constraints="=h,h",
                        vector_constraints="=r,r",
                    ](self)
                return llvm_intrinsic["llvm.fabs", Self, has_side_effect=False](
                    self
                )

            alias mask = FPUtils[dtype].exponent_mantissa_mask()
            return Self.from_bits(self.to_bits() & mask)

    @always_inline("nodebug")
    fn __round__(self) -> Self:
        """Performs elementwise rounding on the elements of a SIMD vector.

        This rounding goes to the nearest integer with ties away from zero.

        Returns:
            The elementwise rounded value of this SIMD vector.
        """

        @parameter
        if dtype.is_integral() or dtype is DType.bool:
            return self

        return llvm_intrinsic["llvm.roundeven", Self, has_side_effect=False](
            self
        )

    @always_inline("nodebug")
    fn __round__(self, ndigits: Int) -> Self:
        """Performs elementwise rounding on the elements of a SIMD vector.

        This rounding goes to the nearest integer with ties away from zero.

        Args:
            ndigits: The number of digits to round to.

        Returns:
            The elementwise rounded value of this SIMD vector.
        """

        @parameter
        if dtype.is_integral() or dtype is DType.bool:
            return self

        var exp = Self(10) ** ndigits
        return (self * exp).__round__() / exp

    fn __hash__(self) -> UInt:
        """Hash the value using builtin hash.

        Returns:
            A 64-bit hash value. This value is _not_ suitable for cryptographic
            uses. Its intended usage is for data structures. See the `hash`
            builtin documentation for more details.
        """
        return _hash_simd(self)

    fn __hash__[H: _Hasher](self, mut hasher: H):
        """Updates hasher with this SIMD value.

        Parameters:
            H: The hasher type.

        Args:
            hasher: The hasher instance.
        """
        hasher._update_with_simd(self)

    @always_inline
    fn __ceildiv__(self, denominator: Self) -> Self:
        """Return the rounded-up result of dividing self by denominator.


        Args:
            denominator: The denominator.

        Returns:
            The ceiling of dividing numerator by denominator.
        """

        @parameter
        if dtype.is_signed():
            return -(self // -denominator)
        return (self + denominator - 1) // denominator

    # ===------------------------------------------------------------------=== #
    # Methods
    # ===------------------------------------------------------------------=== #

    @always_inline("nodebug")
    fn cast[target: DType](self) -> SIMD[target, size]:
        """Casts the elements of the SIMD vector to the target element type.

        Parameters:
            target: The target DType.

        Returns:
            A new SIMD vector whose elements have been casted to the target
            element type.

        Casting behavior:

        ```mojo
        # Basic casting preserves value within range
        Int8(UInt8(127)) == Int8(127)

        # Numbers above signed max wrap to negative using two's complement
        Int8(UInt8(128)) == Int8(-128)
        Int8(UInt8(129)) == Int8(-127)
        Int8(UInt8(256)) == Int8(0)

        # Negative signed cast to unsigned using two's complement
        UInt8(Int8(-128)) == UInt8(128)
        UInt8(Int8(-127)) == UInt8(129)
        UInt8(Int8(-1)) == UInt8(255)

        # Truncate precision after downcast and upcast
        Float64(Float32(Float64(123456789.123456789))) == Float64(123456792.0)

        # Rightmost bits of significand become 0's on upcast
        Float64(Float32(0.3)) == Float64(0.30000001192092896)

        # Numbers equal after truncation of float literal and cast truncation
        Float32(Float64(123456789.123456789)) == Float32(123456789.123456789)

        # Float to int/uint floors
        Int64(Float64(42.2)) == Int64(42)
        ```
        .
        """

        alias Target = SIMD[target, size]

        @parameter
        if dtype is target:
            return rebind[Target](self)

        @parameter
        if is_nvidia_gpu():

            @parameter
            if dtype is DType.bfloat16 and target is DType.float64:
                # Convert to F64 via a Float32 pathway. This would allow us to
                # use the optimizations defined above.
                return self.cast[DType.float32]().cast[target]()

        @parameter
        if target in (DType.float8_e4m3fn, DType.float8_e5m2):
            # TODO(KERN-1488): use gpu (H100) instruction to convert from fp16 to fp8
            return _convert_f32_to_float8[target=target, size=size](
                self.cast[DType.float32]()
            )

        @parameter
        if dtype in (DType.float8_e4m3fn, DType.float8_e5m2):
            constrained[
                target
                in (
                    DType.bfloat16,
                    DType.float16,
                    DType.float32,
                    DType.float64,
                ),
                (
                    String(
                        (
                            "Only FP8->F64, FP8->F32, FP8->F16, and FP8->BF16"
                            " castings are implemented. "
                        ),
                        dtype,
                        "->",
                        target,
                    )
                ),
            ]()

            @parameter
            if target is DType.float16:
                return _convert_float8_to_f16(self).cast[target]()
            return _convert_float8_to_f32(self).cast[target]()

        @parameter
        if has_neon() and (dtype is DType.bfloat16 or target is DType.bfloat16):
            # TODO(KERN-228): support BF16 on neon systems.
            return _unchecked_zero[target, size]()

        @parameter
        if dtype is DType.bool:
            return self.select[target](1, 0)
        elif target is DType.bool:
            return rebind[Target](self != 0)

        @parameter
        if dtype is DType.bfloat16 and (
            is_amd_gpu() or not _has_native_bf16_support()
        ):
            return _bfloat16_to_f32(
                rebind[SIMD[DType.bfloat16, size]](self)
            ).cast[target]()
        elif target is DType.bfloat16 and not _has_native_bf16_support():
            return rebind[Target](_f32_to_bfloat16(self.cast[DType.float32]()))

        return __mlir_op.`pop.cast`[
            _type = Target._mlir_type, fast = __mlir_attr.unit
        ](self.value)

    @always_inline
    fn is_power_of_two(self) -> SIMD[DType.bool, size]:
        """Checks if the input value is a power of 2 for each element of a SIMD vector.

        Constraints:
            The element type of the input vector must be integral.

        Returns:
            A SIMD value where the element at position `i` is True if the integer at
            position `i` of the input value is a power of 2, False otherwise.
        """
        constrained[dtype.is_integral(), "must be integral"]()

        @parameter
        if dtype.is_unsigned():
            return pop_count(self) == 1
        else:
            return (self > 0) & (self & (self - 1) == 0)

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        """
        Formats this SIMD value to the provided Writer.

        Parameters:
            W: A type conforming to the Writable trait.

        Args:
            writer: The object to write to.
        """

        # Write an opening `[`.
        @parameter
        if size > 1:
            writer.write("[")

        # Write each element.
        for i in range(size):
            var element = self[i]
            # Write separators between each element.
            if i != 0:
                writer.write(", ")
            _write_scalar(writer, element)

        # Write a closing `]`.
        @parameter
        if size > 1:
            writer.write("]")

    # FIXME: `_integral_type_of` doesn't work with `DType.bool`.
    @always_inline
    fn to_bits[
        int_dtype: DType = _integral_type_of[dtype]()
    ](self) -> SIMD[int_dtype, size]:
        """Bitcasts the SIMD vector to an integer SIMD vector.

        Parameters:
            int_dtype: The integer type to cast to.

        Returns:
            An integer representation of the floating-point value.
        """
        constrained[
            int_dtype.is_integral(), "the target type must be integral"
        ]()
        constrained[
            bitwidthof[int_dtype]() >= bitwidthof[dtype](),
            (
                "the target integer type must be at least as wide as the source"
                " type"
            ),
        ]()

        return bitcast[_integral_type_of[dtype](), size](self).cast[int_dtype]()

    @staticmethod
    fn from_bytes[
        big_endian: Bool = is_big_endian()
    ](bytes: InlineArray[Byte, dtype.sizeof()]) -> Scalar[dtype]:
        """Converts a byte array to an scalar integer.

        Args:
            bytes: The byte array to convert.

        Parameters:
            big_endian: Whether the byte array is big-endian.

        Returns:
            The integer value.
        """
        var ptr = bytes.unsafe_ptr().bitcast[Scalar[dtype]]()
        var value = ptr[]

        @parameter
        if is_big_endian() != big_endian:
            return byte_swap(value)

        return value

    fn as_bytes[
        big_endian: Bool = is_big_endian()
    ](self) -> InlineArray[Byte, dtype.sizeof()]:
        """Convert the scalar integer to a byte array.

        Parameters:
            big_endian: Whether the byte array should be big-endian.

        Returns:
            The byte array.
        """
        var value = self

        @parameter
        if is_big_endian() != big_endian:
            value = byte_swap(value)

        var ptr = UnsafePointer(to=value)
        var array = InlineArray[Byte, dtype.sizeof()](uninitialized=True)
        memcpy(array.unsafe_ptr(), ptr.bitcast[Byte](), dtype.sizeof())
        return array^

    fn _floor_ceil_trunc_impl[intrinsic: StaticString](self) -> Self:
        constrained[
            intrinsic == "llvm.floor"
            or intrinsic == "llvm.ceil"
            or intrinsic == "llvm.trunc",
            "unsupported intrinsic",
        ]()

        @parameter
        if dtype.is_integral() or dtype is DType.bool:
            return self
        elif has_neon() and dtype is DType.bfloat16:
            # TODO(KERN-228): support BF16 on neon systems.
            # As a workaround, we cast to float32.
            return (
                self.cast[DType.float32]()
                ._floor_ceil_trunc_impl[intrinsic]()
                .cast[dtype]()
            )
        else:
            return llvm_intrinsic[intrinsic, Self, has_side_effect=False](self)

    fn clamp(self, lower_bound: Self, upper_bound: Self) -> Self:
        """Clamps the values in a SIMD vector to be in a certain range.

        Clamp cuts values in the input SIMD vector off at the upper bound and
        lower bound values. For example,  SIMD vector `[0, 1, 2, 3]` clamped to
        a lower bound of 1 and an upper bound of 2 would return `[1, 1, 2, 2]`.

        Args:
            lower_bound: Minimum of the range to clamp to.
            upper_bound: Maximum of the range to clamp to.

        Returns:
            A new SIMD vector containing x clamped to be within lower_bound and
            upper_bound.
        """
        return max(min(self, upper_bound), lower_bound)

    # TODO: Move to global function.
    @always_inline("nodebug")
    fn fma(self, multiplier: Self, accumulator: Self) -> Self:
        """Performs a fused multiply-add operation, i.e.
        `self*multiplier + accumulator`.

        Args:
            multiplier: The value to multiply.
            accumulator: The value to accumulate.

        Returns:
            A new vector whose element at position `i` is computed as
            `self[i]*multiplier[i] + accumulator[i]`.
        """
        constrained[dtype.is_numeric(), "the SIMD type must be numeric"]()

        return __mlir_op.`pop.fma`[
            fastmathFlags = __mlir_attr.`#pop<fmf contract>`
        ](self.value, multiplier.value, accumulator.value)

    @always_inline("nodebug")
    fn _shuffle_variadic[
        *mask: Int, output_size: Int = size
    ](self, other: Self) -> SIMD[dtype, output_size]:
        """Shuffles (also called blend) the values of the current vector with
        the `other` value using the specified mask (permutation). The mask
        values must be within `2 * len(self)`.

        Parameters:
            mask: The permutation to use in the shuffle.
            output_size: The size of the output vector.

        Args:
            other: The other vector to shuffle with.

        Returns:
            A new vector with the same length as the mask where the value at
            position `i` is `(self + other)[permutation[i]]`.
        """

        @parameter
        fn variadic_len[*mask: Int]() -> Int:
            return __mlir_op.`pop.variadic.size`(mask)

        @parameter
        fn _convert_variadic_to_pop_array[
            *mask: Int
        ]() -> __mlir_type[`!pop.array<`, output_size.value, `, `, Int, `>`]:
            var array: __mlir_type[
                `!pop.array<`, output_size.value, `, `, Int, `>`
            ]
            __mlir_op.`lit.ownership.mark_initialized`(
                __get_mvalue_as_litref(array)
            )

            var ptr = UnsafePointer(to=array)

            @parameter
            for i in range(output_size):
                alias idx = mask[i]
                constrained[
                    0 <= idx < 2 * size,
                    "invalid index in the shuffle operation",
                ]()
                __mlir_op.`pop.store`(
                    idx, __mlir_op.`pop.array.gep`(ptr.address, i.value)
                )

            return array

        constrained[
            output_size == variadic_len[*mask](),
            "size of the mask must match the output SIMD size",
        ]()
        return __mlir_op.`pop.simd.shuffle`[
            mask = _convert_variadic_to_pop_array[*mask](),
            _type = SIMD[dtype, output_size]._mlir_type,
        ](self.value, other.value)

    @always_inline("nodebug")
    fn _shuffle_list[
        output_size: Int, mask: StaticTuple[Int, output_size]
    ](self, other: Self) -> SIMD[dtype, output_size]:
        """Shuffles (also called blend) the values of the current vector with
        the `other` value using the specified mask (permutation). The mask
        values must be within `2 * len(self)`.

        Parameters:
            output_size: The output SIMD size.
            mask: The permutation to use in the shuffle.

        Args:
            other: The other vector to shuffle with.

        Returns:
            A new vector with the same length as the mask where the value at
            position `i` is `(self + other)[permutation[i]]`.
        """

        @parameter
        for i in range(output_size):
            constrained[
                0 <= mask[i] < 2 * size,
                "invalid index in the shuffle operation",
            ]()

        return __mlir_op.`pop.simd.shuffle`[
            mask = mask.array,
            _type = SIMD[dtype, output_size]._mlir_type,
        ](self.value, other.value)

    @always_inline("nodebug")
    fn shuffle[*mask: Int](self) -> Self:
        """Shuffles (also called blend) the values of the current vector with
        the `other` value using the specified mask (permutation). The mask
        values must be within `2 * len(self)`.

        Parameters:
            mask: The permutation to use in the shuffle.

        Returns:
            A new vector with the same length as the mask where the value at
            position `i` is `(self)[permutation[i]]`.
        """
        return self._shuffle_variadic[*mask](self)

    @always_inline("nodebug")
    fn shuffle[*mask: Int](self, other: Self) -> Self:
        """Shuffles (also called blend) the values of the current vector with
        the `other` value using the specified mask (permutation). The mask
        values must be within `2 * len(self)`.

        Parameters:
            mask: The permutation to use in the shuffle.

        Args:
            other: The other vector to shuffle with.

        Returns:
            A new vector with the same length as the mask where the value at
            position `i` is `(self + other)[permutation[i]]`.
        """
        return self._shuffle_variadic[*mask](other)

    @always_inline("nodebug")
    fn shuffle[mask: IndexList[size, **_]](self) -> Self:
        """Shuffles (also called blend) the values of the current vector with
        the `other` value using the specified mask (permutation). The mask
        values must be within `2 * len(self)`.

        Parameters:
            mask: The permutation to use in the shuffle.

        Returns:
            A new vector with the same length as the mask where the value at
            position `i` is `(self)[permutation[i]]`.
        """
        return self._shuffle_list[size, mask.as_tuple()](self)

    @always_inline("nodebug")
    fn shuffle[mask: IndexList[size, **_]](self, other: Self) -> Self:
        """Shuffles (also called blend) the values of the current vector with
        the `other` value using the specified mask (permutation). The mask
        values must be within `2 * len(self)`.

        Parameters:
            mask: The permutation to use in the shuffle.

        Args:
            other: The other vector to shuffle with.

        Returns:
            A new vector with the same length as the mask where the value at
            position `i` is `(self + other)[permutation[i]]`.
        """
        return self._shuffle_list[size, mask.as_tuple()](other)

    # Not an overload of shuffle because there is ambiguity
    # with fn shuffle[*mask: Int](self, other: Self) -> Self:
    # TODO: move to the utils directory - see https://github.com/modular/modular/issues/3477
    @always_inline
    fn _dynamic_shuffle[
        mask_size: Int, //
    ](self, mask: SIMD[DType.uint8, mask_size]) -> SIMD[Self.dtype, mask_size]:
        """Shuffles (also called blend) the values of the current vector.

        It's done using the specified mask (permutation). The mask
        values must be within `len(self)`. If that's not the case,
        the behavior is undefined.

        The mask is not known at compile time, unlike the `shuffle` method.

        Note that currently, this function is fast only if the following
        conditions are met:
        1) The SIMD vector `self` is of type uint8 and size 16
        2) The CPU supports SSE4 or NEON

        If that's not the case, the function will fallback on a slower path,
        which is an unrolled for loop.

        The pseudocode of this function is:
        ```
        result = SIMD[Self.type, mask_size]()
        for i in range(mask_size):
            result[i] = self[Int(mask[i])]
        ```

        Parameters:
            mask_size: The size of the mask.

        Args:
            mask: The mask to use. Contains the indices to use to shuffle.

        Returns:
            A new vector with the same length as the mask where the value at
            position `i` is equal to `self[mask[i]]`.
        """

        @parameter
        if (
            # TODO: Allow SSE3 when we have sys.has_sse3()
            (CompilationTarget.has_sse4() or sys.has_neon())
            and dtype is DType.uint8
            and size == 16
        ):
            # The instruction works with mask size of 16
            alias target_mask_size = 16

            # We know that simd sizes are powers of two, so we can use recursivity
            # to iterate on the method until we reach the target size.
            @parameter
            if mask_size < target_mask_size:
                # Make a bigger mask (x2) and retry
                return self._dynamic_shuffle(mask.join({})).slice[mask_size]()
            elif mask_size == target_mask_size:
                alias dt = SIMD[DType.uint8, target_mask_size]
                var res = _pshuf_or_tbl1(rebind[dt](self), rebind[dt](mask))
                return rebind[SIMD[dtype, mask_size]](res)
            elif mask_size > target_mask_size:
                # We split it in two and call dynamic_shuffle twice.
                var fst_mask, snd_mask = mask.split()
                var fst = self._dynamic_shuffle(fst_mask)
                var snd = self._dynamic_shuffle(snd_mask)
                return rebind[SIMD[dtype, mask_size]](fst.join(snd))

        # Slow path, ~3x slower than pshuf for size 16
        var res = SIMD[dtype, mask_size]()

        @parameter
        for i in range(mask_size):
            res[i] = self[Int(mask[i])]
        return res

    @always_inline
    fn slice[
        output_width: Int, /, *, offset: Int = 0
    ](self) -> SIMD[dtype, output_width]:
        """Returns a slice of the vector of the specified width with the given
        offset.

        Constraints:
            `output_width + offset` must not exceed the size of this SIMD
            vector.

        Parameters:
            output_width: The output SIMD vector size.
            offset: The given offset for the slice.

        Returns:
            A new vector whose elements map to
            `self[offset:offset+output_width]`.
        """
        constrained[
            0 <= offset < output_width + offset <= size,
            "output width must be a positive integer less than simd size",
        ]()

        @parameter
        if output_width == 1:
            return self[offset]

        @parameter
        if offset % simdwidthof[dtype]():
            var res = SIMD[dtype, output_width]()

            @parameter
            for i in range(output_width):
                res[i] = self[i + offset]
            return res

        return llvm_intrinsic[
            "llvm.vector.extract",
            SIMD[dtype, output_width],
            has_side_effect=False,
        ](self, Int64(offset))

    @always_inline("nodebug")
    fn insert[*, offset: Int = 0](self, value: SIMD[dtype, _]) -> Self:
        """Returns a new vector where the elements between `offset` and
        `offset + input_width` have been replaced with the elements in `value`.

        Parameters:
            offset: The offset to insert at.

        Args:
            value: The value to be inserted.

        Returns:
            A new vector whose elements at `self[offset:offset+input_width]`
            contain the values of `value`.
        """
        alias input_width = value.size
        constrained[
            0 <= offset < input_width + offset <= size,
            "insertion position must not exceed the size of the vector",
        ]()

        @parameter
        if size == 1:
            constrained[
                input_width == 1, "the input width must be 1 if the size is 1"
            ]()
            return rebind[Self](value)

        # You cannot insert into a SIMD value at positions that are not a
        # multiple of the SIMD width via the `llvm.vector.insert` intrinsic,
        # so resort to a for loop. Note that this can be made more intelligent
        # by dividing the problem into the offset, offset+val, val+input_width
        # where val is a value to align the offset to the simdwidth.
        @parameter
        if offset % simdwidthof[dtype]():
            var res = self

            @parameter
            for i in range(input_width):
                res[i + offset] = value[i]
            return res

        return llvm_intrinsic[
            "llvm.vector.insert", Self, has_side_effect=False
        ](self, value, Int64(offset))

    @always_inline("nodebug")
    fn join(self, other: Self) -> SIMD[dtype, 2 * size]:
        """Concatenates the two vectors together.

        Args:
            other: The other SIMD vector.

        Returns:
            A new vector `self_0, self_1, ..., self_n, other_0, ..., other_n`.
        """

        fn indices() -> StaticTuple[Int, 2 * size]:
            var res = StaticTuple[Int, 2 * size](0)
            for i in range(len(res)):
                res[i] = i
            return res

        return self._shuffle_list[2 * size, indices()](other)

    @always_inline("nodebug")
    fn interleave(self, other: Self) -> SIMD[dtype, 2 * size]:
        """Constructs a vector by interleaving two input vectors.

        Args:
            other: The other SIMD vector.

        Returns:
            A new vector `self_0, other_0, ..., self_n, other_n`.
        """

        @parameter
        if size == 1:
            return [self[0], other[0]]

        return llvm_intrinsic[
            "llvm.vector.interleave2",
            SIMD[dtype, 2 * size],
            has_side_effect=False,
        ](self, other)

    @always_inline("nodebug")
    fn split(self) -> Tuple[SIMD[dtype, size // 2], SIMD[dtype, size // 2]]:
        """Splits the SIMD vector into 2 subvectors.

        Returns:
            A new vector `self_0:N/2, self_N/2:N`.
        """
        constrained[size > 1, "the simd width must be at least 2"]()
        alias half_size = size // 2
        var se = self.slice[half_size]()
        var lf = self.slice[half_size, offset=half_size]()
        return se, lf

    @always_inline("nodebug")
    fn deinterleave(
        self,
    ) -> Tuple[SIMD[dtype, size // 2], SIMD[dtype, size // 2]]:
        """Constructs two vectors by deinterleaving the even and odd lanes of
        the vector.

        Constraints:
            The vector size must be greater than 1.

        Returns:
            Two vectors the first of the form `self_0, self_2, ..., self_{n-2}`
            and the other being `self_1, self_3, ..., self_{n-1}`.
        """

        constrained[size > 1, "the vector size must be greater than 1."]()

        @parameter
        if size == 2:
            return self[0], self[1]

        var res = llvm_intrinsic[
            "llvm.vector.deinterleave2",
            _RegisterPackType[SIMD[dtype, size // 2], SIMD[dtype, size // 2]],
            has_side_effect=False,
        ](self)
        return res[0], res[1]

    # ===------------------------------------------------------------------=== #
    # Reduce operations
    # ===------------------------------------------------------------------=== #

    alias _T = SIMD[dtype, _]

    # TODO: remove when non-capturing can be converted to capturing.
    @always_inline
    fn reduce[
        func: fn[width: Int] (Self._T[width], Self._T[width]) -> Self._T[width],
        size_out: Int = 1,
    ](self) -> Self._T[size_out]:
        """Reduces the vector using a provided reduce operator.

        Parameters:
            func: The reduce function to apply to elements in this SIMD.
            size_out: The width of the reduction.

        Constraints:
            `size_out` must not exceed width of the vector.

        Returns:
            A new scalar which is the reduction of all vector elements.
        """

        @always_inline
        @parameter
        fn body[w: Int](lhs: Self._T[w], rhs: Self._T[w]) -> Self._T[w]:
            return func(lhs, rhs)

        return self.reduce[body, size_out]()

    @always_inline
    fn reduce[
        func: fn[width: Int] (
            Self._T[width], Self._T[width]
        ) capturing -> Self._T[width],
        size_out: Int = 1,
    ](self) -> Self._T[size_out]:
        """Reduces the vector using a provided reduce operator.

        Parameters:
            func: The reduce function to apply to elements in this SIMD.
            size_out: The width of the reduction.

        Constraints:
            `size_out` must not exceed width of the vector.

        Returns:
            A new scalar which is the reduction of all vector elements.
        """
        constrained[size_out <= size, "reduction cannot increase simd width"]()

        @parameter
        if size == size_out:
            return rebind[Self._T[size_out]](self)
        else:
            var lhs, rhs = self.split()
            return func(lhs, rhs).reduce[func, size_out]()

    @always_inline("nodebug")
    fn reduce_max[size_out: Int = 1](self) -> Self._T[size_out]:
        """Reduces the vector using the `max` operator.

        Parameters:
            size_out: The width of the reduction.

        Constraints:
            `size_out` must not exceed width of the vector.
            The element type of the vector must be integer or FP.

        Returns:
            The maximum element of the vector.
        """

        @parameter
        if size == 1:
            return self[0]

        @parameter
        if CompilationTarget.is_x86() or size_out > 1:
            return self.reduce[max[dtype=dtype], size_out]()

        @parameter
        if dtype.is_unsigned():
            return rebind[SIMD[dtype, size_out]](
                llvm_intrinsic[
                    "llvm.vector.reduce.umax",
                    Scalar[dtype],
                    has_side_effect=False,
                ](self)
            )
        elif dtype.is_integral():
            return rebind[SIMD[dtype, size_out]](
                llvm_intrinsic[
                    "llvm.vector.reduce.smax",
                    Scalar[dtype],
                    has_side_effect=False,
                ](self)
            )
        else:
            return rebind[SIMD[dtype, size_out]](
                llvm_intrinsic[
                    "llvm.vector.reduce.fmax",
                    Scalar[dtype],
                    has_side_effect=False,
                ](self)
            )

    @always_inline("nodebug")
    fn reduce_min[size_out: Int = 1](self) -> SIMD[dtype, size_out]:
        """Reduces the vector using the `min` operator.

        Parameters:
            size_out: The width of the reduction.

        Constraints:
            `size_out` must not exceed width of the vector.
            The element type of the vector must be integer or FP.

        Returns:
            The minimum element of the vector.
        """

        @parameter
        if size == 1:
            return self[0]

        @parameter
        if CompilationTarget.is_x86() or size_out > 1:
            return self.reduce[min[dtype=dtype], size_out]()

        @parameter
        if dtype.is_unsigned():
            return rebind[SIMD[dtype, size_out]](
                llvm_intrinsic[
                    "llvm.vector.reduce.umin",
                    Scalar[dtype],
                    has_side_effect=False,
                ](self)
            )
        elif dtype.is_integral():
            return rebind[SIMD[dtype, size_out]](
                llvm_intrinsic[
                    "llvm.vector.reduce.smin",
                    Scalar[dtype],
                    has_side_effect=False,
                ](self)
            )
        else:
            return rebind[SIMD[dtype, size_out]](
                llvm_intrinsic[
                    "llvm.vector.reduce.fmin",
                    Scalar[dtype],
                    has_side_effect=False,
                ](self)
            )

    @always_inline
    fn reduce_add[size_out: Int = 1](self) -> SIMD[dtype, size_out]:
        """Reduces the vector using the `add` operator.

        Parameters:
            size_out: The width of the reduction.

        Constraints:
            `size_out` must not exceed width of the vector.

        Returns:
            The sum of all vector elements.

        """
        return self.reduce[Self._T.__add__, size_out]()

    @always_inline
    fn reduce_mul[size_out: Int = 1](self) -> SIMD[dtype, size_out]:
        """Reduces the vector using the `mul` operator.

        Parameters:
            size_out: The width of the reduction.

        Constraints:
            `size_out` must not exceed width of the vector.
            The element type of the vector must be integer or FP.

        Returns:
            The product of all vector elements.
        """
        return self.reduce[Self._T.__mul__, size_out]()

    @always_inline
    fn reduce_and[size_out: Int = 1](self) -> SIMD[dtype, size_out]:
        """Reduces the vector using the bitwise `&` operator.

        Parameters:
            size_out: The width of the reduction.

        Constraints:
            `size_out` must not exceed width of the vector.
            The element type of the vector must be integer or boolean.

        Returns:
            The reduced vector.
        """
        constrained[
            size_out <= size, "`size_out` must not exceed width of the vector."
        ]()
        constrained[
            dtype.is_integral() or dtype is DType.bool,
            "The element type of the vector must be integer or boolean.",
        ]()

        @parameter
        if size_out > 1:
            return self.reduce[Self._T.__and__, size_out]()

        @parameter
        if size == 1:
            return self[0]

        return llvm_intrinsic[
            "llvm.vector.reduce.and",
            SIMD[dtype, size_out],
            has_side_effect=False,
        ](self)

    @always_inline
    fn reduce_or[size_out: Int = 1](self) -> SIMD[dtype, size_out]:
        """Reduces the vector using the bitwise `|` operator.

        Parameters:
            size_out: The width of the reduction.

        Constraints:
            `size_out` must not exceed width of the vector.
            The element type of the vector must be integer or boolean.

        Returns:
            The reduced vector.
        """
        constrained[
            size_out <= size, "`size_out` must not exceed width of the vector."
        ]()
        constrained[
            dtype.is_integral() or dtype is DType.bool,
            "The element type of the vector must be integer or boolean.",
        ]()

        @parameter
        if size_out > 1:
            return self.reduce[Self._T.__or__, size_out]()

        @parameter
        if size == 1:
            return self[0]

        return llvm_intrinsic[
            "llvm.vector.reduce.or",
            SIMD[dtype, size_out],
            has_side_effect=False,
        ](self)

    @always_inline
    fn reduce_bit_count(self) -> Int:
        """Returns the total number of bits set in the SIMD vector.

        Constraints:
            Must be either an integral or a boolean type.

        Returns:
            Count of set bits across all elements of the vector.
        """
        constrained[
            dtype.is_integral() or dtype is DType.bool,
            "Expected either integral or bool type",
        ]()

        @parameter
        if dtype is DType.bool:
            return Int(self.cast[DType.uint8]().reduce_add())
        else:
            return Int(pop_count(self).reduce_add())

    # ===------------------------------------------------------------------=== #
    # select
    # ===------------------------------------------------------------------=== #

    # TODO (7748): always_inline required to WAR LLVM codegen bug
    @always_inline("nodebug")
    fn select[
        dtype: DType
    ](
        self,
        true_case: SIMD[dtype, size],
        false_case: SIMD[dtype, size],
    ) -> SIMD[dtype, size]:
        """Selects the values of the `true_case` or the `false_case` based on
        the current boolean values of the SIMD vector.

        Parameters:
            dtype: The element type of the input and output SIMD vectors.

        Args:
            true_case: The values selected if the positional value is True.
            false_case: The values selected if the positional value is False.

        Constraints:
            The element type of the vector must be boolean.

        Returns:
            A new vector of the form
            `[true_case[i] if elem else false_case[i] for i, elem in enumerate(self)]`.
        """
        constrained[Self.dtype is DType.bool, "the simd type must be bool"]()
        return __mlir_op.`pop.simd.select`(
            rebind[Self._Mask](self).value,
            true_case.value,
            false_case.value,
        )

    # ===------------------------------------------------------------------=== #
    # Rotation operations
    # ===------------------------------------------------------------------=== #

    @always_inline
    fn rotate_left[shift: Int](self) -> Self:
        """Shifts the elements of a SIMD vector to the left by `shift`
        elements (with wrap-around).

        Constraints:
            `-size <= shift < size`

        Parameters:
            shift: The number of positions by which to rotate the elements of
                   SIMD vector to the left (with wrap-around).

        Returns:
            The SIMD vector rotated to the left by `shift` elements
            (with wrap-around).
        """

        constrained[
            shift >= -size and shift < size,
            "Constraint: -size <= shift < size",
        ]()

        @parameter
        if size == 1:
            constrained[shift == 0, "for scalars the shift must be 0"]()
            return self
        return llvm_intrinsic[
            "llvm.vector.splice", Self, has_side_effect=False
        ](self, self, Int32(shift))

    @always_inline
    fn rotate_right[shift: Int](self) -> Self:
        """Shifts the elements of a SIMD vector to the right by `shift`
        elements (with wrap-around).

        Constraints:
            `-size < shift <= size`

        Parameters:
            shift: The number of positions by which to rotate the elements of
                   SIMD vector to the right (with wrap-around).

        Returns:
            The SIMD vector rotated to the right by `shift` elements
            (with wrap-around).
        """

        constrained[
            shift > -size and shift <= size,
            "Constraint: -size < shift <= size",
        ]()

        @parameter
        if size == 1:
            constrained[shift == 0, "for scalars the shift must be 0"]()
            return self
        return self.rotate_left[-shift]()

    # ===------------------------------------------------------------------=== #
    # Shift operations
    # ===------------------------------------------------------------------=== #

    @always_inline
    fn shift_left[shift: Int](self) -> Self:
        """Shifts the elements of a SIMD vector to the left by `shift`
        elements (no wrap-around, fill with zero).

        Constraints:
            `0 <= shift <= size`

        Parameters:
            shift: The number of positions by which to rotate the elements of
                   SIMD vector to the left (no wrap-around, fill with zero).

        Returns:
            The SIMD vector rotated to the left by `shift` elements (no
            wrap-around, fill with zero).
        """

        constrained[
            0 <= shift <= size,
            (
                "shift must be greater than or equal to 0 and less than equal"
                " to the size"
            ),
        ]()

        @parameter
        if shift == 0:
            return self
        elif shift == size:
            return 0

        return llvm_intrinsic[
            "llvm.vector.splice", Self, has_side_effect=False
        ](self, Self(), Int32(shift))

    @always_inline
    fn shift_right[shift: Int](self) -> Self:
        """Shifts the elements of a SIMD vector to the right by `shift`
        elements (no wrap-around, fill with zero).

        Constraints:
            `0 <= shift <= size`

        Parameters:
            shift: The number of positions by which to rotate the elements of
                   SIMD vector to the right (no wrap-around, fill with zero).

        Returns:
            The SIMD vector rotated to the right by `shift` elements (no
            wrap-around, fill with zero).
        """

        # Note the order of the llvm_intrinsic arguments below differ from
        # shift_left(), so we cannot directly reuse it here.

        constrained[
            0 <= shift <= size,
            (
                "shift must be greater than or equal to 0 and less than equal"
                " to the size"
            ),
        ]()

        @parameter
        if shift == 0:
            return self
        elif shift == size:
            return 0

        return llvm_intrinsic[
            "llvm.vector.splice", Self, has_side_effect=False
        ](Self(), self, Int32(-shift))

    fn reversed(self) -> Self:
        """Reverses the SIMD vector by indexes.

        Returns:
            The by index reversed vector.

        Examples:
        ```mojo
        print(SIMD[DType.uint8, 4](1, 2, 3, 4).reversed()) # [4, 3, 2, 1]
        ```
        .
        """

        fn indices() -> IndexList[size]:
            var res = IndexList[size]()
            for i in range(size):
                res[i] = size - i - 1
            return res

        return self.shuffle[mask = indices()]()


alias U8x16 = SIMD[DType.uint8, 16]


fn _pshuf_or_tbl1(lookup_table: U8x16, indices: U8x16) -> U8x16:
    @parameter
    if CompilationTarget.has_sse4():
        return _pshuf(lookup_table, indices)
    elif sys.has_neon():
        return _tbl1(lookup_table, indices)
    else:
        # TODO: Change the error message when we allow SSE3
        constrained[False, "To call _pshuf_or_tbl1() you need sse4 or neon."]()
        return {}


fn _pshuf(lookup_table: U8x16, indices: U8x16) -> U8x16:
    """Shuffle operation using the SSSE3 `pshuf` instruction.

    See https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_shuffle_epi8&ig_expand=6003
    """
    return llvm_intrinsic[
        "llvm.x86.ssse3.pshuf.b.128", U8x16, has_side_effect=False
    ](lookup_table, indices)


fn _tbl1(lookup_table: U8x16, indices: U8x16) -> U8x16:
    """Shuffle operation using the aarch64 `tbl1` instruction.

    See https://community.arm.com/arm-community-blogs/b/architectures-and-processors-blog/posts/coding-for-neon---part-5-rearranging-vectors
    """
    return llvm_intrinsic[
        "llvm.aarch64.neon.tbl1", U8x16, has_side_effect=False
    ](lookup_table, indices)


# ===----------------------------------------------------------------------=== #
# _pow
# ===----------------------------------------------------------------------=== #


@always_inline
fn _pow[
    simd_width: Int
](base: SIMD[_, simd_width], exp: SIMD[_, simd_width]) -> __type_of(base):
    """Computes the power of the elements of a SIMD vector raised to the
    corresponding elements of another SIMD vector.

    Parameters:
        simd_width: The width of the input and output SIMD vectors.

    Args:
        base: Base of the power operation.
        exp: Exponent of the power operation.

    Returns:
        A vector containing elementwise `base` raised to the power of `exp`.
    """

    @parameter
    if exp.dtype.is_floating_point() and base.dtype is exp.dtype:
        return _powf(base, exp)
    elif exp.dtype.is_integral():
        # Common cases
        if all(exp == 2):
            return base * base
        if all(exp == 3):
            return base * base * base

        var result = __type_of(base)()

        @parameter
        for i in range(simd_width):
            result[i] = _powi(base[i], exp[i].cast[DType.int32]())
        return result
    else:
        constrained[False, "unsupported type combination"]()
        return {}


@always_inline
fn _powf_scalar(base: Scalar, exponent: Scalar) -> __type_of(base):
    constrained[
        exponent.dtype.is_floating_point(), "exponent must be floating point"
    ]()

    var integral, fractional = _modf_scalar(exponent)

    if integral == exponent:
        return _powi(base, integral.cast[DType.int32]())

    if fractional and base < 0:
        return _nan[base.dtype]()

    return math.exp(exponent.cast[base.dtype]() * math.log(base))


@always_inline
fn _powf[
    simd_width: Int
](base: SIMD[_, simd_width], exp: SIMD[_, simd_width]) -> __type_of(base):
    constrained[
        exp.dtype.is_floating_point(), "exponent must be floating point"
    ]()

    var result = __type_of(base)()

    @parameter
    for i in range(simd_width):
        result[i] = _powf_scalar(base[i], exp[i])

    return result


@always_inline
fn _powi(base: Scalar, exp: Int32) -> __type_of(base):
    if base.dtype.is_integral() and exp < 0:
        # Not defined for Integers, this should raise an
        # exception.
        debug_assert(False, "exponent < 0 is undefined for integers")
        return 0

    var a = base
    var b = abs(exp) if base.dtype.is_floating_point() else exp
    var res: Scalar[base.dtype] = 1
    while b > 0:
        if b & 1:
            res *= a
        a *= a
        b >>= 1

    @parameter
    if base.dtype.is_floating_point():
        if exp < 0:
            return 1 / res
    return res


# ===----------------------------------------------------------------------=== #
# float8
# ===----------------------------------------------------------------------=== #


@always_inline
fn _convert_float8_to_f32_scaler[
    dtype: DType,
](x: Scalar[dtype]) -> Float32:
    var kF32_NaN: UInt32 = 0x7FFFFFFF
    var FP8_NUM_BITS = 8
    var IS_E4M3 = dtype is DType.float8_e4m3fn
    var FP8_NUM_MANTISSA_BITS = FPUtils[dtype].mantissa_width()
    var FP8_NUM_EXPONENT_BITS = FPUtils[dtype].exponent_width()
    var FP32_NUM_BITS = 32
    var FP8_EXPONENT_MASK: UInt8 = (1 << FP8_NUM_EXPONENT_BITS) - 1
    var FP8_MANTISSA_MASK: UInt8 = (1 << FP8_NUM_MANTISSA_BITS) - 1
    var FP8_MAX_EXPONENT = 7 if IS_E4M3 else 15
    var FP8_EXPONENT_BIAS = 7 if IS_E4M3 else 15
    var FP32_EXPONENT_BIAS = 127
    var FP32_NUM_MANTISSA_BITS = 23

    var f8: UInt8 = bitcast[DType.uint8](x)

    var sign: UInt32 = (f8.cast[DType.uint32]() >> (FP8_NUM_BITS - 1)) & 1
    var exp: UInt32 = ((f8 >> FP8_NUM_MANTISSA_BITS) & FP8_EXPONENT_MASK).cast[
        DType.uint32
    ]()
    var mantissa: UInt32 = (f8 & FP8_MANTISSA_MASK).cast[DType.uint32]()

    var f: UInt32 = sign << (FP32_NUM_BITS - 1)

    if IS_E4M3 and exp == 15 and mantissa == 0x7:
        f = kF32_NaN
    elif exp > 0 and (
        IS_E4M3 or exp < (FP8_MAX_EXPONENT + FP8_EXPONENT_BIAS + 1)
    ):
        # normal
        exp += FP32_EXPONENT_BIAS - FP8_EXPONENT_BIAS
        f |= exp << FP32_NUM_MANTISSA_BITS
        f |= mantissa << (FP32_NUM_MANTISSA_BITS - FP8_NUM_MANTISSA_BITS)
    elif exp == 0:
        if mantissa:
            # subnormal
            exp += (FP32_EXPONENT_BIAS - FP8_EXPONENT_BIAS) + 1
            while (mantissa & (1 << FP8_NUM_MANTISSA_BITS)) == 0:
                mantissa <<= 1
                exp -= 1
            mantissa = mantissa & FP8_MANTISSA_MASK.cast[DType.uint32]()
            f |= exp << FP32_NUM_MANTISSA_BITS
            f |= mantissa << (FP32_NUM_MANTISSA_BITS - FP8_NUM_MANTISSA_BITS)
        else:
            # sign-preserving zero
            pass
    else:
        if mantissa == 0:
            # Sign-preserving infinity
            f |= 0x7F800000
        else:
            # Canonical NaN
            f = kF32_NaN

    return bitcast[DType.float32](f)


@always_inline
fn _convert_float8_to_f32[
    dtype: DType,
    size: Int,
](val: SIMD[dtype, size]) -> SIMD[DType.float32, size]:
    @parameter
    if _is_sm_9x_or_newer():
        return _convert_float8_to_f16(val).cast[DType.float32]()
    else:

        @always_inline
        @parameter
        fn wrapper_fn[
            input_dtype: DType, result_dtype: DType
        ](val: Scalar[input_dtype]) capturing -> Scalar[result_dtype]:
            return rebind[Scalar[result_dtype]](
                _convert_float8_to_f32_scaler(rebind[Scalar[dtype]](val))
            )

        return _simd_apply[wrapper_fn, DType.float32, size](val)


@always_inline
fn _convert_float8_to_f16[
    dtype: DType,
    size: Int,
](val: SIMD[dtype, size]) -> SIMD[DType.float16, size]:
    @parameter
    if _is_sm_9x_or_newer():
        # do not call `SIMD.cast` here; the inliner will diverge
        return __mlir_op.`pop.cast`[
            _type = SIMD[DType.float16, size]._mlir_type
        ](val.value)
    else:
        return _convert_float8_to_f32(val).cast[DType.float16]()


@always_inline
fn _convert_f32_to_float8[
    dtype: DType,
    target: DType,
    size: Int,
](val: SIMD[dtype, size]) -> SIMD[target, size]:
    @parameter
    if _is_sm_9x_or_newer():
        # do not call `SIMD.cast` here; the inliner will diverge
        return __mlir_op.`pop.cast`[_type = SIMD[target, size]._mlir_type](
            val.value
        )
    else:

        @always_inline
        @parameter
        fn wrapper_fn[
            input_dtype: DType, result_dtype: DType
        ](val: Scalar[input_dtype]) capturing -> Scalar[result_dtype]:
            return rebind[Scalar[result_dtype]](
                _convert_f32_to_float8_scaler[dtype, result_dtype](
                    rebind[Scalar[dtype]](val)
                )
            )

        return _simd_apply[wrapper_fn, target, size](val)


@always_inline
fn _convert_f32_to_float8_scaler[
    dtype: DType,
    target: DType,
](x: Scalar[dtype]) -> Scalar[target]:
    # software implementation rounds toward nearest even

    alias IS_E4M3 = target is DType.float8_e4m3fn
    alias FP8_NUM_MANTISSA_BITS = FPUtils[target].mantissa_width()
    alias FP8_NUM_EXPONENT_BITS = FPUtils[target].exponent_width()
    alias FP32_NUM_BITS = bitwidthof[dtype]()
    alias FP8_EXPONENT_MASK: UInt8 = (1 << FP8_NUM_EXPONENT_BITS) - 1
    alias FP8_MANTISSA_MASK: UInt8 = (1 << FP8_NUM_MANTISSA_BITS) - 1
    alias FP8_MAX_EXPONENT = FPUtils[target].exponent_bias()
    var FP8_MIN_EXPONENT = -6 if IS_E4M3 else -14
    alias FP8_EXPONENT_BIAS = FPUtils[target].exponent_bias()
    alias FP32_EXPONENT_BIAS = FPUtils[dtype].exponent_bias()
    alias FP32_NUM_MANTISSA_BITS = FPUtils[dtype].mantissa_width()
    alias FP8_MAX_FLT = UInt8(0x7E) if IS_E4M3 else UInt8(0x7B)

    # Extract the bits in the FP32 type
    var sign: UInt8 = 0x80 if FPUtils[dtype].get_sign(x) else 0x00
    var exp = Int32(FPUtils[dtype].get_exponent_biased(x)) - FP32_EXPONENT_BIAS
    var mantissa = Int32(FPUtils[dtype].get_mantissa(x))

    var kF8_NaN: UInt8 = 0x7F

    # NaN => NaN
    if _isnan(x):
        return bitcast[target](kF8_NaN)

    # Inf => MAX_FLT (satfinite)
    if _isinf(x):
        return bitcast[target](sign | FP8_MAX_FLT)

    # Special handling
    if exp == -128:
        # int8 range is from -128 to 127
        # So 255(inf) - 127(bias) = 128 - will show up as -128

        # satfinite
        return bitcast[target](sign | FP8_MAX_FLT)

    var sticky_bit: Int32 = 0

    var skip_sign = False
    var may_be_nan = False

    if exp >= FP8_MIN_EXPONENT and exp <= FP8_MAX_EXPONENT:
        # normal fp32 to normal fp8
        exp += FP8_EXPONENT_BIAS
        u = (
            (
                (exp).cast[DType.uint32]()
                & FP8_EXPONENT_MASK.cast[DType.uint32]()
            )
            << FP8_NUM_MANTISSA_BITS
        ).cast[DType.uint8]()
        u = (
            u
            | (
                mantissa >> (FP32_NUM_MANTISSA_BITS - FP8_NUM_MANTISSA_BITS)
            ).cast[DType.uint8]()
        )
    elif exp < FP8_MIN_EXPONENT:
        # normal single-precision to subnormal float8-precision representation
        var rshift: Int32 = FP8_MIN_EXPONENT - exp
        if rshift < FP32_NUM_BITS:
            mantissa |= 1 << FP32_NUM_MANTISSA_BITS
            sticky_bit = ((mantissa & ((1 << rshift) - 1)) != 0).cast[
                DType.int32
            ]()
            mantissa = mantissa >> rshift
            u = (
                mantissa >> (FP32_NUM_MANTISSA_BITS - FP8_NUM_MANTISSA_BITS)
            ).cast[DType.uint8]() & FP8_MANTISSA_MASK
        else:
            mantissa = 0
            u = 0
    # Exponent > FP8_MAX_EXPONENT - this is a special case done to match HW
    # 0x4380_0000 to 0x43e0_0000 - maps from 256 to 448, and does not saturate / inf.
    else:
        if exp == (FP8_MAX_EXPONENT + 1):
            var mantissa_tmp: UInt8 = (
                mantissa >> (FP32_NUM_MANTISSA_BITS - FP8_NUM_MANTISSA_BITS)
            ).cast[DType.uint8]()
            if mantissa_tmp < FP8_MANTISSA_MASK:
                exp = exp + FP8_EXPONENT_BIAS
                u = ((exp).cast[DType.uint32]() << FP8_NUM_MANTISSA_BITS).cast[
                    DType.uint8
                ]() | mantissa_tmp
                may_be_nan = mantissa_tmp == (FP8_MANTISSA_MASK - 1)
            else:
                # satfinite
                return bitcast[target](sign | FP8_MAX_FLT)
        else:
            # satfinite
            return bitcast[target](sign | FP8_MAX_FLT)

    # round to nearest even
    var NUM_BITS_SHIFT: Int32 = FP32_NUM_MANTISSA_BITS - (
        FP8_NUM_MANTISSA_BITS + 1
    )
    var round_bit: Int32 = (mantissa >> NUM_BITS_SHIFT) & 1
    sticky_bit |= ((mantissa & ((1 << NUM_BITS_SHIFT) - 1)) != 0).cast[
        DType.int32
    ]()

    if (round_bit and sticky_bit) or (round_bit and (u & 1)):
        u = (u + 1).cast[DType.uint8]()
        if may_be_nan:
            skip_sign = True

    if u > FP8_MAX_FLT:
        # satfinite
        u = sign | FP8_MAX_FLT

    if not skip_sign:
        u |= sign

    return bitcast[target](u)


# ===----------------------------------------------------------------------=== #
# bfloat16
# ===----------------------------------------------------------------------=== #

alias _fp32_bf16_mantissa_diff = FPUtils[
    DType.float32
].mantissa_width() - FPUtils[DType.bfloat16].mantissa_width()


@always_inline
fn _bfloat16_to_f32_scalar(
    val: BFloat16,
) -> Float32:
    @parameter
    if has_neon():
        # TODO(KERN-228): support BF16 on neon systems.
        return _unchecked_zero[DType.float32, 1]()

    # For bfloat16, we can just do a memcpy to perform the cast to float32.
    @parameter
    if is_nvidia_gpu():
        return inlined_assembly[
            "cvt.f32.bf16 $0, $1;" if _is_sm_9x_or_newer() else "mov.b32 $0, {0, $1};",
            Float32,
            constraints="=f,h",
            has_side_effect=False,
        ](bitcast[DType.int16](val))

    return bitcast[DType.float32, 1](SIMD[DType.bfloat16, 2](0, val))


@always_inline
fn _bfloat16_to_f32[
    size: Int
](val: SIMD[DType.bfloat16, size]) -> SIMD[DType.float32, size]:
    @parameter
    if has_neon():
        # TODO(KERN-228): support BF16 on neon systems.
        return _unchecked_zero[DType.float32, size]()

    @always_inline
    @parameter
    fn wrapper_fn[
        input_dtype: DType, result_dtype: DType
    ](val: Scalar[input_dtype]) capturing -> Scalar[result_dtype]:
        return rebind[Scalar[result_dtype]](
            _bfloat16_to_f32_scalar(rebind[BFloat16](val))
        )

    return _simd_apply[wrapper_fn, DType.float32, size](val)


@always_inline
fn _f32_to_bfloat16_scalar(
    val: Float32,
) -> BFloat16:
    @parameter
    if has_neon():
        # TODO(KERN-228): support BF16 on neon systems.
        return _unchecked_zero[DType.bfloat16, 1]()

    if _isnan(val):
        return _nan[DType.bfloat16]()

    var float_bits = FPUtils[DType.float32].bitcast_to_integer(val)

    var lsb = (float_bits >> _fp32_bf16_mantissa_diff) & 1
    var rounding_bias = 0x7FFF + lsb
    float_bits += rounding_bias

    var bfloat_bits = float_bits >> _fp32_bf16_mantissa_diff

    return FPUtils[DType.bfloat16].bitcast_from_integer(bfloat_bits)


@always_inline
fn _f32_to_bfloat16[
    size: Int
](val: SIMD[DType.float32, size]) -> SIMD[DType.bfloat16, size]:
    @parameter
    if has_neon():
        # TODO(KERN-228): support BF16 on neon systems.
        return _unchecked_zero[DType.bfloat16, size]()

    @always_inline
    @parameter
    fn wrapper_fn[
        input_dtype: DType, result_dtype: DType
    ](val: Scalar[input_dtype]) capturing -> Scalar[result_dtype]:
        return rebind[Scalar[result_dtype]](
            _f32_to_bfloat16_scalar(rebind[Float32](val))
        )

    return _simd_apply[wrapper_fn, DType.bfloat16, size](val)


# ===----------------------------------------------------------------------=== #
# _simd_apply
# ===----------------------------------------------------------------------=== #


@always_inline
fn _simd_apply[
    func: fn[input_dtype: DType, result_dtype: DType] (
        Scalar[input_dtype]
    ) capturing -> Scalar[result_dtype],
    result_dtype: DType,
    simd_width: Int,
](x: SIMD[_, simd_width]) -> SIMD[result_dtype, simd_width]:
    """Returns a value whose elements corresponds to applying `func` to each
    element in the vector.

    Parameter:
      simd_width: Width of the input and output SIMD vectors.
      input_dtype: Type of the input to func.
      result_dtype: Result type of func.
      func: Function to apply to the SIMD vector.

    Args:
      x: the input value.

    Returns:
      A SIMD vector whose element at index `i` is `func(x[i])`.
    """
    var result = SIMD[result_dtype, simd_width]()

    @parameter
    for i in range(simd_width):
        result[i] = func[x.dtype, result_dtype](x[i])

    return result


@always_inline
fn _simd_apply[
    func: fn[lhs_dtype: DType, rhs_dtype: DType, result_dtype: DType] (
        Scalar[lhs_dtype], Scalar[rhs_dtype]
    ) capturing -> Scalar[result_dtype],
    result_dtype: DType,
    simd_width: Int,
](x: SIMD[_, simd_width], y: SIMD[_, simd_width]) -> SIMD[
    result_dtype, simd_width
]:
    """Returns a value whose elements corresponds to applying `func` to each
    element in the vector.

    Parameter:
      simd_width: Width of the input and output SIMD vectors.
      input_dtype: Type of the input to func.
      result_dtype: Result type of func.
      func: Function to apply to the SIMD vector.

    Args:
      x: the lhs input value.
      y: the rhs input value.

    Returns:
      A SIMD vector whose element at index `i` is `func(x[i], y[i])`.
    """
    var result = SIMD[result_dtype, simd_width]()

    @parameter
    for i in range(simd_width):
        result[i] = func[x.dtype, y.dtype, result_dtype](x[i], y[i])

    return result


# ===----------------------------------------------------------------------=== #
# modf
# ===----------------------------------------------------------------------=== #


fn _modf_scalar(x: Scalar) -> Tuple[__type_of(x), __type_of(x)]:
    constrained[
        x.dtype.is_floating_point(), "the type must be floating point"
    ]()
    if x < 1:
        if x < 0:
            var res = _modf_scalar(-x)
            return (-res[0], -res[1])
        if x == 0:
            return (x, x)
        return (Scalar[x.dtype](0), x)

    var f = _floor(x)
    return (f, x - f)


fn _modf(x: SIMD) -> Tuple[__type_of(x), __type_of(x)]:
    constrained[x.dtype.is_numeric(), "the type must be numeric"]()

    @parameter
    if x.dtype.is_integral():
        return (x, __type_of(x)(0))

    var result_int = __type_of(x)()
    var result_frac = __type_of(x)()

    @parameter
    for i in range(x.size):
        var tup = _modf_scalar(x[i])
        result_int[i] = tup[0]
        result_frac[i] = tup[1]

    return (result_int, result_frac)


# ===----------------------------------------------------------------------=== #
# floor
# ===----------------------------------------------------------------------=== #


fn _floor(x: SIMD) -> __type_of(x):
    @parameter
    if x.dtype.is_integral():
        return x

    alias integral_type = FPUtils[x.dtype].integral_type
    alias bitwidth = bitwidthof[x.dtype]()
    alias exponent_width = FPUtils[x.dtype].exponent_width()
    alias mantissa_width = FPUtils[x.dtype].mantissa_width()
    alias mask = FPUtils[x.dtype].exponent_mask()
    alias bias = FPUtils[x.dtype].exponent_bias()
    alias shift_factor = bitwidth - exponent_width - 1

    var bits = x.to_bits()
    var e = ((bits & mask) >> mantissa_width) - bias
    bits = (e < shift_factor).select(
        bits & ~((1 << (shift_factor - e)) - 1),
        bits,
    )
    return __type_of(x).from_bits(bits)


fn _write_scalar[
    dtype: DType,
    W: Writer, //,
](mut writer: W, value: Scalar[dtype]):
    @parameter
    if dtype is DType.bool:
        if value:
            writer.write("True")
        else:
            writer.write("False")

    elif dtype.is_floating_point():
        _write_float(writer, value)

    # TODO(MSTDL-1039): bring in performant integer to string formatter
    elif dtype.is_integral():
        _ = _try_write_int(writer, value)
    else:
        constrained[
            False, "unable to write dtype, only integral/float/bool supported"
        ]()
