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
"""Implements the DType class.

These are Mojo built-ins, so you don't need to import them.
"""


from hashlib.hasher import Hasher
from os import abort
from sys import CompilationTarget, bit_width_of, size_of
from sys.intrinsics import _type_is_eq

comptime _mIsSigned = __mlir_attr.`#pop.simd<1> : !pop.scalar<ui8>`
comptime _mIsInteger = __mlir_attr.`#pop.simd<128> : !pop.scalar<ui8>`
comptime _mIsNotInteger = __mlir_attr.`#pop.simd<127> : !pop.scalar<ui8>`
comptime _mIsFloat = __mlir_attr.`#pop.simd<64> : !pop.scalar<ui8>`


struct DType(
    Equatable,
    Hashable,
    ImplicitlyCopyable,
    KeyElement,
    Representable,
    Stringable,
    TrivialRegisterPassable,
    Writable,
):
    """Represents a data type specification and provides methods for working
    with it.

    `DType` defines a set of compile-time constant that specify the precise
    numeric representation of data in order to prevent runtime errors by
    catching type mismatches at compile time. It directly maps to CPU/GPU
    instruction sets, allowing the compiler to generate optimal SIMD and vector
    operations.

    `DType` behaves like an enum rather than a typical object. You don't
    instantiate it, but instead use its compile-time constants (aliases) to
    declare data types for SIMD vectors, tensors, and other data structures.

    Key usage patterns:

    - **Type specification**: Use aliases like `DType.float32` to specify types
      for SIMD vectors, tensors, and other data structures
    - **Type parameters**: Pass `DType` values as compile-time parameters to
      parameterized types like `SIMD[dtype, size]`
    - **Type introspection**: Call methods like `.is_floating_point()` to query
      type properties at compile time
    - **Type conversion**: Use in casting operations to convert between different
      numeric representations

    **Note:** Not all data types are supported on all platforms. For example,
    `DType.bfloat16` is currently not supported on Apple Silicon.

    Example:

    ```mojo
    var data = SIMD[DType.float16, 4](1.5, 2.5, 3.5, 4.5)
    var dtype = data.dtype

    print("Is float:", dtype.is_floating_point())  # True
    print("Is signed:", dtype.is_signed())         # True
    ```
    """

    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#

    comptime _mlir_type = __mlir_type.`!kgen.dtype`

    var _mlir_value: Self._mlir_type
    """The underlying storage for the DType value."""

    # ===-------------------------------------------------------------------===#
    # Aliases
    # ===-------------------------------------------------------------------===#

    comptime invalid = DType(
        mlir_value=__mlir_attr.`#kgen.dtype.constant<invalid> : !kgen.dtype`
    )
    """Represents an invalid or unknown data type."""

    comptime bool = DType(
        mlir_value=__mlir_attr.`#kgen.dtype.constant<bool> : !kgen.dtype`
    )
    """Represents a boolean data type."""

    comptime int = DType(
        mlir_value=__mlir_attr.`#kgen.dtype.constant<index> : !kgen.dtype`
    )
    """Represents an integral type whose bitwidth is the maximum integral value
    on the system."""

    comptime uint = DType(
        mlir_value=__mlir_attr.`#kgen.dtype.constant<uindex> : !kgen.dtype`
    )
    """Represents an unsigned integral type whose bitwidth is the maximum
    unsigned integral value on the system."""

    comptime _uint1 = DType(
        mlir_value=__mlir_attr.`#kgen.dtype.constant<ui1> : !kgen.dtype`
    )
    comptime _uint2 = DType(
        mlir_value=__mlir_attr.`#kgen.dtype.constant<ui2> : !kgen.dtype`
    )
    comptime _uint4 = DType(
        mlir_value=__mlir_attr.`#kgen.dtype.constant<ui4> : !kgen.dtype`
    )

    comptime uint8 = DType(
        mlir_value=__mlir_attr.`#kgen.dtype.constant<ui8> : !kgen.dtype`
    )
    """Represents an unsigned integer type whose bitwidth is 8."""
    comptime int8 = DType(
        mlir_value=__mlir_attr.`#kgen.dtype.constant<si8> : !kgen.dtype`
    )
    """Represents a signed integer type whose bitwidth is 8."""
    comptime uint16 = DType(
        mlir_value=__mlir_attr.`#kgen.dtype.constant<ui16> : !kgen.dtype`
    )
    """Represents an unsigned integer type whose bitwidth is 16."""
    comptime int16 = DType(
        mlir_value=__mlir_attr.`#kgen.dtype.constant<si16> : !kgen.dtype`
    )
    """Represents a signed integer type whose bitwidth is 16."""
    comptime uint32 = DType(
        mlir_value=__mlir_attr.`#kgen.dtype.constant<ui32> : !kgen.dtype`
    )
    """Represents an unsigned integer type whose bitwidth is 32."""
    comptime int32 = DType(
        mlir_value=__mlir_attr.`#kgen.dtype.constant<si32> : !kgen.dtype`
    )
    """Represents a signed integer type whose bitwidth is 32."""
    comptime uint64 = DType(
        mlir_value=__mlir_attr.`#kgen.dtype.constant<ui64> : !kgen.dtype`
    )
    """Represents an unsigned integer type whose bitwidth is 64."""
    comptime int64 = DType(
        mlir_value=__mlir_attr.`#kgen.dtype.constant<si64> : !kgen.dtype`
    )
    """Represents a signed integer type whose bitwidth is 64."""
    comptime uint128 = DType(
        mlir_value=__mlir_attr.`#kgen.dtype.constant<ui128> : !kgen.dtype`
    )
    """Represents an unsigned integer type whose bitwidth is 128."""
    comptime int128 = DType(
        mlir_value=__mlir_attr.`#kgen.dtype.constant<si128> : !kgen.dtype`
    )
    """Represents a signed integer type whose bitwidth is 128."""
    comptime uint256 = DType(
        mlir_value=__mlir_attr.`#kgen.dtype.constant<ui256> : !kgen.dtype`
    )
    """Represents an unsigned integer type whose bitwidth is 256."""
    comptime int256 = DType(
        mlir_value=__mlir_attr.`#kgen.dtype.constant<si256> : !kgen.dtype`
    )
    """Represents a signed integer type whose bitwidth is 256."""

    comptime float4_e2m1fn = DType(
        mlir_value=__mlir_attr.`#kgen.dtype.constant<f4e2m1fn> : !kgen.dtype`
    )
    """Represents a 4-bit `e2m1` floating point format.

    This type is encoded as `s.ee.m` and defined by the
    [Open Compute MX Format Specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf):

    - (s)ign: 1 bit
    - (e)xponent: 2 bits
    - (m)antissa: 1 bits
    - exponent_bias: 1
    """

    comptime float8_e8m0fnu = DType(
        mlir_value=__mlir_attr.`#kgen.dtype.constant<f8e8m0fnu> : !kgen.dtype`
    )
    """Represents the 8-bit `E8M0Fnu` floating point format.

    This type is defined in section 5.4 of the
    [OFP8 standard](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf),
    encoded as `eeeeeeee`:

    - (e)xponent: 8 bits
    - (m)antissa: 0 bits
    - exponent bias: 127
    - nan: 11111111
    - fn: finite (no inf or -inf encodings)
    - u: no sign or zero value.
    """
    comptime float8_e3m4 = DType(
        mlir_value=__mlir_attr.`#kgen.dtype.constant<f8e3m4> : !kgen.dtype`
    )
    """Represents an 8-bit `e3m4` floating point format.

    This type is encoded as `s.eee.mmmm`:

    - (s)ign: 1 bit
    - (e)xponent: 3 bits
    - (m)antissa: 4 bits
    - exponent bias: 3
    - nan: {0,1}.111.1111
    - fn: finite (no inf or -inf encodings)
    - -0: 1.000.0000
    """
    comptime float8_e4m3fn = DType(
        mlir_value=__mlir_attr.`#kgen.dtype.constant<f8e4m3fn> : !kgen.dtype`
    )
    """Represents the 8-bit `E4M3` floating point format defined in the
    [OFP8 standard](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-12-01-pdf-1).

    This type is named differently across libraries and vendors, for example:
    - Mojo, PyTorch, JAX, and LLVM refer to it as `e4m3fn`.
    - OCP, NVIDIA CUDA, and AMD ROCm refer to it as `e4m3`.

    In these contexts, they are all referring to the same finite type specified
    in the OFP8 standard above, encoded as `s.eeee.mmm`:

    - (s)ign: 1 bit
    - (e)xponent: 4 bits
    - (m)antissa: 3 bits
    - exponent bias: 7
    - nan: {0,1}.1111.111
    - fn: finite (no inf or -inf encodings)
    - -0: 1.0000.000
    """
    comptime float8_e4m3fnuz = DType(
        mlir_value=__mlir_attr.`#kgen.dtype.constant<f8e4m3fnuz> : !kgen.dtype`
    )
    """Represents an 8-bit `e4m3fnuz` floating point format.

    See the [format reference](https://arxiv.org/pdf/2206.02915), encoded as `s.eeee.mmm`:

    - (s)ign: 1 bit
    - (e)xponent: 4 bits
    - (m)antissa: 3 bits
    - exponent bias: 8
    - nan: 1.0000.000
    - fn: finite (no inf or -inf encodings)
    - uz: unsigned zero (no -0 encoding)
    """
    comptime float8_e5m2 = DType(
        mlir_value=__mlir_attr.`#kgen.dtype.constant<f8e5m2> : !kgen.dtype`
    )
    """Represents the 8-bit `E5M2` floating point format.

    This type is defined in the
    [OFP8 standard](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-12-01-pdf-1),
    encoded as `s.eeeee.mm`:

    - (s)ign: 1 bit
    - (e)xponent: 5 bits
    - (m)antissa: 2 bits
    - exponent bias: 15
    - nan: {0,1}.11111.{01,10,11}
    - inf: {0,1}.11111.00
    - -0: 1.00000.00
    """
    comptime float8_e5m2fnuz = DType(
        mlir_value=__mlir_attr.`#kgen.dtype.constant<f8e5m2fnuz> : !kgen.dtype`
    )
    """Represents an 8-bit `e5m2fnuz` floating point format.

    See the [format reference](https://arxiv.org/pdf/2206.02915), encoded as `s.eeeee.mm`:

    - (s)ign: 1 bit
    - (e)xponent: 5 bits
    - (m)antissa: 2 bits
    - exponent bias: 16
    - nan: 1.00000.00
    - fn: finite (no inf or -inf encodings)
    - uz: unsigned zero (no -0 encoding)
    """

    comptime bfloat16 = DType(
        mlir_value=__mlir_attr.`#kgen.dtype.constant<bf16> : !kgen.dtype`
    )
    """Represents a brain floating point value whose bitwidth is 16."""
    comptime float16 = DType(
        mlir_value=__mlir_attr.`#kgen.dtype.constant<f16> : !kgen.dtype`
    )
    """Represents an IEEE754-2008 `binary16` floating point value."""

    comptime float32 = DType(
        mlir_value=__mlir_attr.`#kgen.dtype.constant<f32> : !kgen.dtype`
    )
    """Represents an IEEE754-2008 `binary32` floating point value."""

    comptime float64 = DType(
        mlir_value=__mlir_attr.`#kgen.dtype.constant<f64> : !kgen.dtype`
    )
    """Represents an IEEE754-2008 `binary64` floating point value."""

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @always_inline("builtin")
    fn __init__(out self, *, mlir_value: Self._mlir_type):
        """Construct a DType from MLIR dtype.

        Args:
            mlir_value: The MLIR dtype.
        """
        self._mlir_value = mlir_value

    @staticmethod
    fn _from_str(str: StringSlice) -> DType:
        """Construct a DType from a string.

        Args:
            str: The name of the DType.
        """
        if str.startswith("DType."):
            return Self._from_str(str.removeprefix("DType."))
        elif str == "bool":
            return DType.bool
        elif str == "int":
            return DType.int
        elif str == "uint":
            return DType.uint

        elif str == "uint8":
            return DType.uint8
        elif str == "int8":
            return DType.int8
        elif str == "uint16":
            return DType.uint16
        elif str == "int16":
            return DType.int16
        elif str == "uint32":
            return DType.uint32
        elif str == "int32":
            return DType.int32
        elif str == "uint64":
            return DType.uint64
        elif str == "int64":
            return DType.int64
        elif str == "uint128":
            return DType.uint128
        elif str == "int128":
            return DType.int128
        elif str == "uint256":
            return DType.uint256
        elif str == "int256":
            return DType.int256

        elif str == "float4_e2m1fn":
            return DType.float4_e2m1fn

        elif str == "float8_e3m4":
            return DType.float8_e3m4
        elif str == "float8_e4m3fn":
            return DType.float8_e4m3fn
        elif str == "float8_e4m3fnuz":
            return DType.float8_e4m3fnuz
        elif str == "float8_e8m0fnu":
            return DType.float8_e8m0fnu
        elif str == "float8_e5m2":
            return DType.float8_e5m2
        elif str == "float8_e5m2fnuz":
            return DType.float8_e5m2fnuz

        elif str == "bfloat16":
            return DType.bfloat16
        elif str == "float16":
            return DType.float16

        elif str == "float32":
            return DType.float32

        elif str == "float64":
            return DType.float64

        else:
            return DType.invalid

    @no_inline
    fn __str__(self) -> String:
        """Gets the name of the DType.

        Returns:
            The name of the dtype.
        """

        return String.write(self)

    @no_inline
    fn write_to(self, mut writer: Some[Writer]):
        """
        Formats this dtype to the provided Writer.

        Args:
            writer: The object to write to.
        """

        if self == DType.bool:
            return writer.write("bool")
        elif self == DType.int:
            return writer.write("int")
        elif self == DType.uint:
            return writer.write("uint")

        elif self == DType.uint8:
            return writer.write("uint8")
        elif self == DType.int8:
            return writer.write("int8")
        elif self == DType.uint16:
            return writer.write("uint16")
        elif self == DType.int16:
            return writer.write("int16")
        elif self == DType.uint32:
            return writer.write("uint32")
        elif self == DType.int32:
            return writer.write("int32")
        elif self == DType.uint64:
            return writer.write("uint64")
        elif self == DType.int64:
            return writer.write("int64")
        elif self == DType.uint128:
            return writer.write("uint128")
        elif self == DType.int128:
            return writer.write("int128")
        elif self == DType.uint256:
            return writer.write("uint256")
        elif self == DType.int256:
            return writer.write("int256")

        elif self == DType.float4_e2m1fn:
            return writer.write("float4_e2m1fn")

        elif self == DType.float8_e3m4:
            return writer.write("float8_e3m4")
        elif self == DType.float8_e4m3fn:
            return writer.write("float8_e4m3fn")
        elif self == DType.float8_e4m3fnuz:
            return writer.write("float8_e4m3fnuz")
        elif self == DType.float8_e8m0fnu:
            return writer.write("float8_e8m0fnu")
        elif self == DType.float8_e5m2:
            return writer.write("float8_e5m2")
        elif self == DType.float8_e5m2fnuz:
            return writer.write("float8_e5m2fnuz")

        elif self == DType.bfloat16:
            return writer.write("bfloat16")
        elif self == DType.float16:
            return writer.write("float16")

        elif self == DType.float32:
            return writer.write("float32")

        elif self == DType.float64:
            return writer.write("float64")

        elif self == DType.invalid:
            return writer.write("invalid")

        return writer.write("<<unknown>>")

    @always_inline("nodebug")
    fn write_repr_to(self, mut writer: Some[Writer]):
        """Write the string representation of the DType".

        Args:
            writer: The value to write to.
        """
        writer.write("DType.", self)

    @always_inline("nodebug")
    fn __repr__(self) -> String:
        """Gets the representation of the DType e.g. `"DType.float32"`.

        Returns:
            The representation of the dtype.
        """
        var string = String()
        self.write_repr_to(string)
        return string^

    @always_inline("nodebug")
    fn get_value(self) -> __mlir_type.`!kgen.dtype`:
        """Gets the associated internal kgen.dtype value.

        Returns:
            The kgen.dtype value.
        """
        return self._mlir_value

    @doc_private
    @staticmethod
    @always_inline("builtin")
    fn _from_ui8(ui8: UInt8._mlir_type) -> DType:
        return DType(
            mlir_value=__mlir_op.`pop.dtype.from_ui8`(
                __mlir_op.`pop.cast_to_builtin`[_type = __mlir_type.ui8](ui8)
            )
        )

    @doc_private
    @always_inline("builtin")
    fn _as_ui8(self) -> UInt8._mlir_type:
        return __mlir_op.`pop.cast_from_builtin`[_type = UInt8._mlir_type](
            __mlir_op.`pop.dtype.to_ui8`(self._mlir_value)
        )

    @doc_private
    @always_inline("builtin")
    fn _match(self, mask: UInt8) -> Bool:
        return self._match(mask._mlir_value)

    @doc_private
    @always_inline("builtin")
    fn _match(self, mask: UInt8._mlir_type) -> Bool:
        return Bool(
            mlir_value=__mlir_op.`pop.cmp`[
                pred = __mlir_attr.`#pop<cmp_pred ne>`
            ](
                __mlir_op.`pop.simd.and`(self._as_ui8(), mask),
                __mlir_attr.`#pop.simd<0> : !pop.scalar<ui8>`,
            )
        )

    @always_inline("builtin")
    fn __eq__(self, rhs: DType) -> Bool:
        """Compares one DType to another for equality.

        Args:
            rhs: The DType to compare against.

        Returns:
            True if the DTypes are the same and False otherwise.
        """
        return Bool(
            mlir_value=__mlir_op.`pop.cmp`[
                pred = __mlir_attr.`#pop<cmp_pred eq>`
            ](self._as_ui8(), rhs._as_ui8())
        )

    @always_inline("builtin")
    fn __ne__(self, rhs: DType) -> Bool:
        """Compares one DType to another for inequality.

        Args:
            rhs: The DType to compare against.

        Returns:
            False if the DTypes are the same and True otherwise.
        """
        return Bool(
            mlir_value=__mlir_op.`pop.cmp`[
                pred = __mlir_attr.`#pop<cmp_pred ne>`
            ](self._as_ui8(), rhs._as_ui8())
        )

    fn __hash__[H: Hasher](self, mut hasher: H):
        """Updates hasher with this `DType` value.

        Parameters:
            H: The hasher type.

        Args:
            hasher: The hasher instance.
        """
        hasher._update_with_simd(UInt8(mlir_value=self._as_ui8()))

    @always_inline("builtin")
    fn is_unsigned(self) -> Bool:
        """Returns True if the type parameter is unsigned and False otherwise.

        Returns:
            Returns True if the input type parameter is unsigned.
        """
        return (
            self == DType.uint
            or self._is_non_index_integral()
            and not self._match(_mIsSigned)
        )

    @always_inline("builtin")
    fn is_signed(self) -> Bool:
        """Returns True if the type parameter is signed and False otherwise.

        Returns:
            Returns True if the input type parameter is signed.
        """
        return self.is_floating_point() or (
            self.is_integral() and self._match(_mIsSigned)
        )

    @always_inline("builtin")
    fn _is_non_index_integral(self) -> Bool:
        """Returns True if the type parameter is a non-index integer value and False otherwise.

        Returns:
            Returns True if the input type parameter is a non-index integer.
        """
        return self._match(_mIsInteger)

    @always_inline("builtin")
    fn is_integral(self) -> Bool:
        """Returns True if the type parameter is an integer and False otherwise.

        Returns:
            Returns True if the input type parameter is an integer.
        """
        return (
            self == DType.int
            or self == DType.uint
            or self._is_non_index_integral()
        )

    @always_inline("builtin")
    fn is_floating_point(self) -> Bool:
        """Returns True if the type parameter is a floating-point and False
        otherwise.

        Returns:
            Returns True if the input type parameter is a floating-point.
        """
        return self._match(_mIsFloat)

    @always_inline("builtin")
    fn is_float8(self) -> Bool:
        """Returns True if the dtype is a 8bit-precision floating point type,
        e.g. float8_e5m2, float8_e5m2fnuz, float8_e4m3fn and float8_e4m3fnuz.

        Returns:
            True if the dtype is a 8bit-precision float, false otherwise.
        """

        return (
            self == DType.float8_e8m0fnu
            or self == DType.float8_e3m4
            or self == DType.float8_e4m3fn
            or self == DType.float8_e4m3fnuz
            or self == DType.float8_e5m2
            or self == DType.float8_e5m2fnuz
        )

    @always_inline("builtin")
    fn is_half_float(self) -> Bool:
        """Returns True if the dtype is a half-precision floating point type,
        e.g. either fp16 or bf16.

        Returns:
            True if the dtype is a half-precision float, false otherwise..
        """

        return self == DType.bfloat16 or self == DType.float16

    @always_inline("builtin")
    fn is_numeric(self) -> Bool:
        """Returns True if the type parameter is numeric (i.e. you can perform
        arithmetic operations on).

        Returns:
            Returns True if the input type parameter is either integral or
              floating-point.
        """
        return self.is_integral() or (
            self.is_floating_point()
            and (self != DType.float4_e2m1fn and self != DType.float8_e8m0fnu)
        )

    # ===-------------------------------------------------------------------===#
    # Floating point generics
    # ===-------------------------------------------------------------------===#

    @staticmethod
    @always_inline("nodebug")
    fn mantissa_width[dtype: DType]() -> Int:
        """Returns the mantissa width of a floating point type.

        Parameters:
            dtype: The DType.

        Returns:
            The mantissa width.
        """
        comptime assert (
            dtype.is_floating_point()
        ), "dtype must be floating point"
        return bit_width_of[dtype]() - DType.exponent_width[dtype]() - 1

    @staticmethod
    @always_inline("nodebug")
    fn max_exponent[dtype: DType]() -> Int:
        """Returns the max exponent of a floating point dtype without accounting
        for inf representations. This is not the maximum representable exponent,
        which is generally equal to the exponent_bias.

        Parameters:
            dtype: The DType.

        Returns:
            The max exponent.
        """
        comptime assert (
            dtype.is_floating_point()
        ), "dtype must be floating point"

        comptime if dtype == DType.float4_e2m1fn:
            return 2
        elif dtype in (DType.float8_e4m3fn, DType.float8_e4m3fnuz):
            return 8
        elif dtype in (DType.float8_e5m2, DType.float8_e5m2fnuz, DType.float16):
            return 16
        elif dtype in (DType.bfloat16, DType.float32):
            return 128
        elif dtype == DType.float64:
            return 1024
        else:
            comptime assert False, "unsupported float type"
            return {}

    @staticmethod
    @always_inline("nodebug")
    fn exponent_width[dtype: DType]() -> Int:
        """Returns the exponent width of a floating point type.

        Parameters:
            dtype: The DType.

        Returns:
            The exponent width.
        """
        comptime assert (
            dtype.is_floating_point()
        ), "dtype must be floating point"

        comptime if dtype == DType.float4_e2m1fn:
            return 2
        elif dtype in (DType.float8_e4m3fn, DType.float8_e4m3fnuz):
            return 4
        elif dtype in (DType.float8_e5m2, DType.float8_e5m2fnuz, DType.float16):
            return 5
        elif dtype in (DType.float32, DType.bfloat16):
            return 8
        elif dtype == DType.float64:
            return 11
        else:
            comptime assert False, "unsupported float type"
            return {}

    @staticmethod
    @always_inline
    fn exponent_bias[dtype: DType]() -> Int:
        """Returns the exponent bias of a floating point type.

        Parameters:
            dtype: The DType.

        Returns:
            The exponent bias.
        """

        comptime if dtype in (DType.float8_e4m3fnuz, DType.float8_e5m2fnuz):
            return DType.max_exponent[dtype]()
        else:
            return DType.max_exponent[dtype]() - 1

    # ===-------------------------------------------------------------------===#
    # __mlir_type
    # ===-------------------------------------------------------------------===#

    @always_inline("nodebug")
    fn __mlir_type(self) -> __mlir_type.`!kgen.deferred`:
        """Returns the MLIR type of the current DType as an MLIR type.

        Returns:
            The MLIR type of the current DType.
        """

        if self == DType.bool:
            return __mlir_attr.i1

        if self == DType.int:
            return __mlir_attr.index

        if self == DType.uint8:
            return __mlir_attr.ui8
        if self == DType.int8:
            return __mlir_attr.si8
        if self == DType.uint16:
            return __mlir_attr.ui16
        if self == DType.int16:
            return __mlir_attr.si16
        if self == DType.uint32:
            return __mlir_attr.ui32
        if self == DType.int32:
            return __mlir_attr.si32
        if self == DType.uint64:
            return __mlir_attr.ui64
        if self == DType.int64:
            return __mlir_attr.si64
        if self == DType.uint128:
            return __mlir_attr.ui128
        if self == DType.int128:
            return __mlir_attr.si128
        if self == DType.uint256:
            return __mlir_attr.ui256
        if self == DType.int256:
            return __mlir_attr.si256

        if self == DType.float4_e2m1fn:
            return __mlir_attr.f4E2M1FN

        if self == DType.float8_e8m0fnu:
            return __mlir_attr.f8E8M0FNU
        if self == DType.float8_e3m4:
            return __mlir_attr.f8E3M4
        if self == DType.float8_e4m3fn:
            return __mlir_attr.f8E4M3
        if self == DType.float8_e4m3fnuz:
            return __mlir_attr.f8E4M3FNUZ
        if self == DType.float8_e5m2:
            return __mlir_attr.f8E5M2
        if self == DType.float8_e5m2fnuz:
            return __mlir_attr.f8E5M2FNUZ

        if self == DType.bfloat16:
            return __mlir_attr.bf16
        if self == DType.float16:
            return __mlir_attr.f16

        if self == DType.float32:
            return __mlir_attr.f32

        if self == DType.float64:
            return __mlir_attr.f64

        abort("invalid dtype")

    # ===----------------------------------------------------------------------===#
    # utils
    # ===----------------------------------------------------------------------===#

    @staticmethod
    fn get_dtype[T: AnyType, size: Int = 1]() -> DType:
        """Get the `DType` if the given Type is a `SIMD[_, size]` of a `DType`.

        Parameters:
            T: AnyType.
            size: The SIMD size to compare against.

        Returns:
            The `DType` if matched, otherwise `DType.invalid`.
        """

        comptime if _type_is_eq[T, SIMD[DType.bool, size]]():
            return DType.bool
        elif _type_is_eq[T, SIMD[DType.int, size]]():
            return DType.int
        elif _type_is_eq[T, SIMD[DType.uint, size]]():
            return DType.uint

        elif _type_is_eq[T, SIMD[DType.uint8, size]]():
            return DType.uint8
        elif _type_is_eq[T, SIMD[DType.int8, size]]():
            return DType.int8
        elif _type_is_eq[T, SIMD[DType.uint16, size]]():
            return DType.uint16
        elif _type_is_eq[T, SIMD[DType.int16, size]]():
            return DType.int16
        elif _type_is_eq[T, SIMD[DType.uint32, size]]():
            return DType.uint32
        elif _type_is_eq[T, SIMD[DType.int32, size]]():
            return DType.int32
        elif _type_is_eq[T, SIMD[DType.uint64, size]]():
            return DType.uint64
        elif _type_is_eq[T, SIMD[DType.int64, size]]():
            return DType.int64
        elif _type_is_eq[T, SIMD[DType.uint128, size]]():
            return DType.uint128
        elif _type_is_eq[T, SIMD[DType.int128, size]]():
            return DType.int128
        elif _type_is_eq[T, SIMD[DType.uint256, size]]():
            return DType.uint256
        elif _type_is_eq[T, SIMD[DType.int256, size]]():
            return DType.int256

        elif _type_is_eq[T, SIMD[DType.float4_e2m1fn, size]]():
            return DType.float4_e2m1fn

        elif _type_is_eq[T, SIMD[DType.float8_e8m0fnu, size]]():
            return DType.float8_e8m0fnu
        elif _type_is_eq[T, SIMD[DType.float8_e3m4, size]]():
            return DType.float8_e3m4
        elif _type_is_eq[T, SIMD[DType.float8_e4m3fn, size]]():
            return DType.float8_e4m3fn
        elif _type_is_eq[T, SIMD[DType.float8_e4m3fnuz, size]]():
            return DType.float8_e4m3fnuz
        elif _type_is_eq[T, SIMD[DType.float8_e5m2, size]]():
            return DType.float8_e5m2
        elif _type_is_eq[T, SIMD[DType.float8_e5m2fnuz, size]]():
            return DType.float8_e5m2fnuz

        elif _type_is_eq[T, SIMD[DType.bfloat16, size]]():
            return DType.bfloat16
        elif _type_is_eq[T, SIMD[DType.float16, size]]():
            return DType.float16

        elif _type_is_eq[T, SIMD[DType.float32, size]]():
            return DType.float32

        elif _type_is_eq[T, SIMD[DType.float64, size]]():
            return DType.float64

        else:
            return DType.invalid

    @staticmethod
    fn is_scalar[T: AnyType]() -> Bool:
        """Whether the given Type is a Scalar of a DType.

        Parameters:
            T: AnyType.

        Returns:
            The result.
        """
        return Self.get_dtype[T]() != DType.invalid


# ===-------------------------------------------------------------------===#
# integral_type_of
# ===-------------------------------------------------------------------===#


@always_inline("nodebug")
fn _integral_type_of[dtype: DType]() -> DType:
    """Gets the integral type which has the same bitwidth as the input type."""

    comptime if dtype.is_integral():
        return dtype

    elif dtype.is_float8():
        return DType.int8
    elif dtype.is_half_float():
        return DType.int16
    elif dtype == DType.float32:
        return DType.int32
    elif dtype == DType.float64:
        return DType.int64

    return dtype.invalid


# ===-------------------------------------------------------------------===#
# _unsigned_integral_type_of
# ===-------------------------------------------------------------------===#


@always_inline("nodebug")
fn _unsigned_integral_type_of[dtype: DType]() -> DType:
    """Gets the unsigned integral type which has the same bitwidth as
    the input type."""

    comptime if dtype.is_unsigned():
        return dtype
    elif dtype.is_integral():
        return _uint_type_of_width[bit_width_of[dtype]()]()

    elif dtype.is_float8():
        return DType.uint8
    elif dtype.is_half_float():
        return DType.uint16
    elif dtype == DType.float32:
        return DType.uint32
    elif dtype == DType.float64:
        return DType.uint64

    return dtype.invalid


# ===-------------------------------------------------------------------===#
# _scientific_notation_digits
# ===-------------------------------------------------------------------===#


fn _scientific_notation_digits[
    dtype: DType
]() -> StaticString where dtype.is_floating_point():
    """Get the number of digits as a StaticString for the scientific notation
    representation of a float.
    """

    comptime if dtype.is_float8():
        return "2"
    elif dtype.is_half_float():
        return "4"
    elif dtype == DType.float32:
        return "8"
    else:
        return "16"


# ===-------------------------------------------------------------------===#
# _int_type_of_width
# ===-------------------------------------------------------------------===#


@always_inline
fn _int_type_of_width[width: Int]() -> DType:
    comptime assert width in (
        8,
        16,
        32,
        64,
        128,
        256,
    ), "width must be either 8, 16, 32, 64, 128, or 256"

    comptime if width == 8:
        return DType.int8
    elif width == 16:
        return DType.int16
    elif width == 32:
        return DType.int32
    elif width == 64:
        return DType.int64
    elif width == 128:
        return DType.int128
    else:
        return DType.int256


# ===-------------------------------------------------------------------===#
# _uint_type_of_width
# ===-------------------------------------------------------------------===#


@always_inline("builtin")
fn _uint_type_of_width[width: Int]() -> DType:
    # fmt: off
    comptime assert width in (
        1, 2, 4, 8, 16, 32, 64, 128, 256,
    ), "width must be 1, 2, 4, 8, 16, 32, 64, 128, or 256"
    return (
        DType._uint1 if width == 1 else
        DType._uint2 if width == 2 else
        DType._uint4 if width == 4 else
        DType.uint8 if width == 8 else
        DType.uint16 if width == 16 else
        DType.uint32 if width == 32 else
        DType.uint64 if width == 64 else
        DType.uint128 if width == 128 else
        DType.uint256
    )
    # fmt: on


# ===-------------------------------------------------------------------===#
# printf format
# ===-------------------------------------------------------------------===#


@always_inline
fn _index_printf_format() -> StaticString:
    comptime if bit_width_of[Int]() == 32:
        return "%d"
    else:
        return "%ld"


@always_inline
fn _get_dtype_printf_format[dtype: DType]() -> StaticString:
    comptime if dtype in (DType.bool, DType.int, DType.uint):
        return _index_printf_format()

    elif dtype == DType.uint8:
        return "%hhu"
    elif dtype == DType.int8:
        return "%hhi"
    elif dtype == DType.uint16:
        return "%hu"
    elif dtype == DType.int16:
        return "%hi"
    elif dtype == DType.uint32:
        return "%u"
    elif dtype == DType.int32:
        return "%i"
    elif dtype == DType.int64:
        return "%ld"
    elif dtype == DType.uint64:
        return "%lu"

    elif dtype.is_floating_point():
        return "%.17g"

    else:
        comptime assert False, "invalid dtype"

    return ""
