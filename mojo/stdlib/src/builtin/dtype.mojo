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
"""Implements the DType class.

These are Mojo built-ins, so you don't need to import them.
"""

from collections import KeyElement
from collections.string import StringSlice, StaticString
from hashlib._hasher import _HashableWithHasher, _Hasher
from sys import bitwidthof, os_is_windows, sizeof

alias _mIsSigned = UInt8(1)
alias _mIsInteger = UInt8(1 << 7)
alias _mIsNotInteger = UInt8(~(1 << 7))
alias _mIsFloat = UInt8(1 << 6)


@value
@register_passable("trivial")
struct DType(
    Stringable,
    Writable,
    Representable,
    KeyElement,
    CollectionElementNew,
    EqualityComparableCollectionElement,
    _HashableWithHasher,
):
    """Represents DType and provides methods for working with it."""

    alias type = __mlir_type.`!kgen.dtype`
    var value: Self.type
    """The underlying storage for the DType value."""

    alias invalid = DType(
        __mlir_attr.`#kgen.dtype.constant<invalid> : !kgen.dtype`
    )
    """Represents an invalid or unknown data type."""
    alias bool = DType(__mlir_attr.`#kgen.dtype.constant<bool> : !kgen.dtype`)
    """Represents a boolean data type."""
    alias int8 = DType(__mlir_attr.`#kgen.dtype.constant<si8> : !kgen.dtype`)
    """Represents a signed integer type whose bitwidth is 8."""
    alias uint8 = DType(__mlir_attr.`#kgen.dtype.constant<ui8> : !kgen.dtype`)
    """Represents an unsigned integer type whose bitwidth is 8."""
    alias int16 = DType(__mlir_attr.`#kgen.dtype.constant<si16> : !kgen.dtype`)
    """Represents a signed integer type whose bitwidth is 16."""
    alias uint16 = DType(__mlir_attr.`#kgen.dtype.constant<ui16> : !kgen.dtype`)
    """Represents an unsigned integer type whose bitwidth is 16."""
    alias int32 = DType(__mlir_attr.`#kgen.dtype.constant<si32> : !kgen.dtype`)
    """Represents a signed integer type whose bitwidth is 32."""
    alias uint32 = DType(__mlir_attr.`#kgen.dtype.constant<ui32> : !kgen.dtype`)
    """Represents an unsigned integer type whose bitwidth is 32."""
    alias int64 = DType(__mlir_attr.`#kgen.dtype.constant<si64> : !kgen.dtype`)
    """Represents a signed integer type whose bitwidth is 64."""
    alias uint64 = DType(__mlir_attr.`#kgen.dtype.constant<ui64> : !kgen.dtype`)
    """Represents an unsigned integer type whose bitwidth is 64."""
    alias int128 = DType(
        __mlir_attr.`#kgen.dtype.constant<si128> : !kgen.dtype`
    )
    """Represents a signed integer type whose bitwidth is 128."""
    alias uint128 = DType(
        __mlir_attr.`#kgen.dtype.constant<ui128> : !kgen.dtype`
    )
    """Represents an unsigned integer type whose bitwidth is 128."""
    alias int256 = DType(
        __mlir_attr.`#kgen.dtype.constant<si256> : !kgen.dtype`
    )
    """Represents a signed integer type whose bitwidth is 256."""
    alias uint256 = DType(
        __mlir_attr.`#kgen.dtype.constant<ui256> : !kgen.dtype`
    )
    """Represents an unsigned integer type whose bitwidth is 256."""
    alias float8_e5m2 = DType(
        __mlir_attr.`#kgen.dtype.constant<f8e5m2> : !kgen.dtype`
    )
    """Represents a FP8E5M2 floating point format from the [OFP8
    standard](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-12-01-pdf-1).

    The 8 bits are encoded as `seeeeemm`:
    - (s)ign: 1 bit
    - (e)xponent: 5 bits
    - (m)antissa: 2 bits
    - exponent bias: 15
    - nan: {0,1}11111{01,10,11}
    - inf: 01111100
    - -inf: 11111100
    - -0: 10000000
    """
    alias float8_e5m2fnuz = DType(
        __mlir_attr.`#kgen.dtype.constant<f8e5m2fnuz> : !kgen.dtype`
    )
    """Represents a FP8E5M2FNUZ floating point format.

    The 8 bits are encoded as `seeeeemm`:
    - (s)ign: 1 bit
    - (e)xponent: 5 bits
    - (m)antissa: 2 bits
    - exponent bias: 16
    - nan: 10000000
    - fn: finite (no inf or -inf encodings)
    - uz: unsigned zero (no -0 encoding)
    """
    alias float8_e4m3 = DType(
        __mlir_attr.`#kgen.dtype.constant<f8e4m3> : !kgen.dtype`
    )
    """Represents a FP8E4M3 floating point format from the [OFP8
    standard](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-12-01-pdf-1).

    The 8 bits are encoded as `seeeemmm`:
    - (s)ign: 1 bit
    - (e)xponent: 4 bits
    - (m)antissa: 3 bits
    - exponent bias: 7
    - nan: 01111111, 11111111
    - -0: 10000000
    """
    alias float8_e4m3fn = DType(
        __mlir_attr.`#kgen.dtype.constant<f8e4m3fn> : !kgen.dtype`
    )
    """Represents a FP8E4M3 floating point format from the [OFP8
    standard](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-12-01-pdf-1).

    The 8 bits are encoded as `seeeemmm`:
    - (s)ign: 1 bit
    - (e)xponent: 4 bits
    - (m)antissa: 3 bits
    - exponent bias: 7
    - nan: 01111111, 11111111
    - -0: 10000000
    - fn: finite (no inf or -inf encodings)
    """
    alias float8_e3m4 = DType(
        __mlir_attr.`#kgen.dtype.constant<f8e3m4> : !kgen.dtype`
    )
    """Represents a FP8E3M4 floating point format from the [OFP8
    standard](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-12-01-pdf-1).

    The 8 bits are encoded as `seeeemmm`:
    - (s)ign: 1 bit
    - (e)xponent: 3 bits
    - (m)antissa: 4 bits
    - exponent bias: 6
    - nan: 00111111, 11111111
    - -0: 10000000
    - fn: finite (no inf or -inf encodings)
    """
    alias float8_e4m3fnuz = DType(
        __mlir_attr.`#kgen.dtype.constant<f8e4m3fnuz> : !kgen.dtype`
    )
    """Represents a FP8E4M3FNUZ floating point format.

    The 8 bits are encoded as `seeeemmm`:
    - (s)ign: 1 bit
    - (e)xponent: 4 bits
    - (m)antissa: 3 bits
    - exponent bias: 8
    - nan: 10000000
    - fn: finite (no inf or -inf encodings)
    - uz: unsigned zero (no -0 encoding)
    """
    alias bfloat16 = DType(
        __mlir_attr.`#kgen.dtype.constant<bf16> : !kgen.dtype`
    )
    """Represents a brain floating point value whose bitwidth is 16."""
    alias float16 = DType(__mlir_attr.`#kgen.dtype.constant<f16> : !kgen.dtype`)
    """Represents an IEEE754-2008 `binary16` floating point value."""
    alias float32 = DType(__mlir_attr.`#kgen.dtype.constant<f32> : !kgen.dtype`)
    """Represents an IEEE754-2008 `binary32` floating point value."""
    alias tensor_float32 = DType(
        __mlir_attr.`#kgen.dtype.constant<tf32> : !kgen.dtype`
    )
    """Represents a special floating point format supported by NVIDIA Tensor
    Cores, with the same range as float32 and reduced precision (>=10 bits).
    Note that this dtype is only available on NVIDIA GPUs.
    """
    alias float64 = DType(__mlir_attr.`#kgen.dtype.constant<f64> : !kgen.dtype`)
    """Represents an IEEE754-2008 `binary64` floating point value."""
    alias index = DType(__mlir_attr.`#kgen.dtype.constant<index> : !kgen.dtype`)
    """Represents an integral type whose bitwidth is the maximum integral value
    on the system."""

    @always_inline
    fn copy(self) -> Self:
        """Copy this DType.

        Returns:
            A copy of the value.
        """
        return self

    @always_inline("builtin")
    @implicit
    fn __init__(out self, value: Self.type):
        """Construct a DType from MLIR dtype.

        Args:
            value: The MLIR dtype.
        """
        self.value = value

    @staticmethod
    fn _from_str(str: StringSlice) -> DType:
        """Construct a DType from a string.

        Args:
            str: The name of the DType.
        """
        if str.startswith("DType."):
            return Self._from_str(String(str).removeprefix("DType."))
        elif str == "bool":
            return DType.bool
        elif str == "int8":
            return DType.int8
        elif str == "uint8":
            return DType.uint8
        elif str == "int16":
            return DType.int16
        elif str == "uint16":
            return DType.uint16
        elif str == "int32":
            return DType.int32
        elif str == "uint32":
            return DType.uint32
        elif str == "int64":
            return DType.int64
        elif str == "uint64":
            return DType.uint64
        elif str == "int128":
            return DType.int128
        elif str == "uint128":
            return DType.uint128
        elif str == "int256":
            return DType.int256
        elif str == "uint256":
            return DType.uint256
        elif str == "index":
            return DType.index
        elif str == "float8_e3m4":
            return DType.float8_e3m4
        elif str == "float8_e4m3":
            return DType.float8_e4m3
        elif str == "float8_e5m2":
            return DType.float8_e5m2
        elif str == "float8_e5m2fnuz":
            return DType.float8_e5m2fnuz
        elif str == "float8_e4m3fn":
            return DType.float8_e4m3fn
        elif str == "float8_e4m3fnuz":
            return DType.float8_e4m3fnuz
        elif str == "bfloat16":
            return DType.bfloat16
        elif str == "float16":
            return DType.float16
        elif str == "float32":
            return DType.float32
        elif str == "float64":
            return DType.float64
        elif str == "tensor_float32":
            return DType.tensor_float32
        elif str == "invalid":
            return DType.invalid
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
    fn write_to[W: Writer](self, mut writer: W):
        """
        Formats this dtype to the provided Writer.

        Parameters:
            W: A type conforming to the Writable trait.

        Args:
            writer: The object to write to.
        """

        if self == DType.bool:
            return writer.write("bool")
        elif self == DType.int8:
            return writer.write("int8")
        elif self == DType.uint8:
            return writer.write("uint8")
        elif self == DType.int16:
            return writer.write("int16")
        elif self == DType.uint16:
            return writer.write("uint16")
        elif self == DType.int32:
            return writer.write("int32")
        elif self == DType.uint32:
            return writer.write("uint32")
        elif self == DType.int64:
            return writer.write("int64")
        elif self == DType.uint64:
            return writer.write("uint64")
        elif self == DType.int128:
            return writer.write("int128")
        elif self == DType.uint128:
            return writer.write("uint128")
        elif self == DType.int256:
            return writer.write("int256")
        elif self == DType.uint256:
            return writer.write("uint256")
        elif self == DType.index:
            return writer.write("index")
        elif self == DType.float8_e3m4:
            return writer.write("float8_e3m4")
        elif self == DType.float8_e5m2:
            return writer.write("float8_e5m2")
        elif self == DType.float8_e5m2fnuz:
            return writer.write("float8_e5m2fnuz")
        elif self == DType.float8_e4m3fn:
            return writer.write("float8_e4m3fn")
        elif self == DType.float8_e4m3fnuz:
            return writer.write("float8_e4m3fnuz")
        elif self == DType.float8_e4m3:
            return writer.write("float8_e4m3")
        elif self == DType.bfloat16:
            return writer.write("bfloat16")
        elif self == DType.float16:
            return writer.write("float16")
        elif self == DType.float32:
            return writer.write("float32")
        elif self == DType.tensor_float32:
            return writer.write("tensor_float32")
        elif self == DType.float64:
            return writer.write("float64")
        elif self == DType.invalid:
            return writer.write("invalid")

        return writer.write("<<unknown>>")

    @always_inline("nodebug")
    fn __repr__(self) -> String:
        """Gets the representation of the DType e.g. `"DType.float32"`.

        Returns:
            The representation of the dtype.
        """
        return String.write("DType.", self)

    @always_inline("nodebug")
    fn get_value(self) -> __mlir_type.`!kgen.dtype`:
        """Gets the associated internal kgen.dtype value.

        Returns:
            The kgen.dtype value.
        """
        return self.value

    @staticmethod
    fn _from_ui8(ui8: __mlir_type.ui8) -> DType:
        return __mlir_op.`pop.dtype.from_ui8`(ui8)

    @staticmethod
    fn _from_ui8(ui8: __mlir_type.`!pop.scalar<ui8>`) -> DType:
        return DType._from_ui8(
            __mlir_op.`pop.cast_to_builtin`[_type = __mlir_type.ui8](ui8)
        )

    @always_inline("nodebug")
    fn _as_i8(
        self,
    ) -> __mlir_type.`!pop.scalar<ui8>`:
        var val = __mlir_op.`pop.dtype.to_ui8`(self.value)
        return __mlir_op.`pop.cast_from_builtin`[
            _type = __mlir_type.`!pop.scalar<ui8>`
        ](val)

    @always_inline("nodebug")
    fn __is__(self, rhs: DType) -> Bool:
        """Compares one DType to another for equality.

        Args:
            rhs: The DType to compare against.

        Returns:
            True if the DTypes are the same and False otherwise.
        """
        return self == rhs

    @always_inline("nodebug")
    fn __isnot__(self, rhs: DType) -> Bool:
        """Compares one DType to another for inequality.

        Args:
            rhs: The DType to compare against.

        Returns:
            True if the DTypes are the same and False otherwise.
        """
        return self != rhs

    @always_inline("nodebug")
    fn __eq__(self, rhs: DType) -> Bool:
        """Compares one DType to another for equality.

        Args:
            rhs: The DType to compare against.

        Returns:
            True if the DTypes are the same and False otherwise.
        """
        return __mlir_op.`pop.cmp`[pred = __mlir_attr.`#pop<cmp_pred eq>`](
            self._as_i8(), rhs._as_i8()
        )

    @always_inline("nodebug")
    fn __ne__(self, rhs: DType) -> Bool:
        """Compares one DType to another for inequality.

        Args:
            rhs: The DType to compare against.

        Returns:
            False if the DTypes are the same and True otherwise.
        """
        return __mlir_op.`pop.cmp`[pred = __mlir_attr.`#pop<cmp_pred ne>`](
            self._as_i8(), rhs._as_i8()
        )

    fn __hash__(self) -> UInt:
        """Return a 64-bit hash for this `DType` value.

        Returns:
            A 64-bit integer hash of this `DType` value.
        """
        return hash(UInt8(self._as_i8()))

    fn __hash__[H: _Hasher](self, mut hasher: H):
        """Updates hasher with this `DType` value.

        Parameters:
            H: The hasher type.

        Args:
            hasher: The hasher instance.
        """
        hasher._update_with_simd(UInt8(self._as_i8()))

    @always_inline("nodebug")
    fn is_unsigned(self) -> Bool:
        """Returns True if the type parameter is unsigned and False otherwise.

        Returns:
            Returns True if the input type parameter is unsigned.
        """
        if not self.is_integral():
            return False
        return Bool(
            __mlir_op.`pop.cmp`[pred = __mlir_attr.`#pop<cmp_pred eq>`](
                __mlir_op.`pop.simd.and`(self._as_i8(), _mIsSigned.value),
                UInt8(0).value,
            )
        )

    @always_inline("nodebug")
    fn is_signed(self) -> Bool:
        """Returns True if the type parameter is signed and False otherwise.

        Returns:
            Returns True if the input type parameter is signed.
        """
        if self is DType.index or self.is_floating_point():
            return True
        if not self.is_integral():
            return False
        return Bool(
            __mlir_op.`pop.cmp`[pred = __mlir_attr.`#pop<cmp_pred ne>`](
                __mlir_op.`pop.simd.and`(self._as_i8(), _mIsSigned.value),
                UInt8(0).value,
            )
        )

    @always_inline("nodebug")
    fn _is_non_index_integral(self) -> Bool:
        """Returns True if the type parameter is a non-index integer value and False otherwise.

        Returns:
            Returns True if the input type parameter is a non-index integer.
        """
        return Bool(
            __mlir_op.`pop.cmp`[pred = __mlir_attr.`#pop<cmp_pred ne>`](
                __mlir_op.`pop.simd.and`(self._as_i8(), _mIsInteger.value),
                UInt8(0).value,
            )
        )

    @always_inline("nodebug")
    fn is_integral(self) -> Bool:
        """Returns True if the type parameter is an integer and False otherwise.

        Returns:
            Returns True if the input type parameter is an integer.
        """
        if self is DType.index:
            return True
        return self._is_non_index_integral()

    @always_inline("nodebug")
    fn is_floating_point(self) -> Bool:
        """Returns True if the type parameter is a floating-point and False
        otherwise.

        Returns:
            Returns True if the input type parameter is a floating-point.
        """
        if self.is_integral():
            return False
        return Bool(
            __mlir_op.`pop.cmp`[pred = __mlir_attr.`#pop<cmp_pred ne>`](
                __mlir_op.`pop.simd.and`(self._as_i8(), _mIsFloat.value),
                UInt8(0).value,
            )
        )

    @always_inline("nodebug")
    fn is_float8(self) -> Bool:
        """Returns True if the dtype is a 8bit-precision floating point type,
        e.g. float8_e5m2, float8_e5m2fnuz, float8_e4m3fn and float8_e4m3fnuz.

        Returns:
            True if the dtype is a 8bit-precision float, false otherwise.
        """

        return self in (
            DType.float8_e5m2,
            DType.float8_e4m3,
            DType.float8_e4m3fn,
            DType.float8_e5m2fnuz,
            DType.float8_e4m3fnuz,
            DType.float8_e3m4,
        )

    @always_inline("nodebug")
    fn is_half_float(self) -> Bool:
        """Returns True if the dtype is a half-precision floating point type,
        e.g. either fp16 or bf16.

        Returns:
            True if the dtype is a half-precision float, false otherwise..
        """

        return self in (DType.bfloat16, DType.float16)

    @always_inline("nodebug")
    fn is_numeric(self) -> Bool:
        """Returns True if the type parameter is numeric (i.e. you can perform
        arithmetic operations on).

        Returns:
            Returns True if the input type parameter is either integral or
              floating-point.
        """
        return self.is_integral() or self.is_floating_point()

    @always_inline
    fn sizeof(self) -> Int:
        """Returns the size in bytes of the current DType.

        Returns:
            Returns the size in bytes of the current DType.
        """

        if self._is_non_index_integral():
            return Int(
                UInt8(
                    __mlir_op.`pop.shl`(
                        UInt8(1).value,
                        __mlir_op.`pop.sub`(
                            __mlir_op.`pop.shr`(
                                __mlir_op.`pop.simd.and`(
                                    self._as_i8(), _mIsNotInteger.value
                                ),
                                UInt8(1).value,
                            ),
                            UInt8(3).value,
                        ),
                    )
                )
            )

        if self == DType.bool:
            return sizeof[DType.bool]()
        if self == DType.index:
            return sizeof[DType.index]()
        if self == DType.float8_e5m2:
            return sizeof[DType.float8_e5m2]()
        if self == DType.float8_e5m2fnuz:
            return sizeof[DType.float8_e5m2fnuz]()
        if self == DType.float8_e4m3:
            return sizeof[DType.float8_e4m3]()
        if self == DType.float8_e4m3fn:
            return sizeof[DType.float8_e4m3fn]()
        if self == DType.float8_e4m3fnuz:
            return sizeof[DType.float8_e4m3fnuz]()
        if self == DType.float8_e3m4:
            return sizeof[DType.float8_e3m4]()
        if self == DType.bfloat16:
            return sizeof[DType.bfloat16]()
        if self == DType.float16:
            return sizeof[DType.float16]()
        if self == DType.float32:
            return sizeof[DType.float32]()
        if self == DType.tensor_float32:
            return sizeof[DType.tensor_float32]()
        if self == DType.float64:
            return sizeof[DType.float64]()
        return sizeof[DType.invalid]()

    @always_inline
    fn bitwidth(self) -> Int:
        """Returns the size in bits of the current DType.

        Returns:
            Returns the size in bits of the current DType.
        """
        return 8 * self.sizeof()

    # ===-------------------------------------------------------------------===#
    # dispatch_integral
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn dispatch_integral[
        func: fn[dtype: DType] () capturing [_] -> None
    ](self) raises:
        """Dispatches an integral function corresponding to the current DType.

        Constraints:
            DType must be integral.

        Parameters:
            func: A parametrized on dtype function to dispatch.
        """
        if self is DType.uint8:
            func[DType.uint8]()
        elif self is DType.int8:
            func[DType.int8]()
        elif self is DType.uint16:
            func[DType.uint16]()
        elif self is DType.int16:
            func[DType.int16]()
        elif self is DType.uint32:
            func[DType.uint32]()
        elif self is DType.int32:
            func[DType.int32]()
        elif self is DType.uint64:
            func[DType.uint64]()
        elif self is DType.int64:
            func[DType.int64]()
        elif self is DType.index:
            func[DType.index]()
        else:
            raise Error("only integral types are supported")

    # ===-------------------------------------------------------------------===#
    # dispatch_floating
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn dispatch_floating[
        func: fn[dtype: DType] () capturing [_] -> None
    ](self) raises:
        """Dispatches a floating-point function corresponding to the current DType.

        Constraints:
            DType must be floating-point or integral.

        Parameters:
            func: A parametrized on dtype function to dispatch.
        """
        if self is DType.float16:
            func[DType.float16]()
        # TODO(#15473): Enable after extending LLVM support
        # elif self is DType.bfloat16:
        #     func[DType.bfloat16]()
        elif self is DType.float32:
            func[DType.float32]()
        elif self is DType.float64:
            func[DType.float64]()
        else:
            raise Error("only floating point types are supported")

    @always_inline
    fn _dispatch_bitwidth[
        func: fn[dtype: DType] () capturing [_] -> None,
    ](self) raises:
        """Dispatches a function corresponding to the current DType's bitwidth.
        This should only be used if func only depends on the bitwidth of the dtype,
        and not other properties of the dtype.

        Parameters:
            func: A parametrized on dtype function to dispatch.
        """
        var bitwidth = self.bitwidth()
        if bitwidth == 8:
            func[DType.uint8]()
        elif bitwidth == 16:
            func[DType.uint16]()
        elif bitwidth == 32:
            func[DType.uint32]()
        elif bitwidth == 64:
            func[DType.uint64]()
        else:
            raise Error(
                "bitwidth_dispatch only supports types with bitwidth [8, 16,"
                " 32, 64]"
            )
        return

    @always_inline
    fn _dispatch_custom[
        func: fn[dtype: DType] () capturing [_] -> None, *dtypes: DType
    ](self) raises:
        """Dispatches a function corresponding to current DType if it matches
        any type in the dtypes parameter.

        Parameters:
            func: A parametrized on dtype function to dispatch.
            dtypes: A list of DTypes on which to do dispatch.
        """
        alias dtype_var = VariadicList[DType](dtypes)

        @parameter
        for idx in range(len(dtype_var)):
            alias dtype = dtype_var[idx]
            if self == dtype:
                return func[dtype]()

        raise Error(
            "dispatch_custom: dynamic_type does not match any dtype parameters"
        )

    # ===-------------------------------------------------------------------===#
    # dispatch_arithmetic
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn dispatch_arithmetic[
        func: fn[dtype: DType] () capturing [_] -> None
    ](self) raises:
        """Dispatches a function corresponding to the current DType.

        Parameters:
            func: A parametrized on dtype function to dispatch.
        """
        if self.is_floating_point():
            self.dispatch_floating[func]()
        elif self.is_integral():
            self.dispatch_integral[func]()
        else:
            raise Error("only arithmetic types are supported")


# ===-------------------------------------------------------------------===#
# integral_type_of
# ===-------------------------------------------------------------------===#


@always_inline("nodebug")
fn _integral_type_of[dtype: DType]() -> DType:
    """Gets the integral type which has the same bitwidth as the input type."""

    @parameter
    if dtype.is_integral():
        return dtype
    elif dtype.is_float8():
        return DType.int8
    elif dtype.is_half_float():
        return DType.int16
    elif dtype is DType.float32 or dtype is DType.tensor_float32:
        return DType.int32
    elif dtype is DType.float64:
        return DType.int64

    return dtype.invalid


@always_inline("nodebug")
fn _uint_type_of[dtype: DType]() -> DType:
    """Gets the unsigned integral type which has the same bitwidth as the input
    type."""

    @parameter
    if dtype.is_integral() and dtype.is_unsigned():
        return dtype
    elif dtype.is_float8() or dtype is DType.int8:
        return DType.uint8
    elif dtype.is_half_float() or dtype is DType.int16:
        return DType.uint16
    elif (
        dtype is DType.float32
        or dtype is DType.tensor_float32
        or dtype is DType.int32
    ):
        return DType.uint32
    elif dtype is DType.float64 or dtype is DType.int64:
        return DType.uint64

    return dtype.invalid


# ===-------------------------------------------------------------------===#
# _unsigned_integral_type_of
# ===-------------------------------------------------------------------===#


@always_inline("nodebug")
fn _unsigned_integral_type_of[dtype: DType]() -> DType:
    """Gets the unsigned integral type which has the same bitwidth as
    the input type."""

    @parameter
    if dtype.is_integral():
        return _uint_type_of_width[bitwidthof[dtype]()]()
    elif dtype.is_float8():
        return DType.uint8
    elif dtype.is_half_float():
        return DType.uint16
    elif dtype is DType.float32 or dtype is DType.tensor_float32:
        return DType.uint32
    elif dtype is DType.float64:
        return DType.uint64

    return dtype.invalid


# ===-------------------------------------------------------------------===#
# _scientific_notation_digits
# ===-------------------------------------------------------------------===#


fn _scientific_notation_digits[dtype: DType]() -> StaticString:
    """Get the number of digits as a StaticString for the scientific notation
    representation of a float.
    """
    constrained[dtype.is_floating_point(), "expected floating point type"]()

    @parameter
    if dtype.is_float8():
        return "2"
    elif dtype.is_half_float():
        return "4"
    elif dtype is DType.float32 or dtype is DType.tensor_float32:
        return "8"
    else:
        constrained[dtype is DType.float64, "unknown floating point type"]()
        return "16"


# ===-------------------------------------------------------------------===#
# _int_type_of_width
# ===-------------------------------------------------------------------===#


@parameter
@always_inline
fn _int_type_of_width[width: Int]() -> DType:
    constrained[
        width == 8 or width == 16 or width == 32 or width == 64,
        "width must be either 8, 16, 32, or 64",
    ]()

    @parameter
    if width == 8:
        return DType.int8
    elif width == 16:
        return DType.int16
    elif width == 32:
        return DType.int32
    else:
        return DType.int64


# ===-------------------------------------------------------------------===#
# _uint_type_of_width
# ===-------------------------------------------------------------------===#


@parameter
@always_inline
fn _uint_type_of_width[width: Int]() -> DType:
    constrained[
        width == 8 or width == 16 or width == 32 or width == 64,
        "width must be either 8, 16, 32, or 64",
    ]()

    @parameter
    if width == 8:
        return DType.uint8
    elif width == 16:
        return DType.uint16
    elif width == 32:
        return DType.uint32
    else:
        return DType.uint64


# ===-------------------------------------------------------------------===#
# printf format
# ===-------------------------------------------------------------------===#


@always_inline
fn _index_printf_format() -> StaticString:
    @parameter
    if bitwidthof[Int]() == 32:
        return "%d"
    elif os_is_windows():
        return "%lld"
    else:
        return "%ld"


@always_inline
fn _get_dtype_printf_format[dtype: DType]() -> StaticString:
    @parameter
    if dtype is DType.bool:
        return _index_printf_format()
    elif dtype is DType.uint8:
        return "%hhu"
    elif dtype is DType.int8:
        return "%hhi"
    elif dtype is DType.uint16:
        return "%hu"
    elif dtype is DType.int16:
        return "%hi"
    elif dtype is DType.uint32:
        return "%u"
    elif dtype is DType.int32:
        return "%i"
    elif dtype is DType.int64:

        @parameter
        if os_is_windows():
            return "%lld"
        else:
            return "%ld"
    elif dtype is DType.uint64:

        @parameter
        if os_is_windows():
            return "%llu"
        else:
            return "%lu"
    elif dtype is DType.index:
        return _index_printf_format()

    elif dtype.is_floating_point():
        return "%.17g"

    else:
        constrained[False, "invalid dtype"]()

    return ""
