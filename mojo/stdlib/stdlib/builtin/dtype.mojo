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


from hashlib.hasher import Hasher
from os import abort
from sys import bitwidthof, os_is_windows, sizeof
from sys.intrinsics import _type_is_eq


alias _mIsSigned = UInt8(1)
alias _mIsInteger = UInt8(1 << 7)
alias _mIsNotInteger = UInt8(~(1 << 7))
alias _mIsFloat = UInt8(1 << 6)


@register_passable("trivial")
struct DType(
    Copyable,
    EqualityComparable,
    ExplicitlyCopyable,
    Hashable,
    KeyElement,
    Movable,
    Representable,
    Stringable,
    Writable,
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

    alias index = DType(__mlir_attr.`#kgen.dtype.constant<index> : !kgen.dtype`)
    """Represents an integral type whose bitwidth is the maximum integral value
    on the system."""

    alias _uint1 = DType(__mlir_attr.`#kgen.dtype.constant<ui1> : !kgen.dtype`)
    alias _uint2 = DType(__mlir_attr.`#kgen.dtype.constant<ui2> : !kgen.dtype`)
    alias _uint4 = DType(__mlir_attr.`#kgen.dtype.constant<ui4> : !kgen.dtype`)

    alias uint8 = DType(__mlir_attr.`#kgen.dtype.constant<ui8> : !kgen.dtype`)
    """Represents an unsigned integer type whose bitwidth is 8."""
    alias int8 = DType(__mlir_attr.`#kgen.dtype.constant<si8> : !kgen.dtype`)
    """Represents a signed integer type whose bitwidth is 8."""
    alias uint16 = DType(__mlir_attr.`#kgen.dtype.constant<ui16> : !kgen.dtype`)
    """Represents an unsigned integer type whose bitwidth is 16."""
    alias int16 = DType(__mlir_attr.`#kgen.dtype.constant<si16> : !kgen.dtype`)
    """Represents a signed integer type whose bitwidth is 16."""
    alias uint32 = DType(__mlir_attr.`#kgen.dtype.constant<ui32> : !kgen.dtype`)
    """Represents an unsigned integer type whose bitwidth is 32."""
    alias int32 = DType(__mlir_attr.`#kgen.dtype.constant<si32> : !kgen.dtype`)
    """Represents a signed integer type whose bitwidth is 32."""
    alias uint64 = DType(__mlir_attr.`#kgen.dtype.constant<ui64> : !kgen.dtype`)
    """Represents an unsigned integer type whose bitwidth is 64."""
    alias int64 = DType(__mlir_attr.`#kgen.dtype.constant<si64> : !kgen.dtype`)
    """Represents a signed integer type whose bitwidth is 64."""
    alias uint128 = DType(
        __mlir_attr.`#kgen.dtype.constant<ui128> : !kgen.dtype`
    )
    """Represents an unsigned integer type whose bitwidth is 128."""
    alias int128 = DType(
        __mlir_attr.`#kgen.dtype.constant<si128> : !kgen.dtype`
    )
    """Represents a signed integer type whose bitwidth is 128."""
    alias uint256 = DType(
        __mlir_attr.`#kgen.dtype.constant<ui256> : !kgen.dtype`
    )
    """Represents an unsigned integer type whose bitwidth is 256."""
    alias int256 = DType(
        __mlir_attr.`#kgen.dtype.constant<si256> : !kgen.dtype`
    )
    """Represents a signed integer type whose bitwidth is 256."""

    alias float8_e3m4 = DType(
        __mlir_attr.`#kgen.dtype.constant<f8e3m4> : !kgen.dtype`
    )
    """Represents an 8-bit e3m4 floating point format, encoded as `seeemmmm`:
    - (s)ign: 1 bit
    - (e)xponent: 3 bits
    - (m)antissa: 4 bits
    - exponent bias: 3
    - nan: 00111111, 11111111
    - -0: 10000000
    - fn: finite (no inf or -inf encodings)
    """
    alias float8_e4m3fn = DType(
        __mlir_attr.`#kgen.dtype.constant<f8e4m3fn> : !kgen.dtype`
    )
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
    alias float8_e4m3fnuz = DType(
        __mlir_attr.`#kgen.dtype.constant<f8e4m3fnuz> : !kgen.dtype`
    )
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
    alias float8_e5m2 = DType(
        __mlir_attr.`#kgen.dtype.constant<f8e5m2> : !kgen.dtype`
    )
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
    alias float8_e5m2fnuz = DType(
        __mlir_attr.`#kgen.dtype.constant<f8e5m2fnuz> : !kgen.dtype`
    )
    """Represents an 8-bit floating point format, encoded as `seeeeemm`:
    - (s)ign: 1 bit
    - (e)xponent: 5 bits
    - (m)antissa: 2 bits
    - exponent bias: 16
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
            return Self._from_str(str.removeprefix("DType."))
        elif str == "bool":
            return DType.bool
        elif str == "index":
            return DType.index

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

        elif str == "float8_e3m4":
            return DType.float8_e3m4
        elif str == "float8_e4m3fn":
            return DType.float8_e4m3fn
        elif str == "float8_e4m3fnuz":
            return DType.float8_e4m3fnuz
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
        elif str == "tensor_float32":
            return DType.tensor_float32

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
    fn write_to[W: Writer](self, mut writer: W):
        """
        Formats this dtype to the provided Writer.

        Parameters:
            W: A type conforming to the Writable trait.

        Args:
            writer: The object to write to.
        """

        if self is DType.bool:
            return writer.write("bool")
        elif self is DType.index:
            return writer.write("index")

        elif self is DType.uint8:
            return writer.write("uint8")
        elif self is DType.int8:
            return writer.write("int8")
        elif self is DType.uint16:
            return writer.write("uint16")
        elif self is DType.int16:
            return writer.write("int16")
        elif self is DType.uint32:
            return writer.write("uint32")
        elif self is DType.int32:
            return writer.write("int32")
        elif self is DType.uint64:
            return writer.write("uint64")
        elif self is DType.int64:
            return writer.write("int64")
        elif self is DType.uint128:
            return writer.write("uint128")
        elif self is DType.int128:
            return writer.write("int128")
        elif self is DType.uint256:
            return writer.write("uint256")
        elif self is DType.int256:
            return writer.write("int256")

        elif self is DType.float8_e3m4:
            return writer.write("float8_e3m4")
        elif self is DType.float8_e4m3fn:
            return writer.write("float8_e4m3fn")
        elif self is DType.float8_e4m3fnuz:
            return writer.write("float8_e4m3fnuz")
        elif self is DType.float8_e5m2:
            return writer.write("float8_e5m2")
        elif self is DType.float8_e5m2fnuz:
            return writer.write("float8_e5m2fnuz")

        elif self is DType.bfloat16:
            return writer.write("bfloat16")
        elif self is DType.float16:
            return writer.write("float16")

        elif self is DType.float32:
            return writer.write("float32")
        elif self is DType.tensor_float32:
            return writer.write("tensor_float32")

        elif self is DType.float64:
            return writer.write("float64")

        elif self is DType.invalid:
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

    @doc_private
    @staticmethod
    @always_inline("nodebug")
    fn _from_ui8(ui8: UInt8._mlir_type) -> DType:
        return __mlir_op.`pop.dtype.from_ui8`(
            __mlir_op.`pop.cast_to_builtin`[_type = __mlir_type.ui8](ui8)
        )

    @doc_private
    @always_inline("nodebug")
    fn _as_ui8(self) -> UInt8._mlir_type:
        return __mlir_op.`pop.cast_from_builtin`[_type = UInt8._mlir_type](
            __mlir_op.`pop.dtype.to_ui8`(self.value)
        )

    @doc_private
    @always_inline("nodebug")
    fn _match(self, mask: UInt8) -> Bool:
        return __mlir_op.`pop.cmp`[pred = __mlir_attr.`#pop<cmp_pred ne>`](
            __mlir_op.`pop.simd.and`(self._as_ui8(), mask.value),
            __mlir_attr.`#pop.simd<0> : !pop.scalar<ui8>`,
        )

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
            self._as_ui8(), rhs._as_ui8()
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
            self._as_ui8(), rhs._as_ui8()
        )

    fn __hash__[H: Hasher](self, mut hasher: H):
        """Updates hasher with this `DType` value.

        Parameters:
            H: The hasher type.

        Args:
            hasher: The hasher instance.
        """
        hasher._update_with_simd(UInt8(self._as_ui8()))

    @always_inline("nodebug")
    fn is_unsigned(self) -> Bool:
        """Returns True if the type parameter is unsigned and False otherwise.

        Returns:
            Returns True if the input type parameter is unsigned.
        """
        return self._is_non_index_integral() and not self._match(_mIsSigned)

    @always_inline("nodebug")
    fn is_signed(self) -> Bool:
        """Returns True if the type parameter is signed and False otherwise.

        Returns:
            Returns True if the input type parameter is signed.
        """
        if self.is_floating_point():
            return True
        return self.is_integral() and self._match(_mIsSigned)

    @always_inline("nodebug")
    fn _is_non_index_integral(self) -> Bool:
        """Returns True if the type parameter is a non-index integer value and False otherwise.

        Returns:
            Returns True if the input type parameter is a non-index integer.
        """
        return self._match(_mIsInteger)

    @always_inline("nodebug")
    fn is_integral(self) -> Bool:
        """Returns True if the type parameter is an integer and False otherwise.

        Returns:
            Returns True if the input type parameter is an integer.
        """
        return self is DType.index or self._is_non_index_integral()

    @always_inline("nodebug")
    fn is_floating_point(self) -> Bool:
        """Returns True if the type parameter is a floating-point and False
        otherwise.

        Returns:
            Returns True if the input type parameter is a floating-point.
        """
        return self._match(_mIsFloat)

    @always_inline("nodebug")
    fn is_float8(self) -> Bool:
        """Returns True if the dtype is a 8bit-precision floating point type,
        e.g. float8_e5m2, float8_e5m2fnuz, float8_e4m3fn and float8_e4m3fnuz.

        Returns:
            True if the dtype is a 8bit-precision float, false otherwise.
        """

        return self in (
            DType.float8_e3m4,
            DType.float8_e4m3fn,
            DType.float8_e4m3fnuz,
            DType.float8_e5m2,
            DType.float8_e5m2fnuz,
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
                                    self._as_ui8(), _mIsNotInteger.value
                                ),
                                UInt8(1).value,
                            ),
                            UInt8(3).value,
                        ),
                    )
                )
            )

        elif self is DType.bool:
            return sizeof[DType.bool]()
        elif self is DType.index:
            return sizeof[DType.index]()

        elif self is DType.float8_e3m4:
            return sizeof[DType.float8_e3m4]()
        elif self is DType.float8_e4m3fn:
            return sizeof[DType.float8_e4m3fn]()
        elif self is DType.float8_e4m3fnuz:
            return sizeof[DType.float8_e4m3fnuz]()
        elif self is DType.float8_e5m2:
            return sizeof[DType.float8_e5m2]()
        elif self is DType.float8_e5m2fnuz:
            return sizeof[DType.float8_e5m2fnuz]()

        elif self is DType.bfloat16:
            return sizeof[DType.bfloat16]()
        elif self is DType.float16:
            return sizeof[DType.float16]()

        elif self is DType.float32:
            return sizeof[DType.float32]()
        elif self is DType.tensor_float32:
            return sizeof[DType.tensor_float32]()

        elif self is DType.float64:
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

        # fmt: off
        alias dtypes = [
            DType.index,
            DType.uint8, DType.int8,
            DType.uint16, DType.int16,
            DType.uint32, DType.int32,
            DType.uint64, DType.int64,
            DType.uint128, DType.int128,
            DType.uint256, DType.int256,
        ]
        # fmt: on

        @parameter
        for dtype in dtypes:
            if self is dtype:
                return func[dtype]()
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
        elif self is DType.tensor_float32:
            func[DType.tensor_float32]()
        elif self is DType.float64:
            func[DType.float64]()
        else:
            raise Error(
                "only floating point types with bitwidth in [16, 32, 64] are"
                " supported"
            )

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

        @parameter
        for dtype in VariadicList(dtypes):
            if self is dtype:
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
    # __mlir_type
    # ===-------------------------------------------------------------------===#

    @always_inline("nodebug")
    fn __mlir_type(self) -> __mlir_type.`!kgen.deferred`:
        """Returns the MLIR type of the current DType as an MLIR type.

        Returns:
            The MLIR type of the current DType.
        """

        if self is DType.bool:
            return __mlir_attr.i1

        if self is DType.index:
            return __mlir_attr.index

        if self is DType.uint8:
            return __mlir_attr.ui8
        if self is DType.int8:
            return __mlir_attr.si8
        if self is DType.uint16:
            return __mlir_attr.ui16
        if self is DType.int16:
            return __mlir_attr.si16
        if self is DType.uint32:
            return __mlir_attr.ui32
        if self is DType.int32:
            return __mlir_attr.si32
        if self is DType.uint64:
            return __mlir_attr.ui64
        if self is DType.int64:
            return __mlir_attr.si64
        if self is DType.uint128:
            return __mlir_attr.ui128
        if self is DType.int128:
            return __mlir_attr.si128
        if self is DType.uint256:
            return __mlir_attr.ui256
        if self is DType.int256:
            return __mlir_attr.si256

        if self is DType.float8_e3m4:
            return __mlir_attr.f8E3M4
        if self is DType.float8_e4m3fn:
            return __mlir_attr.f8E4M3
        if self is DType.float8_e4m3fnuz:
            return __mlir_attr.f8E4M3FNUZ
        if self is DType.float8_e5m2:
            return __mlir_attr.f8E5M2
        if self is DType.float8_e5m2fnuz:
            return __mlir_attr.f8E5M2FNUZ

        if self is DType.bfloat16:
            return __mlir_attr.bf16
        if self is DType.float16:
            return __mlir_attr.f16

        if self is DType.float32:
            return __mlir_attr.f32
        if self is DType.tensor_float32:
            return __mlir_attr.tf32

        if self is DType.float64:
            return __mlir_attr.f64

        return abort[__mlir_type.`!kgen.deferred`]("invalid dtype")

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

        @parameter
        if _type_is_eq[T, SIMD[DType.bool, size]]():
            return DType.bool
        elif _type_is_eq[T, SIMD[DType.index, size]]():
            return DType.index

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
        elif _type_is_eq[T, SIMD[DType.tensor_float32, size]]():
            return DType.tensor_float32

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
        return Self.get_dtype[T]() is not DType.invalid


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
    elif dtype in (DType.float32, DType.tensor_float32):
        return DType.int32
    elif dtype is DType.float64:
        return DType.int64

    return dtype.invalid


# ===-------------------------------------------------------------------===#
# _unsigned_integral_type_of
# ===-------------------------------------------------------------------===#


@always_inline("nodebug")
fn _unsigned_integral_type_of[dtype: DType]() -> DType:
    """Gets the unsigned integral type which has the same bitwidth as
    the input type."""

    @parameter
    if dtype.is_unsigned():
        return dtype
    elif dtype.is_integral():
        return _uint_type_of_width[bitwidthof[dtype]()]()

    elif dtype.is_float8():
        return DType.uint8
    elif dtype.is_half_float():
        return DType.uint16
    elif dtype in (DType.float32, DType.tensor_float32):
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
        return "16"


# ===-------------------------------------------------------------------===#
# _int_type_of_width
# ===-------------------------------------------------------------------===#


@always_inline
fn _int_type_of_width[width: Int]() -> DType:
    constrained[
        width in (8, 16, 32, 64, 128, 256),
        "width must be either 8, 16, 32, 64, 128, or 256",
    ]()

    @parameter
    if width == 8:
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


@always_inline
fn _uint_type_of_width[width: Int]() -> DType:
    constrained[
        width in (8, 16, 32, 64, 128, 256),
        "width must be either 8, 16, 32, 64, 128, or 256",
    ]()

    @parameter
    if width == 8:
        return DType.uint8
    elif width == 16:
        return DType.uint16
    elif width == 32:
        return DType.uint32
    elif width == 64:
        return DType.uint64
    elif width == 128:
        return DType.uint128
    else:
        return DType.uint256


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
    elif dtype is DType.index:
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

    elif dtype.is_floating_point():
        return "%.17g"

    else:
        constrained[False, "invalid dtype"]()

    return ""
