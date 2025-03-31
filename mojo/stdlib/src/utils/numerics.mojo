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
"""Defines utilities to work with numeric types.

You can import these APIs from the `utils` package. For example:

```mojo
from utils.numerics import FPUtils
```
"""

from sys import bitwidthof, has_neon, llvm_intrinsic, CompilationTarget
from sys._assembly import inlined_assembly
from sys.ffi import _external_call_const

from builtin.dtype import _integral_type_of, _uint_type_of
from builtin.simd import _simd_apply
from memory import UnsafePointer, bitcast

# ===----------------------------------------------------------------------=== #
# FPUtils
# ===----------------------------------------------------------------------=== #


fn _constrain_fp_type[dtype: DType]():
    constrained[
        dtype.is_floating_point(), "dtype must be a floating point type"
    ]()


struct FPUtils[
    dtype: DType, *, _constraint: NoneType = _constrain_fp_type[dtype]()
]:
    """Collection of utility functions for working with FP values.

    Constraints:
        The dtype is floating point.

    Parameters:
        dtype: The concrete FP dtype (FP32/FP64/etc).
        _constraint: Implements the constraint. Do not pass explicitly.
    """

    alias integral_type = _integral_type_of[dtype]()
    """The equivalent integer dtype of the float type."""

    alias uint_type = _uint_type_of[dtype]()
    """The equivalent uint dtype of the float type."""

    @staticmethod
    @always_inline("nodebug")
    fn mantissa_width() -> Int:
        """Returns the mantissa width of a floating point type.

        Returns:
            The mantissa width.
        """

        return bitwidthof[dtype]() - Self.exponent_width() - 1

    @staticmethod
    @always_inline("nodebug")
    fn max_exponent() -> Int:
        """Returns the max exponent of a floating point dtype without accounting
        for inf representations. This is not
        the maximum representable exponent, which is generally equal to
        the exponent_bias.

        Returns:
            The max exponent.
        """

        @parameter
        if dtype in (DType.float8_e4m3, DType.float8_e4m3fnuz):
            return 7
        elif dtype is DType.float8_e4m3fn:
            return 8
        elif dtype in (DType.float8_e5m2, DType.float8_e5m2fnuz, DType.float16):
            return 16
        elif dtype in (DType.bfloat16, DType.float32):
            return 128
        else:
            constrained[dtype is DType.float64, "unsupported float type"]()
            return 1024

    @staticmethod
    @always_inline("nodebug")
    fn exponent_width() -> Int:
        """Returns the exponent width of a floating point type.

        Returns:
            The exponent width.
        """

        @parameter
        if dtype in (
            DType.float8_e4m3,
            DType.float8_e4m3fn,
            DType.float8_e4m3fnuz,
        ):
            return 4
        elif dtype in (DType.float8_e5m2, DType.float8_e5m2fnuz, DType.float16):
            return 5
        elif dtype in (DType.float32, DType.bfloat16):
            return 8
        else:
            constrained[dtype is DType.float64, "unsupported float type"]()
            return 11

    @staticmethod
    @always_inline
    fn mantissa_mask() -> Int:
        """Returns the mantissa mask of a floating point type.

        Returns:
            The mantissa mask.
        """
        return (1 << Self.mantissa_width()) - 1

    @staticmethod
    @always_inline
    fn exponent_bias() -> Int:
        """Returns the exponent bias of a floating point type.

        Returns:
            The exponent bias.
        """

        @parameter
        if dtype in (DType.float8_e4m3fnuz, DType.float8_e5m2fnuz):
            return Self.max_exponent()
        else:
            return Self.max_exponent() - 1

    @staticmethod
    @always_inline
    fn sign_mask() -> Int:
        """Returns the sign mask of a floating point type.

        It is computed by `1 << (exponent_width + mantissa_width)`.

        Returns:
            The sign mask.
        """
        # convert to `Int` first to bypass overflow check
        return 1 << Int(Self.exponent_width() + Self.mantissa_width())

    @staticmethod
    @always_inline
    fn exponent_mask() -> Int:
        """Returns the exponent mask of a floating point type.

        It is computed by `~(sign_mask | mantissa_mask)`.

        Returns:
            The exponent mask.
        """
        return ~(Self.sign_mask() | Self.mantissa_mask())

    @staticmethod
    @always_inline
    fn exponent_mantissa_mask() -> Int:
        """Returns the exponent and mantissa mask of a floating point type.

        It is computed by `exponent_mask | mantissa_mask`.

        Returns:
            The exponent and mantissa mask.
        """
        return Self.exponent_mask() | Self.mantissa_mask()

    @staticmethod
    @always_inline
    fn quiet_nan_mask() -> Int:
        """Returns the quiet NaN mask for a floating point type.

        The mask is defined by evaluating:

        ```
        (1<<exponent_width-1)<<mantissa_width + 1<<(mantissa_width-1)
        ```

        Returns:
            The quiet NaN mask.
        """
        alias mantissa_width_val = Self.mantissa_width()
        return (1 << Self.exponent_width() - 1) << mantissa_width_val + (
            1 << (mantissa_width_val - 1)
        )

    @staticmethod
    @always_inline
    fn bitcast_to_integer(value: Scalar[dtype]) -> Int:
        """Bitcasts the floating-point value to an integer.

        Args:
            value: The floating-point type.

        Returns:
            An integer representation of the floating-point value.
        """
        return Int(bitcast[Self.integral_type, 1](value))

    @staticmethod
    @always_inline
    fn bitcast_to_uint(value: Scalar[dtype]) -> Scalar[Self.uint_type]:
        """Bitcasts the floating-point value to an integer.

        Args:
            value: The floating-point type.

        Returns:
            An integer representation of the floating-point value.
        """
        return bitcast[Self.uint_type, 1](value)

    @staticmethod
    @always_inline
    fn bitcast_from_integer(value: Int) -> Scalar[dtype]:
        """Bitcasts the floating-point value from an integer.

        Args:
            value: The int value.

        Returns:
            An floating-point representation of the Int.
        """
        return bitcast[dtype, 1](SIMD[Self.integral_type, 1](value))

    @staticmethod
    @always_inline
    fn get_sign(value: Scalar[dtype]) -> Bool:
        """Returns the sign of the floating point value.

        Args:
            value: The floating-point type.

        Returns:
            Returns True if the sign is set and False otherwise.
        """
        return (Self.bitcast_to_integer(value) & Self.sign_mask()) != 0

    @staticmethod
    @always_inline
    fn set_sign(value: Scalar[dtype], sign: Bool) -> Scalar[dtype]:
        """Sets the sign of the floating point value.

        Args:
            value: The floating-point value.
            sign: True to set the sign and false otherwise.

        Returns:
            Returns the floating point value with the sign set.
        """
        var bits = Self.bitcast_to_integer(value)
        var sign_bits = Self.sign_mask()
        bits &= ~sign_bits
        if sign:
            bits |= sign_bits
        return Self.bitcast_from_integer(bits)

    @staticmethod
    @always_inline
    fn get_exponent(value: Scalar[dtype]) -> Int:
        """Returns the exponent bits of the floating-point value.

        Args:
            value: The floating-point value.

        Returns:
            Returns the exponent bits.
        """
        return (
            Self.bitcast_to_integer(value) & Self.exponent_mask()
        ) >> Self.mantissa_width()

    @staticmethod
    @always_inline
    fn get_exponent_biased(value: Scalar[dtype]) -> Int:
        """Returns the biased exponent of the floating-point value as an Int,
        this is how the value is stored before subtracting the exponent bias.

        Args:
            value: The floating-point value.

        Returns:
            The biased exponent as an Int.
        """
        return Int(
            Self.bitcast_to_uint(value) >> Self.mantissa_width()
            & ((1 << Self.exponent_width()) - 1)
        )

    @staticmethod
    @always_inline
    fn set_exponent(value: Scalar[dtype], exponent: Int) -> Scalar[dtype]:
        """Sets the exponent bits of the floating-point value.

        Args:
            value: The floating-point value.
            exponent: The exponent bits.

        Returns:
            Returns the floating-point value with the exponent bits set.
        """
        var bits = Self.bitcast_to_integer(value)
        bits &= ~Self.exponent_mask()
        bits |= (exponent << Self.mantissa_width()) & Self.exponent_mask()
        return Self.bitcast_from_integer(bits)

    @staticmethod
    @always_inline
    fn get_mantissa(value: Scalar[dtype]) -> Int:
        """Gets the mantissa bits of the floating-point value.

        Args:
            value: The floating-point value.

        Returns:
            The mantissa bits.
        """
        return Self.bitcast_to_integer(value) & Self.mantissa_mask()

    @staticmethod
    @always_inline
    fn get_mantissa_uint(value: Scalar[dtype]) -> Scalar[Self.uint_type]:
        """Gets the mantissa bits of the floating-point value.

        Args:
            value: The floating-point value.

        Returns:
            The mantissa bits.
        """
        return Self.bitcast_to_uint(value) & Self.mantissa_mask()

    @staticmethod
    @always_inline
    fn set_mantissa(value: Scalar[dtype], mantissa: Int) -> Scalar[dtype]:
        """Sets the mantissa bits of the floating-point value.

        Args:
            value: The floating-point value.
            mantissa: The mantissa bits.

        Returns:
            Returns the floating-point value with the mantissa bits set.
        """
        var bits = Self.bitcast_to_integer(value)
        bits &= ~Self.mantissa_mask()
        bits |= mantissa & Self.mantissa_mask()
        return Self.bitcast_from_integer(bits)

    @staticmethod
    @always_inline
    fn pack(sign: Bool, exponent: Int, mantissa: Int) -> Scalar[dtype]:
        """Construct a floating-point value from its constituent sign, exponent,
        and mantissa.

        Args:
            sign: The sign of the floating-point value.
            exponent: The exponent of the floating-point value.
            mantissa: The mantissa of the floating-point value.

        Returns:
            Returns the floating-point value.
        """
        var res: Scalar[dtype] = 0
        res = Self.set_sign(res, sign)
        res = Self.set_exponent(res, exponent)
        res = Self.set_mantissa(res, mantissa)
        return res


# ===----------------------------------------------------------------------=== #
# FlushDenormals
# ===----------------------------------------------------------------------=== #


struct FlushDenormals:
    """Flushes and denormals are set to zero within the context and the state
    is restored to the prior value on exit."""

    var state: Int32
    """The current state."""

    @always_inline
    fn __init__(out self):
        """Initializes the FlushDenormals."""
        self.state = Self._current_state()

    @always_inline
    fn __enter__(self):
        """Enters the context. This will set denormals to zero."""
        self._set_flush(True)

    @always_inline
    fn __exit__(self):
        """Exits the context. This will restore the prior FPState."""
        self._set_flush(False, True)

    @always_inline
    fn _set_flush(self, enable: Bool, force: Bool = False):
        @parameter
        if (
            not CompilationTarget.has_sse4() and not has_neon()
        ):  # not supported, so skip
            return
        # Unless we forced to restore the prior state, we check if the flag
        # has already been enabled to avoid calling the intrinsic which can
        # be costly.
        if not force and enable == self._is_set(self.state):
            return

        # If the enable flag is set then we need to argument the register
        # value, otherwise we are in an exit state and we need to restore
        # the prior value.

        @parameter
        if CompilationTarget.has_sse4():
            var mxcsr = self.state
            if enable:
                mxcsr |= 0x8000  # flush to zero
                mxcsr |= 0x40  # denormals are zero
            llvm_intrinsic["llvm.x86.sse.ldmxcsr", NoneType](
                UnsafePointer[Int32].address_of(mxcsr)
            )
            _ = mxcsr
            return

        alias ARM_FPCR_FZ = Int64(1) << 24
        var fpcr = self.state.cast[DType.int64]()
        if enable:
            fpcr |= ARM_FPCR_FZ

        inlined_assembly[
            "msr fpcr, $0",
            NoneType,
            constraints="r",
            has_side_effect=True,
        ](fpcr)

    @always_inline
    fn _is_set(self, state: Int32) -> Bool:
        @parameter
        if CompilationTarget.has_sse4():
            return (state & 0x8000) != 0 and (state & 0x40) != 0

        alias ARM_FPCR_FZ = Int32(1) << 24
        return (state & ARM_FPCR_FZ) != 0

    @always_inline
    @staticmethod
    fn _current_state() -> Int32:
        """Gets the current denormal state."""

        @parameter
        if (
            not CompilationTarget.has_sse4() and not has_neon()
        ):  # not supported, so skip
            return 0

        @parameter
        if CompilationTarget.has_sse4():
            var mxcsr = Int32()
            llvm_intrinsic["llvm.x86.sse.stmxcsr", NoneType](
                UnsafePointer[Int32].address_of(mxcsr)
            )
            return mxcsr

        var fpcr64 = inlined_assembly[
            "mrs $0, fpcr",
            UInt64,
            constraints="=r",
            has_side_effect=True,
        ]()

        return fpcr64.cast[DType.int32]()


# ===----------------------------------------------------------------------=== #
# nan
# ===----------------------------------------------------------------------=== #


@always_inline("nodebug")
fn nan[dtype: DType]() -> Scalar[dtype]:
    """Gets a NaN value for the given dtype.

    Constraints:
        Can only be used for FP dtypes.

    Parameters:
        dtype: The value dtype.

    Returns:
        The NaN value of the given dtype.
    """

    @parameter
    if dtype is DType.float8_e5m2:
        return rebind[__mlir_type[`!pop.scalar<`, dtype.value, `>`]](
            __mlir_op.`kgen.param.constant`[
                _type = __mlir_type[`!pop.scalar<f8e5m2>`],
                value = __mlir_attr[`#pop.simd<"nan"> : !pop.scalar<f8e5m2>`],
            ]()
        )
    elif dtype is DType.float8_e5m2fnuz:
        return rebind[__mlir_type[`!pop.scalar<`, dtype.value, `>`]](
            __mlir_op.`kgen.param.constant`[
                _type = __mlir_type[`!pop.scalar<f8e5m2fnuz>`],
                value = __mlir_attr[
                    `#pop.simd<"nan"> : !pop.scalar<f8e5m2fnuz>`
                ],
            ]()
        )
    elif dtype is DType.float8_e4m3fn:
        return rebind[__mlir_type[`!pop.scalar<`, dtype.value, `>`]](
            __mlir_op.`kgen.param.constant`[
                _type = __mlir_type[`!pop.scalar<f8e4m3>`],
                value = __mlir_attr[`#pop.simd<"nan"> : !pop.scalar<f8e4m3>`],
            ]()
        )
    elif dtype is DType.float8_e4m3fnuz:
        return rebind[__mlir_type[`!pop.scalar<`, dtype.value, `>`]](
            __mlir_op.`kgen.param.constant`[
                _type = __mlir_type[`!pop.scalar<f8e4m3fnuz>`],
                value = __mlir_attr[
                    `#pop.simd<"nan"> : !pop.scalar<f8e4m3fnuz>`
                ],
            ]()
        )
    elif dtype is DType.float16:
        return rebind[__mlir_type[`!pop.scalar<`, dtype.value, `>`]](
            __mlir_op.`kgen.param.constant`[
                _type = __mlir_type[`!pop.scalar<f16>`],
                value = __mlir_attr[`#pop.simd<"nan"> : !pop.scalar<f16>`],
            ]()
        )
    elif dtype is DType.bfloat16:
        return rebind[__mlir_type[`!pop.scalar<`, dtype.value, `>`]](
            __mlir_op.`kgen.param.constant`[
                _type = __mlir_type[`!pop.scalar<bf16>`],
                value = __mlir_attr[`#pop.simd<"nan"> : !pop.scalar<bf16>`],
            ]()
        )
    elif dtype is DType.float32:
        return rebind[__mlir_type[`!pop.scalar<`, dtype.value, `>`]](
            __mlir_op.`kgen.param.constant`[
                _type = __mlir_type[`!pop.scalar<f32>`],
                value = __mlir_attr[`#pop.simd<"nan"> : !pop.scalar<f32>`],
            ]()
        )
    elif dtype is DType.float64:
        return rebind[__mlir_type[`!pop.scalar<`, dtype.value, `>`]](
            __mlir_op.`kgen.param.constant`[
                _type = __mlir_type[`!pop.scalar<f64>`],
                value = __mlir_attr[`#pop.simd<"nan"> : !pop.scalar<f64>`],
            ]()
        )
    else:
        constrained[False, "nan only support on floating point types"]()
        return 0


# ===----------------------------------------------------------------------=== #
# isnan
# ===----------------------------------------------------------------------=== #


@always_inline("nodebug")
fn isnan[
    dtype: DType, simd_width: Int
](val: SIMD[dtype, simd_width]) -> SIMD[DType.bool, simd_width]:
    """Checks if the value is Not a Number (NaN).

    Parameters:
        dtype: The value dtype.
        simd_width: The width of the SIMD vector.

    Args:
        val: The value to check.

    Returns:
        True if val is NaN and False otherwise.
    """

    @parameter
    if not dtype.is_floating_point() or dtype in (
        DType.float8_e4m3fnuz,
        DType.float8_e5m2fnuz,
    ):
        return False

    alias int_dtype = _integral_type_of[dtype]()

    @parameter
    if dtype is DType.float8_e4m3fn:
        return (bitcast[int_dtype, simd_width](val) & 0x7F) == 0x7F
    elif dtype is DType.float8_e5m2:
        # For the float8_e5m2 dtype NaN is limited to 0x7F and 0xFF values.
        # 7D, 7E, 7F are positive NaNs; FD, FE, FF are negative NaNs.
        return (bitcast[int_dtype, simd_width](val) & 0x7F) > 0x7C
    elif dtype is DType.float16:
        var ival = bitcast[int_dtype, simd_width](val)
        return (ival & 0x7C00) == 0x7C00 and (ival & 0x03FF) != 0
    elif dtype is DType.bfloat16:
        alias x7FFF = SIMD[int_dtype, simd_width](0x7FFF)
        alias x7F80 = SIMD[int_dtype, simd_width](0x7F80)
        return bitcast[int_dtype, simd_width](val) & x7FFF > x7F80

    alias signaling_nan_test: UInt32 = 0x0001
    alias quiet_nan_test: UInt32 = 0x0002
    return llvm_intrinsic[
        "llvm.is.fpclass", SIMD[DType.bool, simd_width], has_side_effect=False
    ](val.value, (signaling_nan_test | quiet_nan_test))


# ===----------------------------------------------------------------------=== #
# inf
# ===----------------------------------------------------------------------=== #


@always_inline("nodebug")
fn inf[dtype: DType]() -> Scalar[dtype]:
    """Gets a +inf value for the given dtype.

    Constraints:
        Can only be used for FP dtypes.

    Parameters:
        dtype: The value dtype.

    Returns:
        The +inf value of the given dtype.
    """

    @parameter
    if dtype is DType.float8_e5m2:
        return rebind[__mlir_type[`!pop.scalar<`, dtype.value, `>`]](
            __mlir_op.`kgen.param.constant`[
                _type = __mlir_type[`!pop.scalar<f8e5m2>`],
                value = __mlir_attr[`#pop.simd<"inf"> : !pop.scalar<f8e5m2>`],
            ]()
        )
    elif dtype is DType.float8_e5m2fnuz:
        return rebind[__mlir_type[`!pop.scalar<`, dtype.value, `>`]](
            __mlir_op.`kgen.param.constant`[
                _type = __mlir_type[`!pop.scalar<f8e5m2fnuz>`],
                value = __mlir_attr[
                    `#pop.simd<"inf"> : !pop.scalar<f8e5m2fnuz>`
                ],
            ]()
        )
    elif dtype is DType.float8_e4m3fn:
        return rebind[__mlir_type[`!pop.scalar<`, dtype.value, `>`]](
            __mlir_op.`kgen.param.constant`[
                _type = __mlir_type[`!pop.scalar<f8e4m3>`],
                value = __mlir_attr[`#pop.simd<"inf"> : !pop.scalar<f8e4m3>`],
            ]()
        )
    elif dtype is DType.float8_e4m3fnuz:
        return rebind[__mlir_type[`!pop.scalar<`, dtype.value, `>`]](
            __mlir_op.`kgen.param.constant`[
                _type = __mlir_type[`!pop.scalar<f8e4m3fnuz>`],
                value = __mlir_attr[
                    `#pop.simd<"inf"> : !pop.scalar<f8e4m3fnuz>`
                ],
            ]()
        )
    elif dtype is DType.float16:
        return rebind[__mlir_type[`!pop.scalar<`, dtype.value, `>`]](
            __mlir_op.`kgen.param.constant`[
                _type = __mlir_type[`!pop.scalar<f16>`],
                value = __mlir_attr[`#pop.simd<"inf"> : !pop.scalar<f16>`],
            ]()
        )
    elif dtype is DType.bfloat16:
        return rebind[__mlir_type[`!pop.scalar<`, dtype.value, `>`]](
            __mlir_op.`kgen.param.constant`[
                _type = __mlir_type[`!pop.scalar<bf16>`],
                value = __mlir_attr[`#pop.simd<"inf"> : !pop.scalar<bf16>`],
            ]()
        )
    elif dtype is DType.float32:
        return rebind[__mlir_type[`!pop.scalar<`, dtype.value, `>`]](
            __mlir_op.`kgen.param.constant`[
                _type = __mlir_type[`!pop.scalar<f32>`],
                value = __mlir_attr[`#pop.simd<"inf"> : !pop.scalar<f32>`],
            ]()
        )
    elif dtype is DType.float64:
        return rebind[__mlir_type[`!pop.scalar<`, dtype.value, `>`]](
            __mlir_op.`kgen.param.constant`[
                _type = __mlir_type[`!pop.scalar<f64>`],
                value = __mlir_attr[`#pop.simd<"inf"> : !pop.scalar<f64>`],
            ]()
        )
    else:
        constrained[False, "+inf only support on floating point types"]()
        return 0


# ===----------------------------------------------------------------------=== #
# neg_inf
# ===----------------------------------------------------------------------=== #


@always_inline("nodebug")
fn neg_inf[dtype: DType]() -> Scalar[dtype]:
    """Gets a -inf value for the given dtype.

    Constraints:
        Can only be used for FP dtypes.

    Parameters:
        dtype: The value dtype.

    Returns:
        The -inf value of the given dtype.
    """

    @parameter
    if dtype is DType.float8_e5m2:
        return rebind[__mlir_type[`!pop.scalar<`, dtype.value, `>`]](
            __mlir_op.`kgen.param.constant`[
                _type = __mlir_type[`!pop.scalar<f8e5m2>`],
                value = __mlir_attr[`#pop.simd<"-inf"> : !pop.scalar<f8e5m2>`],
            ]()
        )
    elif dtype is DType.float8_e5m2fnuz:
        return rebind[__mlir_type[`!pop.scalar<`, dtype.value, `>`]](
            __mlir_op.`kgen.param.constant`[
                _type = __mlir_type[`!pop.scalar<f8e5m2fnuz>`],
                value = __mlir_attr[
                    `#pop.simd<"-inf"> : !pop.scalar<f8e5m2fnuz>`
                ],
            ]()
        )
    elif dtype is DType.float8_e4m3fn:
        return rebind[__mlir_type[`!pop.scalar<`, dtype.value, `>`]](
            __mlir_op.`kgen.param.constant`[
                _type = __mlir_type[`!pop.scalar<f8e4m3>`],
                value = __mlir_attr[`#pop.simd<"-inf"> : !pop.scalar<f8e4m3>`],
            ]()
        )
    elif dtype is DType.float8_e4m3fnuz:
        return rebind[__mlir_type[`!pop.scalar<`, dtype.value, `>`]](
            __mlir_op.`kgen.param.constant`[
                _type = __mlir_type[`!pop.scalar<f8e4m3fnuz>`],
                value = __mlir_attr[
                    `#pop.simd<"-inf"> : !pop.scalar<f8e4m3fnuz>`
                ],
            ]()
        )
    elif dtype is DType.float16:
        return rebind[__mlir_type[`!pop.scalar<`, dtype.value, `>`]](
            __mlir_op.`kgen.param.constant`[
                _type = __mlir_type[`!pop.scalar<f16>`],
                value = __mlir_attr[`#pop.simd<"-inf"> : !pop.scalar<f16>`],
            ]()
        )
    elif dtype is DType.bfloat16:
        return rebind[__mlir_type[`!pop.scalar<`, dtype.value, `>`]](
            __mlir_op.`kgen.param.constant`[
                _type = __mlir_type[`!pop.scalar<bf16>`],
                value = __mlir_attr[`#pop.simd<"-inf"> : !pop.scalar<bf16>`],
            ]()
        )
    elif dtype is DType.float32:
        return rebind[__mlir_type[`!pop.scalar<`, dtype.value, `>`]](
            __mlir_op.`kgen.param.constant`[
                _type = __mlir_type[`!pop.scalar<f32>`],
                value = __mlir_attr[`#pop.simd<"-inf"> : !pop.scalar<f32>`],
            ]()
        )
    elif dtype is DType.float64:
        return rebind[__mlir_type[`!pop.scalar<`, dtype.value, `>`]](
            __mlir_op.`kgen.param.constant`[
                _type = __mlir_type[`!pop.scalar<f64>`],
                value = __mlir_attr[`#pop.simd<"-inf"> : !pop.scalar<f64>`],
            ]()
        )
    else:
        constrained[False, "+inf only support on floating point types"]()
        return 0


# ===----------------------------------------------------------------------=== #
# max_finite
# ===----------------------------------------------------------------------=== #


@always_inline
fn max_finite[dtype: DType]() -> Scalar[dtype]:
    """Returns the maximum finite value of type.

    Parameters:
        dtype: The value dtype.

    Returns:
        The maximum representable value of the type. Does not include infinity
        for floating-point types.
    """

    @parameter
    if dtype is DType.int8:
        return 127
    elif dtype is DType.uint8:
        return 255
    elif dtype is DType.int16:
        return 32767
    elif dtype is DType.uint16:
        return 65535
    elif dtype is DType.int32 or (
        dtype is DType.index and bitwidthof[DType.index]() == 32
    ):
        return 2147483647
    elif dtype is DType.uint32:
        return 4294967295
    elif dtype is DType.int64 or (
        dtype is DType.index and bitwidthof[DType.index]() == 64
    ):
        return 9223372036854775807
    elif dtype is DType.uint64:
        return 18446744073709551615
    elif dtype is DType.float8_e4m3fn:
        return 448
    elif dtype is DType.float8_e4m3fnuz:
        return 240
    elif dtype in (DType.float8_e5m2, DType.float8_e5m2fnuz):
        return 57344
    elif dtype is DType.float16:
        return 65504
    elif dtype is DType.bfloat16:
        return 3.38953139e38
    elif dtype is DType.float32:
        return 3.40282346638528859812e38
    elif dtype is DType.float64:
        return 1.79769313486231570815e308
    elif dtype is DType.bool:
        return rebind[Scalar[dtype]](Scalar(True))
    else:
        constrained[False, "max_finite() called on unsupported type"]()
        return 0


# ===----------------------------------------------------------------------=== #
# min_finite
# ===----------------------------------------------------------------------=== #


@always_inline
fn min_finite[dtype: DType]() -> Scalar[dtype]:
    """Returns the minimum (lowest) finite value of type.

    Parameters:
        dtype: The value dtype.

    Returns:
        The minimum representable value of the type. Does not include negative
        infinity for floating-point types.
    """

    @parameter
    if dtype.is_unsigned():
        return 0
    elif dtype is DType.int8:
        return -128
    elif dtype is DType.int16:
        return -32768
    elif dtype is DType.int32 or (
        dtype is DType.index and bitwidthof[DType.index]() == 32
    ):
        return -2147483648
    elif dtype is DType.int64 or (
        dtype is DType.index and bitwidthof[DType.index]() == 64
    ):
        return -9223372036854775808
    elif dtype.is_floating_point():
        return -max_finite[dtype]()
    elif dtype is DType.bool:
        return rebind[Scalar[dtype]](Scalar(False))
    else:
        constrained[False, "min_finite() called on unsupported type"]()
        return 0


# ===----------------------------------------------------------------------=== #
# max_or_inf
# ===----------------------------------------------------------------------=== #


@always_inline("nodebug")
fn max_or_inf[dtype: DType]() -> Scalar[dtype]:
    """Returns the maximum (potentially infinite) value of type.

    Parameters:
        dtype: The value dtype.

    Returns:
        The maximum representable value of the type. Can include infinity for
        floating-point types.
    """

    @parameter
    if dtype.is_floating_point():
        return inf[dtype]()
    else:
        return max_finite[dtype]()


# ===----------------------------------------------------------------------=== #
# min_or_neg_inf
# ===----------------------------------------------------------------------=== #


@always_inline("nodebug")
fn min_or_neg_inf[dtype: DType]() -> Scalar[dtype]:
    """Returns the minimum (potentially negative infinite) value of type.

    Parameters:
        dtype: The value dtype.

    Returns:
        The minimum representable value of the type. Can include negative
        infinity for floating-point types.
    """

    @parameter
    if dtype.is_floating_point():
        return neg_inf[dtype]()
    else:
        return min_finite[dtype]()


# ===----------------------------------------------------------------------=== #
# isinf
# ===----------------------------------------------------------------------=== #


@always_inline("nodebug")
fn isinf[
    dtype: DType, simd_width: Int
](val: SIMD[dtype, simd_width]) -> SIMD[DType.bool, simd_width]:
    """Checks if the value is infinite.

    This is always False for non-FP data types.

    Parameters:
        dtype: The value dtype.
        simd_width: The width of the SIMD vector.

    Args:
        val: The value to check.

    Returns:
        True if val is infinite and False otherwise.
    """

    @parameter
    if not dtype.is_floating_point() or dtype in (
        DType.float8_e4m3fnuz,
        DType.float8_e5m2fnuz,
    ):
        return False
    elif dtype is DType.float8_e5m2:
        # For the float8_e5m2 both 7C and FC are infinity.
        alias int_dtype = _integral_type_of[dtype]()
        return (bitcast[int_dtype, simd_width](val) & 0x7F) == 0x7C

    alias negative_infinity_test: UInt32 = 0x0004
    alias positive_infinity_test: UInt32 = 0x0200
    return llvm_intrinsic["llvm.is.fpclass", SIMD[DType.bool, simd_width]](
        val.value, (negative_infinity_test | positive_infinity_test)
    )


# ===----------------------------------------------------------------------=== #
# isfinite
# ===----------------------------------------------------------------------=== #


@always_inline("nodebug")
fn isfinite[
    dtype: DType, simd_width: Int
](val: SIMD[dtype, simd_width]) -> SIMD[DType.bool, simd_width]:
    """Checks if the value is not infinite.

    This is always True for non-FP data types.

    Parameters:
        dtype: The value dtype.
        simd_width: The width of the SIMD vector.

    Args:
        val: The value to check.

    Returns:
        True if val is finite and False otherwise.
    """

    @parameter
    if not dtype.is_floating_point():
        return True

    return llvm_intrinsic["llvm.is.fpclass", SIMD[DType.bool, simd_width]](
        val.value, UInt32(0x1F8)
    )


# ===----------------------------------------------------------------------=== #
# get_accum_type
# ===----------------------------------------------------------------------=== #


@always_inline
fn get_accum_type[
    dtype: DType, *, preferred_accum_type: DType = DType.float32
]() -> DType:
    """Returns the recommended dtype for accumulation operations.

    Half precision and float8 types can introduce numerical error if they are
    used in reduction/accumulation operations. This method returns a higher
    precision dtype to use for accumulation if a half precision types is
    provided, otherwise it returns the original dtype.

    The rules are as follows:
        - If the dtype is a float8 type, return a float16 type.
        - If the dtype is a bfloat16 precision type, return a float32 type.
        - If the dtype is a float16 precision type, return a float32 dtype if
            the preferred_accum_type is float32, otherwise return a float16
            type.
        - Otherwise, return the original type.

    Parameters:
        dtype: The dtype of some accumulation operation.
        preferred_accum_type: The preferred dtype for accumulation.

    Returns:
        DType.float32 if dtype is a half-precision float, dtype otherwise.
    """

    @parameter
    if dtype.is_float8():
        if preferred_accum_type is DType.float32:
            return preferred_accum_type
        else:
            return DType.float16
    elif dtype is DType.bfloat16:
        return DType.float32
    elif dtype is DType.float16:
        # fp16 accumulation can be done in fp16 or fp32. Use fp16 by default for better
        # performance and use fp32 only when it's specified via preferred type.
        @parameter
        if preferred_accum_type is DType.float32:
            return preferred_accum_type
        else:
            return DType.float16
    else:
        return dtype


# ===----------------------------------------------------------------------=== #
# nextafter
# ===----------------------------------------------------------------------=== #


fn nextafter[
    dtype: DType, simd_width: Int
](arg0: SIMD[dtype, simd_width], arg1: SIMD[dtype, simd_width]) -> SIMD[
    dtype, simd_width
]:
    """Computes next representable value of `arg0` in the direction of `arg1`.

    Constraints:
        The element dtype of the input must be a floating-point type.

    Parameters:
        dtype: The `dtype` of the input and output SIMD vector.
        simd_width: The width of the input and output SIMD vector.

    Args:
        arg0: The first input argument.
        arg1: The second input argument.

    Returns:
        The `nextafter` of the inputs.
    """

    @always_inline("nodebug")
    @parameter
    fn _float32_dispatch[
        lhs_type: DType, rhs_type: DType, result_type: DType
    ](arg0: SIMD[lhs_type, 1], arg1: SIMD[rhs_type, 1]) -> SIMD[result_type, 1]:
        return _external_call_const["nextafterf", SIMD[result_type, 1]](
            arg0, arg1
        )

    @always_inline("nodebug")
    @parameter
    fn _float64_dispatch[
        lhs_type: DType, rhs_type: DType, result_type: DType
    ](arg0: SIMD[lhs_type, 1], arg1: SIMD[rhs_type, 1]) -> SIMD[result_type, 1]:
        return _external_call_const["nextafter", SIMD[result_type, 1]](
            arg0, arg1
        )

    constrained[
        dtype.is_floating_point(), "input dtype must be floating point"
    ]()

    @parameter
    if dtype is DType.float64:
        return _simd_apply[_float64_dispatch, dtype, simd_width](arg0, arg1)
    return _simd_apply[_float32_dispatch, dtype, simd_width](arg0, arg1)


# ===----------------------------------------------------------------------=== #
# ulp
# ===----------------------------------------------------------------------=== #


@always_inline("nodebug")
fn ulp[
    dtype: DType, simd_width: Int
](x: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
    """Computes the ULP (units of last place) or (units of least precision) of
    the number.

    Constraints:
        The element dtype of the inpiut must be a floating-point type.

    Parameters:
        dtype: The `dtype` of the input and output SIMD vector.
        simd_width: The width of the input and output SIMD vector.

    Args:
        x: SIMD vector input.

    Returns:
        The ULP of x.
    """

    constrained[dtype.is_floating_point(), "the dtype must be floating point"]()

    alias inf_val = SIMD[dtype, simd_width](inf[dtype]())

    var nan_mask = isnan(x)
    var xabs = abs(x)
    var inf_mask = isinf(xabs)
    var x2 = nextafter(xabs, inf_val)
    var x2_inf_mask = isinf(x2)

    return nan_mask.select(
        x,
        inf_mask.select(
            xabs,
            x2_inf_mask.select(xabs - nextafter(xabs, -inf_val), x2 - xabs),
        ),
    )
