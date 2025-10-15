# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# GENERATED FILE, DO NOT EDIT MANUALLY!
# ===----------------------------------------------------------------------=== #

import enum

import numpy

class DType(enum.Enum):
    """The tensor data type."""

    bool = 1
    """Boolean data type. Stores ``True`` or ``False`` values."""

    int8 = 135
    """8-bit signed integer. Range: -128 to 127."""

    int16 = 137
    """16-bit signed integer. Range: -32,768 to 32,767."""

    int32 = 139
    """32-bit signed integer. Range: -2,147,483,648 to 2,147,483,647."""

    int64 = 141
    """
    64-bit signed integer. Range: -9,223,372,036,854,775,808 to 9,223,372,036,854,775,807.
    """

    uint8 = 134
    """8-bit unsigned integer. Range: 0 to 255."""

    uint16 = 136
    """16-bit unsigned integer. Range: 0 to 65,535."""

    uint32 = 138
    """32-bit unsigned integer. Range: 0 to 4,294,967,295."""

    uint64 = 140
    """64-bit unsigned integer. Range: 0 to 18,446,744,073,709,551,615."""

    float16 = 70
    """
    16-bit IEEE 754 half-precision floating-point. 1 sign bit, 5 exponent bits, 10 mantissa bits.
    """

    float32 = 72
    """
    32-bit IEEE 754 single-precision floating-point. 1 sign bit, 8 exponent bits, 23 mantissa bits.
    """

    float64 = 73
    """
    64-bit IEEE 754 double-precision floating-point. 1 sign bit, 11 exponent bits, 52 mantissa bits.
    """

    bfloat16 = 71
    """
    16-bit bfloat16 (Brain Float) format. 1 sign bit, 8 exponent bits, 7 mantissa bits.
    """

    float8_e4m3fn = 66
    """
    8-bit floating-point with 4 exponent bits and 3 mantissa bits, finite values only.
    """

    float8_e4m3fnuz = 67
    """
    8-bit floating-point with 4 exponent bits and 3 mantissa bits, finite values only, no negative zero.
    """

    float8_e5m2 = 68
    """8-bit floating-point with 5 exponent bits and 2 mantissa bits."""

    float8_e5m2fnuz = 69
    """
    8-bit floating-point with 5 exponent bits and 2 mantissa bits, finite values only, no negative zero.
    """

    @property
    def align(self) -> int:
        """
        Returns the alignment requirement of the data type in bytes.

        The alignment specifies the memory boundary that values of this data type
        must be aligned to for optimal performance and correctness.
        """

    @property
    def size_in_bytes(self) -> int:
        """
        Returns the size of the data type in bytes.

        This indicates how many bytes are required to store a single value
        of this data type in memory.
        """

    def is_integral(self) -> __builtins__.bool:
        """Checks if the data type is an integer type."""

    def is_unsigned_integral(self) -> __builtins__.bool:
        """Checks if the data type is an unsigned integer type."""

    def is_signed_integral(self) -> __builtins__.bool:
        """Checks if the data type is a signed integer type."""

    def is_float(self) -> __builtins__.bool:
        """Checks if the data type is a floating-point type."""

    def is_float8(self) -> __builtins__.bool:
        """Checks if the data type is an 8-bit floating-point type."""

    def is_half(self) -> __builtins__.bool:
        """Checks if the data type is a half-precision floating-point type."""

    def to_numpy(self) -> numpy.dtype:
        """
        Converts this ``DType`` to the corresponding NumPy dtype.

        Returns:
            DType: The corresponding NumPy dtype object.

        Raises:
            ValueError: If the dtype is not supported.
        """

    @classmethod
    def from_numpy(cls, dtype: numpy.dtype) -> DType:
        """
        Converts a NumPy dtype to the corresponding DType.

        Args:
            dtype (np.dtype): The NumPy dtype to convert.

        Returns:
            DType: The corresponding DType enum value.

        Raises:
            ValueError: If the input dtype is not supported.
        """

    def to_torch(self):
        """
        Converts this ``DType`` to the corresponding torch dtype.

        Returns:
            DType: The corresponding torch dtype object.

        Raises:
            ValueError: If the dtype is not supported.
            ImportError: If `torch` isn't installed.
        """

    @staticmethod
    def from_torch(tensor) -> DType:
        """
        Converts a torch dtype to the corresponding DType.

        Args:
            dtype (np.dtype): The torch dtype to convert.

        Returns:
            DType: The corresponding DType enum value.

        Raises:
            ValueError: If the input dtype is not supported.
            ImportError: If `torch` isn't installed.
        """

    @property
    def _mlir(self) -> str: ...
