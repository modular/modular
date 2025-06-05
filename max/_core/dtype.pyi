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

    int8 = 135

    int16 = 137

    int32 = 139

    int64 = 141

    uint8 = 134

    uint16 = 136

    uint32 = 138

    uint64 = 140

    float16 = 70

    float32 = 72

    float64 = 73

    bfloat16 = 71

    float8_e4m3fn = 66

    float8_e4m3fnuz = 67

    float8_e5m2 = 68

    float8_e5m2fnuz = 69

    _unknown = 0

    @property
    def align(self) -> int:
        """Returns the alignment of the dtype."""

    @property
    def size_in_bytes(self) -> int:
        """Returns the size of the dtype in bytes."""

    def is_integral(self) -> __builtins__.bool:
        """Returns true if the dtype is an integer."""

    def is_unsigned_integral(self) -> __builtins__.bool:
        """Returns true if the dtype is an unsigned integer."""

    def is_signed_integral(self) -> __builtins__.bool:
        """Returns true if the dtype is a signed integer."""

    def is_float(self) -> __builtins__.bool:
        """Returns true if the dtype is floating point."""

    def is_float8(self) -> __builtins__.bool:
        """Returns true if the dtype is any variant of float8."""

    def is_half(self) -> __builtins__.bool:
        """Returns true if the dtype is half-precision floating point."""

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
