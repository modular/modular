# DOC: max/develop/dtypes.mdx

from max.dtype import DType


def validate_weights_dtype(dtype: DType) -> None:
    """Ensure weights use a floating-point type."""
    # API: dtype.is_float()
    #   Returns: True if dtype is any floating-point type
    if not dtype.is_float():
        raise TypeError(f"Weights must be float type, got {dtype}")


def validate_indices_dtype(dtype: DType) -> None:
    """Ensure indices use an integer type."""
    # API: dtype.is_integral()
    #   Returns: True if dtype is any integer type (signed or unsigned)
    if not dtype.is_integral():
        raise TypeError(f"Indices must be integer type, got {dtype}")


# Usage
weights_dtype = DType.float16
indices_dtype = DType.int32

validate_weights_dtype(weights_dtype)  # OK
validate_indices_dtype(indices_dtype)  # OK
