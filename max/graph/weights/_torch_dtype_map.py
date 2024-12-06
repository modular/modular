# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Utility methods for mapping torch.dtype and max.dtypes.

This file is private because it will be moved elsewhere.
"""

from __future__ import annotations

from typing import Any

# Load these methods only if torch and max dtype are available.
try:
    import torch  # type: ignore
    from max.dtype import DType
    from torch._subclasses.fake_tensor import FakeTensor  # type: ignore

    _DATA_TYPES_LIST: list[tuple[DType, torch.dtype, str]] = [
        (DType.bool, torch.bool, "bool"),
        (DType.int8, torch.int8, "si8"),
        (DType.int16, torch.int16, "si16"),
        (DType.int32, torch.int32, "si32"),
        (DType.int64, torch.int64, "si64"),
        (DType.uint8, torch.uint8, "ui8"),
        (DType.float16, torch.float16, "f16"),
        (DType.float32, torch.float32, "f32"),
        (DType.float64, torch.float64, "f64"),
        (DType.bfloat16, torch.bfloat16, "bf16"),
        (DType.f8e4m3, torch.float8_e4m3fn, "f8e4m3"),
        (DType.f8e5m2, torch.float8_e5m2, "f8e5m2"),
    ]

    _modular_to_torch_type_dict = {k: v for k, v, _ in _DATA_TYPES_LIST}
    _torch_to_modular_type_dict = {k: v for v, k, _ in _DATA_TYPES_LIST}
    _modular_dtype_to_str = {k: name for k, _, name in _DATA_TYPES_LIST}
    _torch_dtype_to_str = {k: name for _, k, name in _DATA_TYPES_LIST}

    def modular_to_torch_type(dtype: DType) -> torch.dtype:
        """
        Converts a Modular dtype.DType to a torch.dtype.
        Args:
            dtype (dtype.DType): Modular dtype.DType.
        Returns:
            torch.dtype: The corresponding torch.dtype.
        """
        return _modular_to_torch_type_dict[dtype]

    def torch_to_modular_type(dtype: torch.dtype) -> DType:
        """
        Converts a torch.dtype to a Modular dtype.DType.
        Args:
            dtype (torch.dtype): PyTorch torch.dtype.
        Returns:
            dtype.DType: The corresponding dtype.DType.
        """
        return _torch_to_modular_type_dict[dtype]

    def modular_type_of(object: Any) -> DType:
        """
        Attempts to convert any convertible Python type to a Modular dtype.DType.
        Args:
            object (Any): Any python type
                (currently supported: torch.Tensor, float, int, bool).
        Returns:
            dtype.DType: The corresponding dtype.DType.
        """
        if isinstance(object, torch.Tensor) or isinstance(object, FakeTensor):
            return torch_to_modular_type(object.dtype)
        if isinstance(object, float):
            return DType.float64
        if isinstance(object, int):
            return DType.int64
        if isinstance(object, bool):
            return DType.bool
        raise ValueError(f"{type(object)} currently not supported")

    def modular_dtype_to_str(dtype: DType, default: str | None = None) -> str:
        if default is None:
            return _modular_dtype_to_str[dtype]
        else:
            return _modular_dtype_to_str.get(dtype, default)

    def torch_dtype_to_str(
        dtype: torch.dtype, default: str | None = None
    ) -> str:
        if default is None:
            return _torch_dtype_to_str[dtype]
        else:
            return _torch_dtype_to_str.get(dtype, default)

except ImportError as e:
    error = e

    def modular_to_torch_type(dtype: DType) -> torch.dtype:
        raise error

    def torch_to_modular_type(dtype: torch.dtype) -> DType:
        raise error

    def modular_type_of(object: Any) -> DType:
        raise error

    def modular_dtype_to_str(dtype: DType, default: str | None = None) -> str:
        raise error

    def torch_dtype_to_str(
        dtype: torch.dtype, default: str | None = None
    ) -> str:
        raise error
