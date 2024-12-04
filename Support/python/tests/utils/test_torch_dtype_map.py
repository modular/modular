# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import pytest
import torch
from max.dtype import DType
from modular.utils.misc import (
    modular_dtype_to_str,
    modular_to_torch_type,
    modular_type_of,
    torch_dtype_to_str,
    torch_to_modular_type,
)


def test_modular_to_torch_type() -> None:
    assert modular_to_torch_type(DType.int32) == torch.int32
    assert modular_to_torch_type(DType.uint8) == torch.uint8
    assert modular_to_torch_type(DType.bfloat16) == torch.bfloat16


def test_torch_to_modular_type() -> None:
    assert torch_to_modular_type(torch.int16) == DType.int16
    assert torch_to_modular_type(torch.float32) == DType.float32
    assert torch_to_modular_type(torch.bfloat16) == DType.bfloat16


def test_modular_type_of() -> None:
    assert modular_type_of(123) == DType.int64
    assert modular_type_of(4.56) == DType.float64
    assert (
        modular_type_of(torch.tensor(789, dtype=torch.bfloat16))
        == DType.bfloat16
    )
    with pytest.raises(ValueError):
        modular_type_of([1])


def test_modular_dtype_to_str() -> None:
    assert modular_dtype_to_str(DType.int32) == "si32"
    assert modular_dtype_to_str(DType.uint8) == "ui8"
    assert modular_dtype_to_str(DType.bfloat16) == "bf16"


def test_torch_dtype_to_str() -> None:
    assert torch_dtype_to_str(torch.int32) == "si32"
    assert torch_dtype_to_str(torch.uint8) == "ui8"
    assert torch_dtype_to_str(torch.bfloat16) == "bf16"
    with pytest.raises(KeyError):
        torch_dtype_to_str(torch.complex64)
    assert torch_dtype_to_str(torch.complex64, "complex") == "complex"
