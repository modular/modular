# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

from max.dtype import DType

from .quantization import QuantizationEncoding
from .type import ShapeLike


@dataclass
class Weight:
    """Represents a value in a Graph that can be loaded at a later time."""

    name: str
    dtype: DType
    shape: ShapeLike
    quantization_encoding: Optional[QuantizationEncoding] = None
    align: Optional[int] = None

    filepath: Optional[Path] = None
    offset: int = 0
