# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from dataclasses import dataclass
from os import PathLike
from typing import Optional, Union

from max.dtype import DType

from .quantization import QuantizationEncoding
from .type import ShapeLike


@dataclass
class Weight:
    """Represents a value in a Graph that can be loaded at a later time."""

    name: str
    dtype: DType
    shape: ShapeLike

    filepath: Union[PathLike, str]
    offset: int = 0
    quantization_encoding: Optional[QuantizationEncoding] = None
