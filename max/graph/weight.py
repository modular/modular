# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from dataclasses import dataclass
from os import PathLike
from typing import Optional, Union

from .value import TensorValue
from .quantization import QuantizationEncoding


@dataclass
class Weight:
    """Represents a value in a Graph that can be loaded at a later time."""

    value: TensorValue
    name: str

    filepath: Union[PathLike, str, None]
    offset: Optional[int]

    quantization_encoding: Optional[QuantizationEncoding] = None

    def assign(
        self,
        filepath: Union[PathLike, str, None] = None,
        offset: Optional[int] = None,
    ):
        # TODO: Update weight registry
        self.filepath = filepath
        self.offset = offset
