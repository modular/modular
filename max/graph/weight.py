# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from dataclasses import dataclass
from os import PathLike
from typing import Optional, Union

from .graph_value import GraphValue
from .type import TensorType


@dataclass
class Weight:
    """Represents a value in a Graph that can be loaded at a later time."""

    value: GraphValue
    name: str
    tensor_type: TensorType

    filepath: Union[PathLike, str, None]
    offset: Optional[int]

    def assign(
        self,
        filepath: Union[PathLike, str, None] = None,
        offset: Optional[int] = None,
    ):
        # TODO: Update weight registry
        self.filepath = filepath
        self.offset = offset
