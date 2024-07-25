# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from dataclasses import dataclass
from os import PathLike
from typing import Any, Optional, Union

from .graph_value import GraphValue
from .type import TensorType


@dataclass
class Weight(GraphValue):
    """Represents a value in a Graph that can be loaded at a later time."""

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
