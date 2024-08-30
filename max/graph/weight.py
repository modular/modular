# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
import typing
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Optional, Union

from max import _graph
from max.dtype import DType
from max.mlir.dialects import mo

from .quantization import QuantizationEncoding
from .type import ShapeLike, TensorType

if typing.TYPE_CHECKING:
    from .graph import Graph
    from .value import Value


@dataclass
class Weight:
    """Represents a value in a Graph that can be loaded at a later time."""

    name: str
    dtype: DType
    shape: ShapeLike

    filepath: Union[PathLike, str]
    offset: int = 0
    quantization_encoding: Optional[QuantizationEncoding] = None

    def add_to_graph(self, graph: "Graph") -> "Value":
        """Adds weight to a Graph.

        Args:
            graph: The graph in which to add this weight to.

        Returns:
            `GraphValue` that contains this weight.

        Raises:
            ValueError if a weight with the same name already exists in the
            graph.
        """
        if graph_weight := graph.weights.get(self.name):
            if graph_weight.weight is self:
                return graph_weight.value
            else:
                raise ValueError(
                    f"Weight '{self.name}' already exists in Graph {graph}"
                )

        tensor_type = TensorType(self.dtype, self.shape).to_mlir()
        weights_attr = _graph.weights_attr(
            Path(self.filepath),
            self.offset,
            tensor_type,
            self.name,
        )
        weight_tensor = graph._add_op(
            mo.constant, result=tensor_type, value=weights_attr
        )[0]
        graph.weights[self.name] = GraphWeight(self, weight_tensor)
        return weight_tensor


@dataclass(frozen=True)
class GraphWeight:
    weight: Weight
    value: "Value"
