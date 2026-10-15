# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""What a pipeline model implements to build its decoder as pipeline stages."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

from max import tree
from max.graph import (
    Graph,
    StaticDim,
    SymbolicDim,
    TensorType,
    TensorValue,
    Type,
    Value,
)
from max.pipelines.context import BaseContextType

from ..interfaces import PipelineModel


class StageTraced(Exception):
    """Ends a traced stage's build once it knows what it hands the next."""

    def __init__(self, boundary_types: object) -> None:
        super().__init__("A traced pipeline stage ends before it compiles.")
        self.boundary_types = boundary_types
        """The types of what the stage's decoder returned, in the tree it
        returned them. The next stage's
        :attr:`~max.pipelines.lib.pipeline_stage.PipelineStage.boundary_types`."""


def _crossing_type(value: TensorValue) -> TensorType:
    if odd := [
        d
        for d in value.type.shape
        if not isinstance(d, (StaticDim, SymbolicDim))
    ]:
        raise ValueError(
            f"A tensor crossing a pipeline cut has algebraic dim {odd[0]}."
            " Only static or symbolic dims can cross a cut."
        )
    return value.type


class PipelinedModel(PipelineModel[BaseContextType]):
    """Mixin for a pipeline model whose decoder can run as pipeline stages.

    Every stage's graph takes the whole model's graph inputs, so the engine
    that serves the stages prepares each stage's inputs as it would the
    whole model's. A stage after the first takes what the previous stage's
    decoder returned after them.
    """

    @contextmanager
    def stage_graph(
        self, name: str, input_types: Iterable[Type[Any]], **graph_kwargs: Any
    ) -> Iterator[tuple[Graph, Sequence[Value[Any]], Any]]:
        """Builds this model's graph, or its stage's part of it.

        Yields the graph, the values of ``input_types`` and the boundary
        inputs. A stage after the first takes what the previous stage's
        decoder returned as its last graph inputs, typed by its stage's
        :attr:`~max.pipelines.lib.pipeline_stage.PipelineStage.boundary_types`. The boundary
        inputs are those values in the tree the decoder returned them, or
        ``None`` on the first stage. The model calls its decoder with the
        boundary inputs and outputs what :meth:`stage_outputs` returns.
        Without a pipeline stage this is the plain graph.

        Args:
            name: The graph's name.
            input_types: The model's graph input types, the same on every
                stage.
            graph_kwargs: Passed to :class:`~max.graph.Graph`.

        Raises:
            ValueError: If a stage after the first has no boundary types.
        """
        stage = self.pipeline_config.model.pipeline_stage
        boundary: list[TensorType] = []
        treedef = None
        if stage is not None and not stage.is_first:
            if stage.boundary_types is None:
                raise ValueError(
                    f"Pipeline stage {stage.index} needs the types of what"
                    " the previous stage hands it."
                )
            boundary, treedef = tree.flatten(stage.boundary_types)
        # ModuleV2: the model unpacks the graph's flat inputs itself, where
        # ModuleV3 compiles the decoder on the boundary and input types.
        types = list(input_types)
        with Graph(
            name, input_types=[*types, *boundary], **graph_kwargs
        ) as graph:
            n = len(types)
            boundary_inputs = None
            if treedef is not None:
                received = [v.tensor for v in graph.inputs[n:]]
                boundary_inputs = tree.unflatten(treedef, received)
            yield graph, graph.inputs[:n], boundary_inputs

    def stage_outputs(self, outputs: Any) -> Sequence[Value[Any]]:
        """Returns what a stage graph outputs, from what its decoder returned.

        On the last stage, and without a pipeline stage, that is ``outputs``.
        A stage before the last outputs the leaves of what its decoder
        returned.

        Args:
            outputs: What the decoder returned.

        Raises:
            StageTraced: On a stage that only traces, carrying the types of
                ``outputs``.
            ValueError: If a tensor crossing the cut has an algebraic dim.
        """
        stage = self.pipeline_config.model.pipeline_stage
        if stage is None or stage.is_last:
            return outputs
        boundary_types = tree.map(_crossing_type, outputs)
        if stage.trace_only:
            raise StageTraced(boundary_types)
        return tree.flatten(outputs)[0]
