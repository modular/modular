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
from typing import TYPE_CHECKING, Any, NoReturn

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
from max.nn.graph_groups import HiddenStates
from max.nn.transformer import DecoderOutputs
from max.pipelines.context import BaseContextType
from max.tree import Tree

from ..interfaces import PipelineModel
from ..pipeline_stage import PipelineStage

if TYPE_CHECKING:
    from ..config import PipelineConfig


class StageTraced(Exception):
    """Ends a traced stage's build once it knows what it hands the next."""

    def __init__(self, boundary_types: Tree[TensorType]) -> None:
        super().__init__("A traced pipeline stage ends before it compiles.")
        self.boundary_types = boundary_types
        """The types of what the stage's decoder returned, in the tree it
        returned them. The next stage's
        :attr:`~max.pipelines.lib.pipeline_stage.PipelineStage.boundary_types`."""


def _crossing_type(value: TensorValue) -> TensorType:
    return value.type


def _boundary_type(crossing: Type[Any]) -> TensorType:
    if not isinstance(crossing, TensorType):
        raise ValueError(
            f"Only tensors can cross a pipeline cut, got {crossing}."
        )
    if odd := [
        d for d in crossing.shape if not isinstance(d, (StaticDim, SymbolicDim))
    ]:
        raise ValueError(
            f"A tensor crossing a pipeline cut has algebraic dim {odd[0]}."
            " Only static or symbolic dims can cross a cut."
        )
    return crossing


def raise_stage_traced(boundary_types: Tree[Type[Any]]) -> NoReturn:
    """Ends a stage that only traces, with what it hands the next stage.

    :func:`stage_outputs` calls this for a model that builds through
    :func:`stage_graph`. A model whose graph is traced for it calls it on a
    stage that only traces, with the types of what its decoder returned.

    Args:
        boundary_types: The types of what the stage's decoder returned.

    Raises:
        StageTraced: Always, carrying ``boundary_types``.
        ValueError: If something crossing the cut is not a tensor, or has an
            algebraic dim.
    """
    raise StageTraced(tree.map(_boundary_type, boundary_types))


def _boundary_types(stage: PipelineStage | None) -> Tree[TensorType] | None:
    if stage is None or stage.is_first:
        return None
    if stage.boundary_types is None:
        raise ValueError(
            f"Pipeline stage {stage.index} needs the types of what the"
            " previous stage hands it."
        )
    return stage.boundary_types


def stage_input_types(
    stage: PipelineStage | None, input_types: Iterable[Any]
) -> tuple[Any, ...]:
    """Returns the input types a stage traces on, from the model's.

    Every stage takes ``input_types``. A stage after the first then takes
    one more input, what the previous stage's decoder returned, in the tree
    it returned it, so its leaves are the graph's last inputs. A model
    whose graph is traced from types passes its decoder the graph inputs
    after its own, and on a stage that only traces calls
    :func:`raise_stage_traced`.

    Args:
        stage: The stage to build, or ``None`` for the whole model.
        input_types: The model's input types, the same on every stage.

    Raises:
        ValueError: If a stage after the first has no boundary types.
    """
    boundary_types = _boundary_types(stage)
    if boundary_types is None:
        return tuple(input_types)
    return (*input_types, boundary_types)


def _boundary_inputs(
    stage: PipelineStage | None, inputs: Iterable[Value[Any]]
) -> Any:
    boundary_types = _boundary_types(stage)
    if boundary_types is None:
        return None
    treedef = tree.flatten(boundary_types)[1]
    return tree.unflatten(treedef, [v.tensor for v in inputs])


@contextmanager
def stage_graph(
    stage: PipelineStage | None,
    name: str,
    input_types: Iterable[Type[Any]],
    **graph_kwargs: Any,
) -> Iterator[tuple[Graph, Sequence[Value[Any]], Any]]:
    """Builds a model's graph, or one pipeline stage's part of it.

    Yields the graph, the values of ``input_types`` and the boundary inputs.
    A stage after the first takes what the previous stage's decoder returned
    as its last graph inputs, typed by its
    :attr:`~max.pipelines.lib.pipeline_stage.PipelineStage.boundary_types`.
    The boundary inputs are those values in the tree the decoder returned
    them, or ``None`` on the first stage. The model calls its decoder with the
    boundary inputs and outputs what :func:`stage_outputs` returns. Without a
    stage this is the plain graph.

    Args:
        stage: The stage to build, or ``None`` for the whole model.
        name: The graph's name.
        input_types: The model's graph input types, the same on every stage.
        graph_kwargs: Passed to :class:`~max.graph.Graph`.

    Raises:
        ValueError: If a stage after the first has no boundary types.
    """
    types = list(input_types)
    with Graph(
        name,
        input_types=tree.leaves(stage_input_types(stage, types)),
        **graph_kwargs,
    ) as graph:
        n = len(types)
        yield (
            graph,
            graph.inputs[:n],
            _boundary_inputs(stage, graph.inputs[n:]),
        )


def stage_outputs(stage: PipelineStage | None, outputs: Any) -> Any:
    """Returns what a stage graph outputs, from what its decoder returned.

    On the last stage, and without a stage, that is ``outputs``. A stage
    before the last outputs the leaves of what its decoder returned, as its
    ``hidden_states.stage_boundary``.

    Args:
        stage: The stage being built, or ``None`` for the whole model.
        outputs: What the decoder returned.

    Raises:
        StageTraced: On a stage that only traces, carrying the types of
            ``outputs``.
        ValueError: If a tensor crossing the cut has an algebraic dim.
    """
    if stage is None or stage.is_last:
        return outputs
    if stage.trace_only:
        raise_stage_traced(tree.map(_crossing_type, outputs))
    return DecoderOutputs(
        hidden_states=HiddenStates(stage_boundary=tuple(tree.leaves(outputs)))
    )


class PipelinedModel(PipelineModel[BaseContextType]):
    """Mixin for a pipeline model whose decoder can run as pipeline stages.

    Its graph builds through :func:`stage_graph`, or traces or compiles on
    :func:`stage_input_types`, and outputs what :func:`stage_outputs`
    returns, for the stage its config carries. Every
    stage's graph takes the whole model's graph inputs, so the engine that
    serves the stages prepares each stage's inputs as it would the whole
    model's. A stage after the first takes what the previous stage's
    decoder returned after them.
    """

    @classmethod
    def stage_speculation_error(
        cls, pipeline_config: PipelineConfig
    ) -> str | None:
        """Returns why this model can't build a speculating last stage.

        Returns ``None`` if it can under ``pipeline_config``. A model that
        speculates only inherits a stage's layers from its target, so it
        refuses unless it overrides this. An exporter asks before anything
        loads.
        """
        del pipeline_config
        return f"{cls.__name__} does not speculate on a pipeline stage"
