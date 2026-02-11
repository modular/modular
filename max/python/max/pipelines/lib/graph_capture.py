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
"""Provides device graph capture and replay functionality for model execution.

Device graph capture allows the runtime to record a graph of operations executed
on a device, then replay it efficiently without re-tracing. This is particularly
useful for models with static control flow and fixed input shapes, as it
eliminates Python overhead on repeated executions.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from max.driver import Buffer
from max.engine import Model

if TYPE_CHECKING:
    from .interfaces import ModelInputs

logger = logging.getLogger("max.pipelines")


@dataclass
class _DeviceGraphState:
    """Stores the state of a captured device graph.

    This is an internal dataclass that associates the input buffers used during
    graph capture with the output buffers produced by the captured graph.
    """

    buffers: list[Buffer]
    """The list of input buffers used to capture the graph."""

    outputs: tuple[Buffer, ...]
    """The output buffers produced by the captured graph."""


class InputKey:
    """Represents a unique key for a set of input buffers based on their shapes and dtypes.

    This class creates a hashable key from a sequence of :class:`Buffer` objects
    by extracting their data types and shapes. It enables efficient lookup of
    captured graphs that match a particular input signature.
    """

    def __init__(self, *inputs: Buffer):
        """Initializes an InputKey from input buffers.

        Args:
            *inputs: Variable number of :class:`Buffer` objects whose dtypes
                and shapes will form the key.
        """
        self.keys = tuple((input.dtype, tuple(input.shape)) for input in inputs)

    def __eq__(self, other: Any):
        """Checks equality with another InputKey.

        Args:
            other: The object to compare against.

        Returns:
            ``True`` if ``other`` is an :class:`InputKey` with the same dtypes
            and shapes, ``False`` otherwise.
        """
        if not isinstance(other, InputKey):
            return False
        return self.keys == other.keys

    def __hash__(self) -> int:
        """Returns the hash of this InputKey.

        Returns:
            An integer hash value based on the dtypes and shapes of the input
            buffers.
        """
        return hash(self.keys)


class DeviceGraphExecutor:
    """Manages device graph capture and replay for efficient model execution.

    This class captures device graphs for models with static control flow and
    fixed input shapes, then replays them on subsequent executions to avoid
    re-tracing overhead. Each unique input signature (dtype and shape
    combination) results in a separate captured graph.

    Args:
        trace_fn: A callable that takes :class:`ModelInputs` and returns a list
            of :class:`Buffer` objects representing the traced inputs for graph
            capture.
    """

    def __init__(self, trace_fn: Callable[[ModelInputs], list[Buffer]]):
        self._trace_fn = trace_fn
        self._graph_states: dict[InputKey, _DeviceGraphState] = {}

    def trace(self, model_inputs: ModelInputs) -> tuple[InputKey, list[Buffer]]:
        """Traces model inputs to produce graph inputs and their corresponding key.

        Args:
            model_inputs: The inputs to the model that will be traced.

        Returns:
            A tuple containing an :class:`InputKey` based on the traced buffers
            and the list of traced :class:`Buffer` objects.
        """
        graph_inputs = self._trace_fn(model_inputs)
        return InputKey(*graph_inputs), graph_inputs

    def capture(
        self,
        model: Model,
        model_inputs: list[ModelInputs],
    ) -> None:
        """Captures device graphs for the given model inputs.

        For each unique input signature in ``model_inputs``, this method captures
        a device graph by executing the model and recording the operations. The
        captured graphs are stored internally for later replay. Inputs whose
        signature has already been captured are skipped.

        Args:
            model: The model to capture. Must have a ``capture()`` method.
            model_inputs: A list of :class:`ModelInputs` to capture graphs for.
                Each unique input signature will result in a separate captured
                graph.

        Raises:
            Exception: If graph capture fails for any input. The original
                exception is re-raised after logging.
        """
        for inputs in model_inputs:
            key, graph_inputs = self.trace(inputs)

            if key in self._graph_states:
                continue

            try:
                outputs = model.capture(*graph_inputs)
            except Exception:
                logger.exception("Device graph capture failed for replica.")
                raise

            self._graph_states[key] = _DeviceGraphState(
                buffers=graph_inputs,
                outputs=tuple(outputs),
            )

    def replay(
        self,
        model: Model,
        model_inputs: ModelInputs,
    ) -> tuple[Buffer, ...]:
        """Replays a previously captured device graph.

        Copies the new input data into the captured buffers and replays the
        graph to produce outputs. A graph must have been previously captured
        for the input signature of ``model_inputs``.

        Args:
            model: The model to replay. Must have a ``replay()`` method.
            model_inputs: The inputs for which to replay a captured graph.

        Returns:
            The output buffers from the replayed graph.

        Raises:
            RuntimeError: If no captured graph exists for the input signature.
            ValueError: If input buffers cannot be copied into the captured
                buffers due to shape, dtype, or device mismatch.
        """
        key, graph_inputs = self.trace(model_inputs)

        state = self._graph_states.get(key)

        if state is None:
            raise RuntimeError(
                "No captured device graph found for the given input signature."
            )

        self._copy_graph_inputs(graph_inputs, state.buffers)

        try:
            model.replay(*state.buffers)
        except Exception:
            logger.exception("Device graph replay failed for replica.")
            raise

        return state.outputs

    def _copy_graph_inputs(
        self, src: Sequence[Buffer], dst: Sequence[Buffer]
    ) -> None:
        """Copies source buffers into destination buffers for graph replay.

        Args:
            src: The sequence of source buffers containing new input data.
            dst: The sequence of destination buffers (from the captured graph)
                to copy into.

        Raises:
            ValueError: If the number of source and destination buffers do not
                match, or if any buffer copy fails due to shape, dtype, or
                device incompatibility.
        """
        for src_value, dst_value in zip(src, dst, strict=True):
            if src_value is dst_value:
                continue

            dst_value.inplace_copy_from(src_value)
