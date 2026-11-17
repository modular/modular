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
"""Per-request state a registered decoder carries from one step to the next.

A recurrent or convolutional decoder keeps state for each request beside its
KV cache, such as LFM2's short-convolution states or Nemotron-H's Mamba pools.
A registered ``DECODER`` class declares that state with a ``state_store``
classmethod that returns a :class:`RequestStateStore`, and the framework then:

- adds :meth:`RequestStateStore.input_types` to the graph's inputs, as the
  :attr:`~max.pipelines.lib.role_contracts.DecoderInputs.state` field after
  the KV cache (and LoRA) inputs;
- fills them each step from :meth:`RequestStateStore.prepare`;
- passes the
  :attr:`~max.pipelines.lib.interfaces.pipeline_model.ModuleV3Outputs.state`
  the decoder returns, which the graph outputs after the role's outputs, to
  :meth:`RequestStateStore.update`;
- calls :meth:`RequestStateStore.release` when a request finishes, and
  :meth:`RequestStateStore.release_warmup` after each graph-capture warmup
  probe.

A store either takes the state back from the graph's outputs each step, or
hands the graph mutable buffers that it updates in place, in which case the
decoder returns no state.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

from max.driver import Buffer, Device
from max.graph import BufferType, SymbolicDim, TensorType
from max.nn.legacy.graph_groups import GraphGroup
from max.nn.outputs import ReturnHiddenStates, ReturnLogits
from max.pipelines.context import TextContext
from max.pipelines.modeling.types import RequestID

__all__ = [
    "RequestStateGroup",
    "RequestStateMixin",
    "RequestStateStore",
    "build_state_store",
    "split_state_outputs",
]


class RequestStateGroup(GraphGroup[str, tuple[Any, ...]]):
    """A graph-API decoder's state inputs, or the state outputs it writes back.

    Its one entry, ``state``, holds them in the order the decoder's
    :class:`RequestStateStore` declares, so a serving engine finds them by
    kind.
    """

    kind = "request_state"


class RequestStateStore(ABC):
    """Holds the state a registered decoder keeps for each request in flight.

    It owns every request's state: it assigns each new request a slot,
    assembles a batch's state for the graph, stores what the graph returns,
    and frees a finished request's slot.
    """

    @abstractmethod
    def input_types(self) -> tuple[TensorType | BufferType, ...]:
        """Returns the graph inputs the state adds, in graph order."""

    def max_dims(self) -> Mapping[str, int]:
        """Returns the largest size of each symbolic dim of the state inputs.

        A serving engine allocates each per-step input once at its largest
        size, so every symbolic dim of a :class:`~max.graph.TensorType` input
        needs a bound. A :class:`~max.graph.BufferType` input is allocated
        once by the store and needs none. The default bounds nothing, which
        suits a store whose tensor inputs have static shapes.
        """
        return {}

    @abstractmethod
    def prepare(self, batch: Sequence[TextContext]) -> tuple[Buffer, ...]:
        """Returns the step's state inputs for a batch.

        Gives each request it hasn't seen a fresh slot first; a request it
        holds keeps its state, as across chunked prefill.

        Args:
            batch: The step's contexts, in batch order.

        Returns:
            One buffer per :meth:`input_types` entry, in its order.
        """

    def update(
        self, request_ids: Sequence[RequestID], outputs: Sequence[Buffer]
    ) -> None:
        """Stores the state a step returned into its requests' slots.

        The default is for a store whose graph updates its buffers in place,
        so the decoder returns no state.

        Args:
            request_ids: The step's requests, in batch order.
            outputs: The decoder's state outputs, in their order.

        Raises:
            ValueError: If the decoder returned state.
        """
        del request_ids
        if outputs:
            raise ValueError(
                f"{type(self).__qualname__} updates its state in place, but "
                f"the decoder returned {len(outputs)} state outputs"
            )

    @abstractmethod
    def release(self, request_id: RequestID) -> None:
        """Frees the state a finished request held."""

    def release_warmup(self, request_ids: Sequence[RequestID]) -> None:
        """Frees the state a graph-capture warmup probe claimed."""
        for request_id in request_ids:
            self.release(request_id)


def build_state_store(
    decoder_cls: type[Any],
    config: Any,
    *,
    max_batch_size: int,
    devices: Sequence[Device],
) -> RequestStateStore | None:
    """Returns the state store a registered decoder declares, if any.

    Args:
        decoder_cls: The registered ``DECODER`` class.
        config: The finalized config the decoder is built from.
        max_batch_size: The most requests a step holds.
        devices: The devices the decoder runs on.

    Raises:
        TypeError: If ``state_store`` returns something other than a
            :class:`RequestStateStore`.
        ValueError: If the store leaves a symbolic dim of a tensor input
            unbounded.
    """
    declare = getattr(decoder_cls, "state_store", None)
    if declare is None:
        return None
    store = declare(config, max_batch_size=max_batch_size, devices=devices)
    if not isinstance(store, RequestStateStore):
        raise TypeError(
            f"{decoder_cls.__qualname__}.state_store() returned "
            f"{type(store).__qualname__}, not a RequestStateStore"
        )
    bounds = store.max_dims()
    unbounded = [
        dim.name
        for input_type in store.input_types()
        if isinstance(input_type, TensorType)
        for dim in input_type.shape
        if isinstance(dim, SymbolicDim) and dim.name not in bounds
    ]
    if unbounded:
        raise ValueError(
            f"{type(store).__qualname__}.max_dims() bounds no maximum for "
            f"state input dims {unbounded}"
        )
    return store


def split_state_outputs(
    outputs: Sequence[Any],
    return_logits: ReturnLogits,
    return_hidden_states: ReturnHiddenStates,
) -> tuple[Sequence[Any], Sequence[Any]]:
    """Splits a decoder graph's flat outputs into the role's and the state.

    The role's outputs are the next-token logits, then the logits and their
    offsets when variable or full logits are returned, then the hidden states
    when they are.

    Returns:
        The role's outputs and the state outputs that follow them.
    """
    has_offsets = return_logits in (ReturnLogits.VARIABLE, ReturnLogits.ALL)
    has_hidden_states = return_hidden_states != ReturnHiddenStates.NONE
    count = 1 + 2 * int(has_offsets) + int(has_hidden_states)
    return outputs[:count], outputs[count:]


class RequestStateMixin:
    """Gives a registered pipeline model its decoder's state lifecycle.

    Serving frees a finished request's state through :meth:`release`, and a
    graph-capture warmup's through :meth:`release_warmup_state`.
    """

    _state_store: RequestStateStore | None = None

    @property
    def holds_request_state(self) -> bool:
        """Whether a decoder's state store is bound to this model.

        A bound store hands the graph per-request buffers that the step
        advances in place, so a caller that must not execute a step twice
        over the same state -- launch-trace verification -- refuses while
        this is ``True``.
        """
        return self._state_store is not None

    def release(self, request_id: RequestID) -> None:
        """Frees the state a finished request held."""
        if self._state_store is not None:
            self._state_store.release(request_id)

    def release_warmup_state(self, request_ids: list[RequestID]) -> None:
        """Frees the state a graph-capture warmup probe claimed."""
        if self._state_store is not None:
            self._state_store.release_warmup(request_ids)
