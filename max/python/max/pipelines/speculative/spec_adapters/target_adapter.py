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
"""The call a spec-decode driver makes of one kind of target."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from max.driver import Buffer, Device
from max.graph import BufferType, TensorType, Value
from max.nn.layer import Module
from max.nn.transformer import ReturnHiddenStates
from max.pipelines.context import TextContext

if TYPE_CHECKING:
    from ..block_driver import BlockProposer
    from ..config import SpeculativeConfig
    from ..driver import SequentialProposer
    from . import VerifyFn


STAGE_BOUNDARY = "stage_boundary"
"""The key of a driver batch's ``extra`` that holds a later pipeline stage's
boundary: the previous stage's hidden states per device, over the merged
verify window."""


def _default_ep_input_types(target: Any) -> Sequence[TensorType | BufferType]:
    """Returns no expert-parallel inputs."""
    del target
    return ()


def _default_batch_distributed_buffers(
    target_inputs: Any,
) -> tuple[Sequence[Buffer] | None, Sequence[Buffer]]:
    """Returns the context lengths and EP buffers the target's batching made.

    It reads them under the names DeepseekV3's batch inputs carry them.
    """
    return (
        getattr(target_inputs, "batch_context_lengths", None),
        tuple(getattr(target_inputs, "ep_inputs", None) or ()),
    )


def _default_export_contract(
    speculative: SpeculativeConfig,
    draft_config: Any,
    target_config: Any,
    kv_params: Any,
    input_types: Sequence[TensorType | BufferType],
    structured_output: bool,
) -> None:
    """Returns no engine contract."""
    del speculative, draft_config, target_config, kv_params
    del input_types, structured_output


def _default_tail_buffers(tail: Sequence[object]) -> list[Buffer]:
    """Returns nothing.

    The graph declares every KV child among its KV inputs.
    """
    del tail
    return []


def _default_check(
    speculative: SpeculativeConfig, structured_output: bool
) -> None:
    """Accepts every speculative config."""
    del speculative, structured_output


def _default_trailing_input_types(target_config: Any) -> list[TensorType]:
    """Returns no inputs past the canonical tail."""
    del target_config
    return []


def _default_trailing_buffers(
    target_config: Any,
    contexts: Sequence[TextContext],
    devices: Sequence[Device],
) -> list[Buffer]:
    """Returns no batch inputs past the canonical tail."""
    del target_config, contexts, devices
    return []


def _default_batch_trailing_buffers(target_inputs: object) -> list[Buffer]:
    """Returns no batch inputs from the target's own batch processor."""
    del target_inputs
    return []


def _default_extra(
    target_config: Any, trailing: Sequence[Value[Any]]
) -> dict[str, Any]:
    """Returns nothing for a driver batch's ``extra``."""
    del target_config, trailing
    return {}


def _default_takes_vision_outputs(target_config: Any) -> bool:
    del target_config
    return False


def _default_block_proposer(
    verify: VerifyFn,
    draft: Module,
    draft_config: Any,
    speculative: SpeculativeConfig,
) -> BlockProposer[Any, Any] | None:
    """Returns ``None``: the method's own proposer runs the draft."""
    del verify, draft, draft_config, speculative
    return None


def _default_sequential_proposer(
    verify: VerifyFn,
    draft: Module,
    draft_config: Any,
    speculative: SpeculativeConfig,
) -> SequentialProposer[Any] | None:
    """Returns ``None``: the method's own proposer runs the draft."""
    del verify, draft, draft_config, speculative
    return None


def verify_lengths(contexts: Sequence[TextContext]) -> list[int]:
    """Returns each request's verify-forward length.

    That is its tokens and the drafts it verifies, which the pipeline stages
    on the context first.
    """
    return [
        ctx.tokens.active_length
        + len(ctx.spec_decoding_state.draft_tokens_to_verify)
        for ctx in contexts
    ]


@dataclass(frozen=True, kw_only=True)
class TargetAdapter:
    """How a spec-decode driver calls a target architecture's module.

    A field a target leaves out takes a default: one KV leaf, no
    expert-parallel inputs, nothing past the canonical tail, and the
    method's own proposer. ``verify`` has none.
    """

    primary_kv: str = ""
    """The KV leaf a driver dispatches the target over; ``""`` for a target
    with one leaf."""

    tail_kv: str | None = None
    """The KV child the graph declares past the canonical tail, if any."""

    verify: Callable[[Module, Any, ReturnHiddenStates, Any], VerifyFn]
    """Returns how a built target module verifies under a driver."""

    ep_input_types: Callable[[Any], Sequence[TensorType | BufferType]] = (
        _default_ep_input_types
    )
    """Returns a built target module's expert-parallel inputs."""

    batch_distributed_buffers: Callable[
        [Any], tuple[Sequence[Buffer] | None, Sequence[Buffer]]
    ] = _default_batch_distributed_buffers
    """Returns the per-device context lengths and the expert-parallel
    buffers the target's batch processor prepared."""

    export_contract: Callable[..., Mapping[str, object] | None] = (
        _default_export_contract
    )
    """Returns the record an engine checks the export's inputs against."""

    tail_buffers: Callable[[Sequence[Any]], list[Buffer]] = (
        _default_tail_buffers
    )
    """Returns the inputs of the ``tail_kv`` child, in declared order."""

    check: Callable[[SpeculativeConfig, bool], None] = _default_check
    """Raises if the target can't verify under the speculative config."""

    trailing_input_types: Callable[[Any], Sequence[TensorType | BufferType]] = (
        _default_trailing_input_types
    )
    """Returns the inputs the target's call takes past the canonical tail."""

    trailing_buffers: Callable[
        [Any, Sequence[TextContext], Sequence[Device]], list[Buffer]
    ] = _default_trailing_buffers
    """Returns a batch's trailing values that no KV child holds."""

    batch_trailing_buffers: Callable[[object], list[Buffer]] = (
        _default_batch_trailing_buffers
    )
    """Returns the trailing values the target's batch processor prepared."""

    extra: Callable[[Any, Sequence[Value[Any]]], dict[str, Any]] = (
        _default_extra
    )
    """Returns what the target's call reads of the trailing graph inputs."""

    takes_vision_outputs: Callable[[Any], bool] = _default_takes_vision_outputs
    """Returns whether the graph takes a vision encoder's output that MAX
    doesn't compute, for an engine that runs the encoder itself."""

    block_proposer: Callable[
        [VerifyFn, Module, Any, SpeculativeConfig],
        BlockProposer[Any, Any] | None,
    ] = _default_block_proposer
    """Returns the proposer of a draft that runs on the target's own state,
    or ``None`` for a draft its method's proposer runs."""

    sequential_proposer: Callable[
        [VerifyFn, Module, Any, SpeculativeConfig],
        SequentialProposer[Any] | None,
    ] = _default_sequential_proposer
    """Returns the proposer of a one-token-per-step draft that runs on the
    target's own module, or ``None``."""

    speculates_on_last_stage: bool = False
    """Whether the fused graph builds as the last of several pipeline
    stages: the verify starts the stage's layers from the previous stage's
    hidden states, which reach it in a driver batch's ``extra`` under
    :data:`STAGE_BOUNDARY`, and the draft reads only what the last stage
    holds."""

    by_method: Mapping[str, TargetAdapter] = field(default_factory=dict)
    """The adapter to use under a speculative method, where it differs."""
