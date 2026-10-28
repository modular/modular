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
"""How spec decoding calls each target architecture's module.

The modules MAX architectures build share no calling convention: each takes its
inputs in an order of its own and returns its outputs by position. A driver
can't call a target without knowing which architecture built it, so each
target's package holds a target adapter, its ``spec_adapters`` module, that
holds its call. ``_TARGET_ADAPTERS`` names the module of each target
architecture's :class:`TargetAdapter`, which takes what it leaves out from
Llama 3's and holds a target's differences under one speculative method in
``by_method``. A target whose ModuleV3 variant builds a module of another
kind names that variant's adapter under the variant's own name.

A module is imported only when one of its targets runs, so spec decoding
imports no architecture it doesn't run, and a build that doesn't ship a private
target's package never imports it.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeVar

from max.driver import Buffer, Device
from max.graph import BufferType, TensorType, Value
from max.nn.legacy.kv_cache import (
    MultiKVCacheParams,
    RecurrentStateInputsPerDevice,
)
from max.nn.legacy.layer import Module
from max.nn.legacy.transformer import ReturnHiddenStates
from max.pipelines.context import TextContext

from ..block_driver import BlockBatch, BlockProposer
from ..config import SpeculativeConfig
from ..driver import SequentialBatch, SequentialProposer
from ..spec_target import Verified
from .target_adapter import STAGE_BOUNDARY, TargetAdapter

__all__ = [
    "STAGE_BOUNDARY",
    "VerifyFn",
    "batch_distributed_buffers",
    "batch_replay_buffers",
    "batch_trailing_buffers",
    "block_proposer",
    "check",
    "ep_input_types",
    "export_contract",
    "extra",
    "prepares_every_batch",
    "primary_kv",
    "relaxed_thinking_acceptance",
    "replay_enabled",
    "replay_extra",
    "replay_input_types",
    "sequential_proposer",
    "signal_buffer_index",
    "speculates_on_last_stage",
    "tail_buffers",
    "tail_kv",
    "trailing_buffers",
    "trailing_input_types",
    "verify_for",
]


VerifyFn = Callable[[SequentialBatch | BlockBatch], Verified[Any]]
"""Verifies a driver batch on the target."""

_TARGET_ADAPTERS: Mapping[str, str] = {
    "DeepseekV3ForCausalLM": ".deepseekV3.spec_adapters",
    "DeepseekV4ForCausalLM": ".deepseekV4.spec_adapters",
    "DeepseekV41ForCausalLM": ".deepseekV41.spec_adapters",
    "Gemma4ForConditionalGeneration": ".gemma4.spec_adapters",
    "Gemma4ForConditionalGeneration_ModuleV3": ".gemma4_modulev3.spec_adapters",
    "Gemma4UnifiedForConditionalGeneration": ".gemma4.spec_adapters",
    "Gemma4UnifiedForConditionalGeneration_ModuleV3": (
        ".gemma4_modulev3.spec_adapters"
    ),
    "Glm5NextForConditionalGeneration": ".glm5_next.spec_adapters",
    "GraniteForCausalLM": ".llama3.spec_adapters",
    "HYV4ForCausalLM": ".hy_v4.spec_adapters",
    "InklingForConditionalGeneration": ".inkling.spec_adapters",
    "KimiK25ForConditionalGeneration": ".kimik2_5.spec_adapters",
    "KimiK3ForConditionalGeneration": "kimi_k3_spec",
    "LlamaForCausalLM": ".llama3.spec_adapters",
    "MiMoV2ForCausalLM": ".mimo_v2.spec_adapters",
    "MiMoV2ForCausalLM_ModuleV3": ".mimo_v2_modulev3.spec_adapters",
    "NemotronHForCausalLM": ".nemotron_h_modulev3.spec_adapters",
    "Qwen3_5ForCausalLM": ".qwen3_5.spec_adapters",
    "Qwen3_5ForConditionalGeneration": ".qwen3_5.spec_adapters",
    "Qwen4ExpForConditionalGeneration": "qwen4_exp_spec",
}


def _target_adapter(target_arch: str) -> TargetAdapter:
    # A ModuleV3 variant without an adapter of its own is its target's only
    # variant, as Nemotron-H is, and takes the adapter named for the target.
    if target_arch not in _TARGET_ADAPTERS:
        target_arch = target_arch.removesuffix("_ModuleV3")
    if target_arch not in _TARGET_ADAPTERS:
        raise ValueError(
            f"Spec decoding has no call for a {target_arch} target."
        )
    adapter = importlib.import_module(
        _TARGET_ADAPTERS[target_arch], "max.pipelines.architectures"
    ).ADAPTER
    assert isinstance(adapter, TargetAdapter)
    return adapter


def _adapter(target_arch: str, method: str | None = None) -> TargetAdapter:
    """Returns the target's adapter, or its adapter under ``method``."""
    adapter = _target_adapter(target_arch)
    return adapter.by_method.get(method, adapter) if method else adapter


def verify_for(
    target_arch: str,
    target: Module,
    target_config: Any,
    hidden_states: ReturnHiddenStates,
    draft_config: Any,
) -> VerifyFn:
    """Returns how a built target module verifies under a spec-decode driver.

    Args:
        target_arch: The target checkpoint's architecture name.
        target: The module the target's pipeline model built.
        target_config: The config it was built from.
        hidden_states: The hidden states it was built to return.
        draft_config: The config the draft was built from.
    """
    return _target_adapter(target_arch).verify(
        target, target_config, hidden_states, draft_config
    )


def primary_kv(target_arch: str, method: str) -> str:
    """Returns the name of the KV leaf a driver dispatches the target over.

    The target's other leaves ride in the driver batch's ``passthrough_kv``,
    under their names in the KV tree. A target with one leaf has the name
    ``""``.
    """
    return _adapter(target_arch, method).primary_kv


def ep_input_types(
    target_arch: str, target: Module
) -> Sequence[TensorType | BufferType]:
    """Returns the expert-parallel inputs of a built target module."""
    return _adapter(target_arch).ep_input_types(target)


def batch_distributed_buffers(
    target_arch: str, target_inputs: object
) -> tuple[Sequence[Buffer] | None, Sequence[Buffer]]:
    """Returns the per-device context lengths and the prepared buffers.

    The buffers are the ones in :func:`ep_input_types` that the target's batch
    processor prepared.

    ``None`` context lengths when it prepares none.
    """
    return _adapter(target_arch).batch_distributed_buffers(target_inputs)


def export_contract(
    target_arch: str,
    speculative: SpeculativeConfig,
    draft_config: Any,
    target_config: Any,
    kv_params: MultiKVCacheParams,
    input_types: Sequence[TensorType | BufferType],
    structured_output: bool,
) -> Mapping[str, object] | None:
    """Returns the record a fused-graph engine checks its inputs against.

    Only a target whose engine keeps such a record has one.

    Mach's ``gen_mef`` writes it to the export's sidecar.
    """
    return _adapter(target_arch).export_contract(
        speculative,
        draft_config,
        target_config,
        kv_params,
        input_types,
        structured_output,
    )


def speculates_on_last_stage(target_arch: str, method: str) -> bool:
    """Returns whether ``method`` builds as the last pipeline stage.

    The answer covers the target's fused graph.
    """
    return _adapter(target_arch, method).speculates_on_last_stage


def tail_kv(target_arch: str) -> str | None:
    """Returns the target's KV child declared past the canonical tail.

    The graph declares it there rather than among its KV inputs. The result is
    none when it declares no such child.
    """
    return _adapter(target_arch).tail_kv


def tail_buffers(
    target_arch: str,
    tail: Sequence[RecurrentStateInputsPerDevice[Buffer, Buffer]],
) -> list[Buffer]:
    """Returns the inputs of the :func:`tail_kv` child.

    They come in the order the graph's tail declares them.
    """
    return _adapter(target_arch).tail_buffers(tail)


def check(
    target_arch: str, speculative: SpeculativeConfig, structured_output: bool
) -> None:
    """Raises if the target can't verify under the speculative config."""
    _adapter(target_arch).check(speculative, structured_output)


def relaxed_thinking_acceptance(target_arch: str) -> bool:
    """Returns whether a reasoning model's drafts may be accepted.

    The relaxed in-thinking rule decides it.
    """
    return _adapter(target_arch).relaxed_thinking_acceptance


def prepares_every_batch(target_arch: str) -> bool:
    """Returns whether the target prepares every batch of a pairing.

    The inputs past the canonical tail count.
    """
    return _adapter(target_arch).prepares_every_batch


def signal_buffer_index(target_arch: str) -> int | None:
    """Returns where a fused graph takes its signal buffers.

    The result is fixed across device counts, or none when the graph does not
    fix it.
    """
    return _adapter(target_arch).signal_buffer_index


def trailing_input_types(
    target_arch: str, method: str, target_config: Any
) -> Sequence[TensorType | BufferType]:
    """Returns the inputs the target's call takes past the canonical tail."""
    return _adapter(target_arch, method).trailing_input_types(target_config)


def trailing_buffers(
    target_arch: str,
    method: str,
    target_config: Any,
    contexts: Sequence[TextContext],
    devices: Sequence[Device],
) -> list[Buffer]:
    """Returns the trailing inputs that no KV child holds.

    They are a batch's values of :func:`trailing_input_types`.

    Args:
        target_arch: The target checkpoint's architecture name.
        method: The speculative method.
        target_config: The config the target was built from.
        contexts: The batch, in the order its tokens are merged in.
        devices: The devices the target runs on.
    """
    return _adapter(target_arch, method).trailing_buffers(
        target_config, contexts, devices
    )


def batch_trailing_buffers(
    target_arch: str, method: str, target_inputs: object
) -> list[Buffer]:
    """Returns the trailing inputs that the target prepared.

    They are the batch's values of :func:`trailing_input_types`, for a target
    whose batch processor prepares the pairing's batches.
    """
    return _adapter(target_arch, method).batch_trailing_buffers(target_inputs)


def extra(
    target_arch: str,
    method: str,
    target_config: Any,
    trailing: Sequence[Value[Any]],
) -> dict[str, Any]:
    """Returns what the target's call reads of the trailing graph inputs.

    The driver hands it to the verify and the proposer as the batch's
    ``extra``.
    """
    return _adapter(target_arch, method).extra(target_config, trailing)


def replay_enabled(
    target_arch: str, method: str, pipeline_config: Any, target_config: Any
) -> bool:
    """Returns whether the pairing compiles a replay graph beside the full one."""
    return _adapter(target_arch, method).replay_enabled(
        pipeline_config, target_config
    )


def replay_input_types(
    target_arch: str, method: str, target_config: Any
) -> Sequence[TensorType | BufferType]:
    """Returns the inputs only the replay graph takes, last of all."""
    return _adapter(target_arch, method).replay_input_types(target_config)


def batch_replay_buffers(
    target_arch: str, method: str, target_inputs: object
) -> list[Buffer]:
    """Returns the replay inputs that the target prepared.

    They are the batch's values of :func:`replay_input_types`, and empty for the
    full graph.
    """
    return _adapter(target_arch, method).batch_replay_buffers(target_inputs)


def replay_extra(
    target_arch: str,
    method: str,
    target_config: Any,
    replay: Sequence[Value[Any]],
) -> dict[str, Any]:
    """Returns what the target's call reads of the replay graph's inputs."""
    return _adapter(target_arch, method).replay_extra(target_config, replay)


def takes_vision_outputs(
    target_arch: str, method: str, target_config: Any
) -> bool:
    """Returns whether the graph takes an external vision output.

    Such an output comes from an engine that runs the encoder itself, so MAX
    does not compute it.
    """
    return _adapter(target_arch, method).takes_vision_outputs(target_config)


def block_proposer(
    target_arch: str,
    verify: VerifyFn,
    draft: Module,
    draft_config: Any,
    speculative: SpeculativeConfig,
) -> BlockProposer[Any, Any] | None:
    """Returns the proposer of a draft that runs on its target's own state.

    ``None`` for every other draft, which its method's proposer runs.
    """
    return _adapter(target_arch).block_proposer(
        verify, draft, draft_config, speculative
    )


def sequential_proposer(
    target_arch: str,
    verify: VerifyFn,
    draft: Module,
    draft_config: Any,
    speculative: SpeculativeConfig,
) -> SequentialProposer[Any] | None:
    """Returns the proposer of a one-token-per-step draft.

    The draft runs on its target's own module.

    ``None`` for every other draft, which its method's proposer runs.
    """
    return _adapter(target_arch).sequential_proposer(
        verify, draft, draft_config, speculative
    )
