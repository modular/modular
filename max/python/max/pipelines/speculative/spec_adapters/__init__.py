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
``by_method``.

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
from max.nn.kv_cache import MultiKVCacheParams, RecurrentStateInputsPerDevice
from max.nn.layer import Module
from max.nn.transformer import ReturnHiddenStates
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
    "batch_trailing_buffers",
    "block_proposer",
    "check",
    "ep_input_types",
    "export_contract",
    "extra",
    "primary_kv",
    "sequential_proposer",
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
    "Gemma4ForConditionalGeneration": ".gemma4.spec_adapters",
    "Gemma4UnifiedForConditionalGeneration": ".gemma4.spec_adapters",
    "GraniteForCausalLM": ".llama3.spec_adapters",
    "InklingForConditionalGeneration": ".inkling.spec_adapters",
    "KimiK25ForConditionalGeneration": ".kimik2_5.spec_adapters",
    "KimiK3ForConditionalGeneration": "kimi_k3_spec",
    "LlamaForCausalLM": ".llama3.spec_adapters",
    "MiMoV2ForCausalLM": ".mimo_v2.spec_adapters",
    "Qwen3_5ForCausalLM": ".qwen3_5.spec_adapters",
    "Qwen3_5ForConditionalGeneration": ".qwen3_5.spec_adapters",
    "Qwen4ExpForConditionalGeneration": "qwen4_exp_spec",
}


def _target_adapter(target_arch: str) -> TargetAdapter:
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
    """Returns the per-device context lengths and the buffers of
    :func:`ep_input_types` that the target's batch processor prepared.

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
    """Returns the record an engine that serves the fused graph checks its
    inputs against, for a target whose engine keeps one.

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
    """Returns whether the target's fused graph under ``method`` builds as
    the last of several pipeline stages."""
    return _adapter(target_arch, method).speculates_on_last_stage


def tail_kv(target_arch: str) -> str | None:
    """Returns the target's KV child the graph declares past the canonical
    tail rather than among its KV inputs, if any."""
    return _adapter(target_arch).tail_kv


def tail_buffers(
    target_arch: str,
    tail: Sequence[RecurrentStateInputsPerDevice[Buffer, Buffer]],
) -> list[Buffer]:
    """Returns the inputs of the :func:`tail_kv` child, in the order the
    graph's tail declares them."""
    return _adapter(target_arch).tail_buffers(tail)


def check(
    target_arch: str, speculative: SpeculativeConfig, structured_output: bool
) -> None:
    """Raises if the target can't verify under the speculative config."""
    _adapter(target_arch).check(speculative, structured_output)


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
    """Returns a batch's values of :func:`trailing_input_types` that no KV
    child holds.

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
    """Returns the batch's values of :func:`trailing_input_types` that the
    target's own batch processor prepared, for a target whose batch
    processor prepares the pairing's batches."""
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


def takes_vision_outputs(
    target_arch: str, method: str, target_config: Any
) -> bool:
    """Returns whether the graph takes a vision encoder's output that MAX
    doesn't compute, for an engine that runs the encoder itself."""
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
    """Returns the proposer of a one-token-per-step draft that runs on its
    target's own module.

    ``None`` for every other draft, which its method's proposer runs.
    """
    return _adapter(target_arch).sequential_proposer(
        verify, draft, draft_config, speculative
    )
