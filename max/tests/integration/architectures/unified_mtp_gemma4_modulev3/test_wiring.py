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
"""CPU-only structural wiring tests for the ModuleV3 Gemma4 MTP arch.

These mirror ``unified_mtp_gemma4/test_vision_wiring.py`` but target the
ModuleV3 port, guarding the wiring points that differ from the graph API
arch without a full GPU serve:

* the arch swaps in the ModuleV3 pipeline model and batch processor;
* the fused compiled graph's signature keeps the vision inputs (after
  tokens) and carries NO signal-buffer inputs — a Module.compile-traced
  graph appends its own after every declared input, so
  ``trace_owns_signal_buffers`` must hold and the ModuleV3 inputs
  dataclass's ``buffers`` (signals last) binds one-to-one;
* the proposer adapter declares the Gemma4 MTP contract (Q-only draft
  cross-attending the TARGET's KV caches, thinking-phase flag, skippable
  draft rows).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import numpy as np
import pytest
from max.driver import Buffer
from max.dtype import DType
from max.graph import DeviceRef
from max.pipelines.architectures.gemma4.model_config import (
    Gemma4ForConditionalGenerationConfig,
)
from max.pipelines.architectures.unified_mtp_gemma4_modulev3.arch import (
    unified_mtp_gemma4_modulev3_arch,
)
from max.pipelines.architectures.unified_mtp_gemma4_modulev3.batch_processor import (
    UnifiedMTPGemma4ModuleV3BatchProcessor,
    UnifiedMTPGemma4ModuleV3Inputs,
)
from max.pipelines.architectures.unified_mtp_gemma4_modulev3.model import (
    UnifiedMTPGemma4ModuleV3Model,
)
from max.pipelines.architectures.unified_mtp_gemma4_modulev3.spec_adapters import (
    GLOBAL_KV,
    Gemma4ModuleV3MTPProposer,
)
from max.pipelines.architectures.unified_mtp_gemma4_modulev3.unified_mtp_gemma4_modulev3 import (
    gemma4_modulev3_mtp_input_spec,
)
from max.pipelines.modeling.types import InputModality
from max.pipelines.speculative import build_spec_decode_input_types
from max.pipelines.speculative.driver import DecodeKVSwap, DraftCache

_HIDDEN_SIZE = 128


def _duck_typed_config(
    devices: list[DeviceRef],
) -> Gemma4ForConditionalGenerationConfig:
    """A config exposing only what the input spec reads."""
    return cast(
        Gemma4ForConditionalGenerationConfig,
        SimpleNamespace(
            devices=devices,
            text_config=SimpleNamespace(hidden_size=_HIDDEN_SIZE),
        ),
    )


def _kv_params_stub() -> MagicMock:
    kv_params = MagicMock()
    kv_params.flattened_kv_inputs.return_value = []
    return kv_params


def test_arch_swaps_in_modulev3_model_and_processor() -> None:
    """The arch must be the ModuleV3 model/processor on the V2 base."""
    assert unified_mtp_gemma4_modulev3_arch.name == (
        "UnifiedMTPGemma4ForCausalLM_ModuleV3"
    )
    assert unified_mtp_gemma4_modulev3_arch.pipeline_model is (
        UnifiedMTPGemma4ModuleV3Model
    )
    assert unified_mtp_gemma4_modulev3_arch.batching is (
        UnifiedMTPGemma4ModuleV3BatchProcessor
    )


def test_arch_is_multimodal() -> None:
    """The ModuleV3 arch keeps the V2 multimodal surface."""
    modalities = unified_mtp_gemma4_modulev3_arch.input_modalities
    assert InputModality.TEXT in modalities
    assert InputModality.IMAGE in modalities
    assert InputModality.VIDEO in modalities


@pytest.mark.parametrize("n_devices", [1, 2])
def test_signature_declares_per_device_vision_inputs(n_devices: int) -> None:
    """The signature carries per-device vision inputs right after tokens.

    If ``enable_vision`` regressed, the input right after tokens would be
    the (uint32, 1-D) row-offsets input and these assertions fail.
    """
    devices = [DeviceRef("gpu", i) for i in range(n_devices)]
    input_types = build_spec_decode_input_types(
        gemma4_modulev3_mtp_input_spec(_duck_typed_config(devices)),
        kv_params=_kv_params_stub(),
    )

    # tokens, then per-device image embeddings, then per-device scatter
    # indices.
    image_embeddings = input_types[1 : 1 + n_devices]
    image_indices = input_types[1 + n_devices : 1 + 2 * n_devices]

    assert len(image_embeddings) == n_devices
    assert len(image_indices) == n_devices
    for embed_type in image_embeddings:
        assert embed_type.dtype == DType.bfloat16
        assert int(embed_type.shape[-1]) == _HIDDEN_SIZE
    for index_type in image_indices:
        assert index_type.dtype == DType.int32


@pytest.mark.parametrize("n_devices", [1, 2])
def test_signature_has_no_midlist_signals(n_devices: int) -> None:
    """``trace_owns_signal_buffers`` keeps signal inputs out of the list.

    A Module.compile-traced graph appends its own signal buffers after
    every declared input, so the declared signature must not carry them:
    a stray group would desynchronize ``engine.Model.execute(*buffers)``,
    which passes everything explicitly with the signals last.
    """
    devices = [DeviceRef("gpu", i) for i in range(n_devices)]
    spec = gemma4_modulev3_mtp_input_spec(_duck_typed_config(devices))

    # The helper itself declares the ModuleV3 divergence from V2.
    assert spec.trace_owns_signal_buffers is True
    assert spec.distributed is True
    assert spec.enable_vision is True

    input_types = build_spec_decode_input_types(
        spec,
        kv_params=_kv_params_stub(),
    )
    for input_type in input_types:
        assert input_type.dtype not in {DType.bool, DType.uint8}, (
            "Signature must not carry signal buffers (the traced graph"
            " appends its own)."
        )


def test_proposer_declares_target_kv_cache_contract() -> None:
    """The draft must cross-attend the TARGET's KV caches."""
    assert Gemma4ModuleV3MTPProposer.draft_cache is DraftCache.TARGET
    assert Gemma4ModuleV3MTPProposer.uses_thinking_phase is True
    assert Gemma4ModuleV3MTPProposer.supports_zero_draft_rows is True
    assert Gemma4ModuleV3MTPProposer.decode_swaps == (
        DecodeKVSwap.MAX_PROMPT_LENGTH_ONE,
    )
    assert Gemma4ModuleV3MTPProposer.passthrough_decode_swaps == (GLOBAL_KV,)
    assert Gemma4ModuleV3MTPProposer.reuse is None


def test_inputs_dataclass_carries_signal_and_kv_fields() -> None:
    """The ModuleV3 inputs dataclass keeps every carrier the buffers
    property reads, so the signals-last reordering cannot silently drop
    a field the graph-API parent still writes."""
    fields = set(UnifiedMTPGemma4ModuleV3Inputs.__dataclass_fields__)
    for required in (
        "tokens",
        "signal_buffers",
        "kv_cache_inputs",
        "input_row_offsets",
        "host_input_row_offsets",
        "return_n_logits",
        "batch_context_lengths",
    ):
        assert required in fields, f"Missing inputs field: {required}"


def test_inputs_buffers_put_signals_last() -> None:
    """``buffers`` must order the tuple for a Module.compile graph.

    The graph-API parent declares signal buffers mid-signature (right
    after the data-parallel splits); a Module.compile-traced graph
    appends its own after every declared input, so the override must
    emit them last or ``engine.Model.execute(*buffers)`` misbinds every
    input after the first signal.
    """

    def buf() -> Buffer:
        return Buffer.from_numpy(np.zeros(1, np.uint32))

    tokens, row_offsets, host_offsets, n_logits, dp_splits = (
        buf() for _ in range(5)
    )
    signals = [buf(), buf()]
    context_lengths = [buf(), buf()]
    tail = [buf() for _ in range(8)]
    (
        draft_tokens,
        seed,
        temperature,
        top_k,
        max_k,
        top_p,
        min_top_p,
        in_thinking_phase,
    ) = tail

    inputs = UnifiedMTPGemma4ModuleV3Inputs(
        tokens=tokens,
        input_row_offsets=row_offsets,
        host_input_row_offsets=host_offsets,
        return_n_logits=n_logits,
        data_parallel_splits=dp_splits,
        signal_buffers=signals,
        kv_cache_inputs=(),
        batch_context_lengths=context_lengths,
        draft_tokens=draft_tokens,
        seed=seed,
        temperature=temperature,
        top_k=top_k,
        max_k=max_k,
        top_p=top_p,
        min_top_p=min_top_p,
        in_thinking_phase=in_thinking_phase,
    )

    # Buffer equality is not identity-based, so compare by object id.
    assert [id(b) for b in inputs.buffers] == [
        id(b)
        for b in (
            tokens,
            row_offsets,
            host_offsets,
            n_logits,
            dp_splits,
            *context_lengths,
            *tail,
            *signals,
        )
    ]
