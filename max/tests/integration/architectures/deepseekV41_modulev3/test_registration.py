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
"""Checks registered V4.1 deployment constants and request-state packing."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import NonCallableMock

import numpy as np
import pytest
from abi_contract import hf_config
from max import tree
from max.driver import CPU, Buffer
from max.dtype import DType
from max.graph import DeviceRef, TensorType
from max.nn.kv_cache import KVCacheInputsPerDevice
from max.nn.outputs import ReturnHiddenStates, ReturnLogits
from max.pipelines.architectures.deepseekV41_modulev3 import registration
from max.pipelines.architectures.deepseekV41_modulev3.interface import (
    extra_input_names,
)
from max.pipelines.architectures.deepseekV41_modulev3.registration import (
    DeepseekV41RequestState,
    RegisteredDeepseekV41BatchProcessor,
    RegisteredDeepseekV41Config,
    RegisteredDeepseekV41Decoder,
    RegisteredDeepseekV41Inputs,
)
from max.pipelines.architectures.deepseekV41_modulev3.spec_adapters import (
    batch_trailing_buffers,
)
from max.pipelines.context import TextContext
from max.pipelines.context.tokens import TokenBuffer
from max.pipelines.lib.interfaces.arch_config import ModelOutputConfig
from max.pipelines.lib.interfaces.batch_processor import BatchProcessorRuntime
from max.pipelines.lib.role_contracts import DecoderInputs


def _config() -> RegisteredDeepseekV41Config:
    hf: Any = dict(hf_config("f1"))
    text = hf["text_config"]
    text["engram_layer_ids"] = []
    text["engram_num_embeddings"] = []
    pipeline: Any = SimpleNamespace(
        model=SimpleNamespace(huggingface_config=hf, data_parallel_degree=1),
        runtime=SimpleNamespace(
            max_batch_input_tokens=16,
            enable_chunked_prefill=True,
        ),
    )
    return RegisteredDeepseekV41Config(
        dtype=DType.bfloat16,
        kv_params=NonCallableMock(),
        devices=[DeviceRef.CPU()],
        max_seq_len=128,
        pipeline_config=pipeline,
        vocab_size=text["vocab_size"],
        engram_layer_ids=[],
        engram_num_embeddings=[],
        engram_static_inputs={
            "engram_token_map": Buffer.from_numpy(
                np.zeros(text["vocab_size"], np.int32)
            )
        },
    )


def test_deployment_constants_are_named_and_reused_between_steps() -> None:
    config = _config()
    declared = RegisteredDeepseekV41Decoder.constant_inputs(
        config, devices=[CPU()]
    )
    assert tree.leaves(declared.buffers) == [
        config.engram_static_inputs["engram_token_map"]
    ]
    assert isinstance(declared.input_types, dict)
    assert tuple(declared.input_types) == ("engram_token_map",)
    store = DeepseekV41RequestState(config, max_batch_size=4, devices=[CPU()])
    batch = [TextContext(max_length=128, tokens=TokenBuffer(np.arange(3)))]
    store.set_layout([batch], 1)
    first = store.prepare(batch)
    second = store.prepare(batch)
    names = extra_input_names(store.processor._v41)
    dynamic_names = [name for name in names if name != "engram_token_map"]
    assert len(first) == len(second) == len(store.input_types())
    assert len(first) == len(dynamic_names)
    assert all(
        buffer is not config.engram_static_inputs["engram_token_map"]
        for buffer in first
    )
    store.update([batch[0].request_id], ())
    store.release(batch[0].request_id)


def test_registered_abi_has_a_distinct_name_and_separate_constant_field() -> (
    None
):
    assert RegisteredDeepseekV41Decoder.graph_name == (
        "DeepseekV41_registered_v1"
    )
    order = RegisteredDeepseekV41Decoder.input_layout
    assert order.index("kv") < order.index("constants") < order.index("state")
    assert "constants" in DecoderInputs.__dataclass_fields__
    assert RegisteredDeepseekV41Config.export_ep_init


def test_finalization_consumes_adapter_sidecars_after_pinning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    pinned = config.engram_static_inputs
    token_map = np.arange(config.vocab_size, dtype=np.int32)
    sidecar: Any = NonCallableMock(engram_tables=[object()])
    calls: list[str] = []
    monkeypatch.setattr(
        registration,
        "validate_engram_tables",
        lambda *args: calls.append("validate"),
    )
    monkeypatch.setattr(
        registration, "serving_token_map", lambda *args: token_map
    )

    def pin(*args: object) -> object:
        calls.append("pin")
        return pinned

    monkeypatch.setattr(registration, "engram_static_inputs", pin)
    finalized = config.finalize(NonCallableMock(), sidecar, ModelOutputConfig())
    assert calls == ["validate", "pin"]
    assert sidecar.engram_tables == []
    assert finalized.engram_static_inputs is pinned
    assert finalized.token_map is token_map


def test_composed_batching_binds_providers_and_keeps_the_dspark_extras_view() -> (
    None
):
    config = _config()
    config.dspark_trailing_inputs = [
        ("verify", TensorType(DType.int64, [1], device=DeviceRef.CPU()))
    ]
    assert config.pipeline_config is not None
    runtime = BatchProcessorRuntime(
        pipeline_config=config.pipeline_config,
        devices=[CPU()],
        return_logits=ReturnLogits.LAST_TOKEN,
        max_batch_size=4,
        max_seq_len=128,
        max_batch_input_tokens=16,
    )
    processor = RegisteredDeepseekV41BatchProcessor(config, runtime)
    batch = [TextContext(max_length=128, tokens=TokenBuffer(np.arange(3)))]
    kv_buffer = Buffer.from_numpy(np.zeros(1, np.float32))
    kv = KVCacheInputsPerDevice(
        kv_buffer, kv_buffer, kv_buffer, kv_buffer, kv_buffer, kv_buffer
    )
    inputs = processor.prepare_initial_token_inputs(
        [batch], kv_cache_inputs=(kv,), return_n_logits=2
    )
    assert isinstance(inputs, RegisteredDeepseekV41Inputs)
    assert RegisteredDeepseekV41Decoder.composes_request_state
    assert inputs.constants is not None
    assert isinstance(processor.state_store, DeepseekV41RequestState)
    names = extra_input_names(processor.state_store.processor._v41)
    original_tail = batch_trailing_buffers(inputs)
    assert len(original_tail) == len(names)
    constant_index = names.index("engram_token_map")
    assert (
        original_tail[constant_index]
        is config.engram_static_inputs["engram_token_map"]
    )
    assert len(inputs.state) == len(names) - 1


def test_composition_keeps_trunk_heads_separate_from_adapter_verify_outputs() -> (
    None
):
    config = _config()
    config.return_logits = ReturnLogits.VARIABLE
    config.return_hidden_states = ReturnHiddenStates.SELECTED_LAYERS
    context: Any = SimpleNamespace(target_layer_ids=[])
    composed = config.apply_model_load_context(context)
    assert composed.return_logits == ReturnLogits.LAST_TOKEN
    assert composed.return_hidden_states == ReturnHiddenStates.NONE
    assert composed.dspark_trailing_inputs
