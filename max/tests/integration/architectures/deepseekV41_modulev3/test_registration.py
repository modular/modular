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

from collections.abc import Mapping
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import NonCallableMock

import numpy as np
import pytest
from abi_contract import hf_config
from max import tree
from max.driver import CPU, Buffer
from max.dtype import DType
from max.graph import DeviceRef, TensorType
from max.graph.weights import WeightsFormat
from max.nn.kv_cache import KVCacheInputsPerDevice
from max.nn.outputs import ReturnHiddenStates, ReturnLogits
from max.pipelines.architectures.deepseekV41.arch import deepseekV41_arch
from max.pipelines.architectures.deepseekV41_modulev3 import registration
from max.pipelines.architectures.deepseekV41_modulev3.arch import (
    deepseekV41_modulev3_arch,
)
from max.pipelines.architectures.deepseekV41_modulev3.interface import (
    V41Config,
    extra_input_names,
)
from max.pipelines.architectures.deepseekV41_modulev3.model_config import (
    DeepseekV41Config,
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
from max.pipelines.architectures.deepseekV41_modulev3.weight_adapters import (
    convert_safetensor_state_dict,
)
from max.pipelines.context import TextContext
from max.pipelines.context.tokens import TokenBuffer
from max.pipelines.lib.interfaces.arch_config import ModelOutputConfig
from max.pipelines.lib.interfaces.batch_processor import BatchProcessorRuntime
from max.pipelines.lib.role_contracts import DecoderInputs
from max.pipelines.lib.weight_transform import WeightTransform
from transformers import PretrainedConfig


def _config() -> RegisteredDeepseekV41Config:
    hf: Any = dict(hf_config("f1"))
    text = hf["text_config"]
    text["engram_layer_ids"] = []
    text["engram_num_embeddings"] = []
    pipeline: Any = SimpleNamespace(
        model=SimpleNamespace(huggingface_config=hf, data_parallel_degree=1),
        artifact_export=False,
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


def test_finalization_consumes_adapter_sidecars_after_pinning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    pinned = config.engram_static_inputs
    token_map = np.arange(config.vocab_size, dtype=np.int32)
    sidecar = registration.AdapterResult({}, [cast(Any, object())])
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


def test_the_registration_names_the_v41_format_spec() -> None:
    """``max export`` copies the graph-API build's tool-call and reasoning
    document, which ``formatspec`` also ships for these checkpoints."""
    spec = deepseekV41_modulev3_arch.format_spec
    assert spec == deepseekV41_arch.format_spec
    assert isinstance(spec, Path)
    assert spec.is_file()


def test_raw_checkpoint_state_is_named_without_deployment_inputs() -> None:
    config = _config()
    config.raw_checkpoint_weights = True
    config.static_constants = registration.ConstantInputs(
        input_types={}, buffers={}
    )
    constants = RegisteredDeepseekV41Decoder.constant_inputs(
        config, devices=[CPU()]
    )
    assert not tree.leaves(constants.input_types)
    assert not tree.leaves(constants.buffers)
    store = DeepseekV41RequestState(config, max_batch_size=4, devices=[CPU()])
    (group,) = store.input_types()
    assert group.kind == "batch_context"
    assert tuple(group) == (
        "window_start",
        "write_floor",
        "image_token_mask",
        "lookback_tokens",
        "lookback_mask",
        "window_rows",
    )
    batch = [TextContext(max_length=128, tokens=TokenBuffer(np.arange(3)))]
    store.set_layout([batch], 1)
    (values,) = store.prepare(batch)
    assert isinstance(values, Mapping)
    assert tuple(values) == tuple(group)
    assert len(tree.leaves(values)) == len(tree.leaves(group))
    lookback = values["lookback_tokens"]
    window = values["window_rows"]
    assert isinstance(lookback, Buffer)
    assert isinstance(window, Buffer)
    assert lookback.shape == (1, 3)
    assert window.shape == (3,)


def test_raw_checkpoint_finalization_needs_no_adapter_sidecar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    config.raw_checkpoint_weights = True
    token_map = np.arange(config.vocab_size, dtype=np.int32)
    monkeypatch.setattr(
        registration, "serving_token_map", lambda *args: token_map
    )
    monkeypatch.setattr(
        registration,
        "build_engram_static_inputs",
        lambda *args: registration.ConstantInputs(input_types={}, buffers={}),
    )
    finalized = config.finalize(NonCallableMock(), {}, ModelOutputConfig())
    assert finalized.token_map is token_map
    assert finalized.raw_checkpoint_weights


def test_converted_sidecars_keep_the_legacy_registered_weight_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    token_map = np.arange(config.vocab_size, dtype=np.int32)
    sidecar = registration.AdapterResult({}, [])
    monkeypatch.setattr(
        registration, "serving_token_map", lambda *args: token_map
    )
    monkeypatch.setattr(
        registration, "validate_engram_tables", lambda *args: None
    )
    monkeypatch.setattr(
        registration,
        "engram_static_inputs",
        lambda *args: config.engram_static_inputs,
    )
    finalized = config.finalize(NonCallableMock(), sidecar, ModelOutputConfig())
    assert not finalized.raw_checkpoint_weights
    assert finalized.token_map is token_map


@pytest.mark.parametrize("raw", [False, True])
def test_a_weight_load_the_mode_did_not_ask_for_is_refused(raw: bool) -> None:
    config = _config()
    config.raw_checkpoint_weights = raw
    state_dict = {} if not raw else registration.AdapterResult({}, [])
    with pytest.raises(ValueError, match="raw checkpoint weights"):
        config.finalize(NonCallableMock(), state_dict, ModelOutputConfig())


# The bare-MEF extras after the KV leaves, in order: what Mach's bare
# contract (mach/crates/engine-core/src/deepseek_v41_dspark_full_contract.rs,
# pinned by mach/tools/deepseek-v41-contract) expects of the full checkpoint.
_BARE_EXTRAS = [
    "swa_begin",
    "global_write_floor",
    "image_token_mask",
    "engram_lookback",
    "engram_lookback_dead",
    "engram_table_0_weight",
    "engram_table_0_scale",
    "engram_table_1_weight",
    "engram_table_1_scale",
    "engram_token_map",
    "decoder_rows",
]


@pytest.mark.parametrize("artifact_export", [False, True])
def test_only_an_artifact_export_loads_raw_weights_and_artifact_inputs(
    monkeypatch: pytest.MonkeyPatch, artifact_export: bool
) -> None:
    """Serving and bare MEF exports (gen_mef) keep the adapter and its ABI.

    The registry hands the pipeline the adapter unless the export is an
    artifact, and the registered config's input mode follows the same flag,
    so a bare export declares the inputs Mach's bare contract checks.
    """
    adapter = deepseekV41_modulev3_arch.checkpoint_adapters(
        PretrainedConfig(), artifact_export=artifact_export
    )[WeightsFormat.safetensors]
    if artifact_export:
        assert isinstance(adapter, WeightTransform)
    else:
        assert adapter is convert_safetensor_state_dict

    base = _config()
    trunk = DeepseekV41Config(
        **{
            item.name: getattr(base, item.name)
            for item in fields(DeepseekV41Config)
            if item.init
        }
    )
    monkeypatch.setattr(registration, "check_parallel_layout", lambda _: None)
    monkeypatch.setattr(
        DeepseekV41Config,
        "initialize",
        classmethod(lambda cls, *args, **kwargs: trunk),
    )
    hf = hf_config("full")
    pipeline: Any = SimpleNamespace(
        model=SimpleNamespace(huggingface_config=hf),
        artifact_export=artifact_export,
    )
    config = RegisteredDeepseekV41Config.initialize(pipeline, max_seq_len=128)
    assert config.raw_checkpoint_weights is artifact_export
    names = extra_input_names(
        V41Config.from_hf(hf, artifact_export=artifact_export),
        artifact=config.raw_checkpoint_weights,
    )
    if artifact_export:
        assert "engram_token_map" not in names
        assert "replay.kept_skip" in names
    else:
        assert names == _BARE_EXTRAS


@pytest.mark.parametrize("explicit", [None, False, True])
@pytest.mark.parametrize("artifact_export", [False, True])
def test_artifact_export_turns_on_history_and_replay_unless_named(
    explicit: bool | None, artifact_export: bool
) -> None:
    hf: Any = dict(hf_config("f1"))
    text = dict(hf["text_config"])
    if explicit is not None:
        text.update(engram_lookback_history=explicit, decoder_replay=explicit)
    hf["text_config"] = text
    before = dict(text)
    config = V41Config.from_hf(hf, artifact_export=artifact_export)
    expected = artifact_export if explicit is None else explicit
    assert config.engram_lookback_history is expected
    assert config.decoder_replay is expected
    # The checkpoint's config is read, never written.
    assert text == before
