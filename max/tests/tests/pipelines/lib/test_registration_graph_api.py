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
"""Tests serving a graph-API module registered by role."""

from __future__ import annotations

import pickle
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import NonCallableMock

import numpy as np
import pytest
from max import tree
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, TensorType, TensorValue, Weight
from max.graph.weights import WeightsFormat
from max.nn.legacy.graph_groups import RaggedBatch, TPSignals
from max.nn.legacy.kv_cache import KVCacheInputsPerDevice
from max.nn.legacy.layer import Module
from max.nn.legacy.transformer import ReturnLogits
from max.pipelines.context import TextContext
from max.pipelines.lib import GraphRole, SupportedArchitecture, TextTokenizer
from max.pipelines.lib.interfaces.arch_config import ModelOutputConfig
from max.pipelines.lib.interfaces.batch_processor import BatchProcessorRuntime
from max.pipelines.lib.interfaces.pipeline_model import (
    AlwaysSignalBuffersMixin,
)
from max.pipelines.lib.module_registration import ModuleV3DecoderPipelineModel
from max.pipelines.lib.registration_graph_api import (
    GraphApiDecoderBatchProcessor,
    GraphApiDecoderModelInputs,
    GraphApiDecoderPipelineModel,
    _DecoderGraph,
    graph_api_decoder_input_types,
    graph_api_pipeline_model,
    graph_name,
    registered_pipeline_model,
)
from max.pipelines.lib.role_contracts import DecoderInputs, DecoderOutputs
from max.pipelines.modeling.types import PipelineTask
from transformers import AutoConfig
from typing_extensions import Self

CPU_REF = DeviceRef.CPU()
X = TensorType(DType.float32, [2], device=CPU_REF)


@dataclass(frozen=True)
class ToyConfig:
    finalized: bool = False

    @classmethod
    def initialize(
        cls,
        pipeline_config: Any,
        model_config: Any = None,
        *,
        max_seq_len: int,
    ) -> Self:
        return cls()

    def get_max_seq_len(self) -> int:
        return 16

    @classmethod
    def calculate_max_seq_len(
        cls, huggingface_config: AutoConfig, model_config: Any
    ) -> int:
        return 16

    def finalize(
        self,
        huggingface_config: AutoConfig,
        state_dict: Mapping[str, Any],
        outputs: ModelOutputConfig,
    ) -> ToyConfig:
        return replace(self, finalized=True)


class Scale(Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = Weight("weight", DType.float32, [2], CPU_REF)

    def __call__(self, x: TensorValue) -> TensorValue:
        return x * self.weight


def _toy_logits(
    scale: Scale, inputs: DecoderInputs[TensorValue]
) -> TensorValue:
    """Weighs each input differently, so a misplaced input changes it."""
    logits = (
        scale(inputs.tokens)
        + inputs.input_row_offsets * 10
        + inputs.return_n_logits * 100
        + inputs.kv["a"].tensor * 1000
        - inputs.kv["b"].tensor
    )
    return logits


class ToyDecoder(Module):
    graph_name = "toy"

    def __init__(self, config: ToyConfig) -> None:
        super().__init__()
        self.scale = Scale()

    def __call__(
        self, inputs: DecoderInputs[TensorValue]
    ) -> DecoderOutputs[TensorValue]:
        return DecoderOutputs(next_token_logits=_toy_logits(self.scale, inputs))


class TupleOutputsDecoder(Module):
    """Returns its logits as a graph-API text model does: a sequence."""

    def __init__(self, config: ToyConfig) -> None:
        super().__init__()
        self.scale = Scale()

    def __call__(
        self, inputs: DecoderInputs[TensorValue]
    ) -> tuple[TensorValue, ...]:
        return (_toy_logits(self.scale, inputs),)


class SignalDecoder(ToyDecoder):
    always_signal_buffers = True


def _types(*, signals: int = 0) -> dict[str, object]:
    types: dict[str, object] = {
        "batch": RaggedBatch(tokens=X, row_offsets=X, return_n_logits=X)
    }
    if signals:
        types["signals"] = TPSignals(signal=(X,) * signals)
    types["kv"] = {"a": X, "b": X}
    return types


def _run(decoder: Module, input_types: dict[str, object]) -> np.ndarray:
    decoder.load_state_dict({"scale.weight": np.full(2, 2, np.float32)})
    leaves = tree.leaves(input_types)
    graph = _DecoderGraph(decoder, input_types).trace(*leaves)
    model = InferenceSession(devices=[CPU()]).load(
        graph, weights_registry=decoder.state_dict()
    )
    buffers = [
        Buffer.from_numpy(np.full(2, value, np.float32))
        for value in range(1, len(leaves) + 1)
    ]
    (logits,) = model.execute(*buffers)
    assert isinstance(logits, Buffer)
    return logits.to_numpy()


def test_the_graph_feeds_each_field_and_keeps_weight_names() -> None:
    decoder = ToyDecoder(ToyConfig())
    np.testing.assert_array_equal(
        _run(decoder, _types()),
        np.full(2, 1 * 2 + 2 * 10 + 3 * 100 + 4 * 1000 - 5, np.float32),
    )
    # The weights are the decoder's own, not nested under the graph module.
    assert list(decoder.state_dict()) == ["scale.weight"]
    assert graph_name(ToyDecoder) == "toy"


def test_the_graph_rejects_outputs_outside_the_contract() -> None:
    with pytest.raises(TypeError, match="returns DecoderOutputs"):
        _run(TupleOutputsDecoder(ToyConfig()), _types())


def _kv_params() -> NonCallableMock:
    return NonCallableMock(get_symbolic_inputs=lambda: {"a": X, "b": X})


@pytest.mark.parametrize(
    ("signals", "expected"),
    [(False, ["batch", "kv"]), (True, ["batch", "signals", "kv"])],
)
def test_input_types_take_the_canonical_order(
    signals: bool, expected: list[str]
) -> None:
    devices = [CPU_REF, CPU_REF]
    types = graph_api_decoder_input_types(
        _kv_params(), devices, signals=signals
    )
    assert list(types) == expected
    batch = types["batch"]
    assert isinstance(batch, RaggedBatch)
    assert list(batch) == ["tokens", "row_offsets", "return_n_logits"]
    # One copy of the row offsets on any number of devices, as a ModuleV3
    # decoder takes them.
    assert not isinstance(batch["row_offsets"], tuple)


def _context(*tokens: int) -> TextContext:
    active = np.array(tokens, dtype=np.int64)
    return NonCallableMock(
        spec=TextContext,
        tokens=SimpleNamespace(active=active, active_length=len(active)),
    )


def _batch_processor(
    input_types: dict[str, object], signal_buffers: list[Buffer]
) -> GraphApiDecoderBatchProcessor:
    processor = GraphApiDecoderBatchProcessor(
        ToyConfig(),
        BatchProcessorRuntime(
            pipeline_config=NonCallableMock(),
            devices=[CPU(), CPU()],
            return_logits=ReturnLogits.LAST_TOKEN,
            max_batch_size=4,
            max_seq_len=16,
            max_batch_input_tokens=16,
            signal_buffers=signal_buffers,
        ),
    )
    processor.bind_input_tree(OrderedDict(input_types))
    return processor


def test_the_batch_processor_feeds_the_bound_fields_in_order() -> None:
    kv = NonCallableMock(spec=KVCacheInputsPerDevice)
    signal = Buffer.from_numpy(np.zeros(1, np.uint8))
    processor = _batch_processor(
        _types(signals=2),
        [signal, signal],
    )
    inputs = processor.prepare_initial_token_inputs(
        [[_context(1, 2), _context(3)]], kv_cache_inputs=kv
    )

    assert isinstance(inputs, GraphApiDecoderModelInputs)
    np.testing.assert_array_equal(
        inputs.input_row_offsets.to_numpy(), [0, 2, 3]
    )
    assert inputs.buffers == (
        inputs.tokens,
        inputs.input_row_offsets,
        inputs.return_n_logits,
        signal,
        signal,
        kv,
    )


def test_the_batch_processor_omits_fields_the_graph_does_not_take() -> None:
    kv = NonCallableMock(spec=KVCacheInputsPerDevice)
    inputs = _batch_processor(_types(), []).prepare_initial_token_inputs(
        [[_context(1, 2)]], kv_cache_inputs=kv
    )
    assert inputs.buffers == (
        inputs.tokens,
        inputs.input_row_offsets,
        inputs.return_n_logits,
        kv,
    )


def _arch(**kwargs: Any) -> SupportedArchitecture:
    return SupportedArchitecture(
        name="ToyForCausalLM",
        example_repo_ids=[],
        default_encoding="float32",
        supported_encodings={"float32"},
        task=PipelineTask.TEXT_GENERATION,
        tokenizer=TextTokenizer,
        context_type=TextContext,
        default_weights_format=WeightsFormat.safetensors,
        config=ToyConfig,
        **kwargs,
    )


def test_registered_modules_serve_only_text_generation_so_far() -> None:
    with pytest.raises(NotImplementedError, match="embeddings_generation"):
        registered_pipeline_model(
            PipelineTask.EMBEDDINGS_GENERATION,
            ToyConfig,
            ((GraphRole.DECODER, ToyDecoder),),
            None,
        )


def test_a_registration_builds_a_graph_api_pipeline_model() -> None:
    model = _arch(modules={GraphRole.DECODER: ToyDecoder}).pipeline_model

    assert isinstance(model, type)
    assert issubclass(model, GraphApiDecoderPipelineModel)
    assert not issubclass(model, ModuleV3DecoderPipelineModel)
    assert not issubclass(model, AlwaysSignalBuffersMixin)
    assert model.decoder_cls is ToyDecoder
    assert model.model_config_cls is ToyConfig
    assert model.batch_processor_cls is GraphApiDecoderBatchProcessor
    assert _arch(modules={GraphRole.DECODER: ToyDecoder}).pipeline_model is (
        model
    )
    assert pickle.loads(pickle.dumps(model)) is model


def test_a_decoder_that_always_takes_signals_allocates_them() -> None:
    model = _arch(modules={GraphRole.DECODER: SignalDecoder}).pipeline_model
    assert isinstance(model, type)
    assert issubclass(model, AlwaysSignalBuffersMixin)
    assert pickle.loads(pickle.dumps(model)) is model


@pytest.mark.parametrize(
    ("modules", "error", "match"),
    [
        ({}, ValueError, "needs a DECODER"),
        (
            {GraphRole.DECODER: ToyDecoder, GraphRole.VISION: ToyDecoder},
            NotImplementedError,
            "vision",
        ),
        ({GraphRole.DECODER: ToyConfig}, TypeError, "max.nn.legacy.Module"),
    ],
)
def test_a_malformed_registration_fails(
    modules: dict[str, Any], error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        graph_api_pipeline_model(ToyConfig, tuple(modules.items()), None)
