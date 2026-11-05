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
"""Tests serving embedding architectures that register an ``ENCODER``."""

from __future__ import annotations

import pickle
from collections.abc import Mapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest.mock import NonCallableMock

import numpy as np
import pytest
from max import functional as F
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, TensorType, TensorValue
from max.graph.weights import WeightsFormat
from max.nn import Module as ModuleV3
from max.nn.legacy.layer import Module
from max.nn.outputs import ReturnLogits
from max.pipelines.context import TextContext
from max.pipelines.lib import GraphRole, SupportedArchitecture, TextTokenizer
from max.pipelines.lib.graph_specs import GraphSpec, compile_and_init_graphs
from max.pipelines.lib.interfaces.arch_config import ModelOutputConfig
from max.pipelines.lib.interfaces.batch_processor import BatchProcessorRuntime
from max.pipelines.lib.registration_encoder import (
    EncoderInputs,
    GraphApiEncoderPipelineModel,
    ModuleV3EncoderPipelineModel,
    PaddedEncoderInputs,
    PaddedEncoderProcessor,
    RaggedEncoderInputs,
    RaggedEncoderProcessor,
    _EncoderGraph,
    _ModuleV3EncoderGraph,
    encoder_input_types,
    encoder_pipeline_model,
)
from max.pipelines.modeling.types import PipelineTask
from max.tensor import Tensor
from transformers import AutoConfig
from typing_extensions import Self

CPU_REF = DeviceRef.CPU()


@dataclass(frozen=True)
class ToyConfig:
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
    ) -> None:
        return None


class PaddedEncoder(Module):
    graph_name = "padded"

    def __init__(self, config: ToyConfig) -> None:
        super().__init__()

    def __call__(self, inputs: EncoderInputs[TensorValue]) -> TensorValue:
        assert inputs.attention_mask is not None
        return inputs.tokens.cast(DType.float32) * inputs.attention_mask


class RaggedEncoder(Module):
    ragged_batch = True

    def __init__(self, config: ToyConfig) -> None:
        super().__init__()

    def __call__(self, inputs: EncoderInputs[TensorValue]) -> TensorValue:
        assert inputs.input_row_offsets is not None
        assert inputs.return_n_logits is not None
        return inputs.tokens


class Scale(ModuleV3[[Tensor], Tensor]):
    def __init__(self) -> None:
        super().__init__()
        self.weight = Tensor.zeros([1], dtype=DType.float32, device=CPU())

    def forward(self, x: Tensor) -> Tensor:
        return x * self.weight


class ModuleV3PaddedEncoder(ModuleV3[[EncoderInputs], Tensor]):
    graph_name = "padded_v3"

    def __init__(self, config: ToyConfig) -> None:
        super().__init__()
        self.scale = Scale()

    def forward(self, inputs: EncoderInputs) -> Tensor:
        assert inputs.attention_mask is not None
        return self.scale(F.cast(inputs.tokens, DType.float32)) + (
            inputs.attention_mask * 100
        )


class ModuleV3RaggedEncoder(ModuleV3[[EncoderInputs], Tensor]):
    ragged_batch = True

    def __init__(self, config: ToyConfig) -> None:
        super().__init__()

    def forward(self, inputs: EncoderInputs) -> Tensor:
        return inputs.tokens


def _arch(encoder: type[Any]) -> SupportedArchitecture:
    return SupportedArchitecture(
        name="ToyModel",
        example_repo_ids=[],
        default_encoding="float32",
        supported_encodings={"float32"},
        task=PipelineTask.EMBEDDINGS_GENERATION,
        tokenizer=TextTokenizer,
        context_type=TextContext,
        default_weights_format=WeightsFormat.safetensors,
        config=ToyConfig,
        modules={GraphRole.ENCODER: encoder},
    )


def _context(*tokens: int) -> TextContext:
    active = np.array(tokens, dtype=np.int64)
    return NonCallableMock(
        spec=TextContext,
        tokens=SimpleNamespace(active=active, active_length=len(active)),
    )


def _runtime() -> BatchProcessorRuntime:
    return BatchProcessorRuntime(
        pipeline_config=NonCallableMock(),
        devices=[CPU()],
        return_logits=ReturnLogits.ALL,
        max_batch_size=4,
        max_seq_len=16,
        max_batch_input_tokens=16,
    )


def test_an_encoder_picks_its_batch_layout() -> None:
    padded = encoder_input_types(PaddedEncoder, [CPU_REF])
    assert list(padded) == ["tokens", "attention_mask"]
    assert padded["tokens"] == TensorType(
        DType.int64, ["batch_size", "seq_len"], device=CPU_REF
    )
    assert padded["attention_mask"].dtype == DType.float32

    ragged = encoder_input_types(RaggedEncoder, [CPU_REF])
    assert list(ragged) == ["tokens", "input_row_offsets", "return_n_logits"]


def test_the_encoder_graph_feeds_each_field() -> None:
    input_types = encoder_input_types(PaddedEncoder, [CPU_REF])
    compiled = compile_and_init_graphs(
        InferenceSession(devices=[CPU()]),
        {
            GraphRole.ENCODER: GraphSpec(
                _EncoderGraph(PaddedEncoder(ToyConfig()), list(input_types)),
                "padded",
                tuple(input_types.values()),
            )
        },
    )
    (embeddings,) = compiled[GraphRole.ENCODER].engine_model.execute(
        Buffer.from_numpy(np.array([[1, 2, 3]], np.int64)),
        Buffer.from_numpy(np.array([[1, 1, 0]], np.float32)),
    )
    assert isinstance(embeddings, Buffer)
    np.testing.assert_array_equal(embeddings.to_numpy(), [[1, 2, 0]])


def test_a_modulev3_encoder_graph_feeds_each_field() -> None:
    encoder = ModuleV3PaddedEncoder(ToyConfig())
    input_types = encoder_input_types(ModuleV3PaddedEncoder, [CPU_REF])
    graph = _ModuleV3EncoderGraph(encoder, list(input_types))
    assert [name for name, _ in graph.parameters] == ["scale.weight"]

    run = graph.compile(
        *input_types.values(),
        weights={"scale.weight": np.array([2], np.float32)},
    )
    embeddings = run(
        Buffer.from_numpy(np.array([[1, 2, 3]], np.int64)),
        Buffer.from_numpy(np.array([[1, 1, 0]], np.float32)),
    )
    np.testing.assert_array_equal(
        np.from_dlpack(embeddings.driver_tensor), [[102, 104, 6]]
    )


def test_the_padded_processor_pads_and_masks_the_batch() -> None:
    processor = PaddedEncoderProcessor(ToyConfig(), _runtime())
    inputs = processor.prepare_initial_token_inputs(
        [[_context(5, 6, 7), _context(8)]]
    )
    assert isinstance(inputs, PaddedEncoderInputs)
    assert inputs.buffers == (inputs.tokens, inputs.attention_mask)
    np.testing.assert_array_equal(
        inputs.attention_mask.to_numpy()[1], [1, 0, 0]
    )


def test_the_ragged_processor_packs_the_batch() -> None:
    processor = RaggedEncoderProcessor(ToyConfig(), _runtime())
    inputs = processor.prepare_initial_token_inputs(
        [[_context(5, 6), _context(7)]]
    )
    assert isinstance(inputs, RaggedEncoderInputs)
    assert inputs.buffers == (
        inputs.tokens,
        inputs.input_row_offsets,
        inputs.return_n_logits,
    )
    np.testing.assert_array_equal(
        inputs.input_row_offsets.to_numpy(), [0, 2, 3]
    )


@pytest.mark.parametrize(
    ("encoder", "processor"),
    [
        (PaddedEncoder, PaddedEncoderProcessor),
        (RaggedEncoder, RaggedEncoderProcessor),
    ],
)
def test_a_registration_builds_an_encoder_pipeline_model(
    encoder: type[Module], processor: type[Any]
) -> None:
    model = _arch(encoder).pipeline_model
    assert isinstance(model, type)
    assert issubclass(model, GraphApiEncoderPipelineModel)
    assert model.encoder_cls is encoder
    assert model.model_config_cls is ToyConfig
    assert model.batch_processor_cls is processor
    assert _arch(encoder).pipeline_model is model
    assert pickle.loads(pickle.dumps(model)) is model


@pytest.mark.parametrize(
    ("encoder", "processor"),
    [
        (ModuleV3PaddedEncoder, PaddedEncoderProcessor),
        (ModuleV3RaggedEncoder, RaggedEncoderProcessor),
    ],
)
def test_a_modulev3_registration_builds_a_modulev3_encoder_pipeline_model(
    encoder: type[ModuleV3[..., Any]], processor: type[Any]
) -> None:
    model = _arch(encoder).pipeline_model
    assert isinstance(model, type)
    assert issubclass(model, ModuleV3EncoderPipelineModel)
    assert model.encoder_cls is encoder
    assert model.batch_processor_cls is processor
    assert _arch(encoder).pipeline_model is model
    assert pickle.loads(pickle.dumps(model)) is model


def test_an_encoder_registration_fills_only_the_encoder() -> None:
    with pytest.raises(NotImplementedError, match="decoder"):
        encoder_pipeline_model(
            ToyConfig,
            (
                (GraphRole.ENCODER, PaddedEncoder),
                (GraphRole.DECODER, PaddedEncoder),
            ),
            None,
        )
    with pytest.raises(TypeError, match=r"max\.nn\.Module"):
        encoder_pipeline_model(
            ToyConfig, ((GraphRole.ENCODER, ToyConfig),), None
        )
