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
"""Tests serving architectures that register their modules by role."""

from __future__ import annotations

import pickle
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, ClassVar
from unittest.mock import NonCallableMock

import numpy as np
import pytest
from max import tree
from max.driver import CPU, Buffer
from max.dtype import DType
from max.graph import DeviceRef, TensorType
from max.graph.weights import WeightsFormat
from max.nn import Module
from max.pipelines.context import TextContext
from max.pipelines.lib import GraphRole, SupportedArchitecture, TextTokenizer
from max.pipelines.lib.interfaces.arch_config import ModelOutputConfig
from max.pipelines.lib.interfaces.pipeline_model import PipelineModel
from max.pipelines.lib.module_registration import (
    DecoderBatchProcessor,
    ModuleV3DecoderPipelineModel,
    _DecoderGraph,
    finalize_config,
    graph_name,
    module_pipeline_model,
)
from max.pipelines.lib.role_contracts import DecoderInputs, DecoderOutputs
from max.pipelines.modeling.types import PipelineTask
from max.tensor import Tensor
from transformers import AutoConfig
from typing_extensions import Self

X = TensorType(DType.float32, [2], device=DeviceRef.CPU())
_HF_CONFIG = NonCallableMock(spec=AutoConfig)


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


class Scale(Module[[Tensor], Tensor]):
    def __init__(self) -> None:
        super().__init__()
        self.weight = Tensor.zeros([2], dtype=DType.float32, device=CPU())

    def forward(self, x: Tensor) -> Tensor:
        return x * self.weight


class ToyDecoder(Module[[DecoderInputs], DecoderOutputs]):
    """Weighs each input differently, so a misplaced input changes the output."""

    graph_name: ClassVar[str] = "toy"

    def __init__(self, config: ToyConfig) -> None:
        super().__init__()
        self.scale = Scale()

    def forward(self, inputs: DecoderInputs) -> DecoderOutputs:
        logits = (
            self.scale(inputs.tokens)
            + inputs.input_row_offsets * 10
            + inputs.return_n_logits * 100
            + inputs.kv["a"] * 1000
            - inputs.kv["b"]
        )
        return DecoderOutputs(next_token_logits=logits)


class PlainDecoder(Module[[DecoderInputs], DecoderOutputs]):
    def __init__(self, config: ToyConfig) -> None:
        super().__init__()

    def forward(self, inputs: DecoderInputs) -> DecoderOutputs:
        return DecoderOutputs(next_token_logits=inputs.tokens)


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


def test_the_graph_feeds_each_field_and_keeps_weight_names() -> None:
    decoder = ToyDecoder(ToyConfig())
    names = [name for name, _ in decoder.parameters]
    kv_types = {"a": X, "b": X}
    _, kv_tree = tree.flatten(kv_types)
    adapter = _DecoderGraph(decoder, kv_tree)

    assert [name for name, _ in adapter.parameters] == names == ["scale.weight"]

    run = adapter.compile(
        X, X, X, X, X, weights={"scale.weight": np.full(2, 2, np.float32)}
    )
    tokens, offsets, n_logits, kv_a, kv_b = (
        Buffer.from_numpy(np.full(2, value, np.float32))
        for value in (1, 2, 3, 4, 5)
    )
    outputs = run(tokens, offsets, n_logits, kv_a, kv_b)
    np.testing.assert_array_equal(
        np.from_dlpack(outputs.next_token_logits.driver_tensor),
        np.full(2, 1 * 2 + 2 * 10 + 3 * 100 + 4 * 1000 - 5, np.float32),
    )


def test_graph_name_defaults_to_the_class_name() -> None:
    assert graph_name(PlainDecoder) == "PlainDecoder"
    assert graph_name(ToyDecoder) == "toy"


def test_a_registration_builds_its_pipeline_model() -> None:
    arch = _arch(modules={GraphRole.DECODER: ToyDecoder})

    model = arch.pipeline_model
    assert isinstance(model, type)
    assert issubclass(model, ModuleV3DecoderPipelineModel)
    assert model.decoder_cls is ToyDecoder
    assert model.model_config_cls is ToyConfig
    assert model.batch_processor_cls is DecoderBatchProcessor
    # One registration is one class, so re-registering doesn't fork it.
    assert _arch(modules={GraphRole.DECODER: ToyDecoder}).pipeline_model is (
        model
    )


def test_a_registered_pipeline_model_pickles_as_its_registration() -> None:
    model = _arch(modules={GraphRole.DECODER: ToyDecoder}).pipeline_model
    assert pickle.loads(pickle.dumps(model)) is model


def test_a_derived_architecture_keeps_its_own_pipeline_model() -> None:
    arch = _arch(modules={GraphRole.DECODER: ToyDecoder})

    class Derived(PipelineModel[Any]):
        pass

    assert replace(arch, pipeline_model=Derived).pipeline_model is Derived


@pytest.mark.parametrize(
    ("modules", "error", "match"),
    [
        ({}, ValueError, "needs a DECODER"),
        ({GraphRole.SAMPLER: PlainDecoder}, ValueError, "framework builds"),
        (
            {GraphRole.DECODER: PlainDecoder, GraphRole.VISION: PlainDecoder},
            NotImplementedError,
            "vision",
        ),
        ({GraphRole.DECODER: ToyConfig}, TypeError, "max.nn.Module"),
    ],
)
def test_a_malformed_registration_fails(
    modules: dict[str, Any], error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        module_pipeline_model(ToyConfig, tuple(modules.items()), None)


def test_finalize_config_returns_the_finalized_config() -> None:
    config = ToyConfig()
    finalized = finalize_config(config, _HF_CONFIG, {}, ModelOutputConfig())
    assert isinstance(finalized, ToyConfig)
    assert finalized.finalized
    assert not config.finalized


def test_finalize_config_rejects_another_config_type() -> None:
    @dataclass(frozen=True)
    class Wrong(ToyConfig):
        def finalize(
            self,
            huggingface_config: AutoConfig,
            state_dict: Mapping[str, Any],
            outputs: ModelOutputConfig,
        ) -> Any:
            return object()

    with pytest.raises(TypeError, match="not the config"):
        finalize_config(Wrong(), _HF_CONFIG, {}, ModelOutputConfig())
