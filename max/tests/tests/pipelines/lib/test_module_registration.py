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
from dataclasses import dataclass, fields, replace
from types import SimpleNamespace
from typing import Any, ClassVar
from unittest.mock import NonCallableMock

import numpy as np
import pytest
from max.driver import CPU, Buffer
from max.dtype import DType
from max.graph import DeviceRef, TensorType
from max.graph.weights import WeightsFormat
from max.nn import Module
from max.nn.legacy.comm.ep import EPConfig
from max.nn.legacy.kv_cache import KVCacheInputsPerDevice
from max.nn.legacy.transformer import ReturnLogits
from max.pipelines.context import TextContext
from max.pipelines.lib import GraphRole, SupportedArchitecture, TextTokenizer
from max.pipelines.lib.interfaces.arch_config import ModelOutputConfig
from max.pipelines.lib.interfaces.batch_processor import BatchProcessorRuntime
from max.pipelines.lib.interfaces.pipeline_model import PipelineModel
from max.pipelines.lib.module_registration import (
    DecoderBatchProcessor,
    DecoderModelInputs,
    ModuleV3DecoderPipelineModel,
    _DecoderGraph,
    consumes_vision,
    decoder_input_types,
    finalize_config,
    graph_name,
    lora_targets,
    module_pipeline_model,
    supports_data_parallel,
)
from max.pipelines.lib.registration_graph_api import registered_pipeline_model
from max.pipelines.lib.registration_runtime import (
    DeploymentConfig,
    ExpertParallelDeploymentConfig,
    ModuleRuntime,
    role_weights,
)
from max.pipelines.lib.role_contracts import (
    DecoderInputs,
    DecoderOutputs,
    VisionInputs,
)
from max.pipelines.lib.vision_encoder_cache import vision_encoding
from max.pipelines.lora import LoRAManagerV3, LoRATargetModule
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


class VisionDecoder(PlainDecoder):
    consumes_vision: ClassVar[bool] = True


class ToyVision(PlainDecoder):
    def input_types(self) -> tuple[TensorType, ...]:
        return (X,)


class ToyVisionBatchProcessor(DecoderBatchProcessor):
    def pack_vision_inputs(self, *args: Any, **kwargs: Any) -> None:
        return None

    def vision_execute(self, *args: Any, **kwargs: Any) -> list[Buffer]:
        return []

    def empty_vision_embeddings(self, devices: Any) -> list[Buffer]:
        return []

    def bind_vision_encoder(self, *, vision_model: Any, session: Any) -> None:
        return None


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


def _toy_input_types(**extra: Any) -> dict[str, object]:
    return {
        "tokens": X,
        "input_row_offsets": X,
        "return_n_logits": X,
        "kv": {"a": X, "b": X},
        **extra,
    }


def test_the_graph_feeds_each_field_and_keeps_weight_names() -> None:
    decoder = ToyDecoder(ToyConfig())
    names = [name for name, _ in decoder.parameters]
    adapter = _DecoderGraph(decoder, _toy_input_types())

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


def test_the_graph_binds_lora_inputs_rather_than_reading_them_as_kv() -> None:
    decoder = ToyDecoder(ToyConfig())
    manager = NonCallableMock(spec=LoRAManagerV3)
    adapter = _DecoderGraph(
        decoder, _toy_input_types(lora=(X, X)), lora_manager=manager
    )

    # The LoRA weight adapter nests the checkpoint's keys under `model.`.
    assert [name for name, _ in adapter.parameters] == ["model.scale.weight"]
    run = adapter.compile(
        X,
        X,
        X,
        X,
        X,
        X,
        X,
        weights={"model.scale.weight": np.full(2, 2, np.float32)},
    )
    tokens, offsets, n_logits, kv_a, kv_b, lora_a, lora_b = (
        Buffer.from_numpy(np.full(2, value, np.float32))
        for value in (1, 2, 3, 4, 5, 6, 7)
    )
    outputs = run(tokens, offsets, n_logits, kv_a, kv_b, lora_a, lora_b)
    np.testing.assert_array_equal(
        np.from_dlpack(outputs.next_token_logits.driver_tensor),
        np.full(2, 1 * 2 + 2 * 10 + 3 * 100 + 4 * 1000 - 5, np.float32),
    )
    (bound_model, lora_inputs), _ = manager.bind_inputs.call_args
    # Compiling traces a copy of the module, so the bound decoder is the copy.
    assert type(bound_model) is ToyDecoder
    assert len(lora_inputs) == 2


def test_decoder_inputs_take_the_canonical_order() -> None:
    assert [field.name for field in fields(DecoderInputs)] == [
        "tokens",
        "vision",
        "input_row_offsets",
        "host_input_row_offsets",
        "return_n_logits",
        "data_parallel_splits",
        "signals",
        "kv",
        "batch_context_lengths",
        "ep",
        "lora",
    ]


def test_optional_decoder_inputs_take_their_canonical_place() -> None:
    kv_params = NonCallableMock(get_symbolic_inputs=lambda: (X,))
    types = decoder_input_types(
        kv_params,
        DeviceRef.CPU(),
        vision=NonCallableMock(spec=VisionInputs),
        batch_context_lengths=(X,),
        ep=NonCallableMock(),
        lora=(X,),
    )
    assert list(types) == [
        "tokens",
        "vision",
        "input_row_offsets",
        "return_n_logits",
        "kv",
        "batch_context_lengths",
        "ep",
        "lora",
    ]


@pytest.mark.parametrize(
    ("data_parallel_degree", "lora", "expected"),
    [
        (1, (), ["tokens", "input_row_offsets", "return_n_logits", "kv"]),
        (
            2,
            (),
            [
                "tokens",
                "input_row_offsets",
                "host_input_row_offsets",
                "return_n_logits",
                "data_parallel_splits",
                "kv",
            ],
        ),
        (
            1,
            (X,),
            ["tokens", "input_row_offsets", "return_n_logits", "kv", "lora"],
        ),
    ],
)
def test_decoder_input_types_follow_the_deployment(
    data_parallel_degree: int, lora: tuple[TensorType, ...], expected: list[str]
) -> None:
    kv_params = NonCallableMock(get_symbolic_inputs=lambda: (X,))
    types = decoder_input_types(
        kv_params,
        DeviceRef.CPU(),
        data_parallel_degree=data_parallel_degree,
        lora=lora,
    )
    assert list(types) == expected
    if data_parallel_degree > 1:
        splits = types["data_parallel_splits"]
        assert isinstance(splits, TensorType)
        assert splits.shape == [data_parallel_degree + 1]


def _context(*tokens: int, position: int = 0) -> TextContext:
    active = np.array(tokens, dtype=np.int64)
    return NonCallableMock(
        spec=TextContext,
        tokens=SimpleNamespace(
            active=active,
            active_length=len(active),
            current_position=position,
        ),
    )


def _batch_processor(
    data_parallel_degree: int, page_size: int = 128
) -> DecoderBatchProcessor:
    pipeline_config = NonCallableMock()
    pipeline_config.model.kv_cache.kv_cache_page_size = page_size
    return DecoderBatchProcessor(
        ToyConfig(),
        BatchProcessorRuntime(
            pipeline_config=pipeline_config,
            devices=[CPU()],
            return_logits=ReturnLogits.LAST_TOKEN,
            max_batch_size=4,
            max_seq_len=16,
            max_batch_input_tokens=16,
            data_parallel_degree=data_parallel_degree,
        ),
    )


def test_the_batch_processor_joins_replicas_and_marks_their_splits() -> None:
    kv = NonCallableMock(spec=KVCacheInputsPerDevice)
    inputs = _batch_processor(2).prepare_initial_token_inputs(
        [[_context(1, 2), _context(3)], [_context(4, 5, 6)]],
        kv_cache_inputs=kv,
    )

    np.testing.assert_array_equal(inputs.tokens.to_numpy(), [1, 2, 3, 4, 5, 6])
    np.testing.assert_array_equal(
        inputs.input_row_offsets.to_numpy(), [0, 2, 3, 6]
    )
    assert inputs.host_input_row_offsets is not None
    np.testing.assert_array_equal(
        inputs.host_input_row_offsets.to_numpy(), [0, 2, 3, 6]
    )
    assert inputs.data_parallel_splits is not None
    np.testing.assert_array_equal(
        inputs.data_parallel_splits.to_numpy(), [0, 2, 3]
    )
    assert inputs.buffers == (
        inputs.tokens,
        inputs.input_row_offsets,
        inputs.host_input_row_offsets,
        inputs.return_n_logits,
        inputs.data_parallel_splits,
        kv,
    )


def test_the_batch_processor_omits_data_parallel_inputs_on_one_replica() -> (
    None
):
    kv = NonCallableMock(spec=KVCacheInputsPerDevice)
    inputs = _batch_processor(1).prepare_initial_token_inputs(
        [[_context(1, 2)]], kv_cache_inputs=kv
    )
    assert isinstance(inputs, DecoderModelInputs)
    assert inputs.host_input_row_offsets is None
    assert inputs.data_parallel_splits is None
    assert inputs.buffers == (
        inputs.tokens,
        inputs.input_row_offsets,
        inputs.return_n_logits,
        kv,
    )


def test_the_batch_processor_fills_the_optional_inputs_its_graph_takes() -> (
    None
):
    kv = NonCallableMock(spec=KVCacheInputsPerDevice)
    processor = _batch_processor(2, page_size=4)
    processor.bind_input_tree(
        decoder_input_types(
            NonCallableMock(get_symbolic_inputs=lambda: (X,)),
            DeviceRef.CPU(),
            data_parallel_degree=2,
            batch_context_lengths=(X, X),
            ep=NonCallableMock(),
        )
    )
    ep_buffer = Buffer.zeros((1,), dtype=DType.int32)
    processor.bind_ep_comm_initializer(
        NonCallableMock(model_inputs=lambda: [ep_buffer])
    )
    inputs = processor.prepare_initial_token_inputs(
        [[_context(1, 2, position=5), _context(3, position=1)], []],
        kv_cache_inputs=kv,
    )
    assert inputs.batch_context_lengths is not None
    assert [b.to_numpy()[0] for b in inputs.batch_context_lengths] == [12, 0]
    assert inputs.ep_inputs == (ep_buffer,)
    assert inputs.buffers[-3:] == (*inputs.batch_context_lengths, ep_buffer)


def test_the_batch_processor_runs_an_empty_batch() -> None:
    inputs = _batch_processor(1).prepare_initial_token_inputs(
        [[]], kv_cache_inputs=NonCallableMock(spec=KVCacheInputsPerDevice)
    )
    assert inputs.tokens.to_numpy().shape == (0,)
    np.testing.assert_array_equal(inputs.input_row_offsets.to_numpy(), [0])


@dataclass(frozen=True)
class SplitCheckpointConfig(ToyConfig, DeploymentConfig):
    weight_prefixes: ClassVar[Mapping[str, str]] = {
        GraphRole.DECODER: "language_model."
    }


@dataclass(frozen=True)
class ShardedExpertsConfig(ToyConfig, ExpertParallelDeploymentConfig):
    ep_dispatch_token_layout: ClassVar[str | None] = "tp_sharded"

    def expert_parallel_config(self, runtime: ModuleRuntime) -> EPConfig:
        raise NotImplementedError


def test_a_role_takes_the_weights_under_its_prefix() -> None:
    state = {"language_model.a": 1, "vision_encoder.b": 2}
    config = SplitCheckpointConfig()
    assert role_weights(state, config, GraphRole.DECODER) == {"a": 1}
    assert role_weights(state, config, GraphRole.VISION) == state
    assert role_weights(state, ToyConfig(), GraphRole.DECODER) == state


def test_a_deployment_config_applies_no_runtime_by_default() -> None:
    config = SplitCheckpointConfig()
    assert config.apply_runtime(NonCallableMock(spec=ModuleRuntime)) is config
    assert not config.takes_batch_context_lengths


def test_one_pipeline_model_serves_text_generation_with_or_without_vision() -> (
    None
):
    model = _arch(
        modules={GraphRole.DECODER: VisionDecoder, GraphRole.VISION: ToyVision},
        batching=ToyVisionBatchProcessor,
    ).pipeline_model
    plain = _arch(modules={GraphRole.DECODER: ToyDecoder}).pipeline_model
    assert isinstance(model, type) and isinstance(plain, type)
    assert issubclass(model, ModuleV3DecoderPipelineModel)
    assert issubclass(plain, ModuleV3DecoderPipelineModel)
    assert model.__mro__[1:] == plain.__mro__[1:]
    assert model.vision_cls is ToyVision
    assert plain.vision_cls is None
    assert pickle.loads(pickle.dumps(model)) is model


def test_registered_modules_serve_only_text_generation() -> None:
    with pytest.raises(NotImplementedError, match="embeddings_generation"):
        registered_pipeline_model(
            PipelineTask.EMBEDDINGS_GENERATION,
            ToyConfig,
            ((GraphRole.DECODER, PlainDecoder),),
            None,
        )


def test_a_pipeline_model_without_vision_encodes_no_images() -> None:
    delegating = NonCallableMock(spec=["vision_encoding"], vision_encoding=None)
    assert vision_encoding(delegating) is None
    assert vision_encoding(object()) is None


def test_a_registration_exposes_its_ep_dispatch_token_layout() -> None:
    model = SupportedArchitecture(
        name="ToyForCausalLM",
        example_repo_ids=[],
        default_encoding="float32",
        supported_encodings={"float32"},
        task=PipelineTask.TEXT_GENERATION,
        tokenizer=TextTokenizer,
        context_type=TextContext,
        default_weights_format=WeightsFormat.safetensors,
        config=ShardedExpertsConfig,
        modules={GraphRole.DECODER: PlainDecoder},
    ).pipeline_model
    assert vars(model)["ep_dispatch_token_layout"] == "tp_sharded"
    plain = _arch(modules={GraphRole.DECODER: PlainDecoder}).pipeline_model
    assert not hasattr(plain, "ep_dispatch_token_layout")


def test_the_batch_processor_needs_a_batch_per_replica() -> None:
    with pytest.raises(ValueError, match="expects 2 replica batches"):
        _batch_processor(2).prepare_initial_token_inputs(
            [[_context(1)]],
            kv_cache_inputs=NonCallableMock(spec=KVCacheInputsPerDevice),
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


def test_a_registration_reads_lora_and_data_parallel_support() -> None:
    targets = (LoRATargetModule("self_attn.o_proj", ("o_proj",)),)

    class LoRADecoder(PlainDecoder):
        lora_targets: ClassVar[tuple[LoRATargetModule, ...]] = targets
        supports_data_parallel: ClassVar[bool] = True

    assert lora_targets(LoRADecoder) == targets
    assert supports_data_parallel(LoRADecoder)
    assert lora_targets(PlainDecoder) == ()
    assert not supports_data_parallel(PlainDecoder)

    model = _arch(modules={GraphRole.DECODER: LoRADecoder}).pipeline_model
    assert isinstance(model, type)
    assert issubclass(model, ModuleV3DecoderPipelineModel)
    assert model.lora_modulev3
    assert model.lora_targets == targets
    plain = _arch(modules={GraphRole.DECODER: PlainDecoder}).pipeline_model
    assert isinstance(plain, type)
    assert issubclass(plain, ModuleV3DecoderPipelineModel)
    assert not plain.lora_modulev3


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
            {GraphRole.DECODER: PlainDecoder, GraphRole.ENCODER: PlainDecoder},
            NotImplementedError,
            "encoder",
        ),
        ({GraphRole.DECODER: ToyConfig}, TypeError, "max.nn.Module"),
        ({GraphRole.DECODER: VisionDecoder}, TypeError, "needs a VISION"),
        (
            {GraphRole.DECODER: VisionDecoder, GraphRole.VISION: PlainDecoder},
            TypeError,
            "needs input_types",
        ),
        (
            {GraphRole.DECODER: PlainDecoder, GraphRole.VISION: ToyVision},
            TypeError,
            "doesn't consume image embeddings",
        ),
        (
            {GraphRole.DECODER: VisionDecoder, GraphRole.VISION: ToyVision},
            TypeError,
            "implements VisionBatchProcessor",
        ),
    ],
)
def test_a_malformed_registration_fails(
    modules: dict[str, Any], error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        module_pipeline_model(ToyConfig, tuple(modules.items()), None)


def test_a_vision_registration_rejects_a_text_batch_processor() -> None:
    modules = {GraphRole.DECODER: VisionDecoder, GraphRole.VISION: ToyVision}
    with pytest.raises(TypeError, match="VisionBatchProcessor"):
        module_pipeline_model(
            ToyConfig, tuple(modules.items()), DecoderBatchProcessor
        )
    assert consumes_vision(VisionDecoder)
    assert not consumes_vision(PlainDecoder)


def test_finalize_config_returns_the_finalized_config() -> None:
    config = ToyConfig()
    finalized = finalize_config(config, _HF_CONFIG, {}, ModelOutputConfig())
    assert isinstance(finalized, ToyConfig)
    assert finalized.finalized
    assert not config.finalized


def test_finalize_config_finalizes_an_in_place_config_on_a_copy() -> None:
    @dataclass
    class InPlace:
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
        ) -> None:
            self.finalized = True

    config = InPlace()
    finalized = finalize_config(config, _HF_CONFIG, {}, ModelOutputConfig())
    assert isinstance(finalized, InPlace)
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
