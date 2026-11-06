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
from max import tree
from max._artifact.value_types import treedef_to_json
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import BufferType, DeviceRef, TensorType
from max.graph.weights import WeightData, WeightsFormat
from max.nn import Module
from max.nn.ep_config import EPConfig
from max.nn.kv_cache import KVCacheInputsPerDevice
from max.nn.legacy.graph_groups import DecoderGraphInputs, VisionMerge
from max.nn.outputs import ReturnHiddenStates, ReturnLogits
from max.pipelines.context import TextContext
from max.pipelines.lib import (
    GraphRole,
    SupportedArchitecture,
    TextTokenizer,
    WeightRenaming,
    WeightTransform,
)
from max.pipelines.lib.decoder_scratch import build_decoder_scratch
from max.pipelines.lib.interfaces.arch_config import ModelOutputConfig
from max.pipelines.lib.interfaces.batch_processor import BatchProcessorRuntime
from max.pipelines.lib.interfaces.pipeline_model import PipelineModel
from max.pipelines.lib.module_registration import (
    DecoderBatchProcessor,
    DecoderModelInputs,
    ModuleV3DecoderPipelineModel,
    RegisteredDecoderBuilder,
    _DecoderGraph,
    _registered_weight_naming,
    bind_decoder_scratch,
    consumes_vision,
    decoder_input_types,
    finalize_config,
    graph_name,
    lora_targets,
    module_pipeline_model,
    supports_data_parallel,
    supports_multi_device_replicas,
    supports_pipeline_stages,
)
from max.pipelines.lib.pipeline_parallel import PipelinedModel, StageTraced
from max.pipelines.lib.pipeline_stage import PipelineStage
from max.pipelines.lib.registration_graph_api import registered_pipeline_model
from max.pipelines.lib.registration_runtime import (
    ConstantInputs,
    DeploymentConfig,
    ExpertParallelDeploymentConfig,
    ExpertParallelInputs,
    ModuleRuntime,
    batch_context_length_types,
    ep_input_tree,
    prepare_weights,
    role_weights,
)
from max.pipelines.lib.role_contracts import (
    DecoderInputs,
    DecoderOutputs,
    VisionInputs,
)
from max.pipelines.lib.vision_encoder_cache import (
    SupportsPooledVisionMetrics,
    VideoEncoderMetrics,
    vision_encoding,
)
from max.pipelines.lora import LoRAManagerV3, LoRATargetModule
from max.pipelines.modeling.types import PipelineTask
from max.tensor import Tensor, defaults
from transformers import AutoConfig
from typing_extensions import Self

X = TensorType(DType.float32, [2], device=DeviceRef.CPU())
_KV_BLOCKS = BufferType(DType.float32, [2], device=DeviceRef.CPU())
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


class StatefulDecoder(PlainDecoder):
    @classmethod
    def state_store(cls, *args: Any, **kwargs: Any) -> None:
        return None


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


def _toy_input_types(**extra: Any) -> DecoderGraphInputs:
    return DecoderGraphInputs(
        tokens=X,
        input_row_offsets=X,
        return_n_logits=X,
        kv={"a": X, "b": X},
        **extra,
    )


def test_the_graph_feeds_each_field_and_keeps_weight_names() -> None:
    decoder = ToyDecoder(ToyConfig())
    names = [name for name, _ in decoder.parameters]
    adapter = _DecoderGraph(decoder)

    assert [name for name, _ in adapter.parameters] == names == ["scale.weight"]

    run = adapter.compile(
        _toy_input_types(), weights={"scale.weight": np.full(2, 2, np.float32)}
    )
    tokens, offsets, n_logits, kv_a, kv_b = (
        Buffer.from_numpy(np.full(2, value, np.float32))
        for value in (1, 2, 3, 4, 5)
    )
    (logits,) = run.execute_raw(tokens, offsets, n_logits, kv_a, kv_b)
    np.testing.assert_array_equal(
        np.from_dlpack(logits),
        np.full(2, 1 * 2 + 2 * 10 + 3 * 100 + 4 * 1000 - 5, np.float32),
    )


def test_the_graph_binds_lora_inputs_rather_than_reading_them_as_kv() -> None:
    decoder = ToyDecoder(ToyConfig())
    manager = NonCallableMock(spec=LoRAManagerV3)
    adapter = _DecoderGraph(decoder, lora_manager=manager)

    # The LoRA weight adapter nests the checkpoint's keys under `model.`.
    assert [name for name, _ in adapter.parameters] == ["model.scale.weight"]
    run = adapter.compile(
        _toy_input_types(lora=(X, X)),
        weights={"model.scale.weight": np.full(2, 2, np.float32)},
    )
    tokens, offsets, n_logits, kv_a, kv_b, lora_a, lora_b = (
        Buffer.from_numpy(np.full(2, value, np.float32))
        for value in (1, 2, 3, 4, 5, 6, 7)
    )
    (logits,) = run.execute_raw(
        tokens, offsets, n_logits, kv_a, kv_b, lora_a, lora_b
    )
    np.testing.assert_array_equal(
        np.from_dlpack(logits),
        np.full(2, 1 * 2 + 2 * 10 + 3 * 100 + 4 * 1000 - 5, np.float32),
    )
    (bound_model, lora_inputs), _ = manager.bind_inputs.call_args
    # Compiling traces a copy of the module, so the bound decoder is the copy.
    assert type(bound_model) is ToyDecoder
    assert len(lora_inputs) == 2


class EchoDecoder(PlainDecoder):
    """Returns its inputs, so each output shows which input fed it."""

    def forward(self, inputs: DecoderInputs) -> DecoderOutputs:
        assert inputs.vision is not None
        return DecoderOutputs(
            next_token_logits=inputs.tokens,
            logits=inputs.vision.embeddings,
            logit_offsets=inputs.input_row_offsets,
            hidden_states=inputs.vision.scatter_indices,
            state=inputs.state,
        )


def test_the_graph_names_its_inputs_and_outputs_in_buffer_order() -> None:
    types = decoder_input_types(
        NonCallableMock(
            get_symbolic_inputs=lambda: (
                KVCacheInputsPerDevice(_KV_BLOCKS, X, X, X, X, X),
            )
        ),
        DeviceRef.CPU(),
        vision=VisionInputs(embeddings=X, scatter_indices=X),
        state=(X,),
    )
    run = _DecoderGraph(
        EchoDecoder(ToyConfig()), return_hidden_states=ReturnHiddenStates.LAST
    ).compile(types)

    # An exported graph records both trees, which name every buffer.
    treedef_to_json(run.in_tree)
    (root,) = run.in_tree.children[0].children
    assert root.meta == (DecoderGraphInputs, None)
    assert root.child_keys == (
        "tokens",
        "vision",
        "input_row_offsets",
        "return_n_logits",
        "kv",
        "state",
    )
    vision = root.children[1]
    assert vision.meta == (VisionMerge, None)
    assert vision.child_keys == ("image_embeddings", "image_token_indices")
    leaf = {"kind": "leaf"}
    assert treedef_to_json(run.out_tree) == {
        "kind": "tuple",
        "children": [
            {
                "kind": "node",
                "type": "decoder_outputs",
                "keys": ["logits", "hidden_states"],
                "children": [
                    {
                        "kind": "node",
                        "type": "logits",
                        "keys": [
                            "next_token_logits",
                            "logits",
                            "logit_offsets",
                        ],
                        "children": [leaf, leaf, leaf],
                    },
                    {
                        "kind": "node",
                        "type": "hidden_states",
                        "keys": ["last"],
                        "children": [{"kind": "tuple", "children": [leaf]}],
                    },
                ],
            },
            {
                "kind": "node",
                "type": "request_state",
                "keys": ["state"],
                "children": [{"kind": "tuple", "children": [leaf]}],
            },
        ],
    }

    # The batch's buffers and the graph's outputs keep the role's order.
    tokens = Buffer.from_numpy(np.array([0], np.int64))
    offsets = Buffer.from_numpy(np.array([3, 4], np.uint32))
    n_logits = Buffer.from_numpy(np.array([4], np.int64))
    embeddings, indices, kv_buffer, state = (
        Buffer.from_numpy(np.full(2, value, np.float32))
        for value in (1, 2, 5, 6)
    )
    kv = KVCacheInputsPerDevice(
        kv_buffer, kv_buffer, kv_buffer, kv_buffer, kv_buffer, kv_buffer
    )
    processor = _batch_processor(1)
    processor.bind_input_tree(types)
    inputs = processor.prepare_initial_token_inputs(
        [[_context(1)]], kv_cache_inputs=(kv,)
    )
    inputs.tokens, inputs.input_row_offsets = tokens, offsets
    inputs.return_n_logits, inputs.state = n_logits, (state,)
    inputs.vision_embeddings = [embeddings]
    inputs.vision_scatter_indices = [indices]
    assert inputs.buffers == (
        tokens,
        embeddings,
        indices,
        offsets,
        n_logits,
        *tree.leaves(kv),
        state,
    )
    assert [
        int(np.from_dlpack(output)[0])
        for output in run.execute_raw(*inputs.buffers)
    ] == [0, 1, 3, 2, 6]


def test_decoder_inputs_take_the_canonical_order() -> None:
    assert [field.name for field in fields(DecoderInputs)] == [
        "tokens",
        "vision",
        "input_row_offsets",
        "token_positions",
        "host_input_row_offsets",
        "return_n_logits",
        "data_parallel_splits",
        "signals",
        "kv",
        "batch_context_lengths",
        "ep",
        "lora",
        "constants",
        "state",
        "scratch",
        "stage_boundary",
    ]


def test_optional_decoder_inputs_take_their_canonical_place() -> None:
    kv_params = NonCallableMock(get_symbolic_inputs=lambda: (X,))
    types = decoder_input_types(
        kv_params,
        DeviceRef.CPU(),
        vision=VisionInputs(embeddings=X, scatter_indices=X),
        batch_context_lengths=(X,),
        ep=NonCallableMock(),
        lora=(X,),
        state=(X,),
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
        "state",
    ]


def test_a_decoder_keeps_its_existing_input_layout() -> None:
    layout = (
        "tokens",
        "return_n_logits",
        "input_row_offsets",
        "batch_context_lengths",
        "kv",
    )
    types = decoder_input_types(
        NonCallableMock(
            get_symbolic_inputs=lambda: (
                KVCacheInputsPerDevice(_KV_BLOCKS, X, X, X, X, X),
            )
        ),
        DeviceRef.CPU(),
        batch_context_lengths=(X,),
        input_layout=layout,
    )
    assert tuple(types) == layout
    processor = _batch_processor(1)
    processor.bind_input_tree(types)
    assert processor.get_symbolic_inputs(
        kv_params=NonCallableMock(), device_refs=[DeviceRef.CPU()]
    ) == [
        leaf
        for field_types in types.values()
        for leaf in tree.leaves(field_types)
    ]
    kv_buffer = Buffer.from_numpy(np.zeros(1, np.float32))
    kv = KVCacheInputsPerDevice(
        kv_buffer, kv_buffer, kv_buffer, kv_buffer, kv_buffer, kv_buffer
    )
    inputs = processor.prepare_initial_token_inputs(
        [[_context(1, position=1)]], kv_cache_inputs=(kv,)
    )
    assert inputs.batch_context_lengths is not None
    assert inputs.buffers == (
        inputs.tokens,
        inputs.return_n_logits,
        inputs.input_row_offsets,
        *inputs.batch_context_lengths,
        *tree.leaves(kv),
    )


@pytest.mark.parametrize(
    ("layout", "error"),
    [
        (
            ("tokens", "tokens", "input_row_offsets", "return_n_logits", "kv"),
            "duplicate",
        ),
        (
            ("tokens", "input_row_offsets", "return_n_logits", "other", "kv"),
            "unknown",
        ),
        (("tokens", "input_row_offsets", "return_n_logits"), "missing"),
    ],
)
def test_input_layout_rejects_invalid_fields(
    layout: tuple[str, ...], error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        decoder_input_types(
            NonCallableMock(get_symbolic_inputs=lambda: (X,)),
            DeviceRef.CPU(),
            input_layout=layout,
        )


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
        kv_cache_inputs=(kv,),
    )
    assert inputs.batch_context_lengths is not None
    assert [b.to_numpy()[0] for b in inputs.batch_context_lengths] == [12, 0]
    assert inputs.ep_inputs == (ep_buffer,)
    assert inputs.buffers[-3:] == (*inputs.batch_context_lengths, ep_buffer)


def test_the_batch_processor_fills_each_devices_context_length() -> None:
    lengths = batch_context_length_types(
        2, devices_per_replica=2, dtype=DType.uint32
    )
    assert (
        lengths == (TensorType(DType.uint32, [1], device=DeviceRef.CPU()),) * 4
    )
    processor = _batch_processor(2, page_size=4)
    processor.bind_input_tree(
        decoder_input_types(
            NonCallableMock(get_symbolic_inputs=lambda: (X,)),
            DeviceRef.CPU(),
            data_parallel_degree=2,
            batch_context_lengths=lengths,
        )
    )
    inputs = processor.prepare_initial_token_inputs(
        [[_context(1, 2, position=5), _context(3, position=1)], [_context(4)]],
        kv_cache_inputs=(NonCallableMock(spec=KVCacheInputsPerDevice),),
    )
    assert inputs.batch_context_lengths is not None
    assert [b.dtype for b in inputs.batch_context_lengths] == [DType.uint32] * 4
    # Each replica's length, on each of its two devices.
    assert [b.to_numpy()[0] for b in inputs.batch_context_lengths] == [
        12,
        12,
        0,
        0,
    ]


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


@dataclass(frozen=True)
class SplitVisionCheckpointConfig(ToyConfig, DeploymentConfig):
    weight_prefixes: ClassVar[Mapping[str, str]] = {
        GraphRole.DECODER: "language_model.",
        GraphRole.VISION: "vision_encoder.",
    }


@dataclass(frozen=True)
class PreparedCheckpointConfig(SplitVisionCheckpointConfig):
    def prepare_weights(
        self, state_dict: Mapping[str, WeightData]
    ) -> dict[str, WeightData]:
        return dict(state_dict)


def test_rules_name_each_role_as_it_takes_its_weights() -> None:
    modules = (
        (GraphRole.DECODER, VisionDecoder),
        (GraphRole.VISION, ToyVision),
    )
    rules = WeightTransform([WeightRenaming(r"^lm\.", "language_model.")])
    checkpoint = {
        name: WeightData.from_numpy(np.zeros(1, np.float32), name)
        for name in ("lm.a", "vision_encoder.b")
    }
    cls = module_pipeline_model(
        SplitVisionCheckpointConfig, modules, ToyVisionBatchProcessor
    )
    naming = _registered_weight_naming(cls, rules)
    adapted = rules.apply_data(checkpoint)
    for role in (GraphRole.DECODER, GraphRole.VISION):
        (entry,) = naming[role]
        named = entry.transform.apply_data(checkpoint)
        taken = role_weights(adapted, SplitVisionCheckpointConfig(), role)
        assert taken and {name: named[name] for name in taken} == taken
    # Speculative decoding builds the modules alone, and names them the same.
    builder = RegisteredDecoderBuilder(cls)
    context: Any = SimpleNamespace(adapter=rules)
    assert builder.weight_entries(context, None) == [*naming[GraphRole.DECODER]]
    assert builder.vision_weight_entries(context) == [*naming[GraphRole.VISION]]

    prepared = module_pipeline_model(
        PreparedCheckpointConfig, modules, ToyVisionBatchProcessor
    )
    assert _registered_weight_naming(prepared, rules) == {}


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


def test_registered_modules_serve_no_pixel_generation() -> None:
    with pytest.raises(NotImplementedError, match="pixel_generation"):
        registered_pipeline_model(
            PipelineTask.PIXEL_GENERATION,
            ToyConfig,
            ((GraphRole.DECODER, PlainDecoder),),
            None,
        )


def test_a_pipeline_model_without_vision_encodes_no_images() -> None:
    delegating = NonCallableMock(spec=["vision_encoding"], vision_encoding=None)
    assert vision_encoding(delegating) is None
    assert vision_encoding(object()) is None


class VideoCountingBatchProcessor(ToyVisionBatchProcessor):
    def pop_vision_metrics(self) -> None:
        return None

    def pop_video_metrics(self) -> VideoEncoderMetrics:
        return VideoEncoderMetrics(num_clips_total=3)


def test_a_pipeline_model_reports_its_batch_processors_video_metrics() -> None:
    cls = module_pipeline_model(
        ToyConfig,
        ((GraphRole.DECODER, VisionDecoder), (GraphRole.VISION, ToyVision)),
        VideoCountingBatchProcessor,
    )
    model = cls.__new__(cls)
    assert isinstance(model, SupportsPooledVisionMetrics)
    model._batch_processor = NonCallableMock(spec=DecoderBatchProcessor)
    assert model.pop_video_metrics() is None
    assert model.pop_vision_metrics() is None
    model._batch_processor = VideoCountingBatchProcessor.__new__(
        VideoCountingBatchProcessor
    )
    assert model.pop_video_metrics() == VideoEncoderMetrics(num_clips_total=3)


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


class DataParallelDecoder(PlainDecoder):
    supports_data_parallel: ClassVar[bool] = True


class SharedReplicaDecoder(DataParallelDecoder):
    supports_multi_device_replicas: ClassVar[bool] = True


def _data_parallel_model(
    decoder: type[Module[..., Any]], replicas: int, devices: int
) -> Any:
    cls = module_pipeline_model(
        ToyConfig, ((GraphRole.DECODER, decoder),), None
    )
    model = cls.__new__(cls)
    pipeline_config: Any = SimpleNamespace(
        model=SimpleNamespace(
            data_parallel_degree=replicas, pipeline_stage=None
        )
    )
    model.pipeline_config = pipeline_config
    model.devices = [CPU()] * devices
    return model


def test_replicas_span_devices_only_where_the_decoder_supports_it() -> None:
    assert supports_multi_device_replicas(SharedReplicaDecoder)
    assert not supports_multi_device_replicas(DataParallelDecoder)
    with pytest.raises(ValueError, match="one replica per device"):
        _data_parallel_model(DataParallelDecoder, 2, 4)._instantiate_module(
            ToyConfig()
        )
    with pytest.raises(ValueError, match="one replica per device"):
        _data_parallel_model(SharedReplicaDecoder, 2, 3)._instantiate_module(
            ToyConfig()
        )
    shared = _data_parallel_model(SharedReplicaDecoder, 2, 4)
    assert isinstance(shared._instantiate_module(ToyConfig()), Module)
    mesh = shared._device_mesh()
    assert mesh.mesh_shape == (2, 2)
    assert mesh.axis_names == ("dp", "tp")
    # One replica per device keeps the one-axis mesh.
    assert _data_parallel_model(
        SharedReplicaDecoder, 4, 4
    )._device_mesh().mesh_shape == (4,)


class _TakesContextLengths(DeploymentConfig):
    takes_batch_context_lengths: ClassVar[bool] = True


def test_multi_device_replicas_take_a_context_length_per_device() -> None:
    config = _TakesContextLengths()
    # Two replicas of four: one int32 per device, each its replica's.
    shared = _data_parallel_model(
        SharedReplicaDecoder, 2, 8
    )._batch_context_length_types(config)
    assert shared is not None
    assert len(shared) == 8
    assert {(t.dtype, t.device) for t in shared} == {
        (DType.int32, DeviceRef.CPU())
    }
    # One replica per device: one per replica.
    per_replica = _data_parallel_model(
        SharedReplicaDecoder, 4, 4
    )._batch_context_length_types(config)
    assert per_replica is not None
    assert len(per_replica) == 4


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


class _TailInputs(ExpertParallelInputs):
    def input_tree(self, ep: EPConfig) -> object:
        return ("tail", ep.n_experts)


@dataclass(frozen=True)
class _TailConfig(ShardedExpertsConfig):
    expert_parallel_inputs: ClassVar[ExpertParallelInputs] = _TailInputs()


def test_a_decoder_can_take_the_exchange_in_its_own_layout() -> None:
    ep = NonCallableMock(spec=EPConfig, n_experts=32)
    assert type(ShardedExpertsConfig.expert_parallel_inputs) is (
        ExpertParallelInputs
    )
    assert ep_input_tree(_TailConfig(), ep) == ("tail", 32)


_EP = NonCallableMock(spec=EPConfig)


@dataclass(frozen=True)
class TargetConfig(ToyConfig, ExpertParallelDeploymentConfig):
    """Records what the target builder completes it with."""

    kv_params: object = None
    dtype: DType = DType.float32
    outputs: ModelOutputConfig | None = None
    runtime: ModuleRuntime | None = None

    def finalize(
        self,
        huggingface_config: AutoConfig,
        state_dict: Mapping[str, Any],
        outputs: ModelOutputConfig,
    ) -> TargetConfig:
        return replace(self, finalized=True, outputs=outputs)

    def expert_parallel_config(self, runtime: ModuleRuntime) -> EPConfig:
        return _EP

    def apply_runtime(self, runtime: ModuleRuntime) -> Self:
        return replace(self, runtime=runtime)


class MeshDecoder(ToyDecoder):
    """Records the config and mesh it was constructed with."""

    def __init__(self, config: ToyConfig) -> None:
        super().__init__(config)
        self.config = config
        self.mesh = defaults()[1]


def _target_context(started: list[EPConfig], *, ep_size: int) -> Any:
    return SimpleNamespace(
        pipeline_config=SimpleNamespace(
            model=SimpleNamespace(data_parallel_degree=1),
            runtime=SimpleNamespace(
                max_batch_input_tokens=8,
                ep_size=ep_size,
                ep_use_allreduce=False,
            ),
        ),
        weights=object(),
        adapter=None,
        kv_params=SimpleNamespace(devices=[DeviceRef.CPU()]),
        max_seq_len=16,
        max_batch_total_tokens=32,
        memory_plan=SimpleNamespace(planned_max_batch_size=5),
        huggingface_config=_HF_CONFIG,
        return_logits=ReturnLogits.VARIABLE,
        return_hidden_states=ReturnHiddenStates.ALL_NORMALIZED,
        target_layer_ids=(3, 7),
        start_ep=started.append,
    )


def test_a_composed_decoder_completes_its_config_as_serving_does(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    weight = np.zeros(2, dtype=np.float32)
    monkeypatch.setattr(
        "max.pipelines.lib.module_registration.parse_state_dict_from_weights",
        lambda *args, **kwargs: {"scale.weight": weight},
    )
    model = module_pipeline_model(
        TargetConfig, ((GraphRole.DECODER, MeshDecoder),), None
    )
    started: list[EPConfig] = []
    context = _target_context(started, ep_size=2)
    builder = RegisteredDecoderBuilder(model)
    module, config = builder.build_module(context)
    assert isinstance(module, MeshDecoder) and module.config is config
    assert module.mesh.axis_names == ("tp",)
    assert module.mesh.devices == (CPU(),)
    assert isinstance(config, TargetConfig) and config.finalized
    assert config.kv_params is context.kv_params
    assert config.outputs == ModelOutputConfig(
        return_logits=ReturnLogits.VARIABLE,
        return_hidden_states=ReturnHiddenStates.ALL_NORMALIZED,
        target_layer_ids=(3, 7),
    )
    # Expert parallelism starts through the caller, which shares it.
    assert started == [_EP]
    assert config.runtime is not None
    assert config.runtime.ep is _EP
    assert config.runtime.max_batch_total_tokens == 32
    assert config.runtime.max_batch_size == 5
    assert builder.checkpoint_weights(context, config) == {
        "scale.weight": weight
    }
    assert builder.checkpoint_weights(context, config) == {}


def test_a_composed_decoder_takes_no_expert_parallelism_unasked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "max.pipelines.lib.module_registration.parse_state_dict_from_weights",
        lambda *args, **kwargs: {},
    )
    model = module_pipeline_model(
        TargetConfig, ((GraphRole.DECODER, MeshDecoder),), None
    )
    started: list[EPConfig] = []
    _, config = RegisteredDecoderBuilder(model).build_module(
        _target_context(started, ep_size=1)
    )
    assert started == []
    assert config.runtime is not None and config.runtime.ep is None


def test_a_composed_decoder_refuses_state_it_would_not_get() -> None:
    model: Any = SimpleNamespace(decoder_cls=StatefulDecoder)
    with pytest.raises(NotImplementedError, match="composed module"):
        RegisteredDecoderBuilder(model).build_module(
            _target_context([], ep_size=1)
        )


@dataclass(frozen=True)
class VisionTargetConfig(TargetConfig):
    weight_prefixes: ClassVar[Mapping[str, str]] = {
        GraphRole.DECODER: "decoder.",
        GraphRole.VISION: "vision.",
    }


class VisionMeshDecoder(MeshDecoder):
    consumes_vision: ClassVar[bool] = True


class ScaleVision(Module[[Tensor], Tensor]):
    graph_name: ClassVar[str] = "toy_vision"

    def __init__(self, config: VisionTargetConfig) -> None:
        super().__init__()
        self.scale = Scale()

    def input_types(self) -> tuple[TensorType, ...]:
        return (X,)

    def forward(self, pixels: Tensor) -> Tensor:
        return self.scale(pixels)


class DisabledVision(ScaleVision):
    @classmethod
    def is_enabled(cls, config: VisionTargetConfig) -> bool:
        return False


@pytest.mark.parametrize(
    ("vision", "builds"),
    [(ScaleVision, True), (DisabledVision, False), (None, False)],
)
def test_a_composed_decoder_builds_its_registered_vision_graph(
    monkeypatch: pytest.MonkeyPatch,
    vision: type[ScaleVision] | None,
    builds: bool,
) -> None:
    """A speculative pairing builds the target's VISION module beside it,
    from the config and checkpoint the decoder is built from."""
    vision_weight = np.full(2, 3, dtype=np.float32)
    monkeypatch.setattr(
        "max.pipelines.lib.module_registration.parse_state_dict_from_weights",
        lambda *args, **kwargs: {
            "decoder.scale.weight": np.zeros(2, dtype=np.float32),
            "vision.scale.weight": vision_weight,
        },
    )
    modules: tuple[tuple[str, type[Module[..., Any]]], ...] = (
        (GraphRole.DECODER, VisionMeshDecoder if vision else MeshDecoder),
    )
    if vision is not None:
        modules += ((GraphRole.VISION, vision),)
    model = module_pipeline_model(
        VisionTargetConfig,
        modules,
        ToyVisionBatchProcessor if vision is not None else None,
    )
    context = _target_context([], ep_size=1)
    builder = RegisteredDecoderBuilder(model)
    decoder, config = builder.build_module(context)
    assert isinstance(decoder, MeshDecoder)
    spec = builder.build_vision(context, config, CPU())
    assert (spec is not None) is builds
    if spec is not None:
        assert isinstance(spec.module, ScaleVision)
        assert spec.name == "toy_vision"
        assert tuple(spec.input_types) == (X,)
        assert spec.weights is not None
        assert list(spec.weights) == ["scale.weight"]
        np.testing.assert_array_equal(
            np.from_dlpack(spec.weights["scale.weight"]), vision_weight
        )


@dataclass(frozen=True)
class StagedConfig(ToyConfig):
    """Builds the layers of the stage the pipeline config carries."""

    layers: range = range(2)
    dtype: DType = DType.float32

    @classmethod
    def initialize(
        cls,
        pipeline_config: Any,
        model_config: Any = None,
        *,
        max_seq_len: int,
    ) -> Self:
        stage = pipeline_config.model.pipeline_stage
        return cls() if stage is None else cls(layers=stage.layers)


class StagedDecoder(Module[[DecoderInputs], Any]):
    """Scales its input once per layer, each layer under its global name."""

    supports_pipeline_stages: ClassVar[bool] = True

    def __init__(self, config: StagedConfig) -> None:
        super().__init__()
        self.config = config
        for i in config.layers:
            setattr(self, f"layer_{i}", Scale())

    def forward(self, inputs: DecoderInputs) -> Any:
        h = (
            inputs.kv[0]
            if inputs.stage_boundary is None
            else inputs.stage_boundary[0]
        )
        for i in self.config.layers:
            h = getattr(self, f"layer_{i}")(h)
        if self.config.layers.stop < 2:
            return (h,)
        return DecoderOutputs(next_token_logits=h)


class StagedVisionDecoder(StagedDecoder):
    consumes_vision: ClassVar[bool] = True


def _staged_model(
    decoder: type[Module[..., Any]],
    stage: PipelineStage | None,
    *,
    vision: type[Module[..., Any]] | None = None,
) -> Any:
    modules: tuple[tuple[str, type[Module[..., Any]]], ...] = (
        (GraphRole.DECODER, decoder),
    )
    if vision is not None:
        modules += ((GraphRole.VISION, vision),)
    cls = module_pipeline_model(
        StagedConfig,
        modules,
        ToyVisionBatchProcessor if vision is not None else None,
    )
    model = cls.__new__(cls)
    pipeline_config: Any = SimpleNamespace(
        model=SimpleNamespace(pipeline_stage=stage, data_parallel_degree=1)
    )
    model.pipeline_config = pipeline_config
    model.return_hidden_states = ReturnHiddenStates.NONE
    model._lora_manager = None
    model._batch_processor = None
    model.max_batch_size = 1
    model.devices = [CPU()]
    model.device_refs = [DeviceRef.CPU()]
    model.kv_params = NonCallableMock(get_symbolic_inputs=lambda: (X,))
    model._runtime = NonCallableMock(spec=ModuleRuntime, ep=None)
    model._checkpoint = {
        f"layer_{i}.weight": np.full(2, i + 2, dtype=np.float32)
        for i in range(2)
    }
    return model


def test_a_staged_decoder_is_a_pipelined_model() -> None:
    assert supports_pipeline_stages(StagedDecoder)
    assert not supports_pipeline_stages(PlainDecoder)
    staged = module_pipeline_model(
        StagedConfig, ((GraphRole.DECODER, StagedDecoder),), None
    )
    plain = module_pipeline_model(
        StagedConfig, ((GraphRole.DECODER, PlainDecoder),), None
    )
    assert issubclass(staged, PipelinedModel)
    assert not issubclass(plain, PipelinedModel)


def test_a_stage_before_the_last_traces_what_it_hands_on() -> None:
    stage = PipelineStage(0, 2, range(1), trace_only=True)
    model = _staged_model(StagedDecoder, stage)
    with pytest.raises(StageTraced) as traced:
        model.graph_specs(StagedConfig(layers=stage.layers))
    assert traced.value.boundary_types == (X,)


def test_a_later_stage_starts_from_the_boundary_with_its_own_weights() -> None:
    stage = PipelineStage(1, 2, range(1, 2), boundary_types=(X,))
    model = _staged_model(StagedDecoder, stage)
    spec = model.graph_specs(StagedConfig(layers=stage.layers))[
        GraphRole.DECODER
    ]
    (inputs,) = spec.input_types
    assert isinstance(inputs, DecoderGraphInputs)
    # The boundary comes after every other input.
    assert list(inputs)[-1] == "stage_boundary"
    assert inputs["stage_boundary"] == (X,)
    assert set(spec.weights) == {"layer_1.weight"}
    run = spec.module.compile(*spec.input_types, weights=spec.weights)
    (logits,) = run.execute_raw(
        Buffer.from_numpy(np.zeros(1, np.int64)),
        Buffer.from_numpy(np.array([0, 1], np.uint32)),
        Buffer.from_numpy(np.ones(1, np.int64)),
        Buffer.from_numpy(np.full(2, 4, np.float32)),
        Buffer.from_numpy(np.full(2, 5, np.float32)),
    )
    np.testing.assert_array_equal(np.from_dlpack(logits), np.full(2, 15))


def test_the_bound_inputs_keep_a_nested_stage_boundary_last() -> None:
    types = decoder_input_types(
        NonCallableMock(
            get_symbolic_inputs=lambda: (
                KVCacheInputsPerDevice(_KV_BLOCKS, X, X, X, X, X),
            )
        ),
        DeviceRef.CPU(),
    )
    types["stage_boundary"] = {"h": X, "skip": (X,)}
    processor = _batch_processor(1)
    processor.bind_input_tree(types)
    kv_buffer, h, skip = (
        Buffer.from_numpy(np.full(2, value, np.float32)) for value in (1, 2, 3)
    )
    kv = KVCacheInputsPerDevice(
        kv_buffer, kv_buffer, kv_buffer, kv_buffer, kv_buffer, kv_buffer
    )
    inputs = processor.prepare_initial_token_inputs(
        [[_context(1)]], kv_cache_inputs=(kv,)
    )
    inputs.stage_boundary = {"skip": (skip,), "h": h}
    assert inputs.buffers[-2:] == (h, skip)


class NestedStagedDecoder(StagedDecoder):
    """Hands the next stage a dictionary that nests a tuple."""

    def forward(self, inputs: DecoderInputs) -> Any:
        if inputs.stage_boundary is None:
            h, skip = inputs.kv[0], inputs.kv[0]
        else:
            h, (skip,) = (
                inputs.stage_boundary["h"],
                inputs.stage_boundary["skip"],
            )
        for i in self.config.layers:
            h = getattr(self, f"layer_{i}")(h)
        if self.config.layers.stop < 2:
            return {"h": h, "skip": (skip,)}
        return DecoderOutputs(next_token_logits=h + skip)


def test_the_next_stage_takes_the_tree_the_last_one_returned() -> None:
    first = PipelineStage(0, 2, range(1), trace_only=True)
    with pytest.raises(StageTraced) as traced:
        _staged_model(NestedStagedDecoder, first).graph_specs(
            StagedConfig(layers=first.layers)
        )
    boundary = traced.value.boundary_types
    assert boundary == {"h": X, "skip": (X,)}
    stage = PipelineStage(1, 2, range(1, 2), boundary_types=boundary)
    spec = _staged_model(NestedStagedDecoder, stage).graph_specs(
        StagedConfig(layers=stage.layers)
    )[GraphRole.DECODER]
    run = spec.module.compile(*spec.input_types, weights=spec.weights)
    (logits,) = run.execute_raw(
        Buffer.from_numpy(np.zeros(1, np.int64)),
        Buffer.from_numpy(np.array([0, 1], np.uint32)),
        Buffer.from_numpy(np.ones(1, np.int64)),
        Buffer.from_numpy(np.full(2, 4, np.float32)),
        Buffer.from_numpy(np.full(2, 5, np.float32)),
        Buffer.from_numpy(np.full(2, 7, np.float32)),
    )
    np.testing.assert_array_equal(np.from_dlpack(logits), np.full(2, 22))


def test_only_the_first_stage_encodes_images() -> None:
    stage = PipelineStage(1, 2, range(1, 2), boundary_types=(X,))
    model = _staged_model(StagedVisionDecoder, stage, vision=ToyVision)
    model._batch_processor = NonCallableMock(spec=ToyVisionBatchProcessor)
    model._load_state_dict = dict
    model._create_model_config = lambda state_dict: StagedConfig(
        layers=stage.layers
    )
    models = model.load_graphs(InferenceSession(devices=[CPU()]))
    assert list(models) == [GraphRole.DECODER]
    model._batch_processor.bind_vision_encoder.assert_not_called()


class SignalLayoutDecoder(PlainDecoder):
    """Takes the signal buffers after the batch, as a graph-API graph does."""

    input_layout: ClassVar[tuple[str, ...]] = (
        "tokens",
        "input_row_offsets",
        "return_n_logits",
        "signals",
        "kv",
    )


def test_a_decoder_places_its_signal_buffers_by_its_input_layout() -> None:
    spec = _staged_model(SignalLayoutDecoder, None).graph_specs(StagedConfig())[
        GraphRole.DECODER
    ]
    assert spec.signal_buffer_index == 3
    plain = _staged_model(PlainDecoder, None).graph_specs(StagedConfig())[
        GraphRole.DECODER
    ]
    assert plain.signal_buffer_index is None


@dataclass(frozen=True)
class ContextLengthConfig(StagedConfig, DeploymentConfig):
    takes_batch_context_lengths: ClassVar[bool] = True


class DeviceContextLengthDecoder(PlainDecoder):
    device_context_length_dtype: ClassVar[DType] = DType.uint32


def test_a_decoder_can_take_a_context_length_per_device() -> None:
    def lengths(decoder: type[Module[..., Any]]) -> list[object]:
        spec = _staged_model(decoder, None).graph_specs(ContextLengthConfig())[
            GraphRole.DECODER
        ]
        (inputs,) = spec.input_types
        assert isinstance(inputs, DecoderGraphInputs)
        return list(inputs["batch_context_lengths"])

    cpu = DeviceRef.CPU()
    assert lengths(PlainDecoder) == [TensorType(DType.int32, [1], device=cpu)]
    assert lengths(DeviceContextLengthDecoder) == [
        TensorType(DType.uint32, [1], device=cpu)
    ]


def test_a_decoder_without_stages_refuses_one() -> None:
    stage = PipelineStage(0, 2, range(1))
    with pytest.raises(ValueError, match="doesn't build pipeline stages"):
        _staged_model(PlainDecoder, stage).graph_specs(StagedConfig())


_SCRATCH = BufferType(DType.float32, [2, 3], device=DeviceRef.CPU())


class ScratchDecoder(PlainDecoder):
    """Declares one workspace buffer, sized from the largest batch."""

    @classmethod
    def scratch_types(
        cls, config: ToyConfig, *, max_batch_size: int, devices: Any
    ) -> tuple[BufferType, ...]:
        del config
        return tuple(
            BufferType(DType.float32, [2, max_batch_size], device=device)
            for device in (DeviceRef.from_device(d) for d in devices)
        )


def test_a_decoder_without_workspace_gets_none() -> None:
    assert (
        build_decoder_scratch(
            PlainDecoder, ToyConfig(), max_batch_size=3, devices=[CPU()]
        )
        is None
    )


def test_workspace_is_allocated_once_at_its_declared_size() -> None:
    scratch = build_decoder_scratch(
        ScratchDecoder, ToyConfig(), max_batch_size=3, devices=[CPU()]
    )
    assert scratch is not None
    assert scratch.input_types == (_SCRATCH,)
    (buffer,) = scratch.buffers
    assert buffer.shape == (2, 3)
    assert buffer.dtype == DType.float32


@pytest.mark.parametrize(
    ("declared", "error"),
    [
        (X, TypeError),
        (
            BufferType(DType.float32, ["rows"], device=DeviceRef.CPU()),
            ValueError,
        ),
    ],
)
def test_workspace_must_be_static_buffers(
    declared: Any, error: type[Exception]
) -> None:
    class Declares(PlainDecoder):
        @classmethod
        def scratch_types(cls, config: Any, **kwargs: Any) -> tuple[Any, ...]:
            return (declared,)

    with pytest.raises(error):
        build_decoder_scratch(
            Declares, ToyConfig(), max_batch_size=3, devices=[CPU()]
        )


def test_workspace_follows_the_state_inputs() -> None:
    kv_params = NonCallableMock(get_symbolic_inputs=lambda: (X,))
    types = decoder_input_types(
        kv_params, DeviceRef.CPU(), state=(X,), scratch=(_SCRATCH,)
    )
    assert list(types)[-2:] == ["state", "scratch"]
    assert types["scratch"] == (_SCRATCH,)


def test_every_batch_passes_the_same_workspace_last() -> None:
    processor = _batch_processor(1)
    scratch = bind_decoder_scratch(
        ScratchDecoder,
        ToyConfig(),
        processor,
        max_batch_size=3,
        devices=[CPU()],
    )
    assert scratch is not None
    kv = NonCallableMock(spec=KVCacheInputsPerDevice)
    first, second = (
        processor.prepare_initial_token_inputs(
            [[_context(1, 2)]], kv_cache_inputs=kv
        )
        for _ in range(2)
    )
    assert first.scratch == second.scratch == scratch.buffers
    assert first.buffers[-1] is scratch.buffers[0]
    symbolic = processor.get_symbolic_inputs(
        kv_params=NonCallableMock(get_symbolic_inputs=lambda: (X,)),
        device_refs=[DeviceRef.CPU()],
    )
    assert symbolic[-1] == _SCRATCH


def test_only_the_decoder_batch_processor_passes_workspace() -> None:
    with pytest.raises(TypeError, match="DecoderBatchProcessor"):
        bind_decoder_scratch(
            ScratchDecoder,
            ToyConfig(),
            object(),
            max_batch_size=3,
            devices=[CPU()],
        )


class ScratchReader(Module[[DecoderInputs], DecoderOutputs]):
    def __init__(self, config: ToyConfig) -> None:
        super().__init__()

    def forward(self, inputs: DecoderInputs) -> DecoderOutputs:
        assert inputs.scratch is not None
        (scratch,) = inputs.scratch
        return DecoderOutputs(next_token_logits=inputs.kv[0] + scratch[0])


def test_the_graph_hands_the_decoder_its_workspace() -> None:
    workspace = BufferType(DType.float32, [1, 2], device=DeviceRef.CPU())
    types = _toy_input_types(scratch=(workspace,))
    types["kv"] = (X,)
    run = _DecoderGraph(ScratchReader(ToyConfig())).compile(types)
    (logits,) = run.execute_raw(
        *(Buffer.from_numpy(np.full(2, 1, np.float32)) for _ in range(4)),
        Buffer.from_numpy(np.full((1, 2), 2, np.float32)),
    )
    np.testing.assert_array_equal(np.from_dlpack(logits), np.full(2, 3))


@dataclass(frozen=True)
class RuntimeRecordingConfig(ToyConfig, DeploymentConfig):
    runtime: ModuleRuntime | None = None

    def apply_runtime(self, runtime: ModuleRuntime) -> Self:
        return replace(self, runtime=runtime)


def _deployed_model(config: RuntimeRecordingConfig) -> Any:
    cls = module_pipeline_model(
        type(config), ((GraphRole.DECODER, PlainDecoder),), None
    )
    model: Any = cls.__new__(cls)
    pipeline_config: Any = SimpleNamespace(
        model=SimpleNamespace(
            data_parallel_degree=1, huggingface_config=_HF_CONFIG
        ),
        runtime=SimpleNamespace(
            max_batch_input_tokens=8, ep_size=1, ep_use_allreduce=False
        ),
    )
    model.pipeline_config = pipeline_config
    model.arch_config = config
    model.return_logits = ReturnLogits.LAST_TOKEN
    model.return_hidden_states = ReturnHiddenStates.NONE
    model.devices = [CPU()]
    model.max_batch_size = 7
    model.memory_plan = SimpleNamespace(planned_max_batch_total_tokens=64)
    return model


def test_the_deployment_carries_the_planned_batch_size() -> None:
    config = _deployed_model(RuntimeRecordingConfig())._create_model_config({})
    assert isinstance(config, RuntimeRecordingConfig)
    assert config.runtime is not None
    assert config.runtime.max_batch_size == 7
    assert config.runtime.max_batch_total_tokens == 64


@dataclass(frozen=True)
class LayoutRecordingConfig(RuntimeRecordingConfig):
    def export_records(self, runtime: ModuleRuntime) -> Mapping[str, object]:
        return {"layout_record": {"replicas": runtime.data_parallel_degree}}


def test_the_pipeline_model_publishes_what_the_deployment_records() -> None:
    model = _deployed_model(LayoutRecordingConfig())
    model._create_model_config({})
    assert model.export_records == {"layout_record": {"replicas": 1}}
    plain = _deployed_model(RuntimeRecordingConfig())
    plain._create_model_config({})
    assert plain.export_records == {}


def test_bound_token_positions_follow_each_context() -> None:
    position_type = TensorType(
        DType.uint32, ["total_seq_len"], device=DeviceRef.CPU()
    )
    types = decoder_input_types(
        NonCallableMock(get_symbolic_inputs=lambda: (X,)),
        DeviceRef.CPU(),
        token_positions=position_type,
    )
    processor = _batch_processor(1)
    processor.bind_input_tree(types)
    inputs = processor.prepare_initial_token_inputs(
        [[_context(1, 2, position=5), _context(3, position=9)]],
        kv_cache_inputs=NonCallableMock(spec=KVCacheInputsPerDevice),
    )
    assert inputs.token_positions is not None
    np.testing.assert_array_equal(inputs.token_positions.to_numpy(), [3, 4, 8])
    assert inputs.input_fields().token_positions is inputs.token_positions


def test_constant_input_buffers_share_the_declared_tree() -> None:
    buffer = Buffer.from_numpy(np.zeros(2, np.float32))
    constants = ConstantInputs(
        input_types={"table": (X,)}, buffers={"table": (buffer,)}
    )
    assert constants.buffers == {"table": (buffer,)}
    with pytest.raises(ValueError):
        ConstantInputs(
            input_types={"table": (X,)}, buffers={"other": (buffer,)}
        )


def test_a_disabled_vision_role_is_not_constructed() -> None:
    class DisabledVision(ToyVision):
        @classmethod
        def is_enabled(cls, config: ToyConfig) -> bool:
            return False

    cls = module_pipeline_model(
        ToyConfig,
        (
            (GraphRole.DECODER, VisionDecoder),
            (GraphRole.VISION, DisabledVision),
        ),
        ToyVisionBatchProcessor,
    )
    model: Any = cls.__new__(cls)
    model.pipeline_config = SimpleNamespace(
        model=SimpleNamespace(pipeline_stage=None)
    )
    assert model._vision_graph_spec(ToyConfig()) is None


def test_checkpoint_preparation_precedes_role_prefixes() -> None:
    class PreparedConfig(DeploymentConfig):
        weight_prefixes = {GraphRole.DECODER: "decoder."}

        def prepare_weights(
            self, state_dict: Mapping[str, Any]
        ) -> dict[str, Any]:
            return {
                "decoder." + name: value for name, value in state_dict.items()
            }

    config = PreparedConfig()
    value = NonCallableMock(spec=WeightData)
    prepared = prepare_weights({"weight": value}, config)
    assert role_weights(prepared, config, GraphRole.DECODER) == {
        "weight": value
    }
