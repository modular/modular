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
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Any, ClassVar
from unittest.mock import NonCallableMock

import numpy as np
import pytest
from max import tree
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    TensorType,
    TensorValue,
    Weight,
)
from max.graph.weights import WeightsFormat
from max.nn.ep_config import EPConfig
from max.nn.kv_cache import KVCacheInputsPerDevice
from max.nn.legacy.graph_groups import (
    BatchContext,
    EPPayload,
    RaggedBatch,
    TPSignals,
)
from max.nn.legacy.layer import Module
from max.nn.outputs import ReturnHiddenStates, ReturnLogits
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
    supports_data_parallel,
)
from max.pipelines.lib.registration_runtime import (
    ExpertParallelDeploymentConfig,
    ExpertParallelInputs,
    ModuleRuntime,
    initialize_ep_comm_buffers,
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
    if inputs.data_parallel_splits is not None:
        logits = logits + inputs.data_parallel_splits * 100000
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


class DataParallelDecoder(ToyDecoder):
    supports_data_parallel = True


def _types(
    *, signals: int = 0, data_parallel: bool = False
) -> dict[str, object]:
    batch = RaggedBatch(tokens=X, row_offsets=X, return_n_logits=X)
    if data_parallel:
        batch["data_parallel_splits"] = X
    types: dict[str, object] = {"batch": batch}
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


def test_the_graph_feeds_data_parallel_splits_after_the_logit_count() -> None:
    np.testing.assert_array_equal(
        _run(ToyDecoder(ToyConfig()), _types(data_parallel=True)),
        np.full(
            2,
            1 * 2 + 2 * 10 + 3 * 100 + 4 * 100000 + 5 * 1000 - 6,
            np.float32,
        ),
    )


def test_an_instance_can_name_its_graph() -> None:
    decoder = ToyDecoder(ToyConfig())
    decoder.graph_name = "toy_tp"
    assert graph_name(decoder) == "toy_tp"
    assert graph_name(ToyDecoder) == "toy"
    assert graph_name(TupleOutputsDecoder) == "TupleOutputsDecoder"


def test_a_decoder_opts_into_data_parallelism() -> None:
    assert supports_data_parallel(DataParallelDecoder)
    assert not supports_data_parallel(ToyDecoder)


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


def test_data_parallel_input_types_take_the_splits() -> None:
    types = graph_api_decoder_input_types(
        _kv_params(), [CPU_REF, CPU_REF], data_parallel_degree=2
    )
    assert list(types) == ["batch", "kv"]
    batch = types["batch"]
    assert isinstance(batch, RaggedBatch)
    assert list(batch) == [
        "tokens",
        "row_offsets",
        "return_n_logits",
        "data_parallel_splits",
    ]
    splits = batch["data_parallel_splits"]
    assert isinstance(splits, TensorType)
    assert splits.dtype == DType.int64
    assert list(splits.shape) == [3]
    assert splits.device == CPU_REF


def test_a_legacy_data_parallel_batch_keeps_its_order() -> None:
    types = graph_api_decoder_input_types(
        _kv_params(),
        [CPU_REF, CPU_REF],
        data_parallel_degree=2,
        data_parallel_batch=(
            "tokens",
            "input_row_offsets",
            "return_n_logits",
            "host_input_row_offsets",
            "data_parallel_splits",
        ),
    )
    batch = types["batch"]
    assert isinstance(batch, RaggedBatch)
    assert list(batch) == [
        "tokens",
        "row_offsets",
        "return_n_logits",
        "host_row_offsets",
        "data_parallel_splits",
    ]
    host = batch["host_row_offsets"]
    assert isinstance(host, TensorType)
    assert host.device == CPU_REF


def _context(*tokens: int) -> TextContext:
    active = np.array(tokens, dtype=np.int64)
    return NonCallableMock(
        spec=TextContext,
        tokens=SimpleNamespace(
            active=active, active_length=len(active), current_position=3
        ),
    )


def _batch_processor(
    input_types: dict[str, object],
    signal_buffers: list[Buffer],
    data_parallel_degree: int = 1,
) -> GraphApiDecoderBatchProcessor:
    pipeline_config = NonCallableMock()
    pipeline_config.model.kv_cache.kv_cache_page_size = 4
    processor = GraphApiDecoderBatchProcessor(
        ToyConfig(),
        BatchProcessorRuntime(
            pipeline_config=pipeline_config,
            devices=[CPU(), CPU()],
            return_logits=ReturnLogits.LAST_TOKEN,
            max_batch_size=4,
            max_seq_len=16,
            max_batch_input_tokens=16,
            data_parallel_degree=data_parallel_degree,
            signal_buffers=signal_buffers,
        ),
    )
    # These batching cases supply one mocked KV input, rather than the graph
    # tests' two separately named KV inputs.
    input_types = {**input_types, "kv": X}
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


def test_the_batch_processor_feeds_each_replicas_first_sequence() -> None:
    kv = NonCallableMock(spec=KVCacheInputsPerDevice)
    processor = _batch_processor(
        _types(data_parallel=True), [], data_parallel_degree=2
    )
    inputs = processor.prepare_initial_token_inputs(
        [[_context(1, 2), _context(3)], [_context(4)]], kv_cache_inputs=kv
    )

    splits = inputs.data_parallel_splits
    assert splits is not None
    np.testing.assert_array_equal(splits.to_numpy(), [0, 2, 3])
    np.testing.assert_array_equal(
        inputs.input_row_offsets.to_numpy(), [0, 2, 3, 4]
    )
    assert inputs.buffers == (
        inputs.tokens,
        inputs.input_row_offsets,
        inputs.return_n_logits,
        splits,
        kv,
    )


def test_the_batch_processor_feeds_a_legacy_batch_in_its_order() -> None:
    kv = NonCallableMock(spec=KVCacheInputsPerDevice)
    batch = RaggedBatch(
        tokens=X,
        row_offsets=X,
        return_n_logits=X,
        host_row_offsets=X,
        data_parallel_splits=X,
    )
    processor = _batch_processor(
        {"batch": batch, "kv": {"a": X, "b": X}}, [], data_parallel_degree=2
    )
    inputs = processor.prepare_initial_token_inputs(
        [[_context(1, 2)], [_context(3)]], kv_cache_inputs=kv
    )

    host = inputs.host_input_row_offsets
    splits = inputs.data_parallel_splits
    assert host is not None and splits is not None
    np.testing.assert_array_equal(host.to_numpy(), [0, 2, 3])
    assert inputs.buffers == (
        inputs.tokens,
        inputs.input_row_offsets,
        inputs.return_n_logits,
        host,
        splits,
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


def test_registered_modules_serve_only_text_and_embeddings_so_far() -> None:
    with pytest.raises(NotImplementedError, match="audio_generation"):
        registered_pipeline_model(
            PipelineTask.AUDIO_GENERATION,
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


class _HostEPInputs(ExpertParallelInputs):
    def input_tree(self, ep: EPConfig) -> object:
        return EPPayload(
            atomic_counters=(BufferType(DType.int32, [1], device=CPU_REF),),
            send_buf_ptrs=(X,),
        )


@dataclass(frozen=True)
class _DeploymentConfig(ToyConfig, ExpertParallelDeploymentConfig):
    takes_batch_context_lengths: ClassVar[bool] = True
    ep_dispatch_token_layout: ClassVar[str | None] = "tp_sharded"
    expert_parallel_inputs: ClassVar[ExpertParallelInputs] = _HostEPInputs()
    runtime: ModuleRuntime | None = None
    events: list[str] = field(default_factory=list, compare=False)

    def expert_parallel_config(self, runtime: ModuleRuntime) -> EPConfig:
        assert self.finalized
        self.events.append("size_exchange")
        return NonCallableMock(spec=EPConfig, node_id=-1)

    def apply_runtime(self, runtime: ModuleRuntime) -> Self:
        self.events.append("apply_runtime")
        return replace(self, runtime=runtime)


class _DeploymentDecoder(ToyDecoder):
    def __init__(self, config: _DeploymentConfig) -> None:
        assert config.finalized and config.runtime is not None
        config.events.append("construct_module")
        super().__init__(config)


def _deployment_model(config: ToyConfig, *, ep_size: int) -> Any:
    cls = graph_api_pipeline_model(
        type(config), ((GraphRole.DECODER, _DeploymentDecoder),), None
    )
    model: Any = cls.__new__(cls)
    model.pipeline_config = SimpleNamespace(
        model=SimpleNamespace(
            data_parallel_degree=1, huggingface_config=object()
        ),
        runtime=SimpleNamespace(
            max_batch_input_tokens=8, ep_size=ep_size, ep_use_allreduce=False
        ),
    )
    model.arch_config = config
    model.return_logits = ReturnLogits.LAST_TOKEN
    model.return_hidden_states = ReturnHiddenStates.NONE
    model.memory_plan = SimpleNamespace(planned_max_batch_total_tokens=32)
    model.devices = [CPU(), CPU()]
    model.device_refs = [CPU_REF, CPU_REF]
    model.max_batch_size = 4
    model.kv_params = _kv_params()
    model._batch_processor = None
    model._session = NonCallableMock(spec=InferenceSession)
    return model


def test_deployment_is_completed_before_constructing_the_decoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _DeploymentConfig()
    model = _deployment_model(config, ep_size=2)
    exchange = NonCallableMock()

    def initialize(
        config: _DeploymentConfig, runtime: ModuleRuntime, session: object
    ) -> tuple[ModuleRuntime, object]:
        assert config.finalized
        assert session is model._session
        ep = config.expert_parallel_config(runtime)
        config.events.append("initialize_exchange")
        ep.node_id = 7
        return replace(runtime, ep=ep), exchange

    monkeypatch.setattr(
        "max.pipelines.lib.registration_graph_api.initialize_ep_comm_buffers",
        initialize,
    )
    finalized = model._create_model_config(
        {"scale.weight": np.ones(2, np.float32)}
    )
    assert finalized.runtime.ep.node_id == 7
    assert finalized.runtime.max_batch_total_tokens == 32
    assert finalized.runtime.max_batch_size == 4
    assert finalized.runtime.mesh.devices == (CPU(), CPU())
    spec = model.graph_specs(finalized)[GraphRole.DECODER]
    assert config.events == [
        "size_exchange",
        "initialize_exchange",
        "apply_runtime",
        "construct_module",
    ]
    assert model.ep_comm_initializer is exchange
    assert list(spec.input_types) == [
        "batch",
        "signals",
        "kv",
        "batch_context_lengths",
        "ep",
    ]
    assert isinstance(spec.input_types["batch_context_lengths"], BatchContext)
    assert isinstance(spec.input_types["ep"], EPPayload)
    assert type(model).ep_dispatch_token_layout == "tp_sharded"


def test_a_plain_config_cannot_silently_ignore_expert_parallelism() -> None:
    model = _deployment_model(ToyConfig(), ep_size=2)
    with pytest.raises(ValueError, match="doesn't support expert parallelism"):
        model._create_model_config({})


def test_deployment_without_ep_does_not_initialize_an_exchange(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _DeploymentConfig()
    model = _deployment_model(config, ep_size=1)

    def unexpected(*args: object) -> None:
        raise AssertionError("non-EP deployment initialized an exchange")

    monkeypatch.setattr(
        "max.pipelines.lib.registration_graph_api.initialize_ep_comm_buffers",
        unexpected,
    )
    finalized = model._create_model_config({})
    assert finalized.runtime is not None
    assert finalized.runtime.ep is None
    assert config.events == ["apply_runtime"]


def test_virtual_ep_deployment_sizes_without_allocating_buffers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(_DeploymentConfig(), finalized=True)
    model = _deployment_model(config, ep_size=1)
    finalized = model._create_model_config({})
    runtime = replace(finalized.runtime, ep_size=2)
    monkeypatch.setattr(
        "max.pipelines.lib.registration_runtime.is_virtual_device_mode",
        lambda: True,
    )

    def unexpected(*args: object) -> None:
        raise AssertionError("virtual deployment allocated an EP initializer")

    monkeypatch.setattr(
        "max.pipelines.lib.registration_runtime.EPCommInitializer", unexpected
    )
    initialized, exchange = initialize_ep_comm_buffers(
        config, runtime, model._session
    )
    assert initialized.ep is not None
    assert exchange is None
    assert config.events == ["apply_runtime", "size_exchange"]


def test_batch_processor_passes_the_ep_exchange_and_context_lengths() -> None:
    kv = NonCallableMock(spec=KVCacheInputsPerDevice)
    counter = Buffer.zeros((1,), dtype=DType.int32)
    ep_types = EPPayload(
        atomic_counters=(BufferType(DType.int32, [1], device=CPU_REF),)
    )
    types = graph_api_decoder_input_types(
        _kv_params(), [CPU_REF], batch_context_lengths=(X,), ep=ep_types
    )
    processor = _batch_processor(types, [])
    processor.bind_ep_comm_initializer(
        NonCallableMock(model_inputs=lambda: [counter])
    )
    inputs = processor.prepare_initial_token_inputs(
        [[_context(1, 2)]], kv_cache_inputs=kv
    )
    assert inputs.batch_context_lengths is not None
    assert inputs.batch_context_lengths[0].to_numpy()[0] == 4
    assert inputs.ep_inputs == (counter,)
    assert inputs.buffers[-2:] == (*inputs.batch_context_lengths, counter)
    assert processor.get_symbolic_inputs(
        kv_params=_kv_params(), device_refs=[CPU_REF]
    ) == tree.leaves(processor.input_tree)


class _ExchangeDecoder(ToyDecoder):
    def __call__(
        self, inputs: DecoderInputs[TensorValue]
    ) -> DecoderOutputs[TensorValue]:
        assert inputs.batch_context_lengths is not None
        assert isinstance(inputs.batch_context_lengths[0], TensorValue)
        assert isinstance(inputs.ep, EPPayload)
        assert isinstance(inputs.ep["atomic_counters"][0], BufferValue)
        assert isinstance(inputs.ep["send_buf_ptrs"][0], TensorValue)
        return super().__call__(inputs)


def test_graph_unflattens_ep_buffers_and_host_tensors_by_their_types() -> None:
    ep_types = _HostEPInputs().input_tree(NonCallableMock(spec=EPConfig))
    types = OrderedDict(_types())
    types["batch_context_lengths"] = BatchContext(context_lengths=(X,))
    types["ep"] = ep_types
    graph = _DecoderGraph(_ExchangeDecoder(ToyConfig()), types).trace(
        *tree.leaves(types)
    )
    assert len(graph.inputs) == len(tree.leaves(types))
