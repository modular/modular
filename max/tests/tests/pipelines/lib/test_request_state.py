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
"""Tests the per-request state a registered decoder declares."""

from __future__ import annotations

import pickle
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import NonCallableMock

import numpy as np
import pytest
from max import tree
from max.driver import CPU, Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import BufferType, DeviceRef, TensorType, TensorValue, ops
from max.graph.weights import WeightsFormat
from max.nn import Module as ModuleV3
from max.nn.kv_cache import KVCacheInputsPerDevice
from max.nn.legacy.graph_groups import DecoderGraphInputs, RaggedBatch
from max.nn.legacy.layer import Module
from max.nn.outputs import ReturnHiddenStates, ReturnLogits
from max.pipelines.context import TextContext
from max.pipelines.lib import (
    GraphRole,
    SupportedArchitecture,
    SupportsSSMStateWarmup,
    TextTokenizer,
)
from max.pipelines.lib.graph_specs import GraphSpec, compile_and_init_graphs
from max.pipelines.lib.interfaces.arch_config import ModelOutputConfig
from max.pipelines.lib.interfaces.batch_processor import BatchProcessorRuntime
from max.pipelines.lib.module_registration import (
    DecoderBatchProcessor,
    ModuleV3DecoderPipelineModel,
    _role_outputs,
    bind_state_store,
    decoder_input_types,
)
from max.pipelines.lib.module_registration import (
    _DecoderGraph as ModuleV3DecoderGraph,
)
from max.pipelines.lib.registration_graph_api import (
    GraphApiDecoderBatchProcessor,
    GraphApiDecoderPipelineModel,
    _DecoderGraph,
    graph_api_decoder_input_types,
)
from max.pipelines.lib.request_state import (
    RequestStateGroup,
    RequestStateMixin,
    RequestStateStore,
    build_state_store,
    split_state_outputs,
)
from max.pipelines.lib.role_contracts import DecoderInputs, DecoderOutputs
from max.pipelines.modeling.types import PipelineTask, RequestID
from max.tensor import Tensor
from transformers import AutoConfig
from typing_extensions import Self

CPU_REF = DeviceRef.CPU()
X = TensorType(DType.float32, [2], device=CPU_REF)
POOL = BufferType(DType.float32, [2], device=CPU_REF)


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
    ) -> ToyConfig:
        return replace(self)


class ToyState(RequestStateStore):
    """One state tensor written back each step, and one in-place pool."""

    def __init__(self, input_types: Sequence[TensorType | BufferType]) -> None:
        self.types = tuple(input_types)
        self.held: set[RequestID] = set()
        self.written: list[tuple[Sequence[RequestID], Sequence[Buffer]]] = []

    def input_types(self) -> tuple[TensorType | BufferType, ...]:
        return self.types

    def prepare(self, batch: Sequence[TextContext]) -> tuple[Buffer, ...]:
        self.held.update(ctx.request_id for ctx in batch)
        return tuple(
            Buffer.from_numpy(np.full(2, 7, np.float32)) for _ in self.types
        )

    def update(
        self, request_ids: Sequence[RequestID], outputs: Sequence[Buffer]
    ) -> None:
        self.written.append((request_ids, outputs))

    def release(self, request_id: RequestID) -> None:
        self.held.discard(request_id)


class StatefulDecoder(Module):
    """Adds its state to the logits, writes back the state plus one, and
    stores the logits into its pool."""

    def __init__(self, config: ToyConfig) -> None:
        super().__init__()

    @classmethod
    def state_store(
        cls,
        config: ToyConfig,
        *,
        max_batch_size: int,
        devices: Sequence[Device],
    ) -> ToyState:
        return ToyState((X, POOL))

    def __call__(
        self, inputs: DecoderInputs[TensorValue]
    ) -> DecoderOutputs[TensorValue]:
        assert inputs.state is not None
        state, pool = inputs.state
        logits = inputs.tokens + inputs.kv["a"].tensor * 10 + state * 100
        ops.buffer_store(pool, logits)
        return DecoderOutputs(next_token_logits=logits, state=(state + 1,))


class StatelessDecoder(Module):
    def __init__(self, config: ToyConfig) -> None:
        super().__init__()

    def __call__(
        self, inputs: DecoderInputs[TensorValue]
    ) -> DecoderOutputs[TensorValue]:
        return DecoderOutputs(next_token_logits=inputs.tokens)


class StatefulModuleV3Decoder(ModuleV3[[DecoderInputs], DecoderOutputs]):
    def __init__(self, config: ToyConfig) -> None:
        super().__init__()

    @classmethod
    def state_store(
        cls,
        config: ToyConfig,
        *,
        max_batch_size: int,
        devices: Sequence[Device],
    ) -> ToyState:
        return ToyState((X,))

    def forward(self, inputs: DecoderInputs) -> DecoderOutputs:
        assert inputs.state is not None
        (state,) = inputs.state
        return DecoderOutputs(
            next_token_logits=inputs.tokens + state, state=(state * 2,)
        )


def _arch(decoder: type[Any]) -> SupportedArchitecture:
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
        modules={GraphRole.DECODER: decoder},
    )


def _context(*tokens: int) -> TextContext:
    active = np.array(tokens, dtype=np.int64)
    return NonCallableMock(
        spec=TextContext,
        request_id=RequestID(),
        tokens=SimpleNamespace(active=active, active_length=len(active)),
    )


def _runtime() -> BatchProcessorRuntime:
    return BatchProcessorRuntime(
        pipeline_config=NonCallableMock(),
        devices=[CPU()],
        return_logits=ReturnLogits.LAST_TOKEN,
        max_batch_size=4,
        max_seq_len=16,
        max_batch_input_tokens=16,
    )


def test_a_declared_state_is_built_from_the_config() -> None:
    state = build_state_store(
        StatefulDecoder, ToyConfig(), max_batch_size=4, devices=[CPU()]
    )
    assert isinstance(state, ToyState)
    assert state.input_types() == (X, POOL)
    assert (
        build_state_store(
            StatelessDecoder, ToyConfig(), max_batch_size=4, devices=[CPU()]
        )
        is None
    )


def test_a_declared_state_must_be_a_request_state_store() -> None:
    class Wrong(StatelessDecoder):
        @classmethod
        def state_store(cls, config: ToyConfig, **kwargs: Any) -> object:
            return object()

    with pytest.raises(TypeError, match="not a RequestStateStore"):
        build_state_store(Wrong, ToyConfig(), max_batch_size=4, devices=[CPU()])


def test_an_in_place_state_takes_no_outputs() -> None:
    class InPlace(RequestStateStore):
        def input_types(self) -> tuple[TensorType | BufferType, ...]:
            return (POOL,)

        def prepare(self, batch: Sequence[TextContext]) -> tuple[Buffer, ...]:
            return ()

        def release(self, request_id: RequestID) -> None:
            self.released = request_id

    state = InPlace()
    state.update([RequestID()], [])
    with pytest.raises(ValueError, match="updates its state in place"):
        state.update([RequestID()], [Buffer.zeros([1], DType.float32)])
    request_id = RequestID()
    state.release_warmup([request_id])
    assert state.released == request_id


@pytest.mark.parametrize(
    ("return_logits", "return_hidden_states", "count"),
    [
        (ReturnLogits.LAST_TOKEN, ReturnHiddenStates.NONE, 1),
        (ReturnLogits.VARIABLE, ReturnHiddenStates.NONE, 3),
        (ReturnLogits.ALL, ReturnHiddenStates.LAST, 4),
        (ReturnLogits.LAST_TOKEN, ReturnHiddenStates.ALL, 2),
    ],
)
def test_state_follows_the_role_outputs(
    return_logits: ReturnLogits,
    return_hidden_states: ReturnHiddenStates,
    count: int,
) -> None:
    outputs = list(range(count + 2))
    role, state = split_state_outputs(
        outputs, return_logits, return_hidden_states
    )
    assert list(role) == outputs[:count]
    assert list(state) == [count, count + 1]


def test_graph_api_state_follows_the_kv_inputs() -> None:
    kv_params = NonCallableMock(get_symbolic_inputs=lambda: {"a": X})
    types = graph_api_decoder_input_types(kv_params, [CPU_REF], state=(X, POOL))
    assert list(types) == ["batch", "kv", "state"]
    assert types["state"] == RequestStateGroup(state=(X, POOL))
    assert list(graph_api_decoder_input_types(kv_params, [CPU_REF])) == [
        "batch",
        "kv",
    ]


def test_the_graph_api_graph_feeds_state_and_returns_it_last() -> None:
    # A plain dict would flatten in sorted key order.
    input_types = OrderedDict(
        batch=RaggedBatch(tokens=X, row_offsets=X, return_n_logits=X),
        # Any kinded group stands in for the KV cache's kinded inputs.
        kv=RaggedBatch(a=X),
        state=RequestStateGroup(state=(X, POOL)),
    )
    # Compiles as serving does, which checks every input and output sits in
    # a kinded group.
    compiled = compile_and_init_graphs(
        InferenceSession(devices=[CPU()]),
        {
            GraphRole.DECODER: GraphSpec(
                _DecoderGraph(StatefulDecoder(ToyConfig()), input_types),
                "stateful",
                input_types,
            )
        },
    )
    model = compiled[GraphRole.DECODER].engine_model
    pool = Buffer.from_numpy(np.zeros(2, np.float32))
    tokens, offsets, n_logits, kv_a, state = (
        Buffer.from_numpy(np.full(2, value, np.float32))
        for value in (1, 2, 3, 4, 5)
    )
    logits, new_state = model.execute(
        tokens, offsets, n_logits, kv_a, state, pool
    )
    assert isinstance(logits, Buffer) and isinstance(new_state, Buffer)
    expected = np.full(2, 1 + 4 * 10 + 5 * 100, np.float32)
    np.testing.assert_array_equal(logits.to_numpy(), expected)
    np.testing.assert_array_equal(new_state.to_numpy(), np.full(2, 6))
    np.testing.assert_array_equal(pool.to_numpy(), expected)


def test_the_graph_api_batch_processor_feeds_state_after_the_kv() -> None:
    kv_buffer = Buffer.from_numpy(np.zeros(1, np.float32))
    kv = KVCacheInputsPerDevice(
        kv_buffer, kv_buffer, kv_buffer, kv_buffer, kv_buffer, kv_buffer
    )
    state = ToyState((X,))
    processor = GraphApiDecoderBatchProcessor(ToyConfig(), _runtime())
    processor.bind_input_tree(
        OrderedDict(
            batch=RaggedBatch(tokens=X, row_offsets=X, return_n_logits=X),
            kv=KVCacheInputsPerDevice(
                BufferType(DType.float32, [2], device=CPU_REF), X, X, X, X, X
            ),
            state=(X,),
        )
    )
    processor.state_store = state
    batch = [_context(1, 2), _context(3)]
    inputs = processor.prepare_initial_token_inputs([batch], kv_cache_inputs=kv)

    (state_input,) = inputs.state
    assert inputs.buffers == (
        inputs.tokens,
        inputs.input_row_offsets,
        inputs.return_n_logits,
        *tree.leaves(kv),
        state_input,
    )
    assert inputs.request_ids == tuple(ctx.request_id for ctx in batch)
    assert state.held == {ctx.request_id for ctx in batch}


def test_a_stateful_graph_api_registration_releases_its_state() -> None:
    model = _arch(StatefulDecoder).pipeline_model
    assert isinstance(model, type)
    assert issubclass(model, GraphApiDecoderPipelineModel)
    assert issubclass(model, RequestStateMixin)
    assert issubclass(model, SupportsSSMStateWarmup)
    assert pickle.loads(pickle.dumps(model)) is model

    stateless = _arch(StatelessDecoder).pipeline_model
    assert isinstance(stateless, type)
    assert not issubclass(stateless, RequestStateMixin)
    assert not hasattr(stateless, "release")
    assert not issubclass(stateless, SupportsSSMStateWarmup)

    pipeline_model = model.__new__(model)
    state = ToyState(())
    pipeline_model._state_store = state
    first, second = RequestID(), RequestID()
    state.held = {first, second}
    pipeline_model.release(first)
    assert state.held == {second}
    pipeline_model.release_warmup_state([second])
    assert state.held == set()


def test_modulev3_state_follows_the_lora_inputs() -> None:
    kv_params = NonCallableMock(get_symbolic_inputs=lambda: (X,))
    types = decoder_input_types(kv_params, CPU_REF, lora=(X,), state=(X,))
    assert list(types) == [
        "tokens",
        "input_row_offsets",
        "return_n_logits",
        "kv",
        "lora",
        "state",
    ]


def test_the_modulev3_graph_feeds_state_and_returns_it() -> None:
    run = ModuleV3DecoderGraph(StatefulModuleV3Decoder(ToyConfig())).compile(
        DecoderGraphInputs(
            tokens=X,
            input_row_offsets=X,
            return_n_logits=X,
            kv=(X,),
            state=(X,),
        )
    )
    tokens, offsets, n_logits, kv, state = (
        Buffer.from_numpy(np.full(2, value, np.float32))
        for value in (1, 2, 3, 4, 5)
    )
    grouped = run(
        DecoderGraphInputs(
            tokens=tokens,
            input_row_offsets=offsets,
            return_n_logits=n_logits,
            kv=(kv,),
            state=(state,),
        )
    )
    # The state follows the named outputs.
    assert isinstance(grouped, tuple)
    outputs = _role_outputs(grouped)
    np.testing.assert_array_equal(
        np.from_dlpack(outputs.next_token_logits.driver_tensor), np.full(2, 6)
    )
    assert outputs.state is not None
    (new_state,) = outputs.state
    assert isinstance(new_state, Tensor)
    np.testing.assert_array_equal(
        np.from_dlpack(new_state.driver_tensor), np.full(2, 10)
    )


def test_the_modulev3_batch_processor_feeds_state_last() -> None:
    kv = NonCallableMock(spec=KVCacheInputsPerDevice)
    state = ToyState((X,))
    processor = DecoderBatchProcessor(ToyConfig(), _runtime())
    processor.state_store = state
    batch = [_context(1, 2)]
    inputs = processor.prepare_initial_token_inputs([batch], kv_cache_inputs=kv)
    (state_input,) = inputs.state
    assert inputs.buffers[-1] is state_input
    assert inputs.request_ids == (batch[0].request_id,)
    assert (
        len(
            processor.get_symbolic_inputs(
                kv_params=NonCallableMock(get_symbolic_inputs=lambda: (X,)),
                device_refs=[CPU_REF],
            )
        )
        == 5
    )


def test_a_stateful_modulev3_registration_releases_its_state() -> None:
    model = _arch(StatefulModuleV3Decoder).pipeline_model
    assert isinstance(model, type)
    assert issubclass(model, ModuleV3DecoderPipelineModel)
    assert issubclass(model, RequestStateMixin)
    assert issubclass(model, SupportsSSMStateWarmup)
    assert pickle.loads(pickle.dumps(model)) is model


class RowsState(ToyState):
    """A state whose tensor input has a symbolic row dim."""

    def __init__(self, bounds: Mapping[str, int]) -> None:
        super().__init__(
            (
                TensorType(DType.float32, ["rows", 2], device=CPU_REF),
                BufferType(DType.float32, ["pool_rows", 2], device=CPU_REF),
            )
        )
        self.bounds = bounds

    def max_dims(self) -> Mapping[str, int]:
        return self.bounds


def _declaring(state: RequestStateStore) -> type[Any]:
    class Declaring(StatelessDecoder):
        @classmethod
        def state_store(cls, config: ToyConfig, **kwargs: Any) -> object:
            return state

    return Declaring


def test_a_state_store_bounds_its_tensor_dims() -> None:
    # A buffer's dims are the store's to size, so only "rows" needs a bound.
    bounded = RowsState({"rows": 4})
    assert (
        build_state_store(
            _declaring(bounded), ToyConfig(), max_batch_size=4, devices=[CPU()]
        )
        is bounded
    )
    with pytest.raises(ValueError, match=r"\['rows'\]"):
        build_state_store(
            _declaring(RowsState({})),
            ToyConfig(),
            max_batch_size=4,
            devices=[CPU()],
        )


def test_declared_state_needs_a_decoder_batch_processor() -> None:
    processor = GraphApiDecoderBatchProcessor(ToyConfig(), _runtime())
    store = bind_state_store(
        StatefulDecoder,
        ToyConfig(),
        processor,
        max_batch_size=4,
        devices=[CPU()],
    )
    assert isinstance(store, ToyState)
    assert processor.state_store is store

    for other in (None, NonCallableMock()):
        with pytest.raises(TypeError, match="only a DecoderBatchProcessor"):
            bind_state_store(
                StatefulDecoder,
                ToyConfig(),
                other,
                max_batch_size=4,
                devices=[CPU()],
            )
    assert (
        bind_state_store(
            StatelessDecoder,
            ToyConfig(),
            None,
            max_batch_size=4,
            devices=[CPU()],
        )
        is None
    )


def test_the_batch_processor_bounds_the_state_dims() -> None:
    processor = DecoderBatchProcessor(ToyConfig(), _runtime())
    processor.state_store = RowsState({"rows": 3})
    kv_params = NonCallableMock()
    assert processor.max_dims(["total_seq_len", "rows"], kv_params) == {
        "total_seq_len": _runtime().max_batch_active_tokens,
        "rows": 3,
    }
    processor.state_store = None
    with pytest.raises(ValueError, match="'rows'"):
        processor.max_dims(["rows"], kv_params)
