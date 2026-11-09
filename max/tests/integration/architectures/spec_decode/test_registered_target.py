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
"""Checks a registered ModuleV3 decoder as a speculative pairing's target."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any, ClassVar
from unittest.mock import NonCallableMock

import numpy as np
import pytest
from max import functional as F
from max.driver import CPU, Buffer
from max.dtype import DType
from max.graph import DeviceRef, TensorType
from max.graph.weights import WeightsFormat
from max.nn import Module
from max.nn.outputs import ReturnHiddenStates, ReturnLogits
from max.pipelines.context import TextContext
from max.pipelines.lib import GraphRole, SupportedArchitecture, TextTokenizer
from max.pipelines.lib.graph_specs import GraphSpec
from max.pipelines.lib.module_registration import (
    DecoderBatchProcessor,
    DecoderModelInputs,
    RegisteredDecoderBuilder,
)
from max.pipelines.lib.pipeline_variants import spec_decode_pipeline_model
from max.pipelines.lib.pipeline_variants.spec_decode_pipeline_model import (
    SpecDecodeArchConfig,
    _BatchProcessor,
    _builds_module,
    _module_builder,
    _Modules,
    _share_target_slots,
    _SpecDecodeModuleV3PipelineModel,
    _SpecDecodeModuleV3VisionCachePipelineModel,
    _takes_images,
    spec_decode_architecture,
)
from max.pipelines.lib.role_contracts import DecoderInputs, DecoderOutputs
from max.pipelines.modeling.types import InputModality, PipelineTask
from max.pipelines.speculative import spec_adapters
from max.pipelines.speculative.spec_input_types import SpecDecodeInputTypeSpec
from max.tensor import Tensor
from test_common.context_utils import create_text_context
from typing_extensions import Self

_NAME = "ToyForCausalLM"


@dataclass(frozen=True)
class _Config:
    @classmethod
    def initialize(
        cls, pipeline_config: Any, model_config: Any = None, *, max_seq_len: int
    ) -> Self:
        return cls()

    def get_max_seq_len(self) -> int:
        return 16

    @classmethod
    def calculate_max_seq_len(
        cls, huggingface_config: Any, model_config: Any
    ) -> int:
        return 16


class _Decoder(Module[[DecoderInputs], DecoderOutputs]):
    def __init__(self, config: _Config) -> None:
        super().__init__()

    def forward(self, inputs: DecoderInputs) -> DecoderOutputs:
        return DecoderOutputs(next_token_logits=inputs.tokens)


def _arch(name: str, **kwargs: Any) -> SupportedArchitecture:
    return SupportedArchitecture(
        name=name,
        example_repo_ids=[],
        default_encoding="float32",
        supported_encodings={"float32"},
        task=PipelineTask.TEXT_GENERATION,
        tokenizer=TextTokenizer,
        context_type=TextContext,
        default_weights_format=WeightsFormat.safetensors,
        config=_Config,
        **kwargs,
    )


_DECODER = _arch(f"{_NAME}_ModuleV3", modules={GraphRole.DECODER: _Decoder})

_PIXELS = TensorType(DType.float32, [2], device=DeviceRef.CPU())


class _VisionDecoder(_Decoder):
    consumes_vision: ClassVar[bool] = True


class _Vision(Module[[Tensor], Tensor]):
    def __init__(self, config: _Config) -> None:
        super().__init__()
        self.weight = Tensor.zeros([2], dtype=DType.float32, device=CPU())

    def input_types(self) -> tuple[TensorType, ...]:
        return (_PIXELS,)

    def forward(self, pixels: Tensor) -> Tensor:
        return pixels + self.weight


class _VisionBatching(DecoderBatchProcessor):
    def pack_vision_inputs(self, *args: Any, **kwargs: Any) -> None:
        return None

    def vision_execute(self, *args: Any, **kwargs: Any) -> list[Buffer]:
        return []

    def empty_vision_embeddings(self, devices: Any) -> list[Buffer]:
        return []

    def bind_vision_encoder(self, *, vision_model: Any, session: Any) -> None:
        self.bound = (vision_model, session)


_VISION_TARGET = _arch(
    f"{_NAME}_ModuleV3",
    modules={GraphRole.DECODER: _VisionDecoder, GraphRole.VISION: _Vision},
    batching=_VisionBatching,
    input_modalities={InputModality.TEXT, InputModality.IMAGE},
)


def _calls(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    monkeypatch.setattr(
        spec_adapters, "_TARGET_ADAPTERS", dict.fromkeys(names, "")
    )


def test_a_registered_decoder_targets_only_where_spec_decoding_calls_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The suffixless name is its graph-API twin's, which doesn't call it.
    _calls(monkeypatch, _NAME)
    assert not _builds_module(_DECODER)
    _calls(monkeypatch, _DECODER.name)
    assert _builds_module(_DECODER)
    assert not _builds_module(_DECODER, GraphRole.DRAFT)
    builder = _module_builder(_DECODER)
    assert isinstance(builder, RegisteredDecoderBuilder)
    assert builder.pipeline_model is _DECODER.pipeline_model


def _draft(*modalities: InputModality) -> Any:
    return SimpleNamespace(
        draft=SimpleNamespace(method="mtp"),
        supports_device_graph_capture=True,
        requires_max_batch_context_length=False,
        multi_gpu_supported=True,
        checkpoint_draft_width=None,
        input_modalities=set(modalities),
    )


@pytest.mark.parametrize(
    ("twin", "prefer_module_v3", "pairs"),
    [
        (False, True, True),
        (False, False, True),
        (True, True, True),
        (True, False, False),
    ],
)
def test_a_registered_decoder_pairs_as_the_variant_serving_selects(
    monkeypatch: pytest.MonkeyPatch,
    twin: bool,
    prefer_module_v3: bool,
    pairs: bool,
) -> None:
    """Under the default lookup a graph-API twin serves, and keeps its own
    speculative architectures."""
    _calls(monkeypatch, _DECODER.name)
    graph_api = _arch(_NAME, pipeline_model=SimpleNamespace)
    draft = _draft()
    target = replace(_DECODER, checkpoint_drafts={"mtp": "ToyDraft"})

    def find(name: str, prefer_module_v3: bool = False) -> Any:
        if name == "ToyDraft":
            return draft
        if twin and not prefer_module_v3:
            return graph_api
        return target

    monkeypatch.setattr(spec_decode_pipeline_model.ARCH_LOOKUP, "find", find)
    arch = spec_decode_architecture(
        _NAME, "mtp", None, prefer_module_v3=prefer_module_v3
    )
    assert (arch is not None) is pairs
    if arch is not None:
        assert arch.name == f"{target.name}+mtp"


def test_a_batch_without_context_lengths_takes_each_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A registered decoder's batching keeps its replica's length for its
    own graph only, and the fused graph takes one per device."""
    page = 4
    contexts = [
        create_text_context(np.arange(5, dtype=np.int64)),
        create_text_context(np.arange(2, dtype=np.int64)),
    ]
    for name, value in {
        "batch_distributed_buffers": lambda arch, inputs: (None, ()),
        "batch_trailing_buffers": lambda arch, method, inputs: [],
        "batch_replay_buffers": lambda arch, method, inputs: [],
    }.items():
        monkeypatch.setattr(spec_adapters, name, value)
    batching: Any = object.__new__(_BatchProcessor)
    batching.spec_config = SimpleNamespace(
        input_spec=SpecDecodeInputTypeSpec(
            devices=[DeviceRef.CPU(), DeviceRef.CPU()], distributed=True
        )
    )
    batching.runtime = SimpleNamespace(
        devices=[CPU(), CPU()],
        signal_buffers=[],
        pipeline_config=SimpleNamespace(
            model=SimpleNamespace(
                kv_cache=SimpleNamespace(kv_cache_page_size=page)
            )
        ),
    )
    batching._pairing = SimpleNamespace(adapter_arch=_DECODER.name)
    batching._method = "mtp"
    batching._seed_counter = 0
    empty = Buffer.from_numpy(np.zeros(1, dtype=np.int64))
    target_inputs = SimpleNamespace(
        tokens=empty,
        input_row_offsets=empty,
        return_n_logits=empty,
        kv_cache_inputs=None,
        batch_context_lengths=None,
        ep_inputs=(),
    )
    batching.__dict__["_target_batching"] = SimpleNamespace(
        prepare_initial_token_inputs=lambda *args: target_inputs
    )
    batching.__dict__["_replay_hooks"] = None
    first = batching._distributed_inputs([contexts], None, 1)
    second = batching._distributed_inputs([contexts], None, 1)
    # Each request's 5 and 2 tokens round up to whole pages of 4.
    assert [b.to_numpy()[0] for b in first.batch_context_lengths] == [12, 12]
    # The buffers persist, so a captured graph reads each batch's lengths.
    assert all(
        a is b
        for a, b in zip(
            first.batch_context_lengths,
            second.batch_context_lengths,
            strict=True,
        )
    )
    assert first.batch_context_lengths[0].dtype == DType.int32


@pytest.mark.parametrize("distributed", [True, False])
def test_a_registered_targets_batch_takes_the_pairings_signal_buffers(
    monkeypatch: pytest.MonkeyPatch, distributed: bool
) -> None:
    """A registered decoder's inputs hold no signal buffers, which the fused
    graph takes on every device it spans."""
    for name, value in {
        "batch_distributed_buffers": lambda arch, inputs: (None, ()),
        "batch_trailing_buffers": lambda arch, method, inputs: [],
        "batch_replay_buffers": lambda arch, method, inputs: [],
    }.items():
        monkeypatch.setattr(spec_adapters, name, value)
    signals = [Buffer.from_numpy(np.zeros(1, dtype=np.uint8)) for _ in range(2)]
    batching: Any = object.__new__(_BatchProcessor)
    batching.spec_config = SimpleNamespace(
        input_spec=SpecDecodeInputTypeSpec(
            devices=[DeviceRef.CPU(), DeviceRef.CPU()],
            distributed=distributed,
        )
    )
    batching.runtime = SimpleNamespace(
        devices=[CPU(), CPU()],
        signal_buffers=signals,
        pipeline_config=SimpleNamespace(
            model=SimpleNamespace(
                kv_cache=SimpleNamespace(kv_cache_page_size=4)
            )
        ),
    )
    batching._pairing = SimpleNamespace(adapter_arch=_DECODER.name)
    batching._method = "mtp"
    batching._seed_counter = 0
    tokens = Buffer.from_numpy(np.zeros(1, dtype=np.int64))
    target_inputs = DecoderModelInputs(
        tokens=tokens, input_row_offsets=tokens, return_n_logits=tokens
    )
    assert target_inputs.signal_buffers == ()
    batching.__dict__["_target_batching"] = SimpleNamespace(
        prepare_initial_token_inputs=lambda *args: target_inputs
    )
    batching.__dict__["_replay_hooks"] = None
    prepare = (
        batching._distributed_inputs if distributed else batching._target_inputs
    )
    contexts = [create_text_context(np.arange(3, dtype=np.int64))]
    inputs = prepare([contexts], None, 1)
    assert inputs.signal_buffers == signals


def test_a_registered_vision_target_pairs_through_the_vision_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A registered target with a VISION module encodes a pairing's images
    through the pipeline's vision cache, where its draft takes them too."""
    _calls(monkeypatch, _VISION_TARGET.name)
    draft = _draft(InputModality.TEXT, InputModality.IMAGE)
    target = replace(_VISION_TARGET, checkpoint_drafts={"mtp": "ToyDraft"})
    assert _takes_images(target, draft)
    assert not _takes_images(target, _draft(InputModality.TEXT))
    assert not _takes_images(_DECODER, draft)

    def find(name: str, prefer_module_v3: bool = False) -> Any:
        return draft if name == "ToyDraft" else target

    monkeypatch.setattr(spec_decode_pipeline_model.ARCH_LOOKUP, "find", find)
    arch = spec_decode_architecture(_NAME, "mtp", None, prefer_module_v3=True)
    assert arch is not None
    assert arch.pipeline_model is _SpecDecodeModuleV3VisionCachePipelineModel
    assert arch.input_modalities == target.input_modalities


def test_the_pairing_binds_a_registered_targets_compiled_vision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pairing builds the target's VISION module as it builds serving
    alone, and binds the compiled callable to the target's batching."""
    weight = np.array([1, 2], dtype=np.float32)
    monkeypatch.setattr(
        "max.pipelines.lib.module_registration.parse_state_dict_from_weights",
        lambda *args, **kwargs: {"weight": weight},
    )
    fused = NonCallableMock()
    monkeypatch.setattr(
        _SpecDecodeModuleV3PipelineModel,
        "load_model",
        lambda self, session: fused,
    )
    specs: list[Mapping[str, GraphSpec]] = []
    vision = SimpleNamespace(engine_model=object())

    def compile_and_init(
        session: object, roles: Mapping[str, GraphSpec]
    ) -> Any:
        specs.append(roles)
        return {GraphRole.VISION: vision}

    monkeypatch.setattr(
        spec_decode_pipeline_model, "compile_and_init_graphs", compile_and_init
    )
    model: Any = object.__new__(_SpecDecodeModuleV3VisionCachePipelineModel)
    model.pipeline_config = SimpleNamespace(
        model=SimpleNamespace(pipeline_stage=None)
    )
    model.weights = object()
    model.adapter = None
    model.memory_plan = None
    model.return_logits = ReturnLogits.VARIABLE
    model.return_hidden_states = ReturnHiddenStates.NONE
    model.devices = [CPU()]
    model.compiled_modules = {GraphRole.SPEC_DECODER: fused}
    model.weight_naming = {}
    model.arch_config = object.__new__(SpecDecodeArchConfig)
    model.arch_config.target = _Config()
    model.__dict__["pairing"] = SimpleNamespace(target=_VISION_TARGET)
    # Neither distributed nor preparing every batch, so the target's
    # batching only encodes the images.
    batching: Any = object.__new__(_BatchProcessor)
    batching.spec_config = SimpleNamespace(
        input_spec=SpecDecodeInputTypeSpec(
            devices=[DeviceRef.CPU()], distributed=False
        ),
        target=_Config(),
    )
    batching.runtime = SimpleNamespace(devices=[CPU()])
    batching._pairing = SimpleNamespace(target=_VISION_TARGET)
    batching._takes_images = True
    batching._prepares_batches = False
    model._batch_processor = batching
    session = object()

    assert model.load_model(session) is fused
    (roles,) = specs
    spec = roles[GraphRole.VISION]
    assert isinstance(spec.module, _Vision)
    assert tuple(spec.input_types) == (_PIXELS,)
    assert spec.weights is not None
    np.testing.assert_array_equal(spec.weights["weight"], weight)
    assert model.compiled_modules[GraphRole.VISION] is vision
    assert model.compiled_modules[GraphRole.SPEC_DECODER] is fused
    assert model.vision_model is vision.engine_model
    target_batching = batching._target_batching
    assert isinstance(target_batching, _VisionBatching)
    assert target_batching.bound == (vision, session)
    # The pairing still stages the text itself.
    assert not batching._target_prepares_inputs


class _Slot(Module[[Tensor], Tensor]):
    def __init__(self) -> None:
        super().__init__()
        self.weight = Tensor.zeros([2], dtype=DType.float32, device=CPU())

    def forward(self, x: Tensor) -> Tensor:
        return x


class _LanguageModel(Module[[Tensor], Tensor]):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = _Slot()

    def forward(self, x: Tensor) -> Tensor:
        return x


class _Wrapper(Module[[Tensor], Tensor]):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = _LanguageModel()

    def forward(self, x: Tensor) -> Tensor:
        return x


class _WrappingDecoder(Module[[Tensor], Tensor]):
    """Holds its language model under a name-transparent wrapper."""

    def __init__(self) -> None:
        super().__init__()
        self.model = _Wrapper()
        self.model.name_transparent = True

    def forward(self, x: Tensor) -> Tensor:
        return x


class _SlotDraft(Module[[Tensor], Tensor]):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = _Slot()

    def forward(self, x: Tensor) -> Tensor:
        return x


def test_a_draft_shares_the_slot_a_registered_target_wraps() -> None:
    """A registered decoder may hold its language model deeper than the
    slot's own name, and the draft still reads the target's weight."""
    with F.lazy():
        target, draft = _WrappingDecoder(), _SlotDraft()
    weight = np.ones(2, dtype=np.float32)
    modules = _Modules(
        target,
        None,
        draft,
        None,
        ReturnHiddenStates.NONE,
        {"target.language_model.embed_tokens.weight": weight},
    )
    _share_target_slots(modules)
    assert draft.embed_tokens is target.model.language_model.embed_tokens
    assert modules.weights["draft.embed_tokens.weight"] is weight
