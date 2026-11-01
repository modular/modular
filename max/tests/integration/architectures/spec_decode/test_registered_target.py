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

from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from max.driver import CPU, Buffer
from max.dtype import DType
from max.graph import DeviceRef
from max.graph.weights import WeightsFormat
from max.nn import Module
from max.pipelines.context import TextContext
from max.pipelines.lib import GraphRole, SupportedArchitecture, TextTokenizer
from max.pipelines.lib.module_registration import RegisteredDecoderBuilder
from max.pipelines.lib.pipeline_variants import spec_decode_pipeline_model
from max.pipelines.lib.pipeline_variants.spec_decode_pipeline_model import (
    _BatchProcessor,
    _builds_module,
    _module_builder,
    spec_decode_architecture,
)
from max.pipelines.lib.role_contracts import DecoderInputs, DecoderOutputs
from max.pipelines.modeling.types import PipelineTask
from max.pipelines.speculative import spec_adapters
from max.pipelines.speculative.spec_input_types import SpecDecodeInputTypeSpec
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
    draft = SimpleNamespace(
        draft=SimpleNamespace(method="mtp"),
        supports_device_graph_capture=True,
        requires_max_batch_context_length=False,
        multi_gpu_supported=True,
        checkpoint_draft_width=None,
        input_modalities=set(),
    )
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
