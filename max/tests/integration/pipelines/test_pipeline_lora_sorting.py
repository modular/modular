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

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from enum import Enum
from typing import Any, TypeVar, cast
from unittest.mock import MagicMock, NonCallableMock, patch

import numpy as np
import pytest
from max.driver import CPU, Buffer, Device
from max.dtype import DType
from max.graph import DeviceRef
from max.interfaces import (
    RequestID,
    TextGenerationInputs,
    TextGenerationOutput,
    TokenBuffer,
)
from max.kv_cache import PagedKVCacheManager
from max.nn.legacy.kv_cache import (
    KVCacheInputs,
    KVCacheInputsSequence,
    KVCacheParams,
)
from max.pipelines.core import TextContext, TTSContext
from max.pipelines.lib import (
    KVCacheConfig,
    LoRAConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModel,
    SamplingConfig,
)
from max.pipelines.lib.lora import LoRAManager, LoRAModel
from max.pipelines.lib.pipeline_variants.text_generation import (
    TextGenerationPipeline,
)
from max.pipelines.lib.speech_token_pipeline import (
    SpeechTokenGenerationPipeline,
)
from transformers import AutoConfig

ContextT = TypeVar("ContextT", TextContext, TTSContext)


class PipelineType(Enum):
    TEXT_GENERATION = "text_generation"
    SPEECH_TOKEN = "speech_token"


class MockLoRARequestProcessor:
    def __init__(
        self,
        manager: LoRAManager,
        zmq_endpoint_base: str,
    ) -> None:
        del manager, zmq_endpoint_base

    def process_lora_requests(self) -> None:
        pass


class MockModelInputs(ModelInputs):
    def __init__(
        self,
        batch_size: int,
        kv_cache_inputs: KVCacheInputsSequence | None = None,
    ) -> None:
        self._batch_size = batch_size
        self.kv_cache_inputs = kv_cache_inputs or KVCacheInputsSequence(
            kv_cache_inputs=[]
        )
        self.return_n_logits = 1

    @property
    def active_batch_size(self) -> int:
        return self._batch_size


class MockPipelineModel(PipelineModel[ContextT]):
    def __init__(
        self,
        vocab_size: int = 1000,
        lora_manager: LoRAManager | None = None,
    ) -> None:
        self.vocab_size = vocab_size
        self.encoding = "float32"
        self.devices = [CPU()]
        self.max_seq_len = 2048

        self.kv_manager = MagicMock(spec=PagedKVCacheManager)
        self.kv_manager.contains = MagicMock(return_value=True)
        self.kv_manager.claim = MagicMock()
        self.kv_manager.alloc = MagicMock()
        self.kv_manager.step = MagicMock()
        self.kv_manager.runtime_inputs = MagicMock(return_value=[])
        self.kv_manager.increment_cache_lengths = MagicMock(return_value=[])

        self._lora_manager = lora_manager

    @classmethod
    def calculate_max_seq_len(
        cls, pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        del pipeline_config, huggingface_config
        return 2048

    @classmethod
    def get_kv_params(
        cls,
        huggingface_config: AutoConfig,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        del huggingface_config, kv_cache_config
        return KVCacheParams(
            dtype=cache_dtype,
            n_kv_heads=1,
            head_dim=1,
            num_layers=1,
            enable_prefix_caching=False,
            cache_strategy="paged",
            devices=[DeviceRef.from_device(d) for d in devices],
        )

    @classmethod
    def infer_optional_batch_size(
        cls,
        pipeline_config: PipelineConfig,
        available_cache_memory: int,
        huggingface_config: AutoConfig,
        devices: list[Device],
    ) -> int:
        del pipeline_config, available_cache_memory
        del huggingface_config, devices
        return 16

    @classmethod
    def estimate_weights_size(cls, pipeline_config: PipelineConfig) -> int:
        del pipeline_config
        return 1000000

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        mock_inputs = cast(MockModelInputs, model_inputs)
        batch_size = mock_inputs.active_batch_size
        rand_values = np.random.rand(batch_size, self.vocab_size).astype(
            np.float32
        )
        return ModelOutputs(
            logits=Buffer.from_numpy(rand_values),
            next_token_logits=Buffer.from_numpy(rand_values),
        )

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[ContextT]],
        kv_cache_inputs: KVCacheInputs | None = None,
        return_n_logits: int = 1,
    ) -> ModelInputs:
        if len(replica_batches) > 1:
            raise ValueError("Model does not support DP>1")

        context_batch = replica_batches[0]
        del return_n_logits
        kv_seq = None
        if isinstance(kv_cache_inputs, KVCacheInputsSequence):
            kv_seq = kv_cache_inputs
        return MockModelInputs(
            batch_size=len(context_batch),
            kv_cache_inputs=kv_seq,
        )

    def prepare_next_token_inputs(
        self,
        next_tokens: Buffer,
        prev_model_inputs: ModelInputs,
    ) -> ModelInputs:
        del next_tokens
        mock_prev = cast(MockModelInputs, prev_model_inputs)
        kv_inputs = mock_prev.kv_cache_inputs
        if isinstance(kv_inputs, KVCacheInputsSequence):
            return MockModelInputs(
                batch_size=mock_prev.active_batch_size,
                kv_cache_inputs=kv_inputs,
            )
        return MockModelInputs(
            batch_size=mock_prev.active_batch_size,
            kv_cache_inputs=None,
        )


class MockSamplingProcessor:
    def __init__(self, batch_size: int, num_steps: int = 1) -> None:
        self._batch_size = batch_size
        self._num_steps = num_steps
        self._generated_tokens = np.tile(
            np.arange(batch_size, dtype=np.int32).reshape(-1, 1),
            (1, num_steps),
        )
        self._step = 0

    @property
    def generated_tokens(self) -> Buffer:
        return Buffer.from_numpy(self._generated_tokens)

    @property
    def new_tokens(self) -> Buffer:
        if self._step < self._num_steps:
            tokens = self._generated_tokens[:, self._step]
            self._step += 1
            return Buffer.from_numpy(tokens.astype(np.int64))
        return Buffer.from_numpy(np.zeros(self._batch_size, dtype=np.int64))


def create_context(
    pipeline_type: PipelineType,
    request_id: str,
    model_name: str | None = None,
    max_length: int = 512,
) -> TextContext | TTSContext:
    tokens = [1, 2, 3, 4, 5]

    if pipeline_type == PipelineType.TEXT_GENERATION:
        context: TextContext | TTSContext = TextContext(
            request_id=RequestID(request_id),
            max_length=max_length,
            tokens=TokenBuffer(np.array(tokens, dtype=np.int64)),
        )
    else:
        context = TTSContext(
            request_id=RequestID(request_id),
            max_length=max_length,
            tokens=TokenBuffer(np.array(tokens, dtype=np.int64)),
            streaming=False,
        )

    if model_name is not None:
        context.model_name = model_name

    return context


def create_lora_manager(
    base_model_path: str,
    lora_names: list[str],
) -> LoRAManager:
    config = LoRAConfig(
        enable_lora=True,
        max_num_loras=len(lora_names) + 1,
        max_lora_rank=8,
        lora_paths=[],
    )

    with patch(
        "max.pipelines.lib.lora.LoRARequestProcessor",
        MockLoRARequestProcessor,
    ):
        manager = LoRAManager(
            config=config,
            base_model_path=base_model_path,
            base_dtype=DType.float32,
            n_heads=32,
            n_kv_heads=8,
            head_dim=128,
            zmq_endpoint_base="fake",
        )

    for name in lora_names:
        fake_lora = NonCallableMock(spec=LoRAModel)
        fake_lora.rank = 8
        fake_lora.name = name
        manager._loras[name] = fake_lora
        manager._active_loras.put(name, fake_lora)

    return manager


def create_pipeline_with_lora(
    pipeline_type: PipelineType,
    base_model_path: str,
    lora_names: list[str],
) -> TextGenerationPipeline[TextContext] | SpeechTokenGenerationPipeline:
    lora_manager = create_lora_manager(base_model_path, lora_names)
    pipeline_model: MockPipelineModel[Any] = MockPipelineModel(
        lora_manager=lora_manager
    )

    mock_config = PipelineConfig.model_construct()
    mock_config.model.quantization_encoding = "float32"
    mock_config.sampling = SamplingConfig()
    mock_config.sampling.enable_structured_output = False
    mock_config.sampling.enable_variable_logits = False

    if pipeline_type == PipelineType.TEXT_GENERATION:

        def mock_text_init(
            self: TextGenerationPipeline[TextContext],
            pipeline_config: Any,
            **kwargs: Any,
        ) -> None:
            del kwargs
            self._pipeline_config = pipeline_config
            self._pipeline_model = pipeline_model
            self._devices = [CPU()]
            self._eos_token_id = {999}
            self._tokenizer = MagicMock()
            self.batch_info_output_fname = None
            self.batch_infos = []
            self.vocab_size = 1000
            self._sampler_without_bitmask = MagicMock()
            self._sampler_with_bitmask = None

        with patch.object(TextGenerationPipeline, "__init__", mock_text_init):
            return TextGenerationPipeline(
                pipeline_config=mock_config,
                pipeline_model=MagicMock(),
                eos_token_id=999,
                weight_adapters={},
                tokenizer=MagicMock(),
            )
    else:

        def mock_speech_init(
            self: SpeechTokenGenerationPipeline,
            pipeline_config: Any,
            **kwargs: Any,
        ) -> None:
            del kwargs
            self._pipeline_config = pipeline_config
            self._pipeline_model = pipeline_model
            self._devices = [CPU()]
            self._eos_token_id = {999}
            self._tokenizer = MagicMock()
            self.batch_info_output_fname = None
            self.batch_infos = []
            self.vocab_size = 1000
            self._sampler_without_bitmask = MagicMock()
            self._sampler_with_bitmask = None
            self.d2h_stream = MagicMock()

        with patch.object(
            SpeechTokenGenerationPipeline, "__init__", mock_speech_init
        ):
            return SpeechTokenGenerationPipeline(
                pipeline_config=mock_config,
                pipeline_model=MagicMock(),
                eos_token_id=999,
                weight_adapters={},
                tokenizer=MagicMock(),
            )


def execute_pipeline(
    pipeline_type: PipelineType,
    pipeline: TextGenerationPipeline[TextContext]
    | SpeechTokenGenerationPipeline,
    batch: dict[RequestID, TextContext] | dict[RequestID, TTSContext],
) -> dict[RequestID, TextGenerationOutput]:
    mock_sampling_processor = MockSamplingProcessor(len(batch), 1)

    if pipeline_type == PipelineType.TEXT_GENERATION:
        patch_base = "max.pipelines.lib.pipeline_variants.text_generation"
        inputs: TextGenerationInputs[TextContext] = TextGenerationInputs(
            batches=[list(cast(dict[RequestID, TextContext], batch).values())],
            num_steps=1,
        )
        with (
            patch(
                f"{patch_base}.FusedSamplingProcessor",
                return_value=mock_sampling_processor,
            ),
            patch(f"{patch_base}.apply_logits_processors"),
        ):
            return cast(TextGenerationPipeline[TextContext], pipeline).execute(
                inputs
            )
    else:
        patch_base = "max.pipelines.lib.speech_token_pipeline"
        tokens_to_generate = {ctx.request_id: 1 for ctx in batch.values()}
        with (
            patch(
                f"{patch_base}.FusedSamplingProcessor",
                return_value=mock_sampling_processor,
            ),
            patch(f"{patch_base}.apply_logits_processors"),
        ):
            return cast(
                SpeechTokenGenerationPipeline, pipeline
            ).next_speech_token(
                batch=cast(dict[RequestID, TTSContext], batch),
                num_steps=1,
                tokens_to_generate=tokens_to_generate,
            )


def run_lora_sorting_test(
    pipeline_type: PipelineType,
    lora_names: list[str],
    context_configs: list[tuple[str, str | None]],
    expected_token_mapping: dict[str, int],
) -> None:
    pipeline = create_pipeline_with_lora(
        pipeline_type, "/mock/model", lora_names
    )

    contexts = [
        create_context(pipeline_type, req_id, model_name=model_name)
        for req_id, model_name in context_configs
    ]

    batch: dict[RequestID, Any] = OrderedDict(
        [(ctx.request_id, ctx) for ctx in contexts]
    )

    result = execute_pipeline(pipeline_type, pipeline, batch)

    assert len(result) == len(contexts)
    for ctx in contexts:
        expected_token = expected_token_mapping[str(ctx.request_id)]
        assert result[ctx.request_id].tokens[-1] == expected_token


@pytest.mark.parametrize("pipeline_type", list(PipelineType))
def test_mixed_base_and_lora_batch(pipeline_type: PipelineType) -> None:
    run_lora_sorting_test(
        pipeline_type=pipeline_type,
        lora_names=["lora_a"],
        context_configs=[
            ("base_0", None),
            ("lora_0", "lora_a"),
            ("base_1", None),
            ("lora_1", "lora_a"),
        ],
        expected_token_mapping={
            "lora_0": 0,
            "lora_1": 1,
            "base_0": 2,
            "base_1": 3,
        },
    )


@pytest.mark.parametrize("pipeline_type", list(PipelineType))
def test_without_lora_preserves_order(pipeline_type: PipelineType) -> None:
    run_lora_sorting_test(
        pipeline_type=pipeline_type,
        lora_names=[],
        context_configs=[
            ("base_0", None),
            ("base_1", None),
            ("base_2", None),
        ],
        expected_token_mapping={
            "base_0": 0,
            "base_1": 1,
            "base_2": 2,
        },
    )


@pytest.mark.parametrize("pipeline_type", list(PipelineType))
def test_interleaved_requests(pipeline_type: PipelineType) -> None:
    run_lora_sorting_test(
        pipeline_type=pipeline_type,
        lora_names=["lora_a"],
        context_configs=[
            ("base_0", None),
            ("lora_1", "lora_a"),
            ("base_2", None),
            ("lora_3", "lora_a"),
            ("base_4", None),
            ("lora_5", "lora_a"),
        ],
        expected_token_mapping={
            "lora_1": 0,
            "lora_3": 1,
            "lora_5": 2,
            "base_0": 3,
            "base_2": 4,
            "base_4": 5,
        },
    )


@pytest.mark.parametrize("pipeline_type", list(PipelineType))
def test_multiple_lora_adapters(pipeline_type: PipelineType) -> None:
    run_lora_sorting_test(
        pipeline_type=pipeline_type,
        lora_names=["lora_a", "lora_b"],
        context_configs=[
            ("base_0", None),
            ("lora_0", "lora_a"),
            ("lora_1", "lora_b"),
            ("base_1", None),
        ],
        expected_token_mapping={
            "lora_1": 0,
            "lora_0": 1,
            "base_0": 2,
            "base_1": 3,
        },
    )


@pytest.mark.parametrize("pipeline_type", list(PipelineType))
def test_all_lora_batch(pipeline_type: PipelineType) -> None:
    run_lora_sorting_test(
        pipeline_type=pipeline_type,
        lora_names=["lora_a", "lora_b"],
        context_configs=[
            ("lora_0", "lora_a"),
            ("lora_1", "lora_b"),
        ],
        expected_token_mapping={
            "lora_1": 0,
            "lora_0": 1,
        },
    )


@pytest.mark.parametrize("pipeline_type", list(PipelineType))
def test_all_base_batch(pipeline_type: PipelineType) -> None:
    run_lora_sorting_test(
        pipeline_type=pipeline_type,
        lora_names=["unused_lora"],
        context_configs=[
            ("base_0", None),
            ("base_1", None),
            ("base_2", None),
        ],
        expected_token_mapping={
            "base_0": 0,
            "base_1": 1,
            "base_2": 2,
        },
    )
