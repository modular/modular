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
"""Utilities for working with mock pipeline_model for unit testing"""

from collections.abc import Sequence
from typing import cast
from unittest.mock import MagicMock

import numpy as np
from max.driver import CPU, Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef
from max.graph.weights import Weights, WeightsAdapter
from max.kv_cache import PagedKVCacheManager
from max.nn.legacy.kv_cache import (
    KVCacheInputs,
    KVCacheParams,
)
from max.nn.legacy.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.core import TextContext
from max.pipelines.lib import (
    KVCacheConfig,
    LoRAManager,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModel,
    SupportedEncoding,
)
from transformers import AutoConfig


class MockModelInputs(ModelInputs):
    def __init__(
        self,
        active_batch_size: int,
        eos_prob: float,
        kv_cache_inputs: KVCacheInputs | None = None,
        return_n_logits: int | Buffer = 1,
    ) -> None:
        self.active_batch_size = active_batch_size
        self.eos_prob = eos_prob
        self.tokens = Buffer.from_numpy(
            np.zeros((max(active_batch_size, 1),), dtype=np.int64)
        )
        self.input_row_offsets = Buffer.from_numpy(
            np.array([0, max(active_batch_size, 1)], dtype=np.uint32)
        )
        self.signal_buffers: list[Buffer] = []
        self.kv_cache_inputs = kv_cache_inputs
        if isinstance(return_n_logits, Buffer):
            self.return_n_logits = return_n_logits
        else:
            self.return_n_logits = Buffer.from_numpy(
                np.array([return_n_logits], dtype=np.uint32)
            )

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        kv_cache_inputs = tuple(self.kv_cache_inputs or ())
        return (
            self.tokens,
            self.input_row_offsets,
            self.return_n_logits,
            *self.signal_buffers,
            *kv_cache_inputs,
        )


class MockPipelineModel(PipelineModel):
    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        huggingface_config: AutoConfig,
        encoding: SupportedEncoding,
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        devices: list[Device] = [],  # noqa: B006
        adapter: WeightsAdapter | None = None,
        return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
        return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE,
    ) -> None:
        self.pipeline_config = pipeline_config
        self.huggingface_config = huggingface_config
        self.vocab_size = pipeline_config.vocab_size  # type: ignore
        self.eos_token = pipeline_config.eos_token  # type: ignore
        self.encoding = encoding
        self.kv_cache_config = kv_cache_config
        self.weights = weights
        self.adapter = adapter
        self.return_logits = return_logits
        self.return_hidden_states = return_hidden_states

        if not devices:
            self.devices = [CPU()]
        else:
            self.devices = devices

        # This is required to smuggle these parameters in.
        self.max_length = pipeline_config.max_length
        self.kv_manager = MagicMock(spec=PagedKVCacheManager)

        # These mypy ignores, are needed to smuggle in these settings without
        # reworking these globally.
        self.eos_prob = pipeline_config.eos_prob  # type: ignore
        self.max_seq_len = self.calculate_max_seq_len(
            pipeline_config, huggingface_config
        )
        self._lora_manager = (
            LoRAManager(
                config=self.pipeline_config.lora,
                base_model_path=pipeline_config.model.model_path,
                base_dtype=self.encoding.dtype,
                n_heads=huggingface_config.num_attention_heads,
                n_kv_heads=huggingface_config.num_key_value_heads,
                head_dim=huggingface_config.head_dim,
                zmq_endpoint_base=self.pipeline_config.zmq_endpoint_base,
            )
            if self.pipeline_config.lora
            and self.pipeline_config.lora.enable_lora
            else None
        )

    @classmethod
    def calculate_max_seq_len(
        cls, pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        MAX_LENGTH = 1200
        if pipeline_config.max_length:
            return min(MAX_LENGTH, pipeline_config.max_length)

        return MAX_LENGTH

    @classmethod
    def get_kv_params(
        cls,
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        return KVCacheParams(
            dtype=cache_dtype,
            n_kv_heads=1,
            head_dim=1,
            num_layers=1,
            enable_prefix_caching=False,
            cache_strategy="paged",
            devices=devices,
            data_parallel_degree=pipeline_config.model.data_parallel_degree,
        )

    @classmethod
    def infer_optional_batch_size(
        cls,
        pipeline_config: PipelineConfig,
        available_cache_memory: int,
        huggingface_config: AutoConfig,
        devices: list[Device],
    ) -> int:
        return 16

    @classmethod
    def estimate_weights_size(cls, pipeline_config: PipelineConfig) -> int:
        return 1000000

    def execute(
        self,
        model_inputs: ModelInputs,
    ) -> ModelOutputs:
        model_inputs = cast(MockModelInputs, model_inputs)

        # Generate Random values
        rand_values = np.random.rand(
            model_inputs.active_batch_size,
            self.vocab_size,
        ).astype(np.float32)

        # This will randomly spike the eos token logit probability
        # 10% of the time.
        for i in range(model_inputs.active_batch_size):
            if np.random.uniform() <= model_inputs.eos_prob:
                rand_values[i, self.eos_token] += 0.9

        return ModelOutputs(
            logits=Buffer.from_numpy(rand_values),
            next_token_logits=Buffer.from_numpy(rand_values),
        )

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputs | None = None,
        return_n_logits: int = 1,
    ) -> ModelInputs:
        actual_batch_size = sum(len(batch) for batch in replica_batches)
        return MockModelInputs(
            active_batch_size=actual_batch_size,
            eos_prob=self.eos_prob,
            kv_cache_inputs=kv_cache_inputs,
            return_n_logits=return_n_logits,
        )

    def prepare_next_token_inputs(
        self,
        next_tokens: Buffer,
        prev_model_inputs: ModelInputs,
    ) -> ModelInputs:
        prev_model_inputs = cast(MockModelInputs, prev_model_inputs)
        return MockModelInputs(
            active_batch_size=prev_model_inputs.active_batch_size,
            eos_prob=self.eos_prob,
            kv_cache_inputs=prev_model_inputs.kv_cache_inputs,
            return_n_logits=prev_model_inputs.return_n_logits,
        )
