# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Utilities for working with mock pipeline_model for unit testing"""

from typing import Optional, Sequence, cast
from unittest.mock import MagicMock

import numpy as np
from max.driver import CPU, Device, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.pipelines import (
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModel,
    SupportedEncoding,
    TextContext,
)
from max.pipelines.kv_cache import (
    KVCacheInputs,
    KVCacheParams,
    KVCacheStrategy,
)
from transformers import AutoConfig


class MockModelInputs(ModelInputs):
    def __init__(
        self,
        active_batch_size: int,
        eos_prob: float,
        kv_cache_inputs: Optional[KVCacheInputs] = None,
    ) -> None:
        self.active_batch_size = active_batch_size
        self.eos_prob = eos_prob
        self.kv_cache_inputs = kv_cache_inputs


class MockPipelineModel(PipelineModel):
    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        huggingface_config: AutoConfig,
        encoding: SupportedEncoding,
        kv_cache_config: KVCacheConfig,
        devices: list[Device] = [],
    ) -> None:
        self.pipeline_config = pipeline_config
        self.huggingface_config = huggingface_config
        self.vocab_size = pipeline_config.vocab_size  # type: ignore
        self.eos_token = pipeline_config.eos_token  # type: ignore
        self.encoding = encoding
        self.kv_cache_config = kv_cache_config

        if not devices:
            self.devices = [CPU()]
        else:
            self.devices = devices

        # This is required to smuggle these parameters in.
        self.max_length = pipeline_config.max_length
        self.kv_manager = MagicMock()

        # These mypy ignores, are needed to smuggle in these settings without
        # reworking these globally.
        self.eos_prob = pipeline_config.eos_prob  # type: ignore

    @classmethod
    def calculate_max_seq_len(
        cls, pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        MAX_LENGTH = 1200
        if pipeline_config.max_length:
            return (
                pipeline_config.max_length
                if pipeline_config.max_length < MAX_LENGTH
                else MAX_LENGTH
            )

        return MAX_LENGTH

    @classmethod
    def get_kv_params(
        cls,
        huggingface_config: AutoConfig,
        n_devices: int,
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        return KVCacheParams(
            dtype=DType.float32,
            n_kv_heads=1,
            head_dim=1,
            enable_prefix_caching=False,
            cache_strategy=KVCacheStrategy.CONTINUOUS,
            page_size=None,
            n_devices=n_devices,
        )

    @classmethod
    def get_num_layers(cls, huggingface_config: AutoConfig) -> int:
        return 1

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
            next_token_logits=Tensor.from_numpy(rand_values),
        )

    def prepare_initial_token_inputs(
        self,
        context_batch: Sequence[TextContext],
        kv_cache_inputs: Optional[KVCacheInputs] = None,
    ) -> ModelInputs:
        return MockModelInputs(
            active_batch_size=len(context_batch),
            eos_prob=self.eos_prob,
            kv_cache_inputs=kv_cache_inputs,
        )

    def prepare_next_token_inputs(
        self,
        next_tokens: Tensor,
        prev_model_inputs: ModelInputs,
    ) -> ModelInputs:
        prev_model_inputs = cast(MockModelInputs, prev_model_inputs)
        return MockModelInputs(
            active_batch_size=prev_model_inputs.active_batch_size,
            eos_prob=self.eos_prob,
            kv_cache_inputs=prev_model_inputs.kv_cache_inputs,
        )
