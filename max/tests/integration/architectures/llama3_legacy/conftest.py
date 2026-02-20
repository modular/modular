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

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch
from max.driver import Buffer, Device, DeviceSpec
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef
from max.graph.weights import SafetensorWeights
from max.kv_cache.paged_kv_cache import PagedKVCacheManager
from max.nn.legacy.kv_cache import (
    KVCacheInputs,
    KVCacheInputsSequence,
    KVCacheParams,
)
from max.pipelines.architectures.llama3_legacy.model import Llama3Inputs
from max.pipelines.lib import ModelOutputs
from test_common.context_utils import create_text_context
from test_common.mocks import DummyPipelineConfig
from transformers import LlamaForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig

LLAMA_3_1_HF_REPO_ID = "meta-llama/Llama-3.1-8B-Instruct"


@pytest.fixture
def hf_config() -> LlamaConfig:
    """Small LlamaConfig used by Llama3 graph tests."""
    return LlamaConfig(
        hidden_size=64,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_hidden_layers=2,
        intermediate_size=128,
        vocab_size=256,
        max_position_embeddings=2048,
        rope_theta=500000.0,
        rms_norm_eps=1e-5,
    )


def weights_from_hf_config(hf_config: LlamaConfig) -> SafetensorWeights:
    """Auto-generate zero-valued weights matching a HuggingFace config."""
    hf_model = LlamaForCausalLM(hf_config)
    weight_map = {
        name: param.detach().zero_().to(torch.bfloat16)
        for name, param in hf_model.named_parameters()
    }
    return SafetensorWeights([], _st_weight_map=weight_map)


@pytest.fixture
def weights(hf_config: LlamaConfig) -> SafetensorWeights:
    """Standard SafetensorWeights generated from hf_config."""
    return weights_from_hf_config(hf_config)


@pytest.fixture
def make_pipeline_config(
    hf_config: LlamaConfig,
) -> Callable[..., DummyPipelineConfig]:
    """Factory that creates a fully-configured DummyPipelineConfig."""

    def _make(
        device_specs: list[DeviceSpec],
        max_batch_size: int = 1,
    ) -> DummyPipelineConfig:
        pipeline_config = DummyPipelineConfig(
            model_path=LLAMA_3_1_HF_REPO_ID,
            max_batch_size=max_batch_size,
            max_length=hf_config.max_position_embeddings,
            quantization_encoding="bfloat16",
            kv_cache_strategy="paged",
            device_specs=device_specs,
        )
        pipeline_config.model.kv_cache._available_cache_memory = 1024**4  # 1TB
        pipeline_config.model.kv_cache._cache_dtype = DType.bfloat16
        pipeline_config.model._huggingface_config = hf_config
        pipeline_config.model.weight_path = [Path("fake.safetensors")]
        return pipeline_config

    return _make


@pytest.fixture
def make_kv_inputs(hf_config: LlamaConfig) -> Callable[..., KVCacheInputs]:
    """Factory that creates KVCacheInputs."""

    def _make(
        pipeline_config: DummyPipelineConfig,
        session: InferenceSession,
        device_refs: list[DeviceRef],
        *,
        data_parallel_degree: int | None = None,
        num_replicas: int = 1,
        total_num_pages: int = 1,
        input_seq_len: int = 3,
    ) -> KVCacheInputs:
        kv_params_kwargs = dict(
            dtype=pipeline_config.model.kv_cache._cache_dtype,
            num_layers=hf_config.num_hidden_layers,
            n_kv_heads=hf_config.num_key_value_heads,
            head_dim=hf_config.head_dim,
            devices=device_refs,
        )
        if data_parallel_degree is not None:
            kv_params_kwargs["data_parallel_degree"] = data_parallel_degree

        kv_params = KVCacheParams(**kv_params_kwargs)
        kv_manager = PagedKVCacheManager(
            params=kv_params,
            total_num_pages=total_num_pages,
            session=session,
            max_batch_size=pipeline_config.max_batch_size or 1,
        )

        contexts = []
        batches = []
        for i in range(num_replicas):
            ctx = create_text_context(np.empty(input_seq_len, dtype=np.int64))
            kv_manager.claim(ctx.request_id, replica_idx=i)
            kv_manager.alloc(ctx, replica_idx=i, num_steps=1)
            contexts.append(ctx)
            batches.append([ctx])

        runtime_inputs = kv_manager.get_runtime_inputs(batches)
        kv_inputs: KVCacheInputs
        if len(device_refs) > 1:
            kv_inputs = KVCacheInputsSequence(kv_cache_inputs=runtime_inputs)
        else:
            kv_inputs = runtime_inputs[0]

        return kv_inputs

    return _make


@pytest.fixture
def make_inputs() -> Callable[..., Llama3Inputs]:
    """Factory that creates Llama3Inputs for testing."""

    def _make(
        devices: list[Device],
        kv_inputs: KVCacheInputs,
        *,
        input_seq_len: int = 3,
        num_batches: int = 1,
        signal_buffers: list[Buffer] | None = None,
        data_parallel_splits: Buffer | None = None,
    ) -> Llama3Inputs:
        if signal_buffers is None:
            signal_buffers = []
        total_tokens = num_batches * input_seq_len
        offsets = [i * input_seq_len for i in range(num_batches + 1)]
        return Llama3Inputs(
            tokens=Buffer.from_numpy(
                np.arange(total_tokens, dtype=np.int64)
            ).to(devices[0]),
            input_row_offsets=Buffer.from_numpy(
                np.array(offsets, dtype=np.uint32)
            ).to(devices[0]),
            signal_buffers=signal_buffers,
            return_n_logits=Buffer.from_numpy(np.array([1], dtype=np.int64)),
            kv_cache_inputs=kv_inputs,
            data_parallel_splits=data_parallel_splits,
        )

    return _make


@pytest.fixture
def assert_model_outputs(
    hf_config: LlamaConfig,
) -> Callable[[ModelOutputs, Llama3Inputs, list[Device]], None]:
    """Factory that returns a callable to validate model output types."""

    def _assert(
        outputs: ModelOutputs,
        inputs: Llama3Inputs,
        devices: list[Device],
    ) -> None:
        assert isinstance(outputs, ModelOutputs)
        assert isinstance(outputs.logits, Buffer)
        # With return_n_logits=1, we get logits for the last token only
        assert outputs.logits.shape[0] == inputs.input_row_offsets.shape[0] - 1
        assert outputs.logits.shape[1] == hf_config.vocab_size
        assert outputs.logits.dtype == DType.float32
        assert outputs.logits.device == devices[0]
        assert (outputs.logits.to_numpy() == 0).all(), "Logits should be 0"
        assert isinstance(outputs.next_token_logits, Buffer)
        assert (
            outputs.next_token_logits.shape[0]
            == inputs.input_row_offsets.shape[0] - 1
        )
        assert outputs.next_token_logits.shape[1] == hf_config.vocab_size
        assert outputs.next_token_logits.dtype == DType.float32
        assert outputs.next_token_logits.device == devices[0]
        assert (outputs.next_token_logits.to_numpy() == 0).all(), (
            "Next token logits should be 0"
        )
        assert outputs.logit_offsets is None
        assert outputs.hidden_states is None

    return _assert
