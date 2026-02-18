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
"""Unit test for building a Llama3 graph on GPU."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import torch
from max.driver import load_devices, scan_available_devices
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph.weights import SafetensorWeights
from max.pipelines.architectures.llama3_legacy.model import (
    Llama3Model,
)
from max.pipelines.lib import SupportedEncoding
from test_common.mocks import DummyPipelineConfig
from transformers import LlamaForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig


def weights_from_hf_config(hf_config: LlamaConfig) -> SafetensorWeights:
    """Auto-generate weights with correct names and shapes."""
    hf_model = LlamaForCausalLM(hf_config)
    weight_map = {
        name: param.detach().zero_().to(torch.bfloat16)
        for name, param in hf_model.named_parameters()
    }
    return SafetensorWeights([], _st_weight_map=weight_map)


def test_build_llama3_graph() -> None:
    """Test building a Llama3 graph on GPU."""

    hf_config = LlamaConfig(
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

    device_specs = scan_available_devices()[:1]
    quantization_encoding = SupportedEncoding.bfloat16
    pipeline_config = DummyPipelineConfig(
        model_path="meta-llama/Llama-3.1-8B-Instruct",
        max_batch_size=1,
        max_length=hf_config.max_position_embeddings,
        quantization_encoding=quantization_encoding,
        kv_cache_strategy="paged",
        device_specs=device_specs,
    )
    pipeline_config.model.kv_cache._available_cache_memory = 1024**4  # 1TB
    pipeline_config.model.kv_cache._cache_dtype = DType.bfloat16
    pipeline_config.model._huggingface_config = hf_config
    pipeline_config.model.weight_path = [Path("fake.safetensors")]

    session = MagicMock(spec=InferenceSession)
    session.load.return_value = MagicMock(spec=Model, input_metadata=[])
    devices = load_devices(device_specs)

    Llama3Model(
        pipeline_config=pipeline_config,
        session=session,
        huggingface_config=hf_config,
        encoding=quantization_encoding,
        devices=devices,
        kv_cache_config=pipeline_config.model.kv_cache,
        weights=weights_from_hf_config(hf_config),
    )

    session.load.assert_called()
