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

"""Tests the Qwen2 graph-API registration and its config's finalization."""

from __future__ import annotations

from unittest.mock import NonCallableMock

import numpy as np
from max.dtype import DType
from max.graph.weights import WeightData
from max.nn.legacy.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.architectures.llama3.decoder import Llama3Decoder
from max.pipelines.architectures.qwen2 import Qwen2Config, qwen2_arch
from max.pipelines.lib import GraphRole
from max.pipelines.lib.interfaces.arch_config import ModelOutputConfig
from max.pipelines.lib.module_registration import finalize_config
from max.pipelines.lib.registration_graph_api import (
    GraphApiDecoderPipelineModel,
)
from transformers import PretrainedConfig


def _config() -> Qwen2Config:
    return Qwen2Config(
        hidden_size=4,
        num_attention_heads=1,
        num_key_value_heads=1,
        num_hidden_layers=2,
        rope_theta=10000.0,
        rope_scaling_params=None,
        max_seq_len=16,
        intermediate_size=8,
        interleaved_rope_weights=False,
        vocab_size=8,
        dtype=DType.bfloat16,
        model_quantization_encoding=None,
        quantization_config=None,
        kv_params=NonCallableMock(),
        attention_multiplier=0.5,
        embedding_multiplier=1.0,
        residual_multiplier=1.0,
        devices=[],
        clip_qkv=None,
    )


def test_registers_the_llama3_decoder() -> None:
    assert dict(qwen2_arch.modules) == {GraphRole.DECODER: Llama3Decoder}
    assert qwen2_arch.config is Qwen2Config
    model = qwen2_arch.pipeline_model
    assert isinstance(model, type)
    assert issubclass(model, GraphApiDecoderPipelineModel)
    assert model.decoder_cls is Llama3Decoder


def test_finalize_keeps_attention_bias_and_the_outputs() -> None:
    config = _config()
    huggingface_config = PretrainedConfig(
        rms_norm_eps=1e-6, tie_word_embeddings=False
    )
    huggingface_config.model_type = "qwen2"
    finalized = finalize_config(
        config,
        huggingface_config,
        {"lm_head.weight": WeightData.from_numpy(np.zeros(1, np.float32), "w")},
        ModelOutputConfig(
            return_logits=ReturnLogits.VARIABLE,
            return_hidden_states=ReturnHiddenStates.LAST,
        ),
    )

    assert isinstance(finalized, Qwen2Config)
    assert finalized is not config
    assert finalized.attention_bias
    assert finalized.huggingface_model_type == "qwen2"
    assert finalized.return_logits == ReturnLogits.VARIABLE
    assert finalized.return_hidden_states == ReturnHiddenStates.LAST
    assert not config.attention_bias
