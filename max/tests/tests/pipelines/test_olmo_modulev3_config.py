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

"""Tests the OLMo ModuleV3 registration and its config's finalization."""

from __future__ import annotations

from unittest.mock import NonCallableMock

import numpy as np
from max.dtype import DType
from max.graph.weights import WeightData
from max.nn.outputs import ReturnHiddenStates, ReturnLogits
from max.pipelines.architectures.llama3_modulev3 import Llama3Decoder
from max.pipelines.architectures.olmo_modulev3 import (
    OlmoConfig,
    olmo_modulev3_arch,
)
from max.pipelines.lib import GraphRole
from max.pipelines.lib.interfaces.arch_config import ModelOutputConfig
from max.pipelines.lib.module_registration import (
    ModuleV3DecoderPipelineModel,
    finalize_config,
)
from transformers import PretrainedConfig


def _config() -> OlmoConfig:
    return OlmoConfig(
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
        dtype=DType.float32,
        kv_params=NonCallableMock(),
        attention_multiplier=0.5,
        embedding_multiplier=1.0,
        residual_multiplier=1.0,
        devices=[],
    )


def _weight() -> WeightData:
    return WeightData.from_numpy(np.zeros((1,), np.float32), "w")


def test_registers_the_llama3_decoder() -> None:
    assert dict(olmo_modulev3_arch.modules) == {
        GraphRole.DECODER: Llama3Decoder
    }
    model = olmo_modulev3_arch.pipeline_model
    assert isinstance(model, type)
    assert issubclass(model, ModuleV3DecoderPipelineModel)
    assert model.decoder_cls is Llama3Decoder


def test_finalize_selects_non_parametric_layer_norm() -> None:
    config = _config()
    finalized = finalize_config(
        config,
        PretrainedConfig(tie_word_embeddings=False),
        {"model.lm_head.weight": _weight()},
        ModelOutputConfig(
            return_logits=ReturnLogits.VARIABLE,
            return_hidden_states=ReturnHiddenStates.LAST,
        ),
    )

    assert isinstance(finalized, OlmoConfig)
    assert finalized is not config
    assert finalized.norm_method == "layer_norm"
    assert not finalized.norm_elementwise_affine
    assert finalized.rms_norm_eps is None
    assert finalized.return_logits == ReturnLogits.VARIABLE
    assert finalized.return_hidden_states == ReturnHiddenStates.LAST
    assert not finalized.tie_word_embeddings
    assert config.return_logits == ReturnLogits.LAST_TOKEN
