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

"""Tests the GPT-OSS ModuleV3 registration and config finalization."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import NonCallableMock

import numpy as np
from max.dtype import DType
from max.graph.weights import WeightData
from max.nn.legacy.rotary_embedding import YarnScalingParams
from max.nn.legacy.transformer import ReturnLogits
from max.pipelines.architectures.gpt_oss_modulev3 import (
    GptOss,
    gpt_oss_modulev3_arch,
)
from max.pipelines.architectures.gpt_oss_modulev3.model_config import (
    GptOssConfig,
)
from max.pipelines.lib import GraphRole
from max.pipelines.lib.interfaces.arch_config import ModelOutputConfig
from max.pipelines.lib.module_registration import ModuleV3DecoderPipelineModel


def _config() -> GptOssConfig:
    return GptOssConfig(
        vocab_size=8,
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=4,
        hidden_activation="silu",
        max_position_embeddings=16,
        rms_norm_eps=1e-5,
        rope_theta=150000.0,
        attention_bias=True,
        sliding_window=4,
        num_local_experts=4,
        num_experts_per_tok=2,
        router_aux_loss_coef=0.9,
        layer_types=["sliding_attention", "full_attention"],
        attention_dropout=0.0,
        rope_scaling=YarnScalingParams(
            factor=32.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_max_position_embeddings=4096,
            truncate=False,
        ),
        query_pre_attn_scalar=None,
        final_logit_softcapping=None,
        attn_logit_softcapping=None,
        swiglu_limit=7.0,
        dtype=DType.bfloat16,
        devices=[],
        interleaved_rope_weights=False,
        kv_params=NonCallableMock(),
    )


def _weight() -> WeightData:
    return WeightData.from_numpy(np.zeros((1,), np.float32), "w")


def test_arch_registers_the_decoder_module() -> None:
    assert gpt_oss_modulev3_arch.modules == {GraphRole.DECODER: GptOss}
    model = gpt_oss_modulev3_arch.pipeline_model
    assert isinstance(model, type)
    assert issubclass(model, ModuleV3DecoderPipelineModel)
    assert model.decoder_cls is GptOss


def test_finalize_returns_a_finalized_copy() -> None:
    config = _config()
    finalized = config.finalize(
        SimpleNamespace(tie_word_embeddings=False),
        {"language_model.lm_head.weight": _weight()},
        ModelOutputConfig(return_logits=ReturnLogits.VARIABLE),
    )

    assert finalized is not config
    assert finalized.return_logits == ReturnLogits.VARIABLE
    assert not finalized.tie_word_embeddings
    assert config.return_logits == ReturnLogits.LAST_TOKEN


def test_finalize_ties_embeddings_without_an_lm_head() -> None:
    finalized = _config().finalize(
        SimpleNamespace(tie_word_embeddings=False), {}, ModelOutputConfig()
    )

    assert finalized.tie_word_embeddings
