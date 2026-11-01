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

"""Tests the Gemma 3 ModuleV3 config's checkpoint finalization."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import NonCallableMock

import numpy as np
from max.dtype import DType
from max.graph.weights import WeightData
from max.nn.legacy.transformer import ReturnLogits
from max.pipelines.architectures.gemma3_modulev3.model_config import (
    Gemma3Config,
)
from max.pipelines.lib.interfaces.arch_config import ModelOutputConfig


def _config() -> Gemma3Config:
    return Gemma3Config(
        vocab_size=8,
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=4,
        hidden_activation="gelu_tanh",
        max_position_embeddings=16,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=False,
        query_pre_attn_scalar=None,
        sliding_window=4,
        final_logit_softcapping=None,
        attn_logit_softcapping=None,
        rope_scaling=None,
        rope_local_base_freq=10000.0,
        sliding_window_pattern=2,
        dtype=DType.bfloat16,
        mesh=NonCallableMock(),
        interleaved_rope_weights=False,
        kv_params=NonCallableMock(),
    )


def _weight() -> WeightData:
    return WeightData.from_numpy(np.zeros((1,), np.float32), "w")


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


def test_finalize_reads_a_multimodal_checkpoints_text_config() -> None:
    hf_config = SimpleNamespace(
        tie_word_embeddings=False,
        text_config=SimpleNamespace(tie_word_embeddings=True),
    )
    finalized = _config().finalize(
        hf_config,
        {"language_model.lm_head.weight": _weight()},
        ModelOutputConfig(),
    )

    assert finalized.tie_word_embeddings
