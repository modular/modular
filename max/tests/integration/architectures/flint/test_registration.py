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
"""Tests the Flint module registration and config finalization."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import NonCallableMock

import numpy as np
from max.dtype import DType
from max.graph.weights import WeightData
from max.nn.legacy.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.architectures.flint import Flint, FlintConfig, flint_arch
from max.pipelines.lib import GraphRole
from max.pipelines.lib.interfaces.arch_config import ModelOutputConfig
from max.pipelines.lib.module_registration import ModuleV3DecoderPipelineModel


def _config() -> FlintConfig:
    return FlintConfig(
        hidden_size=4,
        num_attention_heads=1,
        num_key_value_heads=1,
        num_hidden_layers=2,
        rope_theta=10000.0,
        max_seq_len=16,
        intermediate_size=8,
        vocab_size=8,
        dtype=DType.bfloat16,
        kv_params=NonCallableMock(),
        head_dim=4,
        devices=[],
        layer_types=["full_attention", "sliding_attention"],
        sliding_window=4,
        num_experts=0,
        num_experts_per_tok=0,
        moe_intermediate_size=8,
        norm_topk_prob=True,
        hidden_act="silu",
        attention_bias=False,
    )


def _weight() -> WeightData:
    return WeightData.from_numpy(np.zeros((1,), np.float32), "w")


def test_arch_registers_the_decoder_module() -> None:
    assert flint_arch.modules == {GraphRole.DECODER: Flint}
    model = flint_arch.pipeline_model
    assert isinstance(model, type)
    assert issubclass(model, ModuleV3DecoderPipelineModel)
    assert model.decoder_cls is Flint


def test_finalize_returns_a_finalized_copy() -> None:
    config = _config()
    finalized = config.finalize(
        SimpleNamespace(tie_word_embeddings=True),
        {"language_model.lm_head.weight": _weight()},
        ModelOutputConfig(
            return_logits=ReturnLogits.VARIABLE,
            return_hidden_states=ReturnHiddenStates.LAST,
        ),
    )

    assert finalized is not config
    assert finalized.return_logits == ReturnLogits.VARIABLE
    assert finalized.return_hidden_states == ReturnHiddenStates.LAST
    assert not finalized.tie_word_embeddings
    assert config.return_logits == ReturnLogits.LAST_TOKEN
    assert config.return_hidden_states == ReturnHiddenStates.NONE


def test_finalize_ties_embeddings_without_an_lm_head() -> None:
    finalized = _config().finalize(
        SimpleNamespace(tie_word_embeddings=True), {}, ModelOutputConfig()
    )

    assert finalized.tie_word_embeddings
