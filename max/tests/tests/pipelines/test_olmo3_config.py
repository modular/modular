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

"""Tests the OLMo 3 registration and its config's checkpoint finalization."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import NonCallableMock

import numpy as np
from max.dtype import DType
from max.graph.weights import WeightData
from max.nn.legacy.transformer import ReturnLogits
from max.pipelines.architectures.olmo3 import Olmo3, olmo3_arch
from max.pipelines.architectures.olmo3.model_config import Olmo3Config
from max.pipelines.lib import GraphRole
from max.pipelines.lib.interfaces.arch_config import ModelOutputConfig
from max.pipelines.lib.module_registration import (
    ModuleV3DecoderPipelineModel,
    graph_name,
)


def _config() -> Olmo3Config:
    return Olmo3Config(
        vocab_size=8,
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=4,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=4,
        hidden_activation="silu",
        max_position_embeddings=16,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        attention_bias=False,
        sliding_window=4,
        layer_types=["sliding_attention"] * 3 + ["full_attention"],
        attention_dropout=0.0,
        rope_scaling=None,
        rope_scaling_type=None,
        qk_norm_eps=1e-6,
        use_qk_norm=True,
        use_cache=True,
        dtype=DType.bfloat16,
        devices=[],
        interleaved_rope_weights=False,
        return_logits=ReturnLogits.LAST_TOKEN,
        kv_params=NonCallableMock(),
    )


def _weight() -> WeightData:
    return WeightData.from_numpy(np.zeros((1,), np.float32), "w")


def test_registers_the_decoder_module() -> None:
    assert dict(olmo3_arch.modules) == {GraphRole.DECODER: Olmo3}
    assert olmo3_arch.pipeline_model is not None
    assert issubclass(olmo3_arch.pipeline_model, ModuleV3DecoderPipelineModel)
    # The graph name keys precompiled MEFs; it predates registration.
    assert graph_name(Olmo3) == "Olmo3"


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
