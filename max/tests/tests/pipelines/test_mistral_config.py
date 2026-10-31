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

"""Tests the Mistral and Mistral 3 graph-API registrations."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import NonCallableMock, patch

import pytest
from max.dtype import DType
from max.nn.legacy.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.architectures.mistral import (
    MistralConfig,
    MistralDecoder,
    MistralTokenizer,
    mistral_arch,
)
from max.pipelines.architectures.mistral3 import Mistral3Config, mistral3_arch
from max.pipelines.lib import GraphRole, SupportedArchitecture, TextTokenizer
from max.pipelines.lib.interfaces.arch_config import ModelOutputConfig
from max.pipelines.lib.module_registration import finalize_config
from max.pipelines.lib.registration_graph_api import (
    GraphApiDecoderPipelineModel,
)


@pytest.mark.parametrize(
    ("arch", "config"),
    [(mistral_arch, MistralConfig), (mistral3_arch, Mistral3Config)],
)
def test_registers_the_mistral_decoder(
    arch: SupportedArchitecture, config: type[MistralConfig]
) -> None:
    assert dict(arch.modules) == {GraphRole.DECODER: MistralDecoder}
    assert arch.config is config
    model = arch.pipeline_model
    assert isinstance(model, type)
    assert issubclass(model, GraphApiDecoderPipelineModel)
    assert model.decoder_cls is MistralDecoder


def test_finalize_returns_a_copy_with_the_logits_asked_for() -> None:
    config = MistralConfig(
        hidden_size=4,
        num_attention_heads=1,
        num_key_value_heads=1,
        num_hidden_layers=1,
        head_dim=4,
        vocab_size=8,
        rope_theta=10000.0,
        max_seq_len=16,
        rms_norm_eps=1e-5,
        feed_forward_length=8,
        dtype=DType.bfloat16,
        kv_params=NonCallableMock(),
        attention_multiplier=0.5,
        devices=[],
    )
    finalized = finalize_config(
        config,
        SimpleNamespace(),
        {},
        ModelOutputConfig(
            return_logits=ReturnLogits.VARIABLE,
            return_hidden_states=ReturnHiddenStates.LAST,
        ),
    )

    assert isinstance(finalized, MistralConfig)
    assert finalized is not config
    assert finalized.return_logits == ReturnLogits.VARIABLE
    assert config.return_logits == ReturnLogits.LAST_TOKEN


def test_tokenizer_enables_the_llama_whitespace_fix_by_default() -> None:
    assert mistral_arch.tokenizer is MistralTokenizer
    with patch.object(TextTokenizer, "__init__", return_value=None) as init:
        MistralTokenizer("repo", NonCallableMock(), max_length=8)
    assert init.call_args.kwargs["enable_llama_whitespace_fix"] is True
    assert init.call_args.kwargs["max_length"] == 8
