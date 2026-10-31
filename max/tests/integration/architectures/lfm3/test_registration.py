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
"""Tests the LFM3 module registration and config finalization."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import NonCallableMock

from max.dtype import DType
from max.nn.legacy.transformer import ReturnLogits
from max.pipelines.architectures.lfm3 import Lfm3, Lfm3Config, lfm3_arch
from max.pipelines.lib import GraphRole
from max.pipelines.lib.interfaces.arch_config import ModelOutputConfig
from max.pipelines.lib.module_registration import ModuleV3DecoderPipelineModel


def _config() -> Lfm3Config:
    return Lfm3Config(
        vocab_size=8,
        hidden_size=4,
        intermediate_size=8,
        layer_types=["conv", "sliding_attention", "full_attention"],
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=4,
        conv_kernel_size=3,
        sliding_window=4,
        norm_eps=1e-5,
        rope_theta={"full_attention": 10000.0},
        partial_rotary_factor={"full_attention": 1.0},
        use_sandwich_norm=False,
        dtype=DType.bfloat16,
        devices=[],
        max_seq_len=16,
        kv_params=NonCallableMock(),
    )


def test_arch_registers_the_decoder_module() -> None:
    assert lfm3_arch.modules == {GraphRole.DECODER: Lfm3}
    model = lfm3_arch.pipeline_model
    assert isinstance(model, type)
    assert issubclass(model, ModuleV3DecoderPipelineModel)
    assert model.decoder_cls is Lfm3
    assert lfm3_arch.checkpoints_recurrent_state


def test_finalize_returns_a_finalized_copy() -> None:
    config = _config()
    finalized = config.finalize(
        SimpleNamespace(),
        {},
        ModelOutputConfig(return_logits=ReturnLogits.ALL),
    )

    assert finalized is not config
    assert finalized.return_logits == ReturnLogits.ALL
    assert config.return_logits == ReturnLogits.LAST_TOKEN
