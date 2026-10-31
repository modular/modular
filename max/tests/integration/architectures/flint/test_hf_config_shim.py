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
"""Loads the published flint config through the local shim."""

import json
from pathlib import Path

from max.pipelines.architectures import hf_config_shims
from transformers import AutoConfig, PretrainedConfig

TESTDATA = Path(__file__).parent / "testdata"

FULL_ATTENTION = [0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 48]


def _stripped_config() -> PretrainedConfig:
    # The published config states layer_types; drop them so the config
    # object omits the attribute and the model config must derive the
    # default schedule.
    raw = json.loads((TESTDATA / "config.json").read_text())
    del raw["layer_types"]
    return hf_config_shims.FlintHFConfig(**raw)


def _check(config: PretrainedConfig) -> None:
    assert config.model_type == "flint"
    assert config.num_hidden_layers == 49
    assert config.hidden_size == 2048
    assert config.num_attention_heads == 32
    assert config.num_key_value_heads == 4
    assert config.head_dim == 128
    assert config.num_experts == 256
    assert config.num_experts_per_tok == 6
    assert config.sliding_window == 513
    assert config.tie_word_embeddings is False


def test_published_config() -> None:
    config = AutoConfig.from_pretrained(str(TESTDATA / "config.json"))
    _check(config)
    assert [
        i for i, t in enumerate(config.layer_types) if t == "full_attention"
    ] == FULL_ATTENTION
    # The NoPE layers are exactly the full-attention layers.
    assert list(config.no_rope_layer_indices) == FULL_ATTENTION


def test_stripped_config() -> None:
    config = _stripped_config()
    _check(config)
    assert not hasattr(config, "layer_types")
