# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from __future__ import annotations

from unittest.mock import MagicMock

from max.pipelines.architectures.deepseekV3.weight_adapters import (
    convert_safetensor_state_dict,
)
from transformers.models.deepseek_v3.configuration_deepseek_v3 import (
    DeepseekV3Config,
)


def test_convert_safetensor_state_dict() -> None:
    weight = MagicMock()
    state_dict = {
        "model.layers.29.input_layernorm.weight": weight,
        "model.layers.29.post_attention_layernorm.weight": weight,
        "model.layers.29.self_attn.kv_a_layernorm.weight": weight,
        "model.layers.29.self_attn.kv_a_proj_with_mqa.weight": weight,
        "model.layers.29.self_attn.kv_a_proj_with_mqa.weight_scale_inv": weight,
        "model.layers.61.input_layernorm.weight": weight,
        "model.layers.61.post_attention_layernorm.weight": weight,
        "model.layers.61.self_attn.kv_a_layernorm.weight": weight,
        "model.layers.61.self_attn.kv_a_proj_with_mqa.weight": weight,
        "model.layers.61.self_attn.kv_a_proj_with_mqa.weight_scale_inv": weight,
    }

    huggingface_config = DeepseekV3Config(num_hidden_layers=61)
    new_state_dict = convert_safetensor_state_dict(
        state_dict,  # type: ignore
        huggingface_config,
    )
    assert len(new_state_dict) == 5
    for key in new_state_dict:
        assert "61" not in key
