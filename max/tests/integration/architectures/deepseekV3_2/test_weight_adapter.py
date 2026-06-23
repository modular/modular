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

from __future__ import annotations

from unittest.mock import NonCallableMock

from max.pipelines.architectures.deepseekV3_2.weight_adapters import (
    convert_safetensor_state_dict,
)
from transformers.models.deepseek_v3.configuration_deepseek_v3 import (
    DeepseekV3Config,
)


def test_convert_safetensor_state_dict_filters_kv_scales() -> None:
    weight = NonCallableMock()
    state_dict = {
        "model.layers.29.input_layernorm.weight": weight,
        "model.layers.29.post_attention_layernorm.weight": weight,
        "model.layers.29.self_attn.kv_a_layernorm.weight": weight,
        "model.layers.29.self_attn.kv_a_proj_with_mqa.weight": weight,
        "model.layers.29.self_attn.kv_a_proj_with_mqa.weight_scale_inv": weight,
        "model.layers.29.self_attn.k_proj.k_scale": weight,
        "model.layers.29.self_attn.v_proj.v_scale": weight,
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
    # MTP layer (layer 61) and the two k_scale/v_scale entries on layer 29
    # are dropped, leaving the 5 layer-29 weight tensors.
    assert len(new_state_dict) == 5
    for key in new_state_dict:
        assert "61" not in key
        assert not key.endswith(".k_scale")
        assert not key.endswith(".v_scale")
