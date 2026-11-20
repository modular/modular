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

from max.pipelines.architectures.mtp_deepseekV3.weight_adapters import (
    convert_mtp_safetensor_state_dict,
)
from transformers.models.deepseek_v3.configuration_deepseek_v3 import (
    DeepseekV3Config,
)


def test_convert_mtp_safetensor_state_dict_filters_kv_scales() -> None:
    weight = NonCallableMock()
    # num_hidden_layers=61 means layer 61 is the MTP layer in the checkpoint.
    state_dict = {
        # Target (non-MTP) layer weights, including FP8 KV-cache scales.
        "model.layers.0.input_layernorm.weight": weight,
        "model.layers.0.self_attn.k_proj.k_scale": weight,
        "model.layers.0.self_attn.v_proj.v_scale": weight,
        # MTP layer weights, including FP8 KV-cache scales.
        "model.layers.61.enorm.weight": weight,
        "model.layers.61.self_attn.q_proj.weight": weight,
        "model.layers.61.self_attn.k_proj.k_scale": weight,
        "model.layers.61.self_attn.v_proj.v_scale": weight,
    }

    huggingface_config = DeepseekV3Config(num_hidden_layers=61)
    new_state_dict = convert_mtp_safetensor_state_dict(
        state_dict,  # type: ignore
        huggingface_config,
    )

    # The k_scale/v_scale entries are filtered, and the target's layers are
    # left to the target's own adapter.
    assert new_state_dict.keys() == {
        "enorm.weight",
        "decoder_layer.self_attn.q_proj.weight",
    }
