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

from max.graph.weights import WeightData, Weights

OLMO3_SAFETENSOR_MAP: dict[str, str] = {
    "model.embed_tokens.": "",
    "model.norm.": "ln_f.",
    "lm_head.": "",
    "model.layers.": "layers.",
    "self_attn.q_proj.": "attention.wq.",
    "self_attn.k_proj.": "attention.wk.",
    "self_attn.v_proj.": "attention.wv.",
    "self_attn.o_proj.": "attention.wo.",
    "mlp.gate_proj.": "feed_forward.w1.",
    "mlp.up_proj.": "feed_forward.w3.",
    "mlp.down_proj.": "feed_forward.w2.",
    "input_layernorm.": "attention_norm.",
    "post_attention_layernorm.": "ffn_norm.",
}


def convert_safetensor_state_dict(
    state_dict: dict[str, Weights], **kwargs
) -> dict[str, WeightData]:
    """Convert safetensor state dict to MAX format.

    Args:
        state_dict: Dictionary of weight tensors

    Returns:
        Dictionary of converted weight data
    """

    new_state_dict: dict[str, WeightData] = {}

    for weight_name, value in state_dict.items():
        max_name: str = weight_name
        for before, after in OLMO3_SAFETENSOR_MAP.items():
            max_name = max_name.replace(before, after)
        new_state_dict[max_name] = value.data()

    return new_state_dict
