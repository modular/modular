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

# Maps from GGUF to MAX weight names for OLMo2.
OLMO2_GGUF_MAPPING = {
    "token_embd": "embed_tokens",
    "blk": "layers",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
    "ffn_gate": "mlp.gate_proj",
    "ffn_norm": "post_attention_layernorm",
    "attn_norm": "input_layernorm",
    "attn_q": "self_attn.q_proj",
    "attn_v": "self_attn.v_proj",
    "attn_k": "self_attn.k_proj",
    "attn_output": "self_attn.o_proj",
    # OLMo2-specific mappings for key and query norms
    "attn_q_norm": "self_attn.q_norm",
    "attn_k_norm": "self_attn.k_norm",
    "output.weight": "lm_head.weight",
    "output_norm": "norm",
}


def convert_gguf_state_dict(
    state_dict: dict[str, Weights], **unused_kwargs
) -> dict[str, WeightData]:
    new_state_dict: dict[str, WeightData] = {}
    # Map the weight names.
    for gguf_name, value in state_dict.items():
        max_name = gguf_name
        for before, after in OLMO2_GGUF_MAPPING.items():
            max_name = max_name.replace(before, after)
        new_state_dict[max_name] = value.data()
    return new_state_dict
