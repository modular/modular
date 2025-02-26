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
from transformers import LlamaConfig

from ..llama3.weight_adapters import _compute_safetensor_rope_scaling

# Maps Exaone Safetensor to MAX weight names.
EXAONE_SAFETENSOR_MAPPING = {
    "transformer.wte": "embed_tokens",
    "transformer.h": "layers",
    "mlp.c_fc_1": "mlp.up_proj",
    "mlp.c_proj": "mlp.down_proj",
    "mlp.c_fc_0": "mlp.gate_proj",
    "ln_2": "post_attention_layernorm",
    "ln_1": "input_layernorm",
    "attn.attention.q_proj": "self_attn.q_proj",
    "attn.attention.v_proj": "self_attn.v_proj",
    "attn.attention.k_proj": "self_attn.k_proj",
    "attn.attention.out_proj": "self_attn.o_proj",
    "transformer.ln_f": "norm",
}


def convert_exaone_safetensor_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: LlamaConfig,
    **unused_kwargs,
) -> dict[str, WeightData]:
    new_state_dict: dict[str, WeightData] = {}
    # Map the weight names.
    for safetensor_name, value in state_dict.items():
        max_name = safetensor_name
        for before, after in EXAONE_SAFETENSOR_MAPPING.items():
            max_name = max_name.replace(before, after)
        new_state_dict[max_name] = value.data()
    # Add rope scaling to the state dict.
    rope_scaling = _compute_safetensor_rope_scaling(huggingface_config)
    if rope_scaling is not None:
        new_state_dict["rope_freqs.weight"] = WeightData.from_numpy(
            rope_scaling, "rope_freqs.weight"
        )
    return new_state_dict
