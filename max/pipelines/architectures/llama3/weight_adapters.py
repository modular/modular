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

import numpy as np
from max.dtype import DType
from max.graph.weights import WeightData, Weights
from max.pipelines.lib import PipelineConfig, SupportedEncoding
from transformers import LlamaConfig

# Maps from Safetensor to MAX weight names.
LLAMA_SAFETENSOR_MAPPING = {
    "model.": "",  # Removes the "model" prefix.
    "g_idx": "perm_idx",  # Specific to Llama GPT-Q weights.
}


def convert_safetensor_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: LlamaConfig,
    pipeline_config: PipelineConfig,
    **unused_kwargs,
) -> dict[str, WeightData]:
    new_state_dict: dict[str, WeightData] = {}
    # Map the weight names.
    for safetensor_name, value in state_dict.items():
        max_name = safetensor_name
        for before, after in LLAMA_SAFETENSOR_MAPPING.items():
            max_name = max_name.replace(before, after)
        new_state_dict[max_name] = value.data()
    if pipeline_config.model_config._quant_config:
        # hack: argsort the perm_idx array
        for key, weight_data in new_state_dict.items():
            if key.endswith("perm_idx"):
                new_state_dict[key] = WeightData.from_numpy(
                    np.argsort(weight_data.data).astype(np.int32), key
                )
    if (
        pipeline_config.model_config.quantization_encoding
        == SupportedEncoding.gptq
    ):
        for key, weight_data in new_state_dict.items():
            # TODO(E2EOPT-243): gptq models actually have a dtype of float16
            # not bfloat16. Sadly, MMA does not support float16 currently, so
            # we must use bfloat16 for now.
            # That said, leave scale and bias in float16 (apparently that is
            # needed for correctness). The rest must be converted to bfloat16.
            if weight_data.dtype == DType.float16 and not (
                key.endswith("bias") or key.endswith("scales")
            ):
                new_state_dict[key] = weight_data.astype(DType.bfloat16)

    if pipeline_config.model_config.cast_safetensor_weights_from_float32_to_bfloat16:
        for key, weight_data in new_state_dict.items():
            if weight_data.dtype == DType.float32:
                new_state_dict[key] = weight_data.astype(DType.bfloat16)
    return new_state_dict


# Maps from GGUF to MAX weight names.
LLAMA_GGUF_MAPPING = {
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
        for before, after in LLAMA_GGUF_MAPPING.items():
            max_name = max_name.replace(before, after)
        new_state_dict[max_name] = value.data()
    return new_state_dict
