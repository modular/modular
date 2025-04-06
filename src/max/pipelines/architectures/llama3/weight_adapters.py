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

from collections.abc import Sequence
from functools import cached_property
from os import PathLike

import numpy as np
from max.dtype import DType
from max.graph.weights import SafetensorWeights, WeightData, Weights
from max.pipelines import PipelineConfig
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


# Required for Multi-GPU LLama3 until it is migrated to new Layers API

# Map from GGUF tensor names to Safetensor names.
# https://github.com/huggingface/transformers/blob/7bbc62474391aff64f63fcc064c975752d1fa4de/src/transformers/integrations/ggml.py#L36
LLAMA_GGUF_TENSOR_MAPPING = {
    "token_embd": "model.embed_tokens",
    "blk": "model.layers",
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
    "output_norm": "model.norm",
}


class LlamaSafetensorWeights(SafetensorWeights):
    """Loads Safetensor weights with GGUF names.

    Does the following when loading weights:
    (1) converts Safetensor weight names to and from GGUF names. For example,
        the GGUF weight "blk.{i}.attn_q.weight" is instead saved as
        "model.layers.{i}.self_attn.q_proj.weight" in Safetensor.
    (2) Transposes the q_proj and k_proj weights.

    """

    def __init__(
        self,
        filepaths: Sequence[PathLike],
        huggingface_config: LlamaConfig,
        **kwargs,
    ):
        super().__init__(filepaths, **kwargs)
        self._gguf_name_map = LLAMA_GGUF_TENSOR_MAPPING
        self._huggingface_config = huggingface_config

    @classmethod
    def from_safetensor_weights(
        cls, weights: Weights, huggingface_config: LlamaConfig
    ):
        assert isinstance(weights, SafetensorWeights)
        return cls(
            weights._filepaths,
            huggingface_config=huggingface_config,
            tensors=weights._tensors,
            tensors_to_file_idx=weights._tensors_to_file_idx,
            prefix="",
            allocated=weights._allocated,
            _st_weight_map=weights._st_weight_map,
        )

    def items(self):
        """Iterate through all allocable weights that start with the prefix."""
        # `self._tensor` contains Safetensor names, but the Llama pipeline
        # expects GGUF names so this function should return GGUF names.
        for safetensor_name in self._tensors:
            if safetensor_name.startswith(self.name):
                gguf_name = safetensor_name

                # The _gguf_name_map maps gguf -> safetensor names.
                # We want the reverse transformation from safetensor -> gguf.
                for after, before in self._gguf_name_map.items():
                    gguf_name = gguf_name.replace(before, after)
                yield (
                    gguf_name,
                    LlamaSafetensorWeights(
                        self._filepaths,
                        huggingface_config=self._huggingface_config,
                        tensors=self._tensors,
                        tensors_to_file_idx=self._tensors_to_file_idx,
                        prefix=gguf_name,
                        allocated=self._allocated,
                        _st_weight_map=self._st_weight_map,
                    ),
                )

    @cached_property
    def name(self) -> str:
        """The current weight name or prefix."""
        # Convert the prefix, which follows the GGUF naming pattern, to
        # Safetensor weight name.
        name = self._prefix
        for before, after in self._gguf_name_map.items():
            name = name.replace(before, after)
        return name

    def __getattr__(self, attr) -> LlamaSafetensorWeights:
        if self._prefix:
            full_path = f"{self._prefix}.{attr}"
        else:
            full_path = str(attr)
        return LlamaSafetensorWeights(
            self._filepaths,
            huggingface_config=self._huggingface_config,
            tensors=self._tensors,
            tensors_to_file_idx=self._tensors_to_file_idx,
            prefix=full_path,
            allocated=self._allocated,
            _st_weight_map=self._st_weight_map,
        )

    def exists(self) -> bool:
        return self.name in self._tensors_to_file_idx

    def _load_tensor(self, dtype: DType | None = None):
        tensor = super()._load_tensor(dtype)

        return tensor
