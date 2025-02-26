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

import math
from functools import cached_property
from os import PathLike
from typing import Sequence

import numpy as np
import torch
from max.dtype import DType
from max.graph.weights import SafetensorWeights, WeightData, Weights
from max.graph.weights._torch_dtype_map import (
    modular_to_torch_type,
    torch_to_modular_type,
)
from max.pipelines import PipelineConfig
from transformers import LlamaConfig


def _compute_safetensor_rope_scaling(
    huggingface_config: LlamaConfig,
) -> np.ndarray | None:
    # Unlike the `transformers` library's Llama model, MAX Llama expects the
    # rope scaling value to be in the state dict (this is similar to GGUF).
    if rope_scaling := getattr(huggingface_config, "rope_scaling", None):
        if rope_scaling.get("rope_type", "").lower() == "llama3":
            return _compute_rope_scaling(
                rope_scaling, huggingface_config
            ).numpy()
    return None


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
    # Add rope scaling to the state dict.
    rope_scaling = _compute_safetensor_rope_scaling(huggingface_config)
    if rope_scaling is not None:
        new_state_dict["rope_freqs.weight"] = WeightData.from_numpy(
            rope_scaling, "rope_freqs.weight"
        )
    if pipeline_config._quant_config:
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


def _compute_rope_scaling(
    rope_scaling, huggingface_config: LlamaConfig
) -> torch.Tensor:
    # From llama.cpp's HF to GGUF conversion script:
    # https://github.com/ggerganov/llama.cpp/blob/40c6d79fb52f995f47507fedfeaae2ac05d9b35c/convert_hf_to_gguf.py#L1627-L1654
    base = huggingface_config.rope_theta
    dim = huggingface_config.head_dim
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

    factor = rope_scaling.get("factor", 8.0)
    low_freq_factor = rope_scaling.get("low_freq_factor", 1.0)
    high_freq_factor = rope_scaling.get("high_freq_factor", 4.0)
    old_context_len = rope_scaling.get("original_max_position_embeddings", 8192)

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    assert low_freq_wavelen != high_freq_wavelen

    rope_factors = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            rope_factors.append(1)
        elif wavelen > low_freq_wavelen:
            rope_factors.append(factor)
        else:
            smooth = (old_context_len / wavelen - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            rope_factors.append(1 / ((1 - smooth) / factor + smooth))
    return torch.tensor(rope_factors, dtype=torch.float32)


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
    (2) Computes the rope_freqs.weight using the HuggingFace config
    (3) Transposes the q_proj and k_proj weights.

    """

    def __init__(
        self,
        filepaths: Sequence[PathLike],
        huggingface_config: LlamaConfig,
        has_rope_scaling: bool,
        rope_freqs_tensor: torch.Tensor | None,
        **kwargs,
    ):
        super().__init__(filepaths, **kwargs)
        self._gguf_name_map = LLAMA_GGUF_TENSOR_MAPPING
        self._huggingface_config = huggingface_config
        self._has_rope_scaling = has_rope_scaling
        self._rope_freqs_tensor = rope_freqs_tensor

    @classmethod
    def from_safetensor_weights(
        cls, weights: Weights, huggingface_config: LlamaConfig
    ):
        assert isinstance(weights, SafetensorWeights)
        has_rope_scaling = False
        rope_freqs_tensor = None
        if rope_scaling := getattr(huggingface_config, "rope_scaling", None):
            if rope_scaling.get("rope_type", "").lower() == "llama3":
                has_rope_scaling = True
                rope_freqs_tensor = _compute_rope_scaling(
                    rope_scaling, huggingface_config
                )
        return cls(
            weights._filepaths,
            huggingface_config=huggingface_config,
            has_rope_scaling=has_rope_scaling,
            rope_freqs_tensor=rope_freqs_tensor,
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
                        has_rope_scaling=self._has_rope_scaling,
                        rope_freqs_tensor=self._rope_freqs_tensor,
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
            has_rope_scaling=self._has_rope_scaling,
            rope_freqs_tensor=self._rope_freqs_tensor,
            tensors=self._tensors,
            tensors_to_file_idx=self._tensors_to_file_idx,
            prefix=full_path,
            allocated=self._allocated,
            _st_weight_map=self._st_weight_map,
        )

    def exists(self) -> bool:
        return self.name in self._tensors_to_file_idx or (
            self._has_rope_scaling and self.name == "rope_freqs.weight"
        )

    def _load_tensor(self, dtype: DType | None = None):
        if self._has_rope_scaling and self.name == "rope_freqs.weight":
            tensor = self._rope_freqs_tensor
            assert isinstance(tensor, torch.Tensor)
            if (
                dtype is not None
                and torch_to_modular_type(tensor.dtype) != dtype
            ):
                tensor = tensor.to(modular_to_torch_type(dtype))
            return tensor
        tensor = super()._load_tensor(dtype)

        return tensor
