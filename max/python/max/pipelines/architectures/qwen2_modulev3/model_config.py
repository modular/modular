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
"""Qwen2 configuration on ModuleV3."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import ClassVar

from max.dtype import DType
from max.graph import DeviceRef
from max.graph.weights import WeightData
from max.nn.legacy.kv_cache import KVCacheParamInterface
from max.nn.legacy.transformer import ReturnLogits
from max.pipelines.kv_cache import cache_dtype_for_encoding
from max.pipelines.lib import MAXModelConfig, PipelineConfig
from max.pipelines.lib.config.model_config import (
    _interleaved_rope_weights,
    _select_quantization_encoding,
)
from max.pipelines.lib.interfaces.arch_config import (
    ArchConfigWithKVCache,
    ArchConfigWithStoredKVParams,
    ModelOutputConfig,
)
from max.pipelines.lib.pipeline_variants.utils import get_rope_theta
from max.pipelines.modeling.config_enums import (
    SupportedEncoding,
    supported_encoding_dtype,
)
from max.tensor.sharding import DeviceMesh
from transformers import AutoConfig
from typing_extensions import Self, override


def _rope_type(huggingface_config: AutoConfig) -> str:
    # transformers >= 5 reports unscaled RoPE as rope_type "default" in
    # rope_parameters; 4.x leaves rope_scaling unset.
    scaling = getattr(huggingface_config, "rope_scaling", None) or {}
    return scaling.get("rope_type", scaling.get("type", "default"))


@dataclass(kw_only=True)
class Qwen2Config(ArchConfigWithStoredKVParams, ArchConfigWithKVCache):
    """Model configuration for Qwen2 graph construction and execution."""

    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "bfloat16"
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {
        "float32",
        "bfloat16",
    }

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    max_seq_len: int
    interleaved_rope_weights: bool
    dtype: DType
    mesh: DeviceMesh
    """The tensor-parallel mesh the decoder runs on."""
    kv_params: KVCacheParamInterface
    return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN
    tie_word_embeddings: bool = False
    quantization_encoding: SupportedEncoding | None = None

    @property
    def devices(self) -> list[DeviceRef]:
        """Returns the devices in :attr:`mesh`, for the memory planner."""
        return [DeviceRef.from_device(device) for device in self.mesh.devices]

    @override
    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
        *,
        max_seq_len: int,
    ) -> Self:
        model_config = model_config or pipeline_config.model
        huggingface_config = model_config.huggingface_config
        if huggingface_config is None:
            raise ValueError(
                f"HuggingFace config is required for '{model_config.model_path}', "
                "but config could not be loaded. "
                "Please ensure the model repository contains a valid config.json file."
            )
        if (rope_type := _rope_type(huggingface_config)) != "default":
            raise ValueError(
                f"Qwen2 on ModuleV3 does not support rope_type '{rope_type}'."
            )

        quantization_encoding = _select_quantization_encoding(
            model_config, cls.DEFAULT_ENCODING
        )
        cache_dtype = cache_dtype_for_encoding(
            quantization_encoding, model_config.kv_cache.kv_cache_format
        )
        device_refs = [
            DeviceRef(spec.device_type, spec.id)
            for spec in model_config.device_specs
        ]
        mesh = DeviceMesh(
            tuple(device.to_device() for device in device_refs),
            (len(device_refs),),
            ("tp",),
        )
        return cls(
            vocab_size=huggingface_config.vocab_size,
            hidden_size=huggingface_config.hidden_size,
            intermediate_size=huggingface_config.intermediate_size,
            num_hidden_layers=huggingface_config.num_hidden_layers,
            num_attention_heads=huggingface_config.num_attention_heads,
            num_key_value_heads=huggingface_config.num_key_value_heads,
            head_dim=cls.get_head_dim(huggingface_config),
            rms_norm_eps=huggingface_config.rms_norm_eps,
            rope_theta=get_rope_theta(huggingface_config),
            max_seq_len=max_seq_len,
            interleaved_rope_weights=_interleaved_rope_weights(model_config),
            dtype=supported_encoding_dtype(quantization_encoding),
            mesh=mesh,
            kv_params=cls.construct_kv_params(
                huggingface_config=huggingface_config,
                pipeline_config=pipeline_config,
                devices=device_refs,
                kv_cache_config=model_config.kv_cache,
                cache_dtype=cache_dtype,
            ),
            quantization_encoding=quantization_encoding,
        )

    def finalize(
        self,
        huggingface_config: AutoConfig,
        state_dict: Mapping[str, WeightData],
        outputs: ModelOutputConfig,
    ) -> Self:
        """Returns the config completed with checkpoint and output requirements.

        Args:
            huggingface_config: The checkpoint's model configuration.
            state_dict: The adapted checkpoint weights.
            outputs: The outputs the pipeline requires.

        Returns:
            A copy of this config with the checkpoint-dependent fields set.
        """
        tie_word_embeddings = (
            getattr(huggingface_config, "tie_word_embeddings", False)
            or "language_model.lm_head.weight" not in state_dict
        )
        return replace(
            self,
            tie_word_embeddings=tie_word_embeddings,
            return_logits=outputs.return_logits,
        )
