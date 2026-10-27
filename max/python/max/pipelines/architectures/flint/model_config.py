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

"""Configuration for Flint."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from max.dtype import DType
from max.graph import DeviceRef
from max.graph.weights import WeightData
from max.nn.kv_cache import MultiKVCacheParams
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.kv_cache import cache_dtype_for_encoding
from max.pipelines.lib import KVCacheConfig, MAXModelConfig, PipelineConfig
from max.pipelines.lib.config.model_config import (
    _select_quantization_encoding,
)
from max.pipelines.lib.interfaces.arch_config import (
    ArchConfigWithKVCache,
    ArchConfigWithStoredKVParams,
)
from max.pipelines.modeling.config_enums import (
    SupportedEncoding,
    supported_encoding_dtype,
)
from transformers import AutoConfig
from typing_extensions import Self, override

from .layers.hybrid_kv_cache import (
    hybrid_swa_full_kv_params,
    normalize_layer_types,
)
from .layers.tied_embeddings import resolve_tie_word_embeddings


def flint_default_layer_types(num_layers: int) -> list[str]:
    """Returns the layer types the HF config derives when absent.

    Full attention on the first two and last layers, then every fourth.
    """
    return [
        "full_attention"
        if i < 2 or i % 4 == 1 or i == num_layers - 1
        else "sliding_attention"
        for i in range(num_layers)
    ]


@dataclass(kw_only=True)
class FlintConfig(ArchConfigWithStoredKVParams, ArchConfigWithKVCache):
    """Configuration for Flint."""

    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "bfloat16"
    # The published checkpoint is bf16 safetensors. A 4-bit encoding selects a
    # pre-quantized artifact, and the only 4-bit Flint cuts are MLX-affine int4
    # and GGUF, neither of which this architecture registers an adapter for.
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {
        "float32",
        "bfloat16",
    }

    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    num_hidden_layers: int
    rope_theta: float
    max_seq_len: int
    intermediate_size: int
    vocab_size: int
    dtype: DType
    kv_params: MultiKVCacheParams
    head_dim: int
    devices: list[DeviceRef]
    # Mechanism fields carry no dataclass defaults: a config that
    # fails to populate one should fail loudly at initialize()
    # rather than silently compile a different graph.
    layer_types: list[str] = field(default_factory=list)
    sliding_window: int
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    norm_topk_prob: bool
    hidden_act: str
    attention_bias: bool
    # Defaults that match every published flint checkpoint
    # (mlp_only_layers [0, 1], decoder_sparse_step 1,
    # shared-expert 1024).
    mlp_only_layers: list[int] = field(default_factory=lambda: [0, 1])
    decoder_sparse_step: int = 1
    shared_expert_intermediate_size: int = 1024
    # Non-mechanism knobs with defaults.
    return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN
    rms_norm_eps: float | None = None
    tie_word_embeddings: bool = False
    return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE
    quantization_encoding: SupportedEncoding | None = None

    @staticmethod
    def _normalized_layer_types(hf_cfg: AutoConfig) -> list[str]:
        return normalize_layer_types(hf_cfg, default=flint_default_layer_types)

    @classmethod
    def construct_kv_params(
        cls,
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
        *,
        allow_kv_head_replication: bool = False,
    ) -> MultiKVCacheParams:
        """Builds one paged cache per attention type.

        The sliding child is bounded by the sliding window; the full
        child is unbounded.
        """
        return hybrid_swa_full_kv_params(
            layer_types=cls._normalized_layer_types(huggingface_config),
            sliding_window=huggingface_config.sliding_window,
            pipeline_config=pipeline_config,
            devices=devices,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
            n_kv_heads=huggingface_config.num_key_value_heads,
            head_dim=cls.get_head_dim(huggingface_config),
            allow_kv_head_replication=allow_kv_head_replication,
        )

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
                f"HuggingFace config required for "
                f"'{model_config.model_path}' but could not be loaded."
            )
        kv_cache_config = model_config.kv_cache
        quantization_encoding = _select_quantization_encoding(
            model_config, cls.DEFAULT_ENCODING
        )
        dtype = supported_encoding_dtype(quantization_encoding)
        cache_dtype = cache_dtype_for_encoding(
            quantization_encoding, model_config.kv_cache.kv_cache_format
        )
        device_refs = [
            DeviceRef(spec.device_type, spec.id)
            for spec in model_config.device_specs
        ]

        # An explicit empty list means "no dense layers"; only a missing key
        # takes the default.
        mlp_only_layers = getattr(huggingface_config, "mlp_only_layers", None)
        mlp_only = list([0, 1] if mlp_only_layers is None else mlp_only_layers)
        return cls(
            hidden_size=huggingface_config.hidden_size,
            num_attention_heads=huggingface_config.num_attention_heads,
            num_key_value_heads=huggingface_config.num_key_value_heads,
            num_hidden_layers=huggingface_config.num_hidden_layers,
            rope_theta=cls.get_rope_theta(huggingface_config),
            intermediate_size=huggingface_config.intermediate_size,
            vocab_size=huggingface_config.vocab_size,
            head_dim=cls.get_head_dim(huggingface_config),
            dtype=dtype,
            max_seq_len=max_seq_len,
            tie_word_embeddings=getattr(
                huggingface_config, "tie_word_embeddings", False
            ),
            rms_norm_eps=getattr(huggingface_config, "rms_norm_eps", None),
            attention_bias=getattr(huggingface_config, "attention_bias", False),
            layer_types=cls._normalized_layer_types(huggingface_config),
            sliding_window=int(
                getattr(huggingface_config, "sliding_window", -1) or -1
            ),
            mlp_only_layers=mlp_only,
            decoder_sparse_step=int(
                getattr(huggingface_config, "decoder_sparse_step", 1) or 1
            ),
            num_experts=int(getattr(huggingface_config, "num_experts", 0) or 0),
            num_experts_per_tok=int(
                getattr(huggingface_config, "num_experts_per_tok", 0) or 0
            ),
            moe_intermediate_size=int(
                getattr(huggingface_config, "moe_intermediate_size", 512)
            ),
            shared_expert_intermediate_size=int(
                getattr(
                    huggingface_config,
                    "shared_expert_intermediate_size",
                    1024,
                )
            ),
            norm_topk_prob=bool(
                getattr(huggingface_config, "norm_topk_prob", True)
            ),
            hidden_act=str(getattr(huggingface_config, "hidden_act", "silu")),
            kv_params=cls.construct_kv_params(
                huggingface_config=huggingface_config,
                pipeline_config=pipeline_config,
                devices=device_refs,
                kv_cache_config=kv_cache_config,
                cache_dtype=cache_dtype,
            ),
            devices=device_refs,
            quantization_encoding=quantization_encoding,
        )

    def finalize(
        self,
        huggingface_config: AutoConfig,
        state_dict: dict[str, WeightData],
        return_logits: ReturnLogits,
        return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE,
    ) -> None:
        """Resolves the fields that depend on the loaded weights."""
        self.tie_word_embeddings = resolve_tie_word_embeddings(
            huggingface_config, state_dict
        )
        self.return_logits = return_logits
        self.return_hidden_states = return_hidden_states

    @staticmethod
    def get_rope_theta(huggingface_config: AutoConfig) -> float:
        """Returns the rope theta stored in the HF config.

        transformers 5.x may nest the value under ``rope_parameters``, so read
        that first and fall back to the top-level key.
        """
        params = getattr(huggingface_config, "rope_parameters", None)
        if isinstance(params, dict):
            flat = {k: v for k, v in params.items() if not isinstance(v, dict)}
            for key in ("rope_theta", "theta"):
                if flat.get(key) is not None:
                    return float(flat[key])
            # Per-layer-type form: take the first non-None entry.
            for value in params.values():
                if (
                    isinstance(value, dict)
                    and value.get("rope_theta") is not None
                ):
                    return float(value["rope_theta"])
        theta = getattr(huggingface_config, "rope_theta", None)
        return float(theta) if theta is not None else 10000.0

    @staticmethod
    def get_head_dim(huggingface_config: AutoConfig) -> int:
        return getattr(
            huggingface_config,
            "head_dim",
            huggingface_config.hidden_size
            // huggingface_config.num_attention_heads,
        )

    @staticmethod
    def get_num_layers(huggingface_config: AutoConfig) -> int:
        return huggingface_config.num_hidden_layers
