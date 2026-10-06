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

from dataclasses import dataclass, field
from typing import ClassVar

from max.dtype import DType
from max.graph import DeviceRef
from max.graph.weights import WeightData
from max.nn.kv_cache import MultiKVCacheParams
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.architectures.gpt_oss.hybrid_kv_params_util import (
    hybrid_swa_full_kv_params,
)
from max.pipelines.kv_cache import cache_dtype_for_encoding
from max.pipelines.lib import KVCacheConfig, MAXModelConfig, PipelineConfig
from max.pipelines.lib.config.model_config import _select_quantization_encoding
from max.pipelines.lib.interfaces.arch_config import (
    ArchConfigWithBoundedMaxSeqLen,
    ArchConfigWithKVCache,
    ArchConfigWithVisionCache,
)
from max.pipelines.modeling.config_enums import (
    SupportedEncoding,
    supported_encoding_dtype,
)
from transformers import AutoConfig
from typing_extensions import Self, override


def _rope_theta(hf_config: AutoConfig) -> float:
    rope_params = hf_config.rope_parameters
    if rope_params.get("rope_type") != "default":
        raise ValueError(f"Rope parameters {rope_params} not supported")
    return rope_params["rope_theta"]


@dataclass(kw_only=True)
class MuseGlimmerTextConfig(ArchConfigWithBoundedMaxSeqLen):
    """Text decoder configuration for Muse Glimmer."""

    dtype: DType
    devices: list[DeviceRef]
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    num_hidden_layers: int
    intermediate_size: int
    vocab_size: int
    rms_norm_eps: float
    """Epsilon of the input, pre-FFN, embedding, qk and final norms."""
    post_norm_eps: float
    """Epsilon of the post-attention and post-FFN norms."""
    layer_types: list[str]
    layer_rope_theta: list[float]
    """Per-layer RoPE theta; 0 marks a NoPE layer."""
    sliding_window: int
    rope_theta: float
    max_position_embeddings: int
    max_seq_len: int
    qk_scale_factor: float
    output_multiplier: float
    """Scales the final hidden state before ``lm_head``."""
    final_logit_softcapping: float
    tie_word_embeddings: bool = False
    kv_params: MultiKVCacheParams
    return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN
    return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE
    target_layer_ids: list[int] | None = None
    use_rope: list[bool] = field(init=False)

    def __post_init__(self) -> None:
        self.use_rope = [theta != 0 for theta in self.layer_rope_theta]
        sliding = [t == "sliding_attention" for t in self.layer_types]
        # RoPE on sliding layers only is an architecture invariant; a checkpoint
        # that breaks it needs a model change, not a config read.
        if len(self.layer_types) != self.num_hidden_layers or (
            self.use_rope != sliding
        ):
            raise ValueError(
                "layer_rope_theta must be nonzero exactly on the"
                f" sliding_attention layers: layer_types={self.layer_types},"
                f" layer_rope_theta={self.layer_rope_theta}"
            )

    @classmethod
    def construct_kv_params(
        cls,
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> MultiKVCacheParams:
        """Builds the ``sliding_attention`` + ``full_attention`` KV tree."""
        return hybrid_swa_full_kv_params(
            layer_types=huggingface_config.layer_types,
            sliding_window=huggingface_config.sliding_window,
            pipeline_config=pipeline_config,
            devices=devices,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
            n_kv_heads=huggingface_config.num_key_value_heads,
            head_dim=huggingface_config.head_dim,
        )

    @classmethod
    def initialize_from_config(
        cls,
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
        *,
        max_seq_len: int,
        dtype: DType,
        cache_dtype: DType,
    ) -> Self:
        """Initializes from the HuggingFace ``text_config``.

        Args:
            pipeline_config: The MAX Engine pipeline configuration.
            huggingface_config: The HuggingFace text configuration.
            max_seq_len: The effective maximum sequence length.
            dtype: The weight dtype.
            cache_dtype: The KV cache dtype.

        Returns:
            An initialized :obj:`MuseGlimmerTextConfig`.
        """
        devices = [
            DeviceRef(spec.device_type, spec.id)
            for spec in pipeline_config.model.device_specs
        ]
        return cls(
            dtype=dtype,
            devices=devices,
            hidden_size=huggingface_config.hidden_size,
            num_attention_heads=huggingface_config.num_attention_heads,
            num_key_value_heads=huggingface_config.num_key_value_heads,
            head_dim=huggingface_config.head_dim,
            num_hidden_layers=huggingface_config.num_hidden_layers,
            intermediate_size=huggingface_config.intermediate_size,
            vocab_size=huggingface_config.vocab_size,
            rms_norm_eps=huggingface_config.rms_norm_eps,
            post_norm_eps=huggingface_config.post_norm_eps,
            layer_types=list(huggingface_config.layer_types),
            layer_rope_theta=list(huggingface_config.layer_rope_theta),
            sliding_window=huggingface_config.sliding_window,
            rope_theta=_rope_theta(huggingface_config),
            max_position_embeddings=huggingface_config.max_position_embeddings,
            max_seq_len=max_seq_len,
            qk_scale_factor=huggingface_config.qk_scale_factor,
            output_multiplier=huggingface_config.output_multiplier,
            final_logit_softcapping=huggingface_config.final_logit_softcapping,
            tie_word_embeddings=huggingface_config.tie_word_embeddings,
            kv_params=cls.construct_kv_params(
                huggingface_config=huggingface_config,
                pipeline_config=pipeline_config,
                devices=devices,
                kv_cache_config=pipeline_config.model.kv_cache,
                cache_dtype=cache_dtype,
            ),
        )


@dataclass(kw_only=True)
class MuseGlimmerVisionConfig:
    """Vision tower and adapter configuration for Muse Glimmer."""

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    hidden_act: str
    layer_norm_eps: float
    layer_types: list[str]
    """``full_attention`` or ``window_attention`` per layer."""
    patch_size: int
    patch_temporal: int
    merge_size: int
    pos_emb_height: int
    pos_emb_width: int
    rope_theta: float
    projector_hidden_size: int
    window_size_patches: int = 32
    """Side of the square patch window the ``window_attention`` layers use."""

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def patch_dim(self) -> int:
        """Width of one flattened ``temporal x RGB x patch x patch`` input."""
        return self.patch_temporal * 3 * self.patch_size**2

    @property
    def out_hidden_size(self) -> int:
        """Width of one merged token after the pixel shuffle."""
        return self.hidden_size * self.merge_size**2

    @classmethod
    def initialize_from_config(
        cls, huggingface_config: AutoConfig
    ) -> MuseGlimmerVisionConfig:
        """Initializes from the top-level HuggingFace config.

        Args:
            huggingface_config: The top-level HuggingFace configuration, which
                holds the projector widths beside ``vision_config``.

        Returns:
            An initialized :obj:`MuseGlimmerVisionConfig`.
        """
        hf_vision = huggingface_config.vision_config
        config = cls(
            hidden_size=hf_vision.hidden_size,
            intermediate_size=hf_vision.intermediate_size,
            num_hidden_layers=hf_vision.num_hidden_layers,
            num_attention_heads=hf_vision.num_attention_heads,
            hidden_act=hf_vision.hidden_act,
            layer_norm_eps=hf_vision.layer_norm_eps,
            layer_types=list(hf_vision.layer_types),
            patch_size=hf_vision.patch_size,
            patch_temporal=hf_vision.patch_temporal,
            merge_size=hf_vision.merge_size,
            pos_emb_height=hf_vision.pos_emb_height,
            pos_emb_width=hf_vision.pos_emb_width,
            rope_theta=_rope_theta(hf_vision),
            projector_hidden_size=huggingface_config.projector_hidden_size,
        )
        if config.out_hidden_size != huggingface_config.out_hidden_size:
            raise ValueError(
                f"out_hidden_size {huggingface_config.out_hidden_size} does"
                f" not match the merged width {config.out_hidden_size}"
            )
        return config


# processor_config.json's ``max_image_tokens``: merged tokens of the largest
# image the processor emits.
_MAX_IMAGE_TOKENS = 4096


@dataclass(kw_only=True)
class MuseGlimmerConfig(ArchConfigWithKVCache, ArchConfigWithVisionCache):
    """Top-level Muse Glimmer configuration composing text and vision."""

    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "bfloat16"
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {"bfloat16"}

    devices: list[DeviceRef]
    dtype: DType
    kv_params: MultiKVCacheParams
    image_token_id: int
    video_token_id: int
    text_config: MuseGlimmerTextConfig
    vision_config: MuseGlimmerVisionConfig | None
    """``None`` when the checkpoint has no ``vision_config``."""
    quantization_encoding: SupportedEncoding | None = None

    def get_kv_params(self) -> MultiKVCacheParams:
        """Returns the KV cache parameters."""
        return self.kv_params

    def get_max_seq_len(self) -> int:
        """Returns the maximum sequence length of the text decoder."""
        return self.text_config.get_max_seq_len()

    @classmethod
    def estimate_vision_cache_entry_bytes(
        cls, huggingface_config: AutoConfig
    ) -> int:
        """Bytes of one max-resolution image's embeddings, or ``0`` for a
        checkpoint without a vision tower."""
        spec = cls.get_vision_cache_row_spec(huggingface_config)
        if spec is None:
            return 0
        hidden, dtype = spec
        return _MAX_IMAGE_TOKENS * hidden * dtype.size_in_bytes

    @classmethod
    def get_vision_cache_row_spec(
        cls, huggingface_config: AutoConfig
    ) -> tuple[int, DType] | None:
        """One bfloat16 row of the text hidden size per merged image token."""
        if getattr(huggingface_config, "vision_config", None) is None:
            return None
        return (huggingface_config.text_config.hidden_size, DType.bfloat16)

    @staticmethod
    def construct_kv_params(
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> MultiKVCacheParams:
        """Constructs KV cache parameters from the top-level HuggingFace config."""
        return MuseGlimmerTextConfig.construct_kv_params(
            huggingface_config=huggingface_config.text_config,
            pipeline_config=pipeline_config,
            devices=devices,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
        )

    @classmethod
    def calculate_max_seq_len(
        cls,
        huggingface_config: AutoConfig,
        model_config: MAXModelConfig,
    ) -> int:
        """Bounds ``max_length`` by the text decoder's context."""
        return MuseGlimmerTextConfig.calculate_max_seq_len(
            huggingface_config.text_config, model_config
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
        """Initializes from pipeline configuration.

        Args:
            pipeline_config: The MAX Engine pipeline configuration.
            model_config: Optional model config override.
            max_seq_len: The effective maximum sequence length.

        Returns:
            An initialized config instance.
        """
        model_config = model_config or pipeline_config.model
        huggingface_config = model_config.huggingface_config
        if huggingface_config is None:
            raise ValueError(
                "HuggingFace config is required for"
                f" '{model_config.model_path}', but config could not be loaded."
                " Please ensure the model repository contains a valid"
                " config.json file."
            )
        return cls.initialize_from_config(
            pipeline_config, huggingface_config, max_seq_len=max_seq_len
        )

    @classmethod
    def initialize_from_config(
        cls,
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
        *,
        max_seq_len: int,
    ) -> Self:
        """Initializes from pipeline and HuggingFace configs.

        Args:
            pipeline_config: The MAX Engine pipeline configuration.
            huggingface_config: Top-level HuggingFace model configuration.
            max_seq_len: The effective maximum sequence length.

        Returns:
            A config instance ready for finalization.
        """
        quantization_encoding = _select_quantization_encoding(
            pipeline_config.model, cls.DEFAULT_ENCODING
        )
        dtype = supported_encoding_dtype(quantization_encoding)
        cache_dtype = cache_dtype_for_encoding(
            quantization_encoding,
            pipeline_config.model.kv_cache.kv_cache_format,
        )

        hf_text_config = getattr(huggingface_config, "text_config", None)
        if hf_text_config is None:
            raise ValueError("text_config not found in huggingface_config")
        text_config = MuseGlimmerTextConfig.initialize_from_config(
            pipeline_config,
            hf_text_config,
            max_seq_len=max_seq_len,
            dtype=dtype,
            cache_dtype=cache_dtype,
        )
        vision_config = (
            MuseGlimmerVisionConfig.initialize_from_config(huggingface_config)
            if getattr(huggingface_config, "vision_config", None) is not None
            else None
        )

        return cls(
            devices=text_config.devices,
            dtype=dtype,
            kv_params=text_config.kv_params,
            image_token_id=huggingface_config.image_token_id,
            video_token_id=huggingface_config.video_token_id,
            text_config=text_config,
            vision_config=vision_config,
            quantization_encoding=quantization_encoding,
        )

    def finalize(
        self,
        huggingface_config: AutoConfig,
        state_dict: dict[str, WeightData],
        return_logits: ReturnLogits,
    ) -> None:
        """Sets the fields that depend on the loaded weights and pipeline.

        Args:
            huggingface_config: HuggingFace model configuration.
            state_dict: Model weights dictionary.
            return_logits: Return logits configuration.
        """
        self.text_config.return_logits = return_logits
