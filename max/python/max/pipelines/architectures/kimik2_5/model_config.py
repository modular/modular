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
"""Config for Kimi-K2.5 models."""

from __future__ import annotations

from dataclasses import dataclass

from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import (
    KVCacheParamInterface,
)
from max.pipelines.lib import KVCacheConfig, PipelineConfig
from max.pipelines.lib.config.config_enums import supported_encoding_dtype
from max.pipelines.lib.interfaces.arch_config import ArchConfigWithKVCache
from max.pipelines.lib.utils import upper_bounded_default
from transformers import AutoConfig
from typing_extensions import Self, override

from ..deepseekV3.model_config import DeepseekV3Config


@dataclass(kw_only=True)
class KimiK2_5TextConfig(DeepseekV3Config):
    @override
    @classmethod
    def initialize(cls, pipeline_config: PipelineConfig) -> Self:
        """Initializes a DeepseekV3Config instance from pipeline configuration.

        This method creates a config instance with all fields that can be determined
        from the pipeline configuration, without needing the state_dict.
        Fields that depend on the state_dict (like norm_dtype, float8_config, etc.)
        should be set via the `finalize()` method.

        Args:
            pipeline_config: The MAX Engine pipeline configuration.

        Returns:
            An initialized DeepseekV3Config instance.
        """
        assert pipeline_config.model.huggingface_config is not None
        config = pipeline_config.model.huggingface_config.text_config
        if config is None:
            raise ValueError(
                f"HuggingFace config is required for '{pipeline_config.model.model_path}', "
                "but config could not be loaded. "
                "Please ensure the model repository contains a valid config.json file."
            )
        kv_cache_config = pipeline_config.model.kv_cache
        quantization_encoding = pipeline_config.model.quantization_encoding
        if quantization_encoding is None:
            raise ValueError("quantization_encoding must not be None")
        dtype = supported_encoding_dtype(quantization_encoding)
        cache_dtype = pipeline_config.model.kv_cache.cache_dtype

        device_refs = [
            DeviceRef(spec.device_type, spec.id)
            for spec in pipeline_config.model.device_specs
        ]

        kv_params = cls.construct_kv_params(
            huggingface_config=config,
            pipeline_config=pipeline_config,
            devices=device_refs,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
        )

        max_seq_len = upper_bounded_default(
            upper_bound=config.max_position_embeddings,
            default=pipeline_config.model.max_length,
        )

        return cls(
            dtype=dtype,
            kv_params=kv_params,
            devices=device_refs,
            data_parallel_degree=pipeline_config.model.data_parallel_degree,
            use_subgraphs=pipeline_config.model.use_subgraphs,
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            moe_intermediate_size=config.moe_intermediate_size,
            moe_layer_freq=config.moe_layer_freq,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            n_shared_experts=config.n_shared_experts,
            n_routed_experts=config.n_routed_experts,
            routed_scaling_factor=config.routed_scaling_factor,
            kv_lora_rank=config.kv_lora_rank,
            q_lora_rank=config.q_lora_rank,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            qk_nope_head_dim=config.qk_nope_head_dim,
            topk_method=config.topk_method,
            n_group=config.n_group,
            topk_group=config.topk_group,
            num_experts_per_tok=config.num_experts_per_tok,
            first_k_dense_replace=config.first_k_dense_replace,
            norm_topk_prob=config.norm_topk_prob,
            hidden_act=config.hidden_act,
            max_position_embeddings=config.max_position_embeddings,
            max_seq_len=max_seq_len,
            rms_norm_eps=config.rms_norm_eps,
            tie_word_embeddings=config.tie_word_embeddings,
            rope_theta=config.rope_theta,
            rope_scaling=config.rope_scaling,
            rope_interleave=getattr(config, "rope_interleave", True),
            scoring_func=config.scoring_func,
            attention_bias=config.attention_bias,
            attention_dropout=config.attention_dropout,
        )

    @classmethod
    def calculate_max_seq_len(
        cls,
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
    ) -> int:
        """Calculates the maximum sequence length for the Kimi K2.5 language model."""
        try:
            return upper_bounded_default(
                upper_bound=huggingface_config.max_position_embeddings,
                default=pipeline_config.model.max_length,
            )
        except ValueError as e:
            raise ValueError(
                "Unable to infer max_length for Kimi K2.5, the provided "
                f"max_length ({pipeline_config.model.max_length}) exceeds the "
                f"model's max_position_embeddings "
                f"({huggingface_config.max_position_embeddings})."
            ) from e


@dataclass
class VisionConfig:
    """Vision configuration for Kimi-K2.5 models with required fields."""

    dtype: DType
    """DType of the Kimi-K2.5 vision model weights."""
    llm_dtype: DType
    """DType of the Kimi-K2.5 language model weights."""
    devices: list[DeviceRef]
    """Devices that the Kimi-K2.5 vision encoder model is parallelized over."""

    init_pos_emb_height: int
    """Height of the initial position embedding."""
    init_pos_emb_time: int
    """Time of the initial position embedding."""
    init_pos_emb_width: int
    """Width of the initial position embedding."""
    merge_kernel_size: list[int]
    """Kernel size for the merge operation."""
    merge_type: str
    """Type of the merge operation."""
    mm_hidden_size: int
    """Hidden size of the multi-modal hidden layer."""
    mm_projector_type: str
    """Type of the multi-modal projector."""
    model_type: str
    """Type of the model."""
    patch_size: int
    """Size of the patch."""
    pos_emb_type: str
    """Type of the position embedding."""
    projector_hidden_act: str
    """Activation function for the projector."""
    projector_ln_eps: float
    """Epsilon for the layer normalization."""
    text_hidden_size: int
    """Hidden size of the text hidden layer."""
    video_attn_type: str
    """Type of the video attention."""
    vt_hidden_size: int
    """Hidden size of the video hidden layer."""
    vt_intermediate_size: int
    """Intermediate size of the video hidden layer."""
    vt_num_attention_heads: int
    """Number of attention heads of the video hidden layer."""
    vt_num_hidden_layers: int
    """Number of hidden layers of the video hidden layer."""

    @classmethod
    def initialize_from_config(
        cls,
        pipeline_config: PipelineConfig,
        hf_vision_config: AutoConfig,
    ) -> VisionConfig:
        """Initialize VisionConfig from HuggingFace vision config.

        Note: dtype fields will be set to defaults and should be updated
        via finalize() once state_dict is available.
        """
        from max.dtype import DType as MaxDType

        return cls(
            dtype=MaxDType.bfloat16,
            llm_dtype=MaxDType.bfloat16,
            devices=[
                DeviceRef(spec.device_type, spec.id)
                for spec in pipeline_config.model.device_specs
            ],
            init_pos_emb_height=hf_vision_config.init_pos_emb_height,
            init_pos_emb_time=hf_vision_config.init_pos_emb_time,
            init_pos_emb_width=hf_vision_config.init_pos_emb_width,
            merge_kernel_size=hf_vision_config.merge_kernel_size,
            merge_type=hf_vision_config.merge_type,
            mm_hidden_size=hf_vision_config.mm_hidden_size,
            mm_projector_type=hf_vision_config.mm_projector_type,
            model_type=hf_vision_config.model_type,
            patch_size=hf_vision_config.patch_size,
            pos_emb_type=hf_vision_config.pos_emb_type,
            projector_hidden_act=hf_vision_config.projector_hidden_act,
            projector_ln_eps=hf_vision_config.projector_ln_eps,
            text_hidden_size=hf_vision_config.text_hidden_size,
            video_attn_type=hf_vision_config.video_attn_type,
            vt_hidden_size=hf_vision_config.vt_hidden_size,
            vt_intermediate_size=hf_vision_config.vt_intermediate_size,
            vt_num_attention_heads=hf_vision_config.vt_num_attention_heads,
            vt_num_hidden_layers=hf_vision_config.vt_num_hidden_layers,
        )

    def finalize(
        self,
        vision_dtype: DType,
        llm_dtype: DType,
    ) -> None:
        """Finalize VisionConfig with state_dict dependent fields."""
        self.dtype = vision_dtype
        self.llm_dtype = llm_dtype


@dataclass(kw_only=True)
class KimiK2_5Config(ArchConfigWithKVCache):
    """Configuration for Kimi-K2.5 models."""

    devices: list[DeviceRef]
    """Devices that the Kimi-K2.5 model is parallelized over."""

    dtype: DType
    """DType of the Kimi-K2.5 model weights."""

    bos_token_id: int
    """ID of the beginning-of-sequence (BOS) token."""

    eos_token_id: int
    """ID of the end-of-sequence (EOS) token."""

    ignore_index: int
    """Index that should be ignored when calculating loss (e.g., for padding)."""

    media_placeholder_token_id: int
    """Token ID used as a placeholder for media (e.g., images, video frames) within sequences."""

    pad_token_id: int
    """Token ID used for padding sequences to uniform length."""

    tie_word_embeddings: bool
    """Whether to share (tie) the input and output word embeddings in the language model."""

    use_unified_vision_chunk: bool
    """Whether to use a unified chunk for vision inputs."""

    video_placeholder: str
    """Placeholder string used to represent video segments in input text."""

    # Vision encoder configuration.
    vision_config: VisionConfig
    """Vision encoder configuration."""

    llm_config: KimiK2_5TextConfig
    """Language model configuration using DeepseekV3 architecture."""

    def get_kv_params(self) -> KVCacheParamInterface:
        """Returns the KV cache parameters from the embedded LLM config."""
        return self.llm_config.get_kv_params()

    def get_max_seq_len(self) -> int:
        """Returns the maximum sequence length from the embedded LLM config."""
        return self.llm_config.get_max_seq_len()

    @staticmethod
    def construct_kv_params(
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParamInterface:
        # Delegate to KimiK2_5TextConfig for language model parameters.
        llm_config = getattr(
            huggingface_config, "text_config", huggingface_config
        )
        return KimiK2_5TextConfig.construct_kv_params(
            huggingface_config=llm_config,
            pipeline_config=pipeline_config,
            devices=devices,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
        )

    @staticmethod
    def get_num_layers(huggingface_config: AutoConfig) -> int:
        # Delegate to KimiK2_5TextConfig for language model parameters.
        llm_config = getattr(
            huggingface_config, "text_config", huggingface_config
        )
        return KimiK2_5TextConfig.get_num_layers(llm_config)

    @staticmethod
    def calculate_max_seq_len(
        pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        # Delegate to KimiK2_5TextConfig for language model parameters.
        llm_config = getattr(
            huggingface_config, "text_config", huggingface_config
        )
        return KimiK2_5TextConfig.calculate_max_seq_len(
            pipeline_config=pipeline_config,
            huggingface_config=llm_config,
        )

    @override
    @classmethod
    def initialize(cls, pipeline_config: PipelineConfig) -> Self:
        """Initializes a Qwen3VLConfig instance from pipeline configuration.

        Args:
            pipeline_config: The MAX Engine pipeline configuration.

        Returns:
            A Qwen3VLConfig instance with fields initialized from config.
        """
        huggingface_config = pipeline_config.model.huggingface_config
        if huggingface_config is None:
            raise ValueError(
                f"HuggingFace config is required for '{pipeline_config.model.model_path}', "
                "but config could not be loaded. "
                "Please ensure the model repository contains a valid config.json file."
            )
        return cls.initialize_from_config(pipeline_config, huggingface_config)

    @classmethod
    def initialize_from_config(
        cls,
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
        llm_config: KimiK2_5TextConfig | None = None,
    ) -> Self:
        """Initializes a KimiK2_5Config from pipeline and HuggingFace configs.

        This method creates a config instance with all fields that can be
        determined from the pipeline and HuggingFace configurations, without
        needing the state_dict. Fields that depend on the state_dict should
        be set via the `finalize()` method.

        Args:
            pipeline_config: The MAX Engine pipeline configuration.
            huggingface_config: HuggingFace model configuration.
            llm_config: Pre-initialized DeepseekV3 configuration.

        Returns:
            A KimiK2_5Config instance ready for finalization.
        """
        hf_vision_config = getattr(huggingface_config, "vision_config", None)
        if hf_vision_config is None:
            raise ValueError("vision_config not found in huggingface_config")

        text_config = huggingface_config.text_config

        # Get quantization encoding for dtype
        quantization_encoding = pipeline_config.model.quantization_encoding
        if quantization_encoding is None:
            raise ValueError("quantization_encoding must not be None")
        dtype = supported_encoding_dtype(quantization_encoding)

        # Create VisionConfig from the vision config

        # Propagate quantization_config to vision_config if present on main
        # config but not vision_config
        if hasattr(huggingface_config, "quantization_config") and not hasattr(
            hf_vision_config, "quantization_config"
        ):
            hf_vision_config.quantization_config = (
                huggingface_config.quantization_config
            )

        vision_config = VisionConfig.initialize_from_config(
            pipeline_config, hf_vision_config
        )

        # Create KimiK2_5TextConfig for the language model

        # Propagate quantization_config to text_config if present on main config
        # but not text_config
        if hasattr(huggingface_config, "quantization_config") and not hasattr(
            huggingface_config.text_config, "quantization_config"
        ):
            huggingface_config.text_config.quantization_config = (
                huggingface_config.quantization_config
            )

        # For VLM models, tie_word_embeddings on the top-level config determines
        # whether lm_head.weight exists in the checkpoint. The text_config may
        # have a different value (e.g., Qwen3-VL-4B-FP8 has top-level=false but
        # text_config=true). Use top-level value to match actual checkpoint.
        if hasattr(huggingface_config, "tie_word_embeddings"):
            huggingface_config.text_config.tie_word_embeddings = (
                huggingface_config.tie_word_embeddings
            )

        if llm_config is None:
            llm_config = KimiK2_5TextConfig.initialize(pipeline_config)

        return cls(
            dtype=dtype,
            devices=[
                DeviceRef(spec.device_type, spec.id)
                for spec in pipeline_config.model.device_specs
            ],
            # Multimodal parameters
            bos_token_id=huggingface_config.bos_token_id,
            eos_token_id=huggingface_config.eos_token_id,
            ignore_index=huggingface_config.ignore_index,
            media_placeholder_token_id=huggingface_config.media_placeholder_token_id,
            pad_token_id=huggingface_config.pad_token_id,
            tie_word_embeddings=huggingface_config.tie_word_embeddings,
            use_unified_vision_chunk=huggingface_config.use_unified_vision_chunk,
            video_placeholder=huggingface_config.video_placeholder,
            # Vision configuration
            vision_config=vision_config,
            # Composed language model configuration
            llm_config=llm_config,
        )
