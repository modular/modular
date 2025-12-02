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
"""Config for ZImage model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

from max.dtype import DType
from max.graph import DeviceRef
from max.graph.weights import WeightData
from max.nn import ReturnLogits
from max.pipelines.lib import MAXModelConfig, PipelineConfig
from transformers.models.auto.configuration_auto import AutoConfig

from max.pipelines.architectures.qwen3.model_config import Qwen3Config
from max.nn.float8_config import Float8Config, parse_float8_config
from max.nn.kv_cache import KVCacheParams
from max.pipelines.lib import KVCacheConfig 


@dataclass
class SchedulerConfig:
    """Base configuration for scheduler model with required fields."""

    _class_name: str
    """Scheduler class name."""

    _diffusers_version: str
    """Diffusers version."""

    base_image_seq_len: int
    """Base image sequence length."""

    base_shift: float
    """Base shift value."""

    invert_sigmas: bool
    """Invert sigmas flag."""

    max_image_seq_len: int
    """Max image sequence length."""

    max_shift: float
    """Max shift value."""
    
    num_train_timesteps: int
    """Number of training timesteps."""

    shift: float
    """Shift value."""

    shift_terminal: float
    """Shift terminal value."""

    stochastic_sampling: bool
    """Stochastic sampling flag."""

    time_shift_type: str
    """Time shift type."""

    use_beta_sigmas: bool
    """Use beta sigmas flag."""

    use_dynamic_shifting: bool
    """Use dynamic shifting flag."""

    use_exponential_sigmas: bool
    """Use exponential sigmas flag."""

    use_karras_sigmas: bool
    """Use karras sigmas flag."""

    @staticmethod
    def generate(
        scheduler_config: AutoConfig,
    ) -> SchedulerConfig:
        """Generate SchedulerConfig from HuggingFace scheduler config.

        Args:
            scheduler_config: HuggingFace scheduler configuration object.

        Returns:
            Configured SchedulerConfig instance.
        """
        return SchedulerConfig(
            _class_name=scheduler_config._class_name,
            _diffusers_version=scheduler_config._diffusers_version,
            base_image_seq_len=scheduler_config.base_image_seq_len,
            base_shift=scheduler_config.base_shift,
            invert_sigmas=scheduler_config.invert_sigmas,
            max_image_seq_len=scheduler_config.max_image_seq_len,
            max_shift=scheduler_config.max_shift,
            num_train_timesteps=scheduler_config.num_train_timesteps,
            shift=scheduler_config.shift,
            shift_terminal=scheduler_config.shift_terminal,
            stochastic_sampling=scheduler_config.stochastic_sampling,
            time_shift_type=scheduler_config.time_shift_type,
            use_beta_sigmas=scheduler_config.use_beta_sigmas,
            use_dynamic_shifting=scheduler_config.use_dynamic_shifting,
            use_exponential_sigmas=scheduler_config.use_exponential_sigmas,
            use_karras_sigmas=scheduler_config.use_karras_sigmas,
        )


@dataclass
class VAEConfig:
    """Base configuration for VAE model with required fields."""

    _class_name: str
    """VAE class name."""

    _diffusers_version: str
    """Diffusers version."""

    dtype: DType
    """DType of the VAE model weights."""

    devices: list[DeviceRef]
    """Devices that the VAE model is parallelized over."""

    act_fn: str
    """Activation function."""
  
    block_out_channels: Tuple[int]
    """Tuple of block output channels."""

    down_block_types: Tuple[str]
    """Tuple of downsample block types."""

    force_upcast: bool
    """If enabled it will force the VAE to run in float32 for high image resolution pipelines, such as SD-XL. VAE
    can be fine-tuned / trained to a lower range without losing too much precision in which case `force_upcast`
    can be set to `False` - see: https://huggingface.co/madebyollin/sdxl-vae-fp16-fix"""

    in_channels: int
    """Number of channels in the input image."""

    latent_channels: int
    """Number of channels in the latent space."""
    
    latents_mean: Tuple[float]
    """"""
    
    latents_std: Tuple[float]
    """"""
    
    layers_per_block: int
    """"""
    
    mid_block_add_attention: bool
    """If enabled, the mid_block of the Encoder and Decoder will have attention blocks. If set to false, the
    mid_block will only have resnet blocks"""
    
    norm_num_groups: int
    """"""
    
    out_channels: int
    """"""
    
    sample_size: int
    """"""
    
    scaling_factor: float
    """"""
    
    shift_factor: float
    """"""
    
    up_block_types: Tuple[str]
    """"""
    
    use_post_quant_conv: bool
    """"""
    
    use_quant_conv: bool
    """"""

    scaling_factor: float
    """The component-wise standard deviation of the trained latent space computed using the first batch of the
    training set. This is used to scale the latent space to have unit variance when training the diffusion
    model. The latents are scaled with the formula `z = z * scaling_factor` before being passed to the
    diffusion model. When decoding, the latents are scaled back to the original scale with the formula: `z = 1
    / scaling_factor * z`. For more details, refer to sections 4.3.2 and D.1 of the [High-Resolution Image
    Synthesis with Latent Diffusion Models](https://huggingface.co/papers/2112.10752) paper."""

    float8_config: Float8Config | None = None
    """Float8 quantization configuration for the VAE model."""

    @staticmethod
    def generate(
        vae_config: AutoConfig,
        dtype: DType,
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
        vision_state_dict: dict[str, WeightData],
    ) -> VAEConfig:
        """Generate VAEConfig from HuggingFace VAE config.

        Args:
            vae_config: HuggingFace VAE configuration object.

        Returns:
            Configured VAEConfig instance.
        """
        # Parse (if present) a float8 configuration for the vision path.
        v_float8 = parse_float8_config(
            huggingface_config,
            vision_state_dict,
            dtype,
            state_dict_name_prefix="vision_encoder.",
            ignored_modules_prefix="vision_encoder.",
        )
        return VAEConfig(
            _class_name=vae_config._class_name,
            _diffusers_version=vae_config._diffusers_version,
            dtype=dtype,
            devices=[
                DeviceRef(spec.device_type, spec.id)
                for spec in pipeline_config.model_config.device_specs
            ],
            attn_scales=vae_config.attn_scales,
            base_dim=vae_config.base_dim,
            dim_mult=vae_config.dim_mult,
            dropout=vae_config.dropout,
            latents_mean=vae_config.latents_mean,
            latents_std=vae_config.latents_std,
            num_res_blocks=vae_config.num_res_blocks,
            temperal_downsample=vae_config.temperal_downsample,
            z_dim=vae_config.z_dim,
            float8_config=v_float8,
        )


@dataclass
class TransformerConfig:
    """Base configuration for transformer model with required fields."""

    _class_name: str
    """Transformer class name."""

    _diffusers_version: str
    """Diffusers version."""

    dtype: DType
    """DType of the transformer model weights."""

    devices: list[DeviceRef]
    """Devices that the transformer model is parallelized over."""

    attention_head_dim: int
    """Attention head dimension."""

    axes_dims_rope: list[int]
    """Axes dimensions for rope."""

    guidance_embeds: bool
    """Whether to use guidance embeds."""

    in_channels: int
    """Number of input channels."""

    joint_attention_dim: int
    """Joint attention dimension."""

    num_attention_heads: int
    """Number of attention heads."""

    num_layers: int
    """Number of layers."""

    out_channels: int
    """Number of output channels."""

    patch_size: int
    """Patch size."""

    float8_config: Float8Config | None = None
    """Float8 quantization configuration for the transformer model."""

    @staticmethod
    def generate(
        transformer_config: AutoConfig,
        dtype: DType,
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
        state_dict: dict[str, WeightData],
    ) -> TransformerConfig:
        """Generate TransformerConfig from HuggingFace transformer config.

        Args:
            transformer_config: HuggingFace transformer configuration object.

        Returns:
            Configured TransformerConfig instance.
        """
        # Parse (if present) a float8 configuration for the transformer path.
        t_float8 = parse_float8_config(
            huggingface_config,
            state_dict,
            dtype,
            state_dict_name_prefix="transformer.",
            ignored_modules_prefix="transformer.",
        )
        return TransformerConfig(
            _class_name=transformer_config._class_name,
            _diffusers_version=transformer_config._diffusers_version,
            dtype=dtype,
            devices=[
                DeviceRef(spec.device_type, spec.id)
                for spec in pipeline_config.model_config.device_specs
            ],
            float8_config=t_float8,
        )


@dataclass
class ZImageConfigBase:
    """Base configuration for ZImage models with required fields."""

    devices: list[DeviceRef]
    """Devices that the ZImage model is parallelized over."""

    # Multimodal parameters
    image_token_id: int
    """Token ID used for image placeholders in the input sequence."""

    video_token_id: int
    """Token ID used for video placeholders in the input sequence."""

    vision_start_token_id: int
    """Token ID that marks the start of vision content."""

    spatial_merge_size: int
    """Size parameter for spatial merging of vision features."""

    mrope_section: list[int]
    """List of indices for the mrope section."""

    scheduler_config: SchedulerConfig
    """Scheduler configuration."""

    vae_config: VAEConfig
    """VAE configuration."""
    
    text_encoder_config: Qwen3Config
    """Text encoder configuration."""
    
    transformer_config: TransformerConfig
    """Transformer configuration."""


@dataclass
class ZImageConfig(MAXModelConfig, ZImageConfigBase):
    """Implementation of MAXModelConfig for ZImage models."""

    @staticmethod
    def help() -> dict[str, str]:
        """Returns a dictionary describing the configuration parameters."""
        # TODO: Populate this with helpful descriptions based on Args above.
        return {}

    @staticmethod
    def get_kv_params(
        huggingface_config: AutoConfig,
        n_devices: int,
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        # Delegate to Llama3Config for language model parameters.
        llm_config = getattr(
            huggingface_config, "text_config", huggingface_config
        )
        return Llama3Config.get_kv_params(
            huggingface_config=llm_config,
            n_devices=n_devices,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
        )

    @staticmethod
    def get_num_layers(huggingface_config: AutoConfig) -> int:
        # Delegate to Llama3Config for language model parameters.
        llm_config = getattr(
            huggingface_config, "text_config", huggingface_config
        )
        return Llama3Config.get_num_layers(llm_config)

    @staticmethod
    def calculate_max_seq_len(
        pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        """Calculate maximum sequence length for ZImage."""
        # Delegate to Llama3Config for language model parameters.
        llm_config = getattr(
            huggingface_config, "text_config", huggingface_config
        )
        return Llama3Config.calculate_max_seq_len(
            pipeline_config=pipeline_config,
            huggingface_config=llm_config,
        )

    @staticmethod
    def generate(
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
        vae_state_dict: dict[str, WeightData],
        text_encoder_state_dict: dict[str, dict[str, WeightData]],
        denoiser_state_dict: dict[str, WeightData],
        dtype: DType,
        n_devices: int,
        cache_dtype: DType,
        kv_cache_config: KVCacheConfig,
        return_logits: ReturnLogits,
        norm_method: Literal["rms_norm"] | Literal["layer_norm"] = "layer_norm",
    ) -> ZImageConfig:
        """Generate ZImageConfig from pipeline and HuggingFace configs.

        Args:
            pipeline_config: Pipeline configuration.
            huggingface_config: HuggingFace model configuration.
            vae_state_dict: VAE weights dictionary.
            text_encoder_state_dict: Text encoder weights dictionary.
            denoiser_state_dict: Denoiser weights dictionary.
            dtype: Data type for model parameters.
            n_devices: Number of devices.
            cache_dtype: KV cache data type.
            kv_cache_config: KV cache configuration.
            return_logits: Return logits configuration.
            norm_method: Normalization method.

        Returns:
            Configured ZImageConfig instance.
        """
        # Create SchedulerConfig from the scheduler config
        hf_scheduler_config = getattr(huggingface_config, "scheduler_config", None)
        if hf_scheduler_config is None:
            raise ValueError("scheduler_config not found in huggingface_config")
        scheduler_config = SchedulerConfig.generate(
            hf_scheduler_config,
            vae_state_dict["patch_embed.proj.weight"].dtype,
            text_encoder_state_dict["language_model.embed_tokens.weight"].dtype,
            pipeline_config,
        )
        
        # Create VAEConfig from the VAE config
        hf_vae_config = getattr(huggingface_config, "vae_config", None)
        if hf_vae_config is None:
            raise ValueError("vae_config not found in huggingface_config")
        vae_config = VAEConfig.generate(
            hf_vae_config,
            vae_state_dict["patch_embed.proj.weight"].dtype,
            text_encoder_state_dict["language_model.embed_tokens.weight"].dtype,
            pipeline_config,
        )

        # Create Qwen3Config for the text encoder
        text_encoder_config = Qwen3Config.generate(
            pipeline_config,
            huggingface_config,
            text_encoder_state_dict["llm_state_dict"],
            text_encoder_state_dict["vision_state_dict"],
            dtype,
            n_devices,
            cache_dtype,
            kv_cache_config,
            return_logits,
            norm_method,
        )
        
        # Create DonoiserConfig for the denoiser model
        denoiser_config = TransformerConfig.generate(
            huggingface_config.transformer_config,
            dtype,
            pipeline_config,
            huggingface_config,
            denoiser_state_dict,
        )

        return ZImageConfig(
            devices=[
                DeviceRef(spec.device_type, spec.id)
                for spec in pipeline_config.model_config.device_specs
            ],
            # Multimodal parameters
            image_token_id=huggingface_config.image_token_id,
            video_token_id=huggingface_config.video_token_id,
            vision_start_token_id=huggingface_config.vision_start_token_id,
            spatial_merge_size=hf_vae_config.spatial_merge_size,
            mrope_section=huggingface_config.text_config.rope_scaling[
                "mrope_section"
            ],
            # Scheduler configuration
            scheduler_config=scheduler_config,
            # Vision configuration
            vae_config=vae_config,
            # Text encoder configuration
            text_encoder_config=text_encoder_config,
            # Denoising transformer configuration
            denoiser_config=denoiser_config,
        )
