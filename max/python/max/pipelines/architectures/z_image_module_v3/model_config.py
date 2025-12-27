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
from types import SimpleNamespace

from max.dtype import DType
from max.graph.weights import WeightData
from max.nn import ReturnLogits
from max.nn.kv_cache import KVCacheParams
from max.pipelines.architectures.qwen3.model_config import Qwen3Config
from max.pipelines.lib import KVCacheConfig, MAXModelConfig, PipelineConfig
from transformers.models.auto.configuration_auto import AutoConfig


@dataclass
class SchedulerConfig:
    """Base configuration for scheduler model with required fields."""

    base_image_seq_len: int | None = 256
    """Base image sequence length."""

    base_shift: float | None = 0.5
    """Base shift value."""

    invert_sigmas: bool = False
    """Invert sigmas flag."""

    max_image_seq_len: int | None = 4096
    """Max image sequence length."""

    max_shift: float | None = 1.15
    """Max shift value."""

    num_train_timesteps: int = 1000
    """Number of training timesteps."""

    shift: float = 1.0
    """Shift value."""

    shift_terminal: float | None = None
    """Shift terminal value."""

    stochastic_sampling: bool = False
    """Stochastic sampling flag."""

    time_shift_type: str = "exponential"
    """Time shift type."""

    use_beta_sigmas: bool = False
    """Use beta sigmas flag."""

    use_dynamic_shifting: bool = False
    """Use dynamic shifting flag."""

    use_exponential_sigmas: bool | None = False
    """Use exponential sigmas flag."""

    use_karras_sigmas: bool | None = False
    """Use karras sigmas flag."""

    @staticmethod
    def generate(
        scheduler_config: SimpleNamespace,
    ) -> SchedulerConfig:
        """Generate SchedulerConfig from HuggingFace scheduler config.

        Args:
            scheduler_config: HuggingFace scheduler configuration object.

        Returns:
            Configured SchedulerConfig instance.
        """
        return SchedulerConfig(
            base_image_seq_len=getattr(
                scheduler_config, "base_image_seq_len", 256
            ),
            base_shift=getattr(scheduler_config, "base_shift", 0.5),
            invert_sigmas=getattr(scheduler_config, "invert_sigmas", False),
            max_image_seq_len=getattr(
                scheduler_config, "max_image_seq_len", 4096
            ),
            max_shift=getattr(scheduler_config, "max_shift", 1.15),
            num_train_timesteps=getattr(
                scheduler_config, "num_train_timesteps", 1000
            ),
            shift=getattr(scheduler_config, "shift", 1.0),
            shift_terminal=getattr(scheduler_config, "shift_terminal", None),
            stochastic_sampling=getattr(
                scheduler_config, "stochastic_sampling", False
            ),
            time_shift_type=getattr(
                scheduler_config, "time_shift_type", "exponential"
            ),
            use_beta_sigmas=getattr(scheduler_config, "use_beta_sigmas", False),
            use_dynamic_shifting=getattr(
                scheduler_config, "use_dynamic_shifting", False
            ),
            use_exponential_sigmas=getattr(
                scheduler_config, "use_exponential_sigmas", False
            ),
            use_karras_sigmas=getattr(
                scheduler_config, "use_karras_sigmas", False
            ),
        )


@dataclass
class VAEConfig:
    """Base configuration for VAE model with required fields."""

    act_fn: str | None = None
    """Activation function."""

    block_out_channels: list[int] | None = None
    """List of block output channels."""

    down_block_types: list[str] | None = None
    """List of downsample block types."""

    force_upcast: bool | None = None
    """If enabled it will force the VAE to run in float32 for high image resolution pipelines, such as SD-XL. VAE
    can be fine-tuned / trained to a lower range without losing too much precision in which case `force_upcast`
    can be set to `False` - see: https://huggingface.co/madebyollin/sdxl-vae-fp16-fix"""

    in_channels: int | None = None
    """Number of channels in the input image."""

    latent_channels: int | None = None
    """Number of channels in the latent space."""

    latents_mean: list[float] | None = None
    """Latents mean."""

    latents_std: list[float] | None = None
    """Latents standard deviation."""

    layers_per_block: int | None = None
    """Number of layers per block."""

    mid_block_add_attention: bool | None = None
    """If enabled, the mid_block of the Encoder and Decoder will have attention blocks. If set to false, the
    mid_block will only have resnet blocks"""

    norm_num_groups: int | None = None
    """Number of normalization groups."""

    out_channels: int | None = None
    """Number of output channels."""

    sample_size: int | None = None
    """Sample size."""

    shift_factor: float | None = None
    """Shift factor."""

    up_block_types: list[str] | None = None
    """List of upsample block types."""

    use_post_quant_conv: bool | None = None
    """Use post quantization convolution flag."""

    use_quant_conv: bool | None = None
    """Use quantization convolution flag."""

    scaling_factor: float | None = None
    """The component-wise standard deviation of the trained latent space computed using the first batch of the
    training set. This is used to scale the latent space to have unit variance when training the diffusion
    model. The latents are scaled with the formula `z = z * scaling_factor` before being passed to the
    diffusion model. When decoding, the latents are scaled back to the original scale with the formula: `z = 1
    / scaling_factor * z`. For more details, refer to sections 4.3.2 and D.1 of the [High-Resolution Image
    Synthesis with Latent Diffusion Models](https://huggingface.co/papers/2112.10752) paper."""

    @staticmethod
    def generate(
        vae_config: SimpleNamespace,
        dtype: DType,
        pipeline_config: PipelineConfig,
    ) -> VAEConfig:
        """Generate VAEConfig from HuggingFace VAE config.

        Args:
            vae_config: HuggingFace VAE configuration object.

        Returns:
            Configured VAEConfig instance.
        """
        return VAEConfig(
            act_fn=vae_config.act_fn,
            block_out_channels=vae_config.block_out_channels,
            down_block_types=vae_config.down_block_types,
            force_upcast=vae_config.force_upcast,
            in_channels=vae_config.in_channels,
            latent_channels=vae_config.latent_channels,
            latents_mean=vae_config.latents_mean,
            latents_std=vae_config.latents_std,
            layers_per_block=vae_config.layers_per_block,
            mid_block_add_attention=vae_config.mid_block_add_attention,
            norm_num_groups=vae_config.norm_num_groups,
            out_channels=vae_config.out_channels,
            sample_size=vae_config.sample_size,
            scaling_factor=vae_config.scaling_factor,
            shift_factor=vae_config.shift_factor,
            up_block_types=vae_config.up_block_types,
            use_post_quant_conv=vae_config.use_post_quant_conv,
            use_quant_conv=vae_config.use_quant_conv,
        )


@dataclass
class TransformerConfig:
    """Base configuration for transformer model with required fields."""

    all_f_patch_size: list[int] | None = None
    """All f patch size."""

    all_patch_size: list[int] | None = None
    """All patch size."""

    axes_dims: list[int] | None = None
    """Axes dimensions."""

    axes_lens: list[int] | None = None
    """Axes lengths."""

    cap_feat_dim: int | None = None
    """Capacity feature dimension."""

    dim: int | None = None
    """Dimension."""

    in_channels: int | None = None
    """Number of input channels."""

    n_heads: int | None = None
    """Number of heads."""

    n_kv_heads: int | None = None
    """Number of KV heads."""

    n_layers: int | None = None
    """Number of layers."""

    n_refiner_layers: int | None = None
    """Number of refiner layers."""

    norm_eps: float | None = None
    """Normalization epsilon."""

    qk_norm: bool | None = None
    """Query-Key normalization flag."""

    rope_theta: float | None = None
    """RoPE theta."""

    t_scale: float | None = None
    """Time scale."""

    @staticmethod
    def generate(
        transformer_config: SimpleNamespace,
        dtype: DType,
        pipeline_config: PipelineConfig,
    ) -> TransformerConfig:
        """Generate TransformerConfig from HuggingFace transformer config.

        Args:
            transformer_config: HuggingFace transformer configuration object.

        Returns:
            Configured TransformerConfig instance.
        """
        return TransformerConfig(
            all_f_patch_size=transformer_config.all_f_patch_size,
            all_patch_size=transformer_config.all_patch_size,
            axes_dims=transformer_config.axes_dims,
            axes_lens=transformer_config.axes_lens,
            cap_feat_dim=transformer_config.cap_feat_dim,
            dim=transformer_config.dim,
            in_channels=transformer_config.in_channels,
            n_heads=transformer_config.n_heads,
            n_kv_heads=transformer_config.n_kv_heads,
            n_layers=transformer_config.n_layers,
            n_refiner_layers=transformer_config.n_refiner_layers,
            norm_eps=transformer_config.norm_eps,
            qk_norm=transformer_config.qk_norm,
            rope_theta=transformer_config.rope_theta,
            t_scale=transformer_config.t_scale,
        )


@dataclass
class ZImageConfigBase:
    """Base configuration for ZImage models with required fields."""

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
        # Delegate to Qwen3Config for language model parameters.
        llm_config = getattr(
            huggingface_config, "text_config", huggingface_config
        )
        return Qwen3Config.get_kv_params(
            huggingface_config=llm_config,
            n_devices=n_devices,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
        )

    @staticmethod
    def get_num_layers(huggingface_config: AutoConfig) -> int:
        # Delegate to Qwen3Config for language model parameters.
        llm_config = getattr(
            huggingface_config, "text_config", huggingface_config
        )
        return Qwen3Config.get_num_layers(llm_config)

    @staticmethod
    def calculate_max_seq_len(
        pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        """Calculate maximum sequence length for ZImage."""
        # Delegate to Qwen3Config for language model parameters.
        llm_config = getattr(
            huggingface_config, "text_config", huggingface_config
        )
        return Qwen3Config.calculate_max_seq_len(
            pipeline_config=pipeline_config,
            huggingface_config=llm_config,
        )

    @staticmethod
    def generate(
        pipeline_config: PipelineConfig,
        scheduler_config: SchedulerConfig,
        vae_config: VAEConfig,
        text_encoder_config: AutoConfig,
        transformer_config: SimpleNamespace,
        vae_state_dict: dict[str, WeightData],
        text_encoder_state_dict: dict[str, dict[str, WeightData]],
        transformer_state_dict: dict[str, WeightData],
        dtype: DType,
        n_devices: int,
        cache_dtype: DType,
        kv_cache_config: KVCacheConfig,
        return_logits: ReturnLogits,
    ) -> ZImageConfig:
        """Generate ZImageConfig from pipeline and HuggingFace configs.

        Args:
            pipeline_config: Pipeline configuration.
            scheduler_config: Scheduler configuration.
            vae_config: VAE configuration.
            text_encoder_config: Text encoder configuration.
            transformer_config: Transformer configuration.
            vae_state_dict: VAE weights dictionary.
            text_encoder_state_dict: Text encoder weights dictionary.
            transformer_state_dict: Transformer weights dictionary.
            dtype: Data type for model parameters.
            n_devices: Number of devices.
            cache_dtype: KV cache data type.
            kv_cache_config: KV cache configuration.
            return_logits: Return logits configuration.
            norm_method: Normalization method.

        Returns:
            Configured ZImageConfig instance.
        """
        # Create SchedulerConfig from the scheduler_config
        scheduler_config = SchedulerConfig.generate(scheduler_config)

        # Create VAEConfig from the vae_config
        vae_config = VAEConfig.generate(
            vae_config,
            vae_state_dict["encoder.conv_in.weight"].dtype,
            pipeline_config,
        )

        # Create Qwen3Config for the text encoder
        # Use ReturnHiddenStates.SECOND_TO_LAST to get hidden_states[-2]
        # (second-to-last layer) for Z-Image conditioning - matching diffusers behavior
        from max.nn import ReturnHiddenStates

        text_encoder_config = Qwen3Config.generate(
            pipeline_config,
            text_encoder_config,
            text_encoder_state_dict["llm_state_dict"],
            dtype,
            n_devices,
            cache_dtype,
            kv_cache_config,
            return_logits,
            return_hidden_states=ReturnHiddenStates.SECOND_TO_LAST,
        )

        # Create TransformerConfig for the backbone of the pipeline
        transformer_config = TransformerConfig.generate(
            transformer_config,
            transformer_state_dict["layers.0.feed_forward.w1.weight"],
            pipeline_config,
        )

        # Return a new ZImageConfig with the corrected parameters
        return ZImageConfig(
            scheduler_config=scheduler_config,
            vae_config=vae_config,
            text_encoder_config=text_encoder_config,
            transformer_config=transformer_config,
        )
