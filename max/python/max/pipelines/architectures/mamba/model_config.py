# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Qwerky AI Inc. All rights reserved.
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
"""Config for Mamba models."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, Any

logger = logging.getLogger("max.pipelines")

from max.dtype import DType
from max.graph import DeviceRef
from max.nn import (
    ReturnHiddenStates,
    ReturnLogits,
)
from max.nn.float8_config import Float8Config
from max.pipelines.lib import (
    KVCacheConfig,
    MAXModelConfig,
    MAXModelConfigBase,
    PipelineConfig,
    upper_bounded_default,
)
from transformers import AutoConfig


@dataclass
class MambaConfigBase(MAXModelConfigBase):
    """Base configuration for Mamba models."""

    # Required fields
    max_seq_len: int
    """Maximum sequence length for the model."""
    vocab_size: int
    """Vocabulary size of the model."""
    dtype: DType
    """Data type for model parameters."""
    devices: list[DeviceRef]
    """List of devices to run the model on."""
    float8_config: Float8Config | None
    """Float8 quantization configuration, if applicable."""
    # Model architecture fields
    hidden_size: int = 2560
    """Hidden dimension size (d_model in HuggingFace config)."""
    intermediate_size: int = 0
    """Intermediate dimension for SSM expansion (d_inner or d_intermediate in HuggingFace config)."""
    num_hidden_layers: int = 64
    """Number of hidden layers (n_layer in HuggingFace config)."""
    # SSM-specific fields
    d_state: int = 16
    """State space dimension (state_size in HuggingFace config)."""
    dt_rank: int | str | None = None
    """Rank of delta projection (time_step_rank in HuggingFace config). 'auto' sets it to ceil(hidden_size / 16)."""
    conv_kernel: int = 4
    """Convolution kernel size."""
    x_proj_dim: int | None = None
    """Output dimension of x_proj (inferred from weights if None)."""
    # Time step parameters for SSM
    time_step_min: float = 0.001
    """Minimum value for dt initialization (time_step_min in HuggingFace config)."""
    time_step_max: float = 0.1
    """Maximum value for dt initialization (time_step_max in HuggingFace config)."""
    time_step_init_scheme: str = "random"
    """Initialization method for dt_proj weights (time_step_init_scheme in HuggingFace config)."""
    time_step_scale: float = 1.0
    """Scale factor for dt_proj weight initialization (time_step_scale in HuggingFace config)."""
    time_step_floor: float = 1e-4
    """Floor value for dt initialization (time_step_floor in HuggingFace config)."""
    # Normalization fields
    norm_method: Literal["rms_norm"] | Literal["layer_norm"] = "rms_norm"
    """Normalization method to use."""
    rms_norm_eps: float | None = None
    """RMS normalization epsilon (layer_norm_epsilon in HuggingFace config)."""
    norm_dtype: DType | None = None
    """Data type for normalization layers (uses dtype if None)."""
    # Other configuration fields
    expand: int = 2
    """Expansion factor for intermediate size (expand in HuggingFace config)."""
    use_bias: bool = False
    """Whether to use bias in linear projections (use_bias in HuggingFace config)."""
    use_conv_bias: bool = True
    """Whether to use bias in conv1d (use_conv_bias in HuggingFace config)."""
    hidden_act: str = "silu"
    """Hidden activation function (hidden_act in HuggingFace config)."""
    rescale_prenorm_residual: bool = False
    """Whether to rescale pre-normalization residual (rescale_prenorm_residual in HuggingFace config)."""
    # Legacy/compatibility fields (kept for backward compatibility)
    d_model: int = 2560
    """Deprecated: Use hidden_size instead. Kept for compatibility."""
    d_intermediate: int = 0
    """Deprecated: Use intermediate_size instead. Kept for compatibility."""
    n_layer: int = 64
    """Deprecated: Use num_hidden_layers instead. Kept for compatibility."""
    ssm_cfg: dict[str, Any] | None = None
    """SSM configuration dictionary."""
    attn_layer_idx: list[int] | None = None
    """Indices of attention layers (if any)."""
    attn_cfg: dict[str, Any] | None = None
    """Attention configuration dictionary."""
    rms_norm: bool = True
    """Whether to use RMS normalization (deprecated: use norm_method instead)."""
    residual_in_fp32: bool = True
    """Whether to compute residuals in fp32."""
    fused_add_norm: bool = True
    """Whether to use fused add-norm operations for better performance."""
    pad_vocab_size_multiple: int = 8
    """Pad vocabulary size to multiple of this value."""
    tie_embeddings: bool = True
    """Deprecated: Use tie_word_embeddings instead. Kept for compatibility."""
    tie_word_embeddings: bool = True
    """Whether to tie word embeddings with output layer."""
    # Output configuration
    return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN
    """Which logits to return from the model."""
    return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE
    """Whether to return hidden states from the model."""


    @staticmethod
    def help() -> dict[str, str]:
        return {}


@dataclass
class MambaConfig(MAXModelConfig, MambaConfigBase):
    """Implementation of MAXModelConfig for Mamba models."""

    @staticmethod
    def get_num_layers(huggingface_config: AutoConfig) -> int:
        """Get the number of layers from HuggingFace config.
        
        Checks both num_hidden_layers and n_layer for compatibility.
        """
        return getattr(huggingface_config, "num_hidden_layers", None) or getattr(
            huggingface_config, "n_layer", 64
        )

    @staticmethod
    def calculate_max_seq_len(
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
    ) -> int:
        try:
            return upper_bounded_default(
                upper_bound=getattr(
                    huggingface_config, "max_position_embeddings", 2048
                ),
                default=pipeline_config.max_length,
            )
        except ValueError as e:
            raise ValueError(
                "Unable to infer max_length for Mamba, the provided "
                f"max_length ({pipeline_config.max_length}) exceeds the "
                f"model's max_position_embeddings "
                f"({getattr(huggingface_config, 'max_position_embeddings', 2048)})."
            ) from e

    @staticmethod
    def generate(
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
        state_dict: dict,
        dtype: DType,
        n_devices: int,
        norm_method: Literal["rms_norm"] | Literal["layer_norm"] = "rms_norm",
        cache_dtype: DType | None = None,
        kv_cache_config: KVCacheConfig | None = None,
        return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
        return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE,
    ) -> MambaConfig:
        device_refs = [
            DeviceRef(spec.device_type, spec.id)
            for spec in pipeline_config.model_config.device_specs[:n_devices]
        ]

        # Parse the float8 config from compressed-tensors or FBGEMM.
        from max.pipelines.lib.float8 import parse_float8_config

        float8_config = parse_float8_config(
            huggingface_config, state_dict, dtype
        )

        # Map model architecture fields
        # hidden_size can come from d_model or hidden_size
        hidden_size = getattr(huggingface_config, "hidden_size", None) or getattr(
            huggingface_config, "d_model", 2560
        )
        
        # expand factor for intermediate size calculation
        expand = getattr(huggingface_config, "expand", 2)
        
        # intermediate_size can come from d_inner, d_intermediate, or intermediate_size
        # If not provided or 0, calculate as expand * hidden_size (matches reference: d_inner = int(expand * d_model))
        intermediate_size = (
            getattr(huggingface_config, "intermediate_size", None)
            or getattr(huggingface_config, "d_inner", None)
            or getattr(huggingface_config, "d_intermediate", None)
        )
        if intermediate_size is None or intermediate_size == 0:
            intermediate_size = int(expand * hidden_size)
        
        # num_hidden_layers can come from num_hidden_layers or n_layer
        num_hidden_layers = getattr(
            huggingface_config, "num_hidden_layers", None
        ) or getattr(huggingface_config, "n_layer", 64)

        # SSM-specific fields
        d_state = getattr(huggingface_config, "state_size", 16)
        dt_rank = getattr(huggingface_config, "time_step_rank", None)
        conv_kernel = getattr(huggingface_config, "conv_kernel", 4)
        x_proj_dim = getattr(huggingface_config, "x_proj_dim", None)

        # Time step parameters
        time_step_min = getattr(huggingface_config, "time_step_min", 0.001)
        time_step_max = getattr(huggingface_config, "time_step_max", 0.1)
        time_step_init_scheme = getattr(
            huggingface_config, "time_step_init_scheme", "random"
        )
        time_step_scale = getattr(huggingface_config, "time_step_scale", 1.0)
        time_step_floor = getattr(huggingface_config, "time_step_floor", 1e-4)

        # Normalization fields
        rms_norm_eps = getattr(
            huggingface_config, "layer_norm_epsilon", None
        ) or getattr(huggingface_config, "rms_norm_eps", None)

        # Other configuration fields
        use_bias = getattr(huggingface_config, "use_bias", False)
        use_conv_bias = getattr(huggingface_config, "use_conv_bias", True)
        hidden_act = getattr(huggingface_config, "hidden_act", "silu")
        rescale_prenorm_residual = getattr(
            huggingface_config, "rescale_prenorm_residual", False
        )

        # Legacy/compatibility fields
        d_model = hidden_size
        d_intermediate = intermediate_size
        n_layer = num_hidden_layers
        ssm_cfg = getattr(huggingface_config, "ssm_cfg", None)
        attn_layer_idx = getattr(huggingface_config, "attn_layer_idx", None)
        attn_cfg = getattr(huggingface_config, "attn_cfg", None)
        rms_norm = getattr(huggingface_config, "rms_norm", True)
        residual_in_fp32 = getattr(huggingface_config, "residual_in_fp32", True)
        fused_add_norm = getattr(huggingface_config, "fused_add_norm", True)
        pad_vocab_size_multiple = getattr(
            huggingface_config, "pad_vocab_size_multiple", 8
        )
        tie_embeddings = getattr(huggingface_config, "tie_embeddings", True)
        tie_word_embeddings = tie_embeddings

        return MambaConfig(
            vocab_size=huggingface_config.vocab_size,
            dtype=dtype,
            max_seq_len=MambaConfig.calculate_max_seq_len(
                pipeline_config, huggingface_config=huggingface_config
            ),
            # Model architecture fields
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            # SSM-specific fields
            d_state=d_state,
            dt_rank=dt_rank,
            conv_kernel=conv_kernel,
            x_proj_dim=x_proj_dim,
            # Time step parameters
            time_step_min=time_step_min,
            time_step_max=time_step_max,
            time_step_init_scheme=time_step_init_scheme,
            time_step_scale=time_step_scale,
            time_step_floor=time_step_floor,
            # Normalization fields
            norm_method=norm_method,
            rms_norm_eps=rms_norm_eps,
            norm_dtype=None,  # Can be set from pipeline config if needed
            # Other configuration fields
            expand=expand,
            use_bias=use_bias,
            use_conv_bias=use_conv_bias,
            hidden_act=hidden_act,
            rescale_prenorm_residual=rescale_prenorm_residual,
            # Legacy/compatibility fields
            d_model=d_model,
            d_intermediate=d_intermediate,
            n_layer=n_layer,
            ssm_cfg=ssm_cfg,
            attn_layer_idx=attn_layer_idx,
            attn_cfg=attn_cfg,
            rms_norm=rms_norm,
            residual_in_fp32=residual_in_fp32,
            fused_add_norm=fused_add_norm,
            pad_vocab_size_multiple=pad_vocab_size_multiple,
            tie_embeddings=tie_embeddings,
            tie_word_embeddings=tie_word_embeddings,
            devices=device_refs,
            float8_config=float8_config,
            use_subgraphs=pipeline_config.model_config.use_subgraphs,
            data_parallel_degree=pipeline_config.model_config.data_parallel_degree,
            return_logits=return_logits,
            return_hidden_states=return_hidden_states,
        )

