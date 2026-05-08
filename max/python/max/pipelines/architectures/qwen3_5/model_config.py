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
"""Config for Qwen3.5 models (hybrid linear/full attention)."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from max.driver import Device, load_devices
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import KVCacheParams
from max.pipelines.lib import KVCacheConfig, MAXModelConfig, PipelineConfig
from transformers.models.auto.configuration_auto import AutoConfig
from typing_extensions import Self, override

from ..llama3.model_config import Llama3Config
from ..qwen3vl_moe.model_config import VisionConfig

__all__ = ["Qwen3_5Config", "VisionConfig"]

logger = logging.getLogger("max.pipelines")


@dataclass(kw_only=True)
class Qwen3_5Config(Llama3Config):
    """Configuration for Qwen3.5 hybrid attention models.

    Qwen3.5 uses a hybrid architecture with both full (standard) attention
    and linear attention (Gated DeltaNet) layers. Every full_attention_interval-th
    layer uses full attention, and the rest use linear attention.
    """

    # Hybrid attention parameters
    layer_types: list[str] = field(default_factory=list)
    """Per-layer attention type: 'full_attention' or 'linear_attention'."""

    full_attention_interval: int = 4
    """Every N-th layer uses full attention."""

    # Linear attention (Gated DeltaNet) parameters
    linear_key_head_dim: int = 128
    """Key head dimension for linear attention layers."""

    linear_value_head_dim: int = 128
    """Value head dimension for linear attention layers."""

    linear_num_key_heads: int = 16
    """Number of key heads for linear attention layers."""

    linear_num_value_heads: int = 48
    """Number of value heads for linear attention layers."""

    linear_conv_kernel_dim: int = 4
    """Causal conv1d kernel size for linear attention layers."""

    # Qwen3.5-specific full attention parameters
    partial_rotary_factor: float = 0.25
    """Fraction of head_dim that gets rotary position embedding."""

    attn_output_gate: bool = True
    """Whether full attention layers use a sigmoid output gate."""

    mamba_ssm_dtype: DType = DType.float32
    """Dtype for SSM (state space model) computations in linear attention layers."""

    # Vision encoder (optional - text-only models leave these None)
    vision_config: VisionConfig | None = None
    """Vision encoder configuration; None for text-only models."""

    image_token_id: int | None = None
    """Token ID used for image placeholders in the input sequence."""

    video_token_id: int | None = None
    """Token ID used for video placeholders in the input sequence."""

    vision_start_token_id: int | None = None
    """Token ID that marks the start of vision content."""

    mrope_section: list[int] | None = None
    """MRoPE section lengths for multimodal rotary position encoding."""

    @staticmethod
    def _get_text_config(huggingface_config: AutoConfig) -> AutoConfig:
        """Extract text config, handling both multimodal and text-only models."""
        return getattr(huggingface_config, "text_config", huggingface_config)

    @staticmethod
    def _get_layer_types(text_config: AutoConfig) -> list[str]:
        """Return the per-layer attention type list for the model.

        Uses `layer_types` from the config when present; otherwise generates
        it from `full_attention_interval`.
        """
        layer_types = getattr(text_config, "layer_types", [])
        if layer_types:
            return list(layer_types)
        full_interval = getattr(text_config, "full_attention_interval", 4)
        num_layers = text_config.num_hidden_layers
        return [
            "full_attention"
            if (i + 1) % full_interval == 0
            else "linear_attention"
            for i in range(num_layers)
        ]

    @staticmethod
    def construct_kv_params(
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        """Construct KV cache parameters for full attention layers only.

        Only allocates KV cache entries for full-attention layers; linear
        attention layers use separate conv/recurrent state buffers instead.
        The forward pass maps each full-attention layer to a sequential KV
        cache index (0, 1, 2, ...) independent of the absolute layer index.
        """
        text_config = Qwen3_5Config._get_text_config(huggingface_config)
        data_parallel_degree = pipeline_config.model.data_parallel_degree
        if data_parallel_degree > 1:
            raise ValueError(
                "Data parallelism is not supported for Qwen3.5 models"
            )
        layer_types = Qwen3_5Config._get_layer_types(text_config)
        num_full_attention_layers = sum(
            1 for lt in layer_types if lt == "full_attention"
        )
        return kv_cache_config.to_params(
            dtype=cache_dtype,
            n_kv_heads=text_config.num_key_value_heads,
            head_dim=text_config.head_dim,
            num_layers=num_full_attention_layers,
            devices=devices,
            data_parallel_degree=data_parallel_degree,
        )

    @staticmethod
    def calculate_attention_multiplier(
        huggingface_config: AutoConfig,
    ) -> float:
        """Compute attention scaling factor using explicit head_dim."""
        text_config = Qwen3_5Config._get_text_config(huggingface_config)
        return getattr(
            text_config,
            "attention_multiplier",
            math.sqrt(1.0 / float(text_config.head_dim)),
        )

    @staticmethod
    def get_num_layers(huggingface_config: AutoConfig) -> int:
        text_config = Qwen3_5Config._get_text_config(huggingface_config)
        return text_config.num_hidden_layers

    def _per_request_state_bytes(self) -> int:
        """Return GPU bytes for one request's linear-attention state (all linear layers).

        Each linear-attention layer stores two state arrays per active request:
        - Conv state:       `(1, conv_dim, kernel-1)` model dtype
        - Recurrent state:  `(1, nv, kd, vd)`         model dtype

        States are stored in the model's native dtype (typically bfloat16).
        Computation is promoted to float32 inside GatedDeltaNet.__call__().
        These buffers are NOT included in the KV-cache budget.
        """
        num_linear = sum(
            1 for lt in self.layer_types if lt == "linear_attention"
        )
        if num_linear == 0:
            return 0
        conv_dim = (
            2 * self.linear_key_head_dim * self.linear_num_key_heads
            + self.linear_value_head_dim * self.linear_num_value_heads
        )
        # States are stored in the model's native dtype (typically bfloat16).
        dtype_bytes = self.dtype.size_in_bytes
        bytes_per_layer = (
            # conv state: (conv_dim * (kernel-1)) elements
            conv_dim * (self.linear_conv_kernel_dim - 1) * dtype_bytes
            # recurrent state: (nv * kd * vd) elements
            + self.linear_num_value_heads
            * self.linear_key_head_dim
            * self.linear_value_head_dim
            * dtype_bytes
        )
        return num_linear * bytes_per_layer

    def infer_optimal_batch_size(
        self,
        devices: list[Device],
        *,
        weights_size: int,
        device_memory_utilization: float,
    ) -> int:
        """Return a memory-safe default `max_batch_size` for this architecture.

        Qwen3.5 allocates GPU memory for GatedDeltaNet recurrent states with
        three distinct cost centres per active request:

        1. **Persistent pool** (`max_batch x per_req`): pre-allocated once at
           startup and lives for the full server lifetime.
        2. **Input working buffers** (`batch x per_req`): gathered from the
           pool into dense batch tensors by `get_states()` each step.
        3. **Output working buffers** (`batch x per_req`): produced by the
           model kernel and scattered back to the pool by `update_states()`.

        Worst-case simultaneous footprint is therefore **3 x max_batch x per_req**
        (pool + both working copies). We split the post-weights utilization
        budget evenly: the state pool gets up to half, the KV cache absorbs
        the rest. This uses the same ``device_memory_utilization`` headroom
        factor as the rest of the pipeline.

        This is consistent with `estimate_activation_memory()` which reserves
        `3 x max_batch x per_req` bytes before the KV-cache allocator runs.

        Falls back to 32—safe for the 27B model on H100/A100 (80 GB)—when
        the device query fails.
        """
        per_req = self._per_request_state_bytes()
        try:
            free_bytes = int(
                sum(d.stats.get("free_memory", 0) for d in devices)
            )
        except Exception:
            free_bytes = 0
        if free_bytes <= 0:
            # Conservative fallback: safe for Qwen3.5-27B on H100/A100 (80 GB).
            return 32
        budget = int(free_bytes * device_memory_utilization) - weights_size
        if budget <= 0:
            return 1
        # Divide by 3*per_req: pool + input working + output working.
        max_batch = max(1, (budget // 2) // (3 * per_req))
        return min(512, max_batch)

    @override
    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
    ) -> Self:
        model_config = model_config or pipeline_config.model
        huggingface_config = model_config.huggingface_config
        if huggingface_config is None:
            raise ValueError(
                f"HuggingFace config is required for "
                f"'{model_config.model_path}', "
                "but config could not be loaded."
            )
        return cls.initialize_from_config(
            pipeline_config, huggingface_config, model_config
        )

    @override
    @classmethod
    def initialize_from_config(
        cls,
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
        model_config: MAXModelConfig | None = None,
    ) -> Self:
        """Initialize config from pipeline and HuggingFace configurations.

        Handles both multimodal (Qwen3_5ForConditionalGeneration) and
        text-only (Qwen3_5ForCausalLM) configs by extracting the text config.
        """
        model_config = model_config or pipeline_config.model
        text_config = Qwen3_5Config._get_text_config(huggingface_config)

        # Get base Llama3Config from the text config
        base_config = Llama3Config.initialize_from_config(
            pipeline_config, text_config
        )

        kv_cache_config = model_config.kv_cache
        # The MHA kernel selects tile_size == head_dim. The KV cache
        # page_size must be >= tile_size. Qwen3.5 has head_dim=256.
        if text_config.head_dim > 128:
            kv_cache_config.kv_cache_page_size = max(
                kv_cache_config.kv_cache_page_size, text_config.head_dim
            )

        quantization_encoding = model_config.quantization_encoding
        if quantization_encoding is None:
            raise ValueError("quantization_encoding must not be None")
        cache_dtype = model_config.kv_cache.cache_dtype
        n_devices = len(model_config.device_specs)
        device_refs = [
            DeviceRef(spec.device_type, spec.id)
            for spec in model_config.device_specs[:n_devices]
        ]

        # Override KV params and attention multiplier
        kv_params = Qwen3_5Config.construct_kv_params(
            huggingface_config=huggingface_config,
            pipeline_config=pipeline_config,
            devices=device_refs,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
        )
        attention_multiplier = Qwen3_5Config.calculate_attention_multiplier(
            huggingface_config=huggingface_config,
        )

        # Extract rope_theta and partial_rotary_factor.
        # Priority: top-level text_config.rope_theta > rope_parameters.rope_theta
        # > base config default.  Qwen3.5 stores these inside rope_parameters;
        # some fine-tuned variants may promote rope_theta to the top level.
        rope_theta = base_config.rope_theta
        partial_rotary_factor = 0.25
        rope_params = getattr(text_config, "rope_parameters", None)
        if rope_params is not None:
            if isinstance(rope_params, dict):
                rope_theta = rope_params.get("rope_theta", rope_theta)
                partial_rotary_factor = rope_params.get(
                    "partial_rotary_factor", partial_rotary_factor
                )
            else:
                rope_theta = getattr(rope_params, "rope_theta", rope_theta)
                partial_rotary_factor = getattr(
                    rope_params, "partial_rotary_factor", partial_rotary_factor
                )

        # Top-level text_config.rope_theta takes explicit priority when present.
        if hasattr(text_config, "rope_theta"):
            rope_theta = text_config.rope_theta

        # Hybrid attention parameters
        layer_types = Qwen3_5Config._get_layer_types(text_config)

        # Linear attention parameters
        linear_key_head_dim = getattr(text_config, "linear_key_head_dim", 128)
        linear_value_head_dim = getattr(
            text_config, "linear_value_head_dim", 128
        )
        linear_num_key_heads = getattr(text_config, "linear_num_key_heads", 16)
        linear_num_value_heads = getattr(
            text_config, "linear_num_value_heads", 48
        )
        linear_conv_kernel_dim = getattr(
            text_config, "linear_conv_kernel_dim", 4
        )
        attn_output_gate = getattr(text_config, "attn_output_gate", True)

        _mamba_dtype_map: dict[str, DType] = {
            "float32": DType.float32,
            "bfloat16": DType.bfloat16,
            "float16": DType.float16,
        }
        mamba_ssm_dtype_str = getattr(text_config, "mamba_ssm_dtype", "float32")
        mamba_ssm_dtype = _mamba_dtype_map.get(
            mamba_ssm_dtype_str, DType.float32
        )

        # Handle tie_word_embeddings from top-level config
        tie_word_embeddings = getattr(
            huggingface_config, "tie_word_embeddings", False
        )

        # Vision encoder (only present in multimodal checkpoints)
        hf_vision_config = getattr(huggingface_config, "vision_config", None)
        vision_cfg: VisionConfig | None = None
        if hf_vision_config is not None and hasattr(
            hf_vision_config, "patch_size"
        ):
            vision_cfg = VisionConfig.initialize_from_config(
                pipeline_config, hf_vision_config
            )

        # Multimodal token IDs and MRoPE section
        image_token_id = getattr(huggingface_config, "image_token_id", None)
        video_token_id = getattr(huggingface_config, "video_token_id", None)
        vision_start_token_id = getattr(
            huggingface_config, "vision_start_token_id", None
        )
        mrope_section: list[int] | None = None
        rope_params = getattr(text_config, "rope_parameters", None)
        if rope_params is not None:
            raw_section = (
                rope_params.get("mrope_section")
                if isinstance(rope_params, dict)
                else getattr(rope_params, "mrope_section", None)
            )
            if raw_section is not None:
                mrope_section = list(raw_section)

        config_instance = cls(
            hidden_size=base_config.hidden_size,
            num_attention_heads=base_config.num_attention_heads,
            num_key_value_heads=base_config.num_key_value_heads,
            num_hidden_layers=base_config.num_hidden_layers,
            rope_theta=rope_theta,
            rope_scaling_params=base_config.rope_scaling_params,
            rms_norm_eps=base_config.rms_norm_eps,
            intermediate_size=base_config.intermediate_size,
            # Partial RoPE requires interleaved pattern in the kernel
            interleaved_rope_weights=True,
            vocab_size=base_config.vocab_size,
            dtype=base_config.dtype,
            model_quantization_encoding=base_config.model_quantization_encoding,
            quantization_config=base_config.quantization_config,
            max_seq_len=base_config.max_seq_len,
            kv_params=kv_params,
            attention_multiplier=attention_multiplier,
            embedding_multiplier=base_config.embedding_multiplier,
            residual_multiplier=base_config.residual_multiplier,
            devices=base_config.devices,
            clip_qkv=base_config.clip_qkv,
            use_subgraphs=base_config.use_subgraphs,
            tie_word_embeddings=tie_word_embeddings,
            # Hybrid attention parameters
            layer_types=layer_types,
            full_attention_interval=getattr(
                text_config, "full_attention_interval", 4
            ),
            linear_key_head_dim=linear_key_head_dim,
            linear_value_head_dim=linear_value_head_dim,
            linear_num_key_heads=linear_num_key_heads,
            linear_num_value_heads=linear_num_value_heads,
            linear_conv_kernel_dim=linear_conv_kernel_dim,
            partial_rotary_factor=partial_rotary_factor,
            attn_output_gate=attn_output_gate,
            mamba_ssm_dtype=mamba_ssm_dtype,
            # Vision (optional)
            vision_config=vision_cfg,
            image_token_id=image_token_id,
            video_token_id=video_token_id,
            vision_start_token_id=vision_start_token_id,
            mrope_section=mrope_section,
        )

        # Set a safe default max_batch_size if not explicitly set by user.
        # Qwen3.5 has per-request GPU overhead (recurrent-state buffers)
        # beyond the KV cache, so the framework's default 512 can OOM us;
        # compute a bound that fits actual free memory.
        if pipeline_config.runtime.max_batch_size is None:
            try:
                actual_devices = load_devices(model_config.device_specs)
            except Exception:
                actual_devices = []
            try:
                weights_bytes = model_config.weights_size()
            except (FileNotFoundError, ValueError, RuntimeError) as e:
                logger.warning(
                    "Qwen3.5: weights_size() failed (%s); assuming 0 bytes "
                    "for max_batch_size inference. The state-pool budget "
                    "may be over-allocated and risk OOM.",
                    e,
                )
                weights_bytes = 0
            pipeline_config.runtime.max_batch_size = (
                config_instance.infer_optimal_batch_size(
                    actual_devices,
                    weights_size=weights_bytes,
                    device_memory_utilization=(
                        model_config.kv_cache.device_memory_utilization
                    ),
                )
            )

        return config_instance
