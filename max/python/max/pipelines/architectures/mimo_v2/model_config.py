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
"""Config for MiMo-V2.6-Flash (``MiMoV2ForCausalLM``).

The model runs FP8 W8A8 dense linears and W4A8 experts, with Q/K (192) and V
(128) zero-padded to a 256-wide KV cache for the fast attention kernels.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

from max.driver import DeviceSpec, load_devices
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import KVCacheParamInterface, MultiKVCacheParams
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.architectures.gpt_oss.hybrid_kv_params_util import (
    hybrid_swa_full_kv_params,
)
from max.pipelines.kv_cache import cache_dtype_for_encoding
from max.pipelines.lib import KVCacheConfig, MAXModelConfig, PipelineConfig
from max.pipelines.lib.config.model_config import (
    _select_quantization_encoding,
)
from max.pipelines.lib.interfaces.arch_config import (
    ArchConfigWithKVCache,
    ArchConfigWithStoredKVParams,
)
from max.pipelines.modeling.config_enums import SupportedEncoding
from transformers import AutoConfig
from typing_extensions import Self, override

from .quant import MiMoV2QuantScheme, moe_layers, parse_quant_scheme
from .weight_adapters import QkvChunkLayout, qkv_chunk_layout

SLIDING = "sliding_attention"
FULL = "full_attention"

# TODO(DISTINF-639): The padding makes the KV cache and each full-attention
# decode step's KV reads 1.6x their unpadded size (512 values per KV head per
# token instead of 320), and prefill attention does 1.6x the MMA work.
# Separate K and V head dims in the KV params, or a native 192/128 attention
# kernel, would remove it.
KV_HEAD_DIM = 256
"""The KV cache head dim, which Q/K (192) and V (128) are zero-padded to:
MAX's paged attention takes one head dim for Q, K and V, and it has no fast
kernel at 192."""


def layer_types(huggingface_config: AutoConfig) -> list[str]:
    """Returns each decoder layer's KV group, from ``hybrid_layer_pattern``."""
    pattern = huggingface_config.hybrid_layer_pattern
    return [
        SLIDING if pattern[i] == 1 else FULL
        for i in range(huggingface_config.num_hidden_layers)
    ]


def attention_head_dim(huggingface_config: AutoConfig) -> int:
    """Returns the KV cache head dim, :data:`KV_HEAD_DIM`.

    Raises:
        ValueError: If a Q/K or V head is wider than it.
    """
    widest = max(huggingface_config.head_dim, huggingface_config.v_head_dim)
    if widest > KV_HEAD_DIM:
        raise ValueError(
            f"MiMo-V2: head dim {widest} does not fit the "
            f"{KV_HEAD_DIM}-wide KV cache."
        )
    return KV_HEAD_DIM


def validate_devices(device_specs: Sequence[DeviceSpec]) -> None:
    """Raises unless every device is an NVIDIA SM100 (B200-class) GPU.

    The W4A8 experts' grouped matmul and MXFP8 activation quantize are SM100
    kernels; elsewhere the graph fails late, with an error that does not name
    the cause.

    Args:
        device_specs: The devices to run on.

    Raises:
        ValueError: If a device is not an SM100 GPU.
    """
    for device in load_devices(device_specs):
        arch = device.architecture_name if device.api == "cuda" else device.api
        if not arch.startswith("sm_10"):
            raise ValueError(
                "MiMo-V2 runs only on NVIDIA SM100 (B200-class) GPUs, whose "
                f"kernels its W4A8 experts use; {device.label}:{device.id} "
                f"is {arch!r}."
            )


def _validate(config: AutoConfig) -> None:
    """Raises on any config value this implementation does not handle."""
    expected = {
        "scoring_func": "sigmoid",
        "topk_method": "noaux_tc",
        "hidden_act": "silu",
        "attention_bias": False,
        "tie_word_embeddings": False,
        "attention_projection_layout": "fused_qkv",
    }
    for key, value in expected.items():
        if getattr(config, key, None) != value:
            raise ValueError(
                f"MiMo-V2: {key} is {getattr(config, key, None)!r}; only "
                f"{value!r} is implemented."
            )
    if (config.n_group or 1, config.topk_group or 1) != (1, 1):
        raise ValueError(
            f"MiMo-V2: n_group {config.n_group} and topk_group "
            f"{config.topk_group} are not the single-group router."
        )
    if getattr(config, "n_shared_experts", None):
        raise ValueError("MiMo-V2: shared experts are not implemented.")
    if config.routed_scaling_factor not in (None, 1.0):
        raise ValueError(
            f"MiMo-V2: routed_scaling_factor {config.routed_scaling_factor}"
            " is not implemented."
        )
    # The reference reads the RoPE type from either key and would apply
    # YaRN or any other scaling named there.
    for key in ("rope_parameters", "rope_scaling"):
        rope = getattr(config, key, None) or {}
        rope_type = rope.get("rope_type", rope.get("type", "default"))
        if rope_type != "default":
            raise ValueError(
                f"MiMo-V2: {key} has rope_type {rope_type!r}; only default "
                "RoPE is implemented."
            )
    # Sliding and full layers share every width except their KV head count.
    for key in ("num_attention_heads", "head_dim", "v_head_dim"):
        if getattr(config, f"swa_{key}") != getattr(config, key):
            raise ValueError(
                f"MiMo-V2: swa_{key} differs from {key}; only the KV head "
                "count may differ between sliding and full layers."
            )


@dataclass(kw_only=True)
class MiMoV2Config(ArchConfigWithStoredKVParams, ArchConfigWithKVCache):
    """Model configuration for MiMo-V2.6-Flash.

    ``attention_chunk_size`` in the checkpoint config is a leftover key and is
    deliberately not read: the model has no chunked local attention.
    """

    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "float4_e2m1fnx2"
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {"float4_e2m1fnx2"}

    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    rms_norm_eps: float
    layer_types: list[str]
    """Each decoder layer's KV group, ``sliding_attention`` or
    ``full_attention``."""

    num_attention_heads: int
    head_dim: int
    """Q and K head dim."""
    v_head_dim: int
    qkv_layouts: dict[str, QkvChunkLayout]
    """The fused ``qkv_proj`` chunk layout of each KV group."""
    rotary_dim: int
    """Leading Q/K dims that RoPE rotates, NeoX style."""
    rope_thetas: dict[str, float]
    sliding_window: int
    """Keys a sliding layer attends to, the query's own included."""
    attention_value_scale: float
    sinks: dict[str, bool]
    """Whether each KV group has a learned per-head attention sink."""

    intermediate_size: int
    """Dense MLP width."""
    moe_layers: set[int]
    n_routed_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    norm_topk_prob: bool

    quant: MiMoV2QuantScheme
    dtype: DType
    devices: list[DeviceRef]
    kv_params: KVCacheParamInterface
    max_seq_len: int

    return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN
    return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE
    target_layer_ids: list[int] | None = None
    """For ``ReturnHiddenStates.SELECTED_LAYERS``, the layers whose outputs
    (post-residual, before the final norm) are returned."""

    @property
    def attention_scale(self) -> float:
        """The softmax scale, from the unpadded Q/K head dim."""
        return self.head_dim**-0.5

    @override
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
        """Builds the ``{sliding_attention, full_attention}`` KV tree."""
        return hybrid_swa_full_kv_params(
            layer_types=layer_types(huggingface_config),
            sliding_window=huggingface_config.sliding_window,
            pipeline_config=pipeline_config,
            devices=devices,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
            n_kv_heads=huggingface_config.num_key_value_heads,
            head_dim=attention_head_dim(huggingface_config),
            allow_kv_head_replication=allow_kv_head_replication,
            sliding_n_kv_heads=huggingface_config.swa_num_key_value_heads,
            full_n_kv_heads=huggingface_config.num_key_value_heads,
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
        """Initializes the config from the pipeline config.

        Args:
            pipeline_config: The pipeline configuration.
            model_config: The model configuration to read. Defaults to
                ``pipeline_config.model``.
            max_seq_len: The maximum sequence length to build for.

        Returns:
            The initialized config.

        Raises:
            ValueError: If a device is not an SM100 GPU, or the checkpoint is
                not one this implementation handles.
        """
        model_config = model_config or pipeline_config.model
        validate_devices(model_config.device_specs)
        huggingface_config = model_config.huggingface_config
        if huggingface_config is None:
            raise ValueError(
                f"HuggingFace config is required for "
                f"'{model_config.model_path}', but it could not be loaded."
            )
        encoding = _select_quantization_encoding(
            model_config, cls.DEFAULT_ENCODING
        )
        cache_dtype = cache_dtype_for_encoding(
            encoding, model_config.kv_cache.kv_cache_format
        )
        devices = [
            DeviceRef(spec.device_type, spec.id)
            for spec in model_config.device_specs
        ]
        kv_params = cls.construct_kv_params(
            huggingface_config,
            pipeline_config,
            devices,
            model_config.kv_cache,
            cache_dtype,
        )
        return cls.from_huggingface_config(
            huggingface_config,
            devices=devices,
            kv_params=kv_params,
            max_seq_len=max_seq_len,
        )

    @classmethod
    def from_huggingface_config(
        cls,
        huggingface_config: AutoConfig,
        *,
        devices: list[DeviceRef],
        kv_params: KVCacheParamInterface,
        max_seq_len: int,
    ) -> Self:
        """Builds the config from the checkpoint's HuggingFace config.

        Args:
            huggingface_config: The checkpoint's top-level config.
            devices: The devices to build for.
            kv_params: The KV cache tree, from :meth:`construct_kv_params`.
            max_seq_len: The maximum sequence length to build for.

        Returns:
            The config.

        Raises:
            ValueError: If the config describes a variant this implementation
                does not handle, or a quantization other than the NVFP4
                export's.
        """
        hf = huggingface_config
        _validate(hf)
        num_devices = len(devices)
        layouts = {
            SLIDING: qkv_chunk_layout(hf, sliding=True),
            FULL: qkv_chunk_layout(hf, sliding=False),
        }
        for group, layout in layouts.items():
            # Every rank must hold whole qkv_proj chunks for the FP8 block
            # grid to stay aligned with its shard.
            if layout.chunks % num_devices:
                raise ValueError(
                    f"MiMo-V2: {num_devices} devices do not divide the "
                    f"{layout.chunks} {group} qkv_proj chunks."
                )
        if hf.moe_intermediate_size % (128 * num_devices):
            raise ValueError(
                f"MiMo-V2: expert width {hf.moe_intermediate_size} does not "
                f"split into whole 128-row scale granules over {num_devices} "
                "devices."
            )
        rotary_dim = int(hf.head_dim * hf.partial_rotary_factor)
        if rotary_dim % 2:
            raise ValueError(f"MiMo-V2: rotary dim {rotary_dim} is odd.")
        return cls(
            vocab_size=hf.vocab_size,
            hidden_size=hf.hidden_size,
            num_hidden_layers=hf.num_hidden_layers,
            rms_norm_eps=hf.layernorm_epsilon,
            layer_types=layer_types(hf),
            num_attention_heads=hf.num_attention_heads,
            head_dim=hf.head_dim,
            v_head_dim=hf.v_head_dim,
            qkv_layouts=layouts,
            rotary_dim=rotary_dim,
            rope_thetas={SLIDING: hf.swa_rope_theta, FULL: hf.rope_theta},
            sliding_window=hf.sliding_window,
            attention_value_scale=hf.attention_value_scale,
            sinks={
                SLIDING: bool(hf.add_swa_attention_sink_bias),
                FULL: bool(hf.add_full_attention_sink_bias),
            },
            intermediate_size=hf.intermediate_size,
            moe_layers=set(moe_layers(hf)),
            n_routed_experts=hf.n_routed_experts,
            num_experts_per_tok=hf.num_experts_per_tok,
            moe_intermediate_size=hf.moe_intermediate_size,
            norm_topk_prob=hf.norm_topk_prob,
            quant=parse_quant_scheme(hf),
            dtype=DType.bfloat16,
            devices=devices,
            kv_params=kv_params,
            max_seq_len=max_seq_len,
        )
