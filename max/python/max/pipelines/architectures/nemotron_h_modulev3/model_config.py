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
"""Config for Nemotron-H hybrid Mamba-2, attention and MoE models."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import (
    MultiKVCacheParams,
    RecurrentStateParams,
    RecurrentStateRegion,
)
from max.nn.transformer import ReturnLogits
from max.pipelines.kv_cache import cache_dtype_for_encoding
from max.pipelines.lib import KVCacheConfig, MAXModelConfig, PipelineConfig
from max.pipelines.lib.config.model_config import _select_quantization_encoding
from max.pipelines.lib.interfaces import (
    ArchConfigWithKVCache,
    ArchConfigWithStoredKVParams,
)
from max.pipelines.modeling.config_enums import SupportedEncoding
from max.pipelines.weights import resolve_hf_quant_config
from transformers import AutoConfig
from typing_extensions import Self

from .quantization import NemotronHQuantScheme, parse_quant_scheme

logger = logging.getLogger("max.pipelines")

ATTN_CACHE_KEY = "attn"
STATE_CACHE_KEY = "state"

# transformers main renamed the block types; both spellings name the same
# mixers.
_BLOCK_KINDS: dict[str, str] = {
    "mamba": "mamba",
    "linear_attention": "mamba",
    "attention": "attention",
    "full_attention": "attention",
    "moe": "moe",
    "mlp": "mlp",
}

# The only values of these fields this implementation builds.
_REQUIRED: dict[str, object] = {
    "attention_bias": False,
    "mlp_bias": False,
    "mamba_proj_bias": False,
    "use_conv_bias": True,
    "tie_word_embeddings": False,
    "residual_in_fp32": False,
    "mlp_hidden_act": "relu2",
    "mamba_hidden_act": "silu",
    "mamba_ssm_cache_dtype": "float32",
    "n_group": 1,
    "topk_group": 1,
    # A latent MoE is Nemotron-3 Super's, a different model.
    "moe_latent_size": None,
}


def parse_layer_kinds(block_types: Sequence[str]) -> list[str]:
    """Maps a ``layers_block_type`` list to per-layer mixer kinds.

    Returns ``"mamba"``, ``"attention"``, ``"moe"`` or ``"mlp"`` per layer.

    Raises:
        ValueError: If a block type is not one of the known spellings.
    """
    kinds = []
    for block_type in block_types:
        kind = _BLOCK_KINDS.get(block_type)
        if kind is None:
            raise ValueError(
                f"unknown Nemotron-H block type {block_type!r}; expected one "
                f"of {sorted(_BLOCK_KINDS)}"
            )
        kinds.append(kind)
    return kinds


@dataclass(kw_only=True)
class NemotronHConfig(ArchConfigWithStoredKVParams, ArchConfigWithKVCache):
    """Configuration for a Nemotron-H hybrid decoder.

    Every layer is a pre-norm residual block around one mixer: Mamba-2,
    NoPE grouped-query attention, a relu2 MoE or a relu2 MLP.
    """

    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "bfloat16"
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {
        "bfloat16",
        "float4_e2m1fnx2",
    }

    hidden_size: int
    vocab_size: int
    layer_kinds: list[str]
    layer_norm_epsilon: float

    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int

    intermediate_size: int

    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    moe_shared_expert_intermediate_size: int
    routed_scaling_factor: float
    norm_topk_prob: bool

    mamba_num_heads: int
    mamba_head_dim: int
    n_groups: int
    ssm_state_size: int
    conv_kernel: int

    quant_scheme: NemotronHQuantScheme

    dtype: DType
    devices: list[DeviceRef]
    max_seq_len: int
    kv_params: MultiKVCacheParams
    return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN

    @property
    def mamba_intermediate_size(self) -> int:
        return self.mamba_num_heads * self.mamba_head_dim

    @property
    def conv_dim(self) -> int:
        return (
            self.mamba_intermediate_size
            + 2 * self.n_groups * self.ssm_state_size
        )

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
        """Returns the attention leaf beside the Mamba state.

        The attention layers index the KV cache 0, 1, 2, ... in layer order;
        the Mamba layers do the same in the state leaves.
        """
        data_parallel_degree = pipeline_config.model.data_parallel_degree
        if data_parallel_degree > 1:
            raise ValueError("Nemotron-H does not support data parallelism")
        kinds = parse_layer_kinds(huggingface_config.layers_block_type)
        attn = kv_cache_config.to_params(
            allow_kv_head_replication=allow_kv_head_replication,
            dtype=cache_dtype,
            n_kv_heads=huggingface_config.num_key_value_heads,
            head_dim=huggingface_config.head_dim,
            num_layers=kinds.count("attention"),
            devices=devices,
            data_parallel_degree=data_parallel_degree,
        )
        hf = huggingface_config
        num_mamba_layers = kinds.count("mamba")
        state = RecurrentStateParams(
            # Separate leaves because the conv and SSM kernels each index
            # their own uniformly strided pool, at different dtypes.
            # NemotronHBackbone unpacks them in this order.
            regions=(
                RecurrentStateRegion(
                    leaf_id="mamba/conv",
                    num_layers=num_mamba_layers,
                    row_shape=(
                        hf.mamba_num_heads * hf.mamba_head_dim
                        + 2 * hf.n_groups * hf.ssm_state_size,
                        hf.conv_kernel - 1,
                    ),
                    dtype=DType.bfloat16,
                ),
                RecurrentStateRegion(
                    leaf_id="mamba/ssm",
                    num_layers=num_mamba_layers,
                    row_shape=(
                        hf.mamba_num_heads,
                        hf.mamba_head_dim,
                        hf.ssm_state_size,
                    ),
                    dtype=DType.float32,
                ),
            ),
            devices=attn.devices,
            data_parallel_degree=attn.data_parallel_degree,
        )
        return MultiKVCacheParams.from_params(
            {ATTN_CACHE_KEY: attn, STATE_CACHE_KEY: state}
        )

    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
        *,
        max_seq_len: int,
    ) -> Self:
        model_config = model_config or pipeline_config.model
        hf = model_config.huggingface_config
        if hf is None:
            raise ValueError(
                f"HuggingFace config is required for "
                f"'{model_config.model_path}', but it could not be loaded."
            )
        encoding = _select_quantization_encoding(
            model_config, cls.DEFAULT_ENCODING
        )
        kv_cache_format = model_config.kv_cache.kv_cache_format
        devices = [
            DeviceRef(spec.device_type, spec.id)
            for spec in model_config.device_specs
        ]
        kv_params = cls.construct_kv_params(
            huggingface_config=hf,
            pipeline_config=pipeline_config,
            devices=devices,
            kv_cache_config=model_config.kv_cache,
            cache_dtype=cache_dtype_for_encoding(encoding, kv_cache_format),
        )
        config = cls.from_huggingface(
            hf, kv_params=kv_params, devices=devices, max_seq_len=max_seq_len
        )
        hf_quant_config = resolve_hf_quant_config(hf, {}) or {}
        if hf_quant_config.get("kv_cache_scheme") and kv_cache_format is None:
            logger.info(
                "Nemotron-H: the checkpoint declares KV-cache scales, which "
                "are not applied; the KV cache stays BF16. Pass "
                "--kv-cache-format float8_e4m3fn for an unscaled FP8 cache."
            )
        return config

    @classmethod
    def from_huggingface(
        cls,
        hf: AutoConfig,
        *,
        kv_params: MultiKVCacheParams,
        devices: list[DeviceRef],
        max_seq_len: int,
    ) -> Self:
        """Reads the architecture out of a Hugging Face config.

        Raises:
            NotImplementedError: If the config asks for a variant this
                implementation does not build.
        """
        for name, expected in _REQUIRED.items():
            value = getattr(hf, name, expected)
            if value != expected:
                raise NotImplementedError(
                    f"Nemotron-H supports {name}={expected!r}, but the "
                    f"checkpoint sets {value!r}"
                )
        limit = tuple(getattr(hf, "time_step_limit", ()))
        if limit and limit != (0.0, math.inf):
            raise NotImplementedError(
                f"Nemotron-H does not clamp dt, but the checkpoint sets "
                f"time_step_limit={limit}"
            )
        return cls(
            hidden_size=hf.hidden_size,
            vocab_size=hf.vocab_size,
            layer_kinds=parse_layer_kinds(hf.layers_block_type),
            layer_norm_epsilon=hf.layer_norm_epsilon,
            num_attention_heads=hf.num_attention_heads,
            num_key_value_heads=hf.num_key_value_heads,
            head_dim=hf.head_dim,
            intermediate_size=hf.intermediate_size,
            num_experts=hf.n_routed_experts,
            num_experts_per_tok=hf.num_experts_per_tok,
            moe_intermediate_size=hf.moe_intermediate_size,
            moe_shared_expert_intermediate_size=(
                hf.moe_shared_expert_intermediate_size
            ),
            routed_scaling_factor=hf.routed_scaling_factor,
            norm_topk_prob=hf.norm_topk_prob,
            mamba_num_heads=hf.mamba_num_heads,
            mamba_head_dim=hf.mamba_head_dim,
            n_groups=hf.n_groups,
            ssm_state_size=hf.ssm_state_size,
            conv_kernel=hf.conv_kernel,
            quant_scheme=parse_quant_scheme(resolve_hf_quant_config(hf, {})),
            # Quantization is per module: activations and every unquantized
            # weight stay BF16 whatever the checkpoint's encoding.
            dtype=DType.bfloat16,
            devices=devices,
            max_seq_len=max_seq_len,
            kv_params=kv_params,
        )
