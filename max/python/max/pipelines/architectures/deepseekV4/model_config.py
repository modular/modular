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
"""Config for DeepSeek-V4-Flash models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import (
    KVCacheParamInterface,
    KVCacheParams,
    KVCacheQuantizationConfig,
    MultiKVCacheParams,
    spec_decode_cache_slack,
)
from max.nn.quant_config import QuantConfig
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.kv_cache import cache_dtype_for_encoding
from max.pipelines.lib import KVCacheConfig, MAXModelConfig, PipelineConfig
from max.pipelines.lib.config.model_config import _select_quantization_encoding
from max.pipelines.lib.interfaces.arch_config import ArchConfigWithKVCache
from max.pipelines.lib.pipeline_variants.utils import get_rope_theta
from max.pipelines.lib.utils import upper_bounded_default
from max.pipelines.modeling.config_enums import (
    SupportedEncoding,
    supported_encoding_dtype,
)
from max.pipelines.speculative.config import SpeculativeMethod
from transformers import AutoConfig
from typing_extensions import Self, override


@dataclass(kw_only=True)
class DeepseekV4Config(ArchConfigWithKVCache):
    """Configuration for DeepSeek-V4-Flash models.

    V4 is *not* a V3.2 variant despite the shared lineage. It replaces V3.2's
    MLA (``kv_lora_rank`` + ``qk_nope_head_dim`` + ``v_head_dim`` + ``kv_b_proj``)
    with a single ``head_dim``-wide shared latent produced by ``wkv``: query and
    key live in the same 512-dim space, the last ``rope_head_dim`` dims carry
    RoPE, and there is no per-head KV up-projection at all. It also adds
    sliding-window + compressed-token sparse attention (CSA), a grouped low-rank
    output projection, hyper-connection residuals (mHC), and hash routing on the
    first ``num_hash_layers`` MoE layers. Hence a standalone config rather than a
    subclass of :class:`DeepseekV3_2Config`.
    """

    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "float8_e4m3fn"
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {"float8_e4m3fn"}

    # MAX specific fields.
    dtype: DType
    kv_params: KVCacheParamInterface
    devices: list[DeviceRef]
    use_subgraphs: bool = True
    data_parallel_degree: int = 1
    quantization_encoding: SupportedEncoding | None = None

    max_seq_len: int

    # Core dimensions. Reference ``ModelArgs`` names in parentheses.
    vocab_size: int = 129280
    hidden_size: int = 4096  # dim
    num_hidden_layers: int = 43  # n_layers
    num_attention_heads: int = 64  # n_heads
    num_key_value_heads: int = 1
    rms_norm_eps: float = 1e-6  # norm_eps
    hidden_act: str = "silu"
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    attention_dropout: float = 0.0

    # Attention. ``head_dim`` is the full shared latent width; the trailing
    # ``qk_rope_head_dim`` dims of it carry RoPE, the leading
    # ``head_dim - qk_rope_head_dim`` dims do not.
    head_dim: int = 512
    qk_rope_head_dim: int = 64  # rope_head_dim
    q_lora_rank: int = 1024
    o_lora_rank: int = 1024
    o_groups: int = 8
    sliding_window: int = 128  # window_size

    # Compressed sparse attention (CSA). One entry per layer, including the
    # trailing DSpark/MTP layers. 0 = sliding window only, 4 = compressor +
    # lightning indexer, 128 = compressor only (strided selection).
    compress_ratios: list[int] = field(default_factory=list)
    compress_rope_theta: float = 160000.0

    # Lightning indexer, only instantiated on ``compress_ratio == 4`` layers.
    index_head_dim: int = 128
    index_n_heads: int = 64
    index_topk: int = 512

    # MoE. Every layer is MoE: V4 has no dense prefix, so there is no
    # ``intermediate_size`` / ``first_k_dense_replace`` here.
    moe_intermediate_size: int = 2048  # moe_inter_dim
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 6  # n_activated_experts
    routed_scaling_factor: float = 1.5  # route_scale
    scoring_func: str = "sqrtsoftplus"
    norm_topk_prob: bool = True
    swiglu_limit: float = 10.0
    expert_dtype: str | None = "fp4"

    # Hash routing: layers ``[0, num_hash_layers)`` read expert indices from a
    # ``tid2eid`` table keyed on token id instead of scoring the gate.
    num_hash_layers: int = 3

    # Hyper-connections (mHC).
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6

    # DSpark speculative head.
    dspark_block_size: int = 5
    dspark_noise_token_id: int = 128799
    dspark_target_layer_ids: list[int] = field(default_factory=list)
    dspark_markov_rank: int = 256

    # RoPE.
    max_position_embeddings: int = 1048576
    rope_theta: float = 10000.0
    rope_scaling: dict[str, Any] | None = None
    rope_interleave: bool = True

    # Populated from the state dict after ``initialize``.
    norm_dtype: DType = DType.bfloat16
    gate_dtype: DType | None = None
    correction_bias_dtype: DType | None = None
    max_batch_context_length: int = 131072
    quant_config: QuantConfig | None = None

    graph_mode: str = "auto"  # "auto" | "prefill" | "decode"
    return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN
    return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE

    def __post_init__(self) -> None:
        if self.hidden_act != "silu":
            raise ValueError(
                "'silu' is the only hidden_act currently supported"
            )
        if self.tie_word_embeddings:
            raise ValueError("tie_word_embeddings is not supported yet")
        if self.scoring_func != "sqrtsoftplus":
            raise ValueError(
                "'sqrtsoftplus' is the only scoring_func DeepSeek-V4 ships, got "
                f"'{self.scoring_func}'"
            )
        if self.n_shared_experts != 1:
            raise ValueError(
                "DeepSeek-V4 assumes exactly one shared expert, got "
                f"{self.n_shared_experts}"
            )
        if len(self.compress_ratios) < self.num_hidden_layers:
            raise ValueError(
                f"compress_ratios has {len(self.compress_ratios)} entries but "
                f"the model has {self.num_hidden_layers} layers"
            )
        for ratio in self.compress_ratios:
            if ratio not in (0, 4, 128):
                raise ValueError(
                    f"unsupported compress_ratio {ratio}; DeepSeek-V4 ships "
                    "only 0 (window-only), 4 (compressor+indexer) and 128 "
                    "(compressor-only)"
                )
        num_devices = len(self.devices)
        if self.data_parallel_degree not in (1, num_devices):
            raise ValueError(
                f"data_parallel_degree for DeepSeek-V4 ({self.data_parallel_degree}) "
                f"must be 1 (TP attention) or equal to the device count "
                f"({num_devices})"
            )

    def get_kv_params(self) -> KVCacheParamInterface:
        return self.kv_params

    def get_max_seq_len(self) -> int:
        return self.max_seq_len

    @property
    def qk_nope_head_dim(self) -> int:
        """Width of the non-RoPE portion of the shared latent."""
        return self.head_dim - self.qk_rope_head_dim

    def layer_compress_ratio(self, layer_idx: int) -> int:
        return self.compress_ratios[layer_idx]

    def layer_has_indexer(self, layer_idx: int) -> bool:
        return self.compress_ratios[layer_idx] == 4

    def layer_is_hash_routed(self, layer_idx: int) -> bool:
        return layer_idx < self.num_hash_layers

    @classmethod
    def calculate_max_seq_len(
        cls,
        huggingface_config: AutoConfig,
        model_config: MAXModelConfig,
    ) -> int:
        return upper_bounded_default(
            upper_bound=huggingface_config.max_position_embeddings,
            default=model_config.max_length,
        )

    @staticmethod
    def get_num_layers(huggingface_config: AutoConfig) -> int:
        return huggingface_config.num_hidden_layers

    @staticmethod
    def construct_kv_params(
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParamInterface:
        """Build the attention + indexer caches.

        The attention cache stores one ``head_dim``-wide latent per token; unlike
        V3.2 there is no separate ``qk_rope_head_dim`` tail appended to it, the
        RoPE dims live inside ``head_dim``.

        NOTE: the reference implementation sizes each layer's cache as
        ``window_size + max_seq_len // compress_ratio``, which differs per layer
        (window-only layers need just ``window_size``). MAX's ``KVCacheParams``
        is uniform across layers, so this over-allocates the window-only layers.
        See OPEN QUESTIONS in the bringup progress log.
        """
        data_parallel_degree = pipeline_config.model.data_parallel_degree

        kvcache_quant_config = None
        if cache_dtype in (DType.float8_e4m3fn, DType.float8_e4m3fnuz):
            kvcache_quant_config = KVCacheQuantizationConfig(
                scale_dtype=DType.float32, quantization_granularity=32
            )

        speculative_method: SpeculativeMethod | None = None
        num_draft_tokens: int = 0
        if pipeline_config.speculative:
            speculative_method = pipeline_config.speculative.speculative_method
            num_draft_tokens = (
                pipeline_config.speculative.num_speculative_tokens or 0
            )

        num_layers = DeepseekV4Config.get_num_layers(huggingface_config)

        attn_kv_params = kv_cache_config.to_params(
            dtype=cache_dtype,
            # A single shared latent per token, exactly like MLA's absorbed form.
            n_kv_heads=1,
            head_dim=huggingface_config.head_dim,
            num_layers=num_layers,
            devices=devices,
            data_parallel_degree=data_parallel_degree,
            is_mla=True,
            num_q_heads=huggingface_config.num_attention_heads,
            kvcache_quant_config=kvcache_quant_config,
            speculative_method=speculative_method,
            num_draft_tokens=num_draft_tokens,
        )
        assert isinstance(attn_kv_params, KVCacheParams)

        indexer_kv_params = kv_cache_config.to_params(
            # The indexer always keeps its K cache in float8_e4m3fn.
            dtype=DType.float8_e4m3fn,
            n_kv_heads=1,
            head_dim=huggingface_config.index_head_dim,
            num_layers=num_layers,
            devices=devices,
            data_parallel_degree=data_parallel_degree,
            is_mla=True,
            num_q_heads=huggingface_config.num_attention_heads,
            kvcache_quant_config=KVCacheQuantizationConfig(
                scale_dtype=DType.float32, quantization_granularity=32
            ),
            speculative_method=speculative_method,
            num_draft_tokens=num_draft_tokens,
        )
        assert isinstance(indexer_kv_params, KVCacheParams)

        return MultiKVCacheParams.from_params(
            {"mla": attn_kv_params, "indexer": indexer_kv_params}
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
        config = model_config.huggingface_config
        if config is None:
            raise ValueError(
                f"HuggingFace config is required for '{model_config.model_path}', "
                "but config could not be loaded. "
                "Please ensure the model repository contains a valid config.json file."
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

        kv_params = cls.construct_kv_params(
            huggingface_config=config,
            pipeline_config=pipeline_config,
            devices=device_refs,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
        )

        return cls(
            dtype=dtype,
            kv_params=kv_params,
            devices=device_refs,
            use_subgraphs=model_config.use_subgraphs,
            data_parallel_degree=model_config.data_parallel_degree,
            quantization_encoding=quantization_encoding,
            max_seq_len=max_seq_len,
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            rms_norm_eps=config.rms_norm_eps,
            hidden_act=config.hidden_act,
            tie_word_embeddings=config.tie_word_embeddings,
            attention_bias=config.attention_bias,
            attention_dropout=config.attention_dropout,
            head_dim=config.head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            q_lora_rank=config.q_lora_rank,
            o_lora_rank=config.o_lora_rank,
            o_groups=config.o_groups,
            sliding_window=config.sliding_window,
            compress_ratios=list(config.compress_ratios),
            compress_rope_theta=config.compress_rope_theta,
            index_head_dim=config.index_head_dim,
            index_n_heads=config.index_n_heads,
            index_topk=config.index_topk,
            moe_intermediate_size=config.moe_intermediate_size,
            n_routed_experts=config.n_routed_experts,
            n_shared_experts=config.n_shared_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            routed_scaling_factor=config.routed_scaling_factor,
            scoring_func=config.scoring_func,
            norm_topk_prob=config.norm_topk_prob,
            swiglu_limit=config.swiglu_limit,
            expert_dtype=getattr(config, "expert_dtype", None),
            num_hash_layers=config.num_hash_layers,
            hc_mult=config.hc_mult,
            hc_sinkhorn_iters=config.hc_sinkhorn_iters,
            hc_eps=config.hc_eps,
            dspark_block_size=config.dspark_block_size,
            dspark_noise_token_id=config.dspark_noise_token_id,
            dspark_target_layer_ids=list(config.dspark_target_layer_ids),
            dspark_markov_rank=config.dspark_markov_rank,
            max_position_embeddings=config.max_position_embeddings
            + spec_decode_cache_slack(kv_params),
            rope_theta=get_rope_theta(config),
            rope_scaling=config.rope_scaling,
            rope_interleave=getattr(config, "rope_interleave", True),
        )
