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

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import (
    KVCacheParamInterface,
    KVCacheParams,
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


@dataclass(frozen=True)
class KVLeafSpec:
    """One paged-cache leaf of the V4 KV state (see ``layers/cache.py``).

    ``kind`` names the stream: ``swa`` (the per-token latent window), ``comp``
    / ``state`` (a compressed zone and its compressor's open rows, per ratio),
    ``idx_comp`` / ``idx_state`` (the same for the ratio-4 indexer).
    """

    key: str
    kind: str
    ratio: int
    window_size: int | None
    head_dim: int
    kv_dim: int
    num_layers: int
    float32: bool

    def slots_per_page(self, page_size: int) -> int:
        """Storage slots per page: one per token, or one per ``ratio`` tokens."""
        return (
            page_size // self.ratio
            if self.kind in ("comp", "idx_comp")
            else page_size
        )


def kv_leaf_specs(
    *,
    compress_ratios: Sequence[int],
    num_hidden_layers: int,
    head_dim: int,
    index_head_dim: int,
    sliding_window: int,
    num_dspark_stages: int,
) -> list[KVLeafSpec]:
    """The leaves a V4 model needs, in the order the params tree declares them.

    A compressor's open state is the last ``coff * ratio`` raw projections, so
    it is a sliding window of that many tokens holding ``wkv`` as K and
    ``wgate`` as V -- ``coff = 2`` for the overlapping ratio-4 compressor.
    Ratios the model does not use produce no leaf.
    """
    trunk = list(compress_ratios[:num_hidden_layers])
    specs = [
        KVLeafSpec(
            key="swa",
            kind="swa",
            ratio=0,
            window_size=sliding_window,
            head_dim=head_dim,
            kv_dim=1,
            num_layers=num_hidden_layers + num_dspark_stages,
            float32=False,
        )
    ]
    for ratio in (4, 128):
        layers = trunk.count(ratio)
        if layers == 0:
            continue
        coff = 2 if ratio == 4 else 1
        specs.append(
            KVLeafSpec(
                key=f"c{ratio}a",
                kind="comp",
                ratio=ratio,
                window_size=None,
                head_dim=head_dim,
                kv_dim=1,
                num_layers=layers,
                float32=False,
            )
        )
        specs.append(
            KVLeafSpec(
                key=f"c{ratio}a_state",
                kind="state",
                ratio=ratio,
                window_size=coff * ratio,
                head_dim=coff * head_dim,
                kv_dim=2,
                num_layers=layers,
                float32=True,
            )
        )
        if ratio == 4:
            specs.append(
                KVLeafSpec(
                    key="idx_c4a",
                    kind="idx_comp",
                    ratio=ratio,
                    window_size=None,
                    head_dim=index_head_dim,
                    kv_dim=1,
                    num_layers=layers,
                    float32=False,
                )
            )
            specs.append(
                KVLeafSpec(
                    key="idx_c4a_state",
                    kind="idx_state",
                    ratio=ratio,
                    window_size=coff * ratio,
                    head_dim=coff * index_head_dim,
                    kv_dim=2,
                    num_layers=layers,
                    float32=True,
                )
            )
    return specs


def build_kv_params(
    specs: Sequence[KVLeafSpec],
    *,
    kv_cache_config: KVCacheConfig,
    cache_dtype: DType,
    devices: Sequence[DeviceRef],
    num_q_heads: int,
    data_parallel_degree: int = 1,
    speculative_method: SpeculativeMethod | None = None,
    num_draft_tokens: int = 0,
) -> MultiKVCacheParams:
    """One ``MultiKVCacheParams`` with a leaf per spec, all paged by token.

    The window and zone leaves store one latent per slot (MLA-shaped, K only)
    in ``cache_dtype``; the state leaves are MHA-shaped float32 with K and V.
    Every leaf shares the page size, so a zone leaf's page holds
    ``page_size // ratio`` entries and the block table needs no translation.
    """
    page_size = kv_cache_config.kv_cache_page_size
    children: dict[str, KVCacheParams] = {}
    for spec in specs:
        children[spec.key] = kv_cache_config.to_params(
            dtype=DType.float32 if spec.float32 else cache_dtype,
            n_kv_heads=1,
            head_dim=spec.head_dim,
            num_layers=spec.num_layers,
            devices=devices,
            data_parallel_degree=data_parallel_degree,
            is_mla=spec.kv_dim == 1,
            num_q_heads=num_q_heads,
            speculative_method=speculative_method,
            num_draft_tokens=num_draft_tokens,
            slots_per_page=spec.slots_per_page(page_size),
            window_size=spec.window_size,
        )
    return MultiKVCacheParams.from_params(children)


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

    def compressed_layer_index(self, layer_idx: int) -> int:
        """Rank of a compressed layer among trunk layers sharing its ratio.

        That rank is its layer index within the ratio's zone and state leaves.
        """
        ratio = self.compress_ratios[layer_idx]
        return sum(
            1 for i in range(layer_idx) if self.compress_ratios[i] == ratio
        )

    def kv_leaf_specs(self) -> list[KVLeafSpec]:
        return kv_leaf_specs(
            compress_ratios=self.compress_ratios,
            num_hidden_layers=self.num_hidden_layers,
            head_dim=self.head_dim,
            index_head_dim=self.index_head_dim,
            sliding_window=self.sliding_window,
            num_dspark_stages=len(self.dspark_target_layer_ids),
        )

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
        """Build the seven-leaf KV tree (see ``kv_leaf_specs``).

        The reference sizes each layer's cache as ``window_size + max_seq_len
        // compress_ratio``. Here the window is one sliding-window leaf over
        every layer and each ratio's compressed entries are a zone leaf with
        ``page_size // ratio`` slots per page, so no layer over-allocates.
        """
        speculative_method: SpeculativeMethod | None = None
        num_draft_tokens: int = 0
        if pipeline_config.speculative:
            speculative_method = pipeline_config.speculative.speculative_method
            num_draft_tokens = (
                pipeline_config.speculative.num_speculative_tokens or 0
            )
        specs = kv_leaf_specs(
            compress_ratios=list(huggingface_config.compress_ratios),
            num_hidden_layers=huggingface_config.num_hidden_layers,
            head_dim=huggingface_config.head_dim,
            index_head_dim=huggingface_config.index_head_dim,
            sliding_window=huggingface_config.sliding_window,
            num_dspark_stages=len(huggingface_config.dspark_target_layer_ids),
        )
        return build_kv_params(
            specs,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
            devices=devices,
            num_q_heads=huggingface_config.num_attention_heads,
            data_parallel_degree=pipeline_config.model.data_parallel_degree,
            speculative_method=speculative_method,
            num_draft_tokens=num_draft_tokens,
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
