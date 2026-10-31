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
"""Config for DeepSeek-V4.1-Flash models (ModuleV3).

Unlike V4, the published ``config.json`` nests every model field under
``text_config`` (and ``vision_config``, unused until vision lands); see
``hf_config_shims.DeepseekV41HFConfig``. This config flattens the fields it
needs from ``text_config`` onto itself -- so an arch module always sees a
flat ``DeepseekV41Config``, never a sub-config reach-through. It subclasses
``ArchConfigWithKVCache`` directly, *not* ``DeepseekV3Config``: V3's
``__post_init__`` rejects everything but silu/YaRN, which does not fit
V4.1's compress-RoPE schedule.

Only the fields the layer-role resolver and the KV interface need are
resolved here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, ClassVar

from max.driver import Buffer
from max.dtype import DType
from max.graph import BufferType, DeviceRef, TensorType
from max.nn.legacy.kv_cache import KVCacheParamInterface
from max.nn.legacy.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.lib import KVCacheConfig, MAXModelConfig, PipelineConfig
from max.pipelines.lib.config.model_config import _select_quantization_encoding
from max.pipelines.lib.interfaces.arch_config import (
    ArchConfigWithKVCache,
)
from max.pipelines.lib.pipeline_variants.utils import get_rope_theta
from max.pipelines.lib.utils import upper_bounded_default
from max.pipelines.modeling.config_enums import (
    SupportedEncoding,
    supported_encoding_dtype,
)
from max.pipelines.speculative.config import SpeculativeMethod
from transformers import AutoConfig
from typing_extensions import Self, override

from . import interface

# DSpark draft stages when the checkpoint config does not say.
_DEFAULT_DSPARK_STAGES = 3


def num_dspark_stages(text_config: Any) -> int:
    """The number of DSpark draft stages the checkpoint declares.

    Checkpoint configs name the count either ``n_mtp_layers`` or
    ``num_nextn_predict_layers``. Zero is a meaningful declaration (a
    checkpoint with no drafter), so only a missing field falls through
    to the default.
    """
    n_mtp_layers = getattr(text_config, "n_mtp_layers", None)
    if n_mtp_layers is not None:
        return n_mtp_layers
    num_nextn_layers = getattr(text_config, "num_nextn_predict_layers", None)
    if num_nextn_layers is not None:
        return num_nextn_layers
    return _DEFAULT_DSPARK_STAGES


def dspark_expert_counts(text_config: Any) -> tuple[int, int]:
    """The drafter MoE's (routed experts, experts per token).

    Read as the V2 port reads them: 0 or absent falls back to the trunk's
    count, and the top-k key appears under either
    ``dspark_num_experts_per_tok`` or ``dspark_n_activated_experts``.
    """
    routed = getattr(text_config, "dspark_n_routed_experts", 0) or int(
        text_config.n_routed_experts
    )
    top_k = (
        getattr(text_config, "dspark_num_experts_per_tok", 0)
        or getattr(text_config, "dspark_n_activated_experts", 0)
        or int(text_config.num_experts_per_tok)
    )
    return int(routed), int(top_k)


def dspark_draft_width(
    speculative: Any,
    target_huggingface_config: Any,
    draft_huggingface_config: Any,
) -> int:
    """The DSpark draft width: the checkpoint's ``dspark_block_size``.

    The KV cache reserves room for this many draft tokens per step.

    Raises:
        ValueError: If a different ``num_speculative_tokens`` is requested.
    """
    del draft_huggingface_config
    text_config = DeepseekV41Config.get_text_config(target_huggingface_config)
    block_size = int(getattr(text_config, "dspark_block_size", 5))
    requested = speculative.num_speculative_tokens
    if requested is not None and requested != block_size:
        raise ValueError(
            "DeepSeek-V4.1 DSpark drafts exactly dspark_block_size="
            f"{block_size} tokens per step (the block size the checkpoint "
            "was trained with); got "
            f"num_speculative_tokens={requested}"
        )
    return block_size


@dataclass(kw_only=True)
class DeepseekV41Config(ArchConfigWithKVCache):
    """Configuration for DeepSeek-V4.1-Flash models (ModuleV3).

    Core dimensions mirror ``text_config`` in the published ``config.json``.
    The KV leaves come from :func:`interface.build_kv_params`, which this
    class delegates to.
    """

    # The published NVFP4 checkpoint's dense FP8 + expert NVFP4 blocks; the
    # float32 entry is a checkpoint dequantized offline.
    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "float4_e2m1fnx2"
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {
        "float4_e2m1fnx2",
    }

    # MAX-specific fields.
    dtype: DType
    kv_params: KVCacheParamInterface
    devices: list[DeviceRef]
    max_seq_len: int
    quantization_encoding: SupportedEncoding | None = None
    data_parallel_degree: int = 1
    return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN
    return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE
    # Core dimensions (``text_config``).
    vocab_size: int = 129280
    hidden_size: int = 5120
    num_hidden_layers: int = 40
    rms_norm_eps: float = 1e-20

    # Attention geometry (``text_config``; ``V41Config`` mirrors these for
    # the interface side).
    num_attention_heads: int = 64
    head_dim: int = 512
    q_lora_rank: int = 1280
    o_groups: int = 8
    o_lora_rank: int = 1024
    hc_mult: int = 4
    # CSA2's SWA leaf window; ``interface.kv_leaf_specs`` reads the same
    # field off its own ``V41Config`` copy of the text config.
    sliding_window: int = 128

    # MoE. Both expert counts are published checkpoint keys, so the config
    # keeps them even where a count has no reader yet.
    n_routed_experts: int = 384
    n_shared_experts: int = 1
    moe_intermediate_size: int = 2304
    num_experts_per_tok: int = 6
    routed_scaling_factor: float = 1.5
    swiglu_limit: float = 10.0

    # Hash routing is gone in V4.1 (every layer is score-routed); kept as a
    # defaulted-absent field so a checkpoint without the key resolves to 0
    # instead of raising.
    num_hash_layers: int = 0

    # Compressed sparse attention (CSA2). One entry per trunk + DSpark
    # layer; the ratio values are 0 (SWA-only) / 1 / 2.
    compress_ratios: list[int] = field(default_factory=list)

    # Per-layer KV/index source and candidate-pool wiring the layer resolver
    # reads; ``candidate_source_layer_id`` default -1 matches the reference's
    # "no candidate pool" sentinel.
    kv_source_layer_ids: list[int] = field(default_factory=list)
    index_source_layer_ids: list[int] = field(default_factory=list)
    candidate_source_layer_id: int = -1
    candidate_topk_blocks: int = 0
    candidate_block_size: int = 0

    # Engram placement (the layout itself is ABI inputs, not config state).
    engram_layer_ids: list[int] = field(default_factory=list)
    engram_num_embeddings: list[int] = field(default_factory=list)

    # DSpark draft-target taps, read by the DSpark drafter. The SWA leaf's
    # draft rows are sized by the draft-stage count
    # (``interface.V41Config.num_dspark_stages``), not this list: the two
    # only coincide on the published checkpoint.
    dspark_target_layer_ids: list[int] = field(default_factory=list)

    # DSpark draft stage count, resolved from ``n_mtp_layers`` /
    # ``num_nextn_predict_layers``. Declared independent of
    # ``dspark_target_layer_ids``: the checkpoint names both, and only the
    # drafter's build asserts they agree.
    num_dspark_layers: int = 3
    # Token id filling the non-anchor draft-block slots; no safe default.
    dspark_noise_token_id: int | None = None
    # Tokens per draft block, anchor included; also the number of
    # speculative tokens per step (:func:`dspark_draft_width`).
    dspark_block_size: int = 5
    # Width of the Markov head's bigram embedding (``markov_head.embed``).
    dspark_markov_rank: int = 256
    # The drafter MoE's expert counts; 0/absent falls back to the trunk's.
    dspark_n_routed_experts: int = 128
    dspark_num_experts_per_tok: int = 3

    # What a DSpark pairing's batching and graph take beside the weights,
    # which the trunk's ``build_module`` sets: the pinned Engram tables and
    # token map by input name, and the trunk's extras the fused graph takes
    # past the canonical tail.
    engram_static_inputs: dict[str, Buffer] = field(default_factory=dict)
    dspark_trailing_inputs: list[tuple[str, TensorType | BufferType]] = field(
        default_factory=list
    )

    # Sparse indexer top-k, one global constant (not per-layer).
    index_topk: int | None = None
    # Indexer geometry (``text_config``): the MXFP4 index-key width and
    # the scoring heads. The interface's ``V41Config`` keeps its own
    # ``index_head_dim`` copy for the KV leaf geometry; both views
    # require the checkpoint's fields (``resolve_text_fields`` reads
    # them like ``V41Config.from_hf`` does).
    index_n_heads: int = 32
    index_head_dim: int = 128

    # RoPE / YaRN: theta 160000 + YaRN factor 16 on compressed layers,
    # theta 10000 on sliding-window-only layers.
    rope_theta: float = 10000.0
    rope_scaling: dict[str, Any] | None = None
    compress_rope_theta: float = 160000.0

    def get_kv_params(self) -> KVCacheParamInterface:
        """The KV cache parameters for this config."""
        return self.kv_params

    def get_max_seq_len(self) -> int:
        """The maximum sequence length the config allows."""
        return self.max_seq_len

    @property
    def target_layer_ids(self) -> list[int]:
        """Returns the trunk layers whose states the DSpark stages read.

        A speculative pairing reads them off its draft's config.
        """
        return self.dspark_target_layer_ids

    @property
    def mask_token_id(self) -> int:
        """The token that fills the DSpark block past its anchor."""
        assert self.dspark_noise_token_id is not None
        return self.dspark_noise_token_id

    @staticmethod
    def get_text_config(huggingface_config: Any) -> Any:
        """Normalizes any supported config form to the text config.

        DeepSeek-V4.1-Flash nests every text field under ``text_config``.
        Callers pass either the ``DeepseekV41HFConfig`` wrapper, its already
        unwrapped text config, or a raw mapping -- either the wrapper dict
        ``to_dict`` emits or a bare text mapping. Mappings become
        attribute-accessible so downstream ``getattr``/field reads work.
        """
        if isinstance(huggingface_config, Mapping):
            if "text_config" in huggingface_config:
                huggingface_config = huggingface_config["text_config"]
            if isinstance(huggingface_config, Mapping):
                return SimpleNamespace(**huggingface_config)
            return huggingface_config
        # No ``text_config`` attribute means the input already *is* the
        # text config; return it unchanged instead of unwrapping again.
        return getattr(huggingface_config, "text_config", huggingface_config)

    @staticmethod
    def construct_kv_params(
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParamInterface:
        """Delegates to :func:`interface.build_kv_params`.

        ``PipelineModel.get_kv_params`` (the pre-``initialize`` early KV-param
        query) calls ``model_config_cls.construct_kv_params`` on the *raw*
        top-level ``huggingface_config``; ``interface.build_kv_params`` needs
        the attributes DeepSeek-V4.1-Flash nests under ``text_config``, so
        this re-reads them into the interface's ``V41Config`` view.
        ``V41Config.from_hf`` takes a JSON mapping (the shape
        ``to_dict`` produces, with registered sub-configs serialized back
        to nested dicts), so the live ``PretrainedConfig`` is normalized
        with ``to_dict`` first.

        A speculative configuration resolves to its method and draft width
        the way the V2 pipeline's ``kv_leaves.construct_kv_params`` does,
        so every leaf's pool params plan for the draft rows a verify step
        appends.

        Raises:
            ValueError: If the speculative method is anything but DSpark,
                the only pairing this trunk has, or DSpark is named without
                its ``num_speculative_tokens`` resolved: the draft width
                sizes the pool's per-request blocks and the state ring's
                depth check, so it cannot stay unset.
        """
        del cache_dtype
        speculative_method: SpeculativeMethod | None = None
        num_draft_tokens = 0
        if pipeline_config.speculative:
            speculative_method = pipeline_config.speculative.speculative_method
            if speculative_method not in (None, "dspark"):
                raise ValueError(
                    "DeepSeek-V4.1 (ModuleV3) speculates only with dspark, "
                    f"not {speculative_method}."
                )
            drafts = pipeline_config.speculative.num_speculative_tokens
            if speculative_method is not None and drafts is None:
                raise ValueError(
                    f"{speculative_method} needs num_speculative_tokens "
                    "resolved before the KV cache is sized."
                )
            num_draft_tokens = drafts or 0
        config = huggingface_config.to_dict()
        return interface.build_kv_params(
            config=interface.V41Config.from_hf(config),
            kv_cache_config=kv_cache_config,
            devices=devices,
            data_parallel_degree=pipeline_config.model.data_parallel_degree,
            speculative_method=speculative_method,
            num_draft_tokens=num_draft_tokens,
        )

    @classmethod
    def calculate_max_seq_len(
        cls,
        huggingface_config: AutoConfig,
        model_config: MAXModelConfig,
    ) -> int:
        """The longest sequence that fits the KV cache."""
        text_config = cls.get_text_config(huggingface_config)
        return upper_bounded_default(
            upper_bound=text_config.max_position_embeddings,
            default=model_config.max_length,
        )

    @classmethod
    def resolve_text_fields(cls, text_config: AutoConfig) -> dict[str, Any]:
        """Flattens the ``text_config`` fields this config resolves, verbatim.

        Returns the kwargs :meth:`initialize` merges with the fields it
        resolves itself (``dtype``, ``kv_params``, ``devices``, ...). Split
        out so a test can exercise field resolution directly, without
        building a full ``PipelineConfig``.
        """
        return dict(
            vocab_size=text_config.vocab_size,
            hidden_size=text_config.hidden_size,
            num_hidden_layers=text_config.num_hidden_layers,
            rms_norm_eps=text_config.rms_norm_eps,
            num_attention_heads=text_config.num_attention_heads,
            head_dim=text_config.head_dim,
            q_lora_rank=text_config.q_lora_rank,
            o_groups=text_config.o_groups,
            o_lora_rank=text_config.o_lora_rank,
            hc_mult=text_config.hc_mult,
            sliding_window=text_config.sliding_window,
            n_routed_experts=text_config.n_routed_experts,
            n_shared_experts=getattr(text_config, "n_shared_experts", 1),
            num_hash_layers=getattr(text_config, "num_hash_layers", 0),
            moe_intermediate_size=text_config.moe_intermediate_size,
            num_experts_per_tok=text_config.num_experts_per_tok,
            routed_scaling_factor=float(
                getattr(text_config, "routed_scaling_factor", 1.5)
            ),
            swiglu_limit=float(getattr(text_config, "swiglu_limit", 10.0)),
            compress_ratios=list(text_config.compress_ratios),
            kv_source_layer_ids=list(text_config.kv_source_layer_ids),
            index_source_layer_ids=list(text_config.index_source_layer_ids),
            candidate_source_layer_id=text_config.candidate_source_layer_id,
            candidate_topk_blocks=getattr(
                text_config, "candidate_topk_blocks", 0
            ),
            candidate_block_size=getattr(
                text_config, "candidate_block_size", 0
            ),
            engram_layer_ids=list(text_config.engram_layer_ids),
            engram_num_embeddings=list(text_config.engram_num_embeddings),
            dspark_target_layer_ids=list(
                getattr(text_config, "dspark_target_layer_ids", [])
            ),
            num_dspark_layers=num_dspark_stages(text_config),
            dspark_noise_token_id=(
                None
                if (
                    noise := getattr(text_config, "dspark_noise_token_id", None)
                )
                is None
                else int(noise)
            ),
            dspark_block_size=int(getattr(text_config, "dspark_block_size", 5)),
            dspark_markov_rank=int(
                getattr(text_config, "dspark_markov_rank", 256)
            ),
            dspark_n_routed_experts=dspark_expert_counts(text_config)[0],
            dspark_num_experts_per_tok=dspark_expert_counts(text_config)[1],
            index_topk=text_config.index_topk,
            index_n_heads=text_config.index_n_heads,
            index_head_dim=text_config.index_head_dim,
            rope_theta=get_rope_theta(text_config),
            rope_scaling=text_config.rope_scaling,
            compress_rope_theta=text_config.compress_rope_theta,
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
        """Initializes a DeepseekV41Config from pipeline configuration."""
        model_config = model_config or pipeline_config.model
        hf_config = model_config.huggingface_config
        if hf_config is None:
            raise ValueError(
                "HuggingFace config is required for"
                f" '{model_config.model_path}', but config could not be"
                " loaded. Please ensure the model repository contains a"
                " valid config.json file."
            )
        text_config = cls.get_text_config(hf_config)
        kv_cache_config = model_config.kv_cache
        quantization_encoding = _select_quantization_encoding(
            model_config, cls.DEFAULT_ENCODING
        )
        dtype = supported_encoding_dtype(quantization_encoding)

        device_refs = [
            DeviceRef(spec.device_type, spec.id)
            for spec in model_config.device_specs
        ]

        kv_params = cls.construct_kv_params(
            huggingface_config=hf_config,
            pipeline_config=pipeline_config,
            devices=device_refs,
            kv_cache_config=kv_cache_config,
            cache_dtype=dtype,
        )

        return cls(
            dtype=dtype,
            kv_params=kv_params,
            devices=device_refs,
            max_seq_len=max_seq_len,
            quantization_encoding=quantization_encoding,
            data_parallel_degree=model_config.data_parallel_degree,
            **cls.resolve_text_fields(text_config),
        )
