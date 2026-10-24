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
"""Config for DeepSeek-V4.1-Flash models.

The checkpoint's ``config.json`` nests model fields under ``text_config``;
this module flattens the fields the model needs into one config object.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, ClassVar

from max.driver import accelerator_api
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.comm.ep import EPConfig
from max.nn.comm.ep.ep_config import calculate_ep_max_tokens_per_rank
from max.nn.kv_cache import KVCacheParamInterface, spec_decode_cache_slack
from max.nn.quant_config import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)
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
from transformers import AutoConfig
from typing_extensions import Self, override

from . import kv_leaves

# Precision switches. All default on.
WOA_MXFP8_ENV = "MAX_DSV41_WOA_MXFP8"
Q_MXFP8_ENV = "MAX_DSV41_Q_MXFP8"
HEAD_BF16_ENV = "MAX_DSV41_HEAD_BF16"


def _env_switch(name: str) -> bool:
    return os.environ.get(name, "1").strip().lower() not in (
        "0",
        "false",
        "off",
    )


def woa_mxfp8_enabled() -> bool:
    """Whether ``attn.wo_a`` runs as an MXFP8 grouped matmul.

    Covers the trunk and DSpark's ``mtp.*`` stages. On (default): the FP8
    weight loads as-is, its E8M0 block scale is row-expanded and
    tile-interleaved on the host, and the input is quantized to MXFP8. Off
    (``MAX_DSV41_WOA_MXFP8=0``): the weight is dequantized to bf16 on the
    host.
    """
    return _env_switch(WOA_MXFP8_ENV)


def q_mxfp8_enabled() -> bool:
    """Whether the query-side projections run as MXFP8 matmuls.

    Covers ``attn.wq_a``, ``attn.wkv``, ``attn.wq_b`` and
    ``attn.indexer.wq_b`` in the trunk and DSpark's ``mtp.*`` stages. On
    (default): the FP8 weights load as-is with row-expanded E8M0 scales,
    and the inputs are quantized to MXFP8. Off (``MAX_DSV41_Q_MXFP8=0``):
    the weights are dequantized to bf16 on the host.
    """
    return _env_switch(Q_MXFP8_ENV)


# MXFP8 projections on the cluster split-K matmul (default on).
MXFP8_SPLITK_ENV = "MAX_DSV41_MXFP8_SPLITK"


def mxfp8_splitk_enabled() -> bool:
    """Whether ``attn.wo_b`` and the shared expert run on the cluster
    split-K MXFP8 matmul.

    On (default): each weight shape and M takes cluster split-K or a
    pair-CTA GEMM (decode or prefill) where measured faster, and the
    production block-scaled matmul otherwise. Off
    (``MAX_DSV41_MXFP8_SPLITK=0``): production at every size.
    """
    return _env_switch(MXFP8_SPLITK_ENV)


ENGRAM_HOST_GATHER_ENV = "MAX_ENGRAM_HOST_GATHER"


def engram_host_gather_enabled() -> bool:
    """Whether full-table Engram lookups hash and gather on host threads.

    On (default, NVIDIA GPUs): from 1,024 tokens CPU threads read the pinned
    tables and the operands reach the GPU in chunks, so no SM reads the
    tables, and data-parallel forwards run the lookup on a side stream that
    overlaps the layers before the first Engram layer. Smaller steps and
    steps a device graph captures run the GPU kernels, on that side stream
    too. Off (``MAX_ENGRAM_HOST_GATHER=0``, and on other GPUs): the GPU
    kernels at every size, on the main stream.
    """
    return accelerator_api() == "cuda" and _env_switch(ENGRAM_HOST_GATHER_ENV)


# Interleave routed-expert NVFP4 block scales on the host (default on). "0"
# interleaves in-graph, which the compiler hoists into a second resident copy.
NVFP4_SF_PREINTERLEAVE_ENV = "MAX_DSV41_NVFP4_SF_PREINTERLEAVE"
# Scale-factor tile ("SF atom") of the interleaved layout: 128 rows
# (32 x 4) by 4 scale columns.
NVFP4_SF_ATOM_ROWS = 128
NVFP4_SF_ATOM_COLS = 4


def nvfp4_sf_preinterleave_enabled() -> bool:
    """Whether stacked routed-expert NVFP4 scales are host-interleaved."""
    return _env_switch(NVFP4_SF_PREINTERLEAVE_ENV)


# Elements per E4M3 block scale of NVFP4 (``quantization_config.group_size``).
NVFP4_BLOCK = 16

# NVFP4 activations: a static tensor-wise input scale and dynamic per-16 E4M3
# block scales. What the expert-parallel dispatch and the fused SwiGLU
# quantize read off their config.
NVFP4_ACTIVATION_QUANT_CONFIG = QuantConfig(
    input_scale=InputScaleSpec(
        granularity=ScaleGranularity.BLOCK,
        origin=ScaleOrigin.STATIC,
        dtype=DType.float32,
        block_size=(1, NVFP4_BLOCK),
    ),
    weight_scale=WeightScaleSpec(
        granularity=ScaleGranularity.BLOCK,
        dtype=DType.float8_e4m3fn,
        block_size=(1, NVFP4_BLOCK),
    ),
    mlp_quantized_layers=set(),
    attn_quantized_layers=set(),
    format=QuantFormat.NVFP4,
)


def ep_dispatch_active(data_parallel_degree: int) -> bool:
    """Whether the routed experts run behind expert-parallel dispatch."""
    return data_parallel_degree > 1


def data_parallel_ep_config(
    pipeline_config: PipelineConfig, text_config: Any
) -> EPConfig | None:
    """The routed experts' expert-parallel dispatch, or ``None`` without DP.

    Under data parallelism each device quantizes its rows to NVFP4 against
    one global gate/up input scale and sends the routed copies to the
    devices owning their experts. Each rank holds up to
    ``max_batch_input_tokens`` rows, plus with DSpark the draft rows that
    verification adds for each of up to ``max_batch_size`` requests.

    Raises:
        ValueError: If the devices are not ``0`` to ``n - 1``, a device
            would hold more than :data:`MEGAFFN_MAX_LOCAL_EXPERTS` routed
            experts, chunked prefill is off, so a forward step's rows are not
            bounded by ``max_batch_input_tokens``, or DSpark runs without
            ``max_batch_size``.
    """
    model_config = pipeline_config.model
    dp = int(model_config.data_parallel_degree)
    if not ep_dispatch_active(dp):
        return None
    device_ids = [int(spec.id) for spec in model_config.device_specs]
    if device_ids != list(range(len(device_ids))):
        raise ValueError(
            "DeepSeek-V4.1 expert-parallel dispatch under data parallelism "
            f"needs devices 0 to {len(device_ids) - 1}, got {device_ids}"
        )
    local_experts = int(text_config.n_routed_experts) // dp
    if local_experts > MEGAFFN_MAX_LOCAL_EXPERTS:
        raise ValueError(
            "DeepSeek-V4.1 data parallelism runs each device's routed "
            f"experts in one fused kernel, which holds at most "
            f"{MEGAFFN_MAX_LOCAL_EXPERTS}; data_parallel_degree {dp} leaves "
            f"{local_experts} per device"
        )
    runtime = pipeline_config.runtime
    if not runtime.enable_chunked_prefill:
        # The receive buffers hold max_batch_input_tokens rows per rank, which
        # only chunked prefill bounds a forward step to.
        raise ValueError(
            "DeepSeek-V4.1 expert-parallel dispatch under data parallelism "
            "needs chunked prefill, which bounds each forward step to "
            "max_batch_input_tokens rows per replica"
        )
    rows = runtime.max_batch_input_tokens
    speculative = pipeline_config.speculative
    if speculative is not None and speculative.is_dspark():
        # Verification runs each request's drafts beside its active tokens,
        # and the scheduler's token budget counts only the active tokens.
        if runtime.max_batch_size is None:
            raise ValueError(
                "DeepSeek-V4.1 DSpark under data parallelism needs "
                "max_batch_size, which bounds the draft rows a verification "
                "step adds to max_batch_input_tokens"
            )
        rows += runtime.max_batch_size * (
            speculative.num_speculative_tokens or 0
        )
    n_devices = len(model_config.device_specs)
    return EPConfig(
        dispatch_dtype=DType.uint8,
        dispatch_quant_config=NVFP4_ACTIVATION_QUANT_CONFIG,
        combine_dtype=DType.bfloat16,
        hidden_size=int(text_config.hidden_size),
        top_k=int(text_config.num_experts_per_tok),
        n_experts=int(text_config.n_routed_experts),
        max_tokens_per_rank=calculate_ep_max_tokens_per_rank(
            max_batch_input_tokens=rows,
            ep_size=n_devices,
            data_parallel_degree=dp,
        ),
        n_gpus_per_node=n_devices,
        n_nodes=1,
    )


# The most experts one device's fused gate/up + down kernel schedules.
MEGAFFN_MAX_LOCAL_EXPERTS = 128


def gate_up_rows_interleaved(
    n_routed_experts: int, data_parallel_degree: int
) -> bool:
    """Whether the trunk's routed ``gate_up_proj`` rows alternate gate, up.

    Under data parallelism the routed experts run behind expert-parallel
    dispatch, whose statically sized receive buffers let each device's
    gate/up GEMM, SwiGLU and down GEMM fuse into one kernel when it holds at
    most :data:`MEGAFFN_MAX_LOCAL_EXPERTS` experts. That kernel reads each
    gate row next to its up row.
    """
    return (
        ep_dispatch_active(data_parallel_degree)
        and n_routed_experts // data_parallel_degree
        <= MEGAFFN_MAX_LOCAL_EXPERTS
    )


def nvfp4_sf_preinterleaved(rows: int, scale_cols: int) -> bool:
    """Whether an ``[rows, scale_cols]`` expert scale is stored interleaved.

    Only shapes that tile the scale-factor tile exactly are pre-interleaved,
    so the weight keeps its checkpoint shape. Weight loading and graph
    construction must both use this predicate so their layouts agree.
    """
    return (
        nvfp4_sf_preinterleave_enabled()
        and rows % NVFP4_SF_ATOM_ROWS == 0
        and scale_cols % NVFP4_SF_ATOM_COLS == 0
    )


# DSpark draft stages when the checkpoint config does not say.
_DEFAULT_DSPARK_STAGES = 3


def num_dspark_stages(text_config: Any) -> int:
    """Returns the number of DSpark draft stages the checkpoint declares.

    Checkpoint configs name the count either ``n_mtp_layers`` or
    ``num_nextn_predict_layers``.
    """
    return (
        getattr(text_config, "n_mtp_layers", None)
        or getattr(text_config, "num_nextn_predict_layers", None)
        or _DEFAULT_DSPARK_STAGES
    )


def head_bf16_enabled() -> bool:
    """Whether the LM head keeps the checkpoint's bf16 weight.

    On (default): bf16 matmul with float32 accumulation, logits rounded to
    bf16 then widened to float32. Off (``MAX_DSV41_HEAD_BF16=0``): the weight
    is widened to float32 at load.
    """
    return _env_switch(HEAD_BF16_ENV)


# The CSA2 compressor's projections on bf16 operands (default on).
COMPRESSOR_BF16_ENV = "MAX_DSV41_COMPRESSOR_BF16"


def compressor_bf16_enabled() -> bool:
    """Whether the compressor's ``wkv``/``wgate`` keep the checkpoint's bf16.

    On (default): bf16 matmuls with float32 accumulation and output. Off
    (``MAX_DSV41_COMPRESSOR_BF16=0``): the weights are widened to float32
    at load and the bf16 input to float32 in the graph. The products of
    bf16 values are exact in both, so only the accumulation order differs.
    """
    return _env_switch(COMPRESSOR_BF16_ENV)


# Each CSA2 layer's index list in one kernel launch (default on).
INDEX_LIST_FUSED_ENV = "MAX_DSV41_INDEX_LIST_FUSED"


def index_list_fused_enabled() -> bool:
    """Whether a GPU layer builds its CSA2 index list in one launch.

    On (default): one ``mo.csa2.index_list`` kernel writes the layer's
    ``[T, 640]`` window and compressed slots
    (:func:`~.layers.csa2.index_list`). Off
    (``MAX_DSV41_INDEX_LIST_FUSED=0``): the graph ops of
    :func:`~.layers.csa2.index_list_ops`, which build the same list bit for
    bit.
    """
    return _env_switch(INDEX_LIST_FUSED_ENV)


@dataclass(kw_only=True)
class DeepseekV41Config(ArchConfigWithKVCache):
    """Configuration for DeepSeek-V4.1-Flash models.

    Core dimensions come from ``text_config`` in the checkpoint's
    ``config.json``.
    """

    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "float8_e4m3fn"
    # float32: an offline-dequantized checkpoint. float4_e2m1fnx2: an NVFP4
    # checkpoint, whose per-layer weight formats are resolved at load.
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {
        "float8_e4m3fn",
        "float32",
        "float4_e2m1fnx2",
    }

    # MAX-specific fields.
    dtype: DType
    kv_params: KVCacheParamInterface
    devices: list[DeviceRef]
    max_seq_len: int
    quantization_encoding: SupportedEncoding | None = None
    data_parallel_degree: int = 1
    ep_config: EPConfig | None = None
    """Expert-parallel dispatch of the routed experts; only under data
    parallelism (:func:`data_parallel_ep_config`)."""
    forward_max_rows: int | None = None
    """Most query rows one forward step carries (``None``: ``max_seq_len``).

    Bounds the indexer's row-tile count.
    """

    # Core dimensions (``text_config``).
    vocab_size: int = 129280
    hidden_size: int = 5120
    num_hidden_layers: int = 40
    rms_norm_eps: float = 1e-20
    # Sliding-window attention size, which is also the length of the
    # sliding-window KV cache.
    sliding_window: int = 128

    # MoE.
    n_routed_experts: int = 384
    # One always-active SwiGLU FFN per layer, summed with the routed output.
    n_shared_experts: int = 1
    moe_intermediate_size: int = 2304
    num_experts_per_tok: int = 6
    routed_scaling_factor: float = 1.5
    swiglu_limit: float = 10.0

    # V4.1 has no hash-routed layers; checkpoints omit the key.
    num_hash_layers: int = 0

    # DSpark speculative decoding. Number of draft stages, independent of
    # ``dspark_target_layer_ids``.
    num_dspark_layers: int = 3
    # Token id filling the non-anchor draft-block slots; no safe default.
    dspark_noise_token_id: int | None = None
    # Trunk layers whose *inputs* are captured as DSpark's aux hidden states.
    dspark_target_layer_ids: list[int] = field(default_factory=list)
    # Tokens per draft block, anchor included; also the number of speculative
    # tokens per step (:meth:`validate_num_speculative_tokens`).
    dspark_block_size: int = 5
    # Width of the Markov head's bigram embedding (``markov_head.embed``).
    dspark_markov_rank: int = 256
    # The drafter MoE's expert counts; 0/absent falls back to the trunk's.
    dspark_n_routed_experts: int = 128
    dspark_num_experts_per_tok: int = 3

    # From ``quantization_config`` (a sibling of ``text_config``); ``None``
    # for an unquantized checkpoint.
    expert_dtype: str | None = None
    moe_quant_algo: str | None = None

    # KV compression ratio per trunk and DSpark layer: 0 (sliding window
    # only), 1 or 2.
    compress_ratios: list[int] = field(default_factory=list)

    # Which layers compute their own KV and indexer results (later layers
    # reuse them); ``candidate_source_layer_id == -1`` means no candidate
    # pool.
    kv_source_layer_ids: list[int] = field(default_factory=list)
    index_source_layer_ids: list[int] = field(default_factory=list)
    candidate_source_layer_id: int = -1
    # Candidate-block prefilter: the candidate-source layer keeps its
    # ``candidate_topk_blocks`` highest-scoring blocks of
    # ``candidate_block_size`` compressed positions, and later index-source
    # layers score only positions inside them. Zero for either turns it off.
    candidate_topk_blocks: int = 0
    candidate_block_size: int = 0

    # Engram placement and hash-table layout.
    # ``engram_vocab_size`` is the floor each head's prime bucket count is
    # drawn above, distinct from ``engram_num_embeddings`` (per-layer row
    # counts) and ``vocab_size``.
    engram_layer_ids: list[int] = field(default_factory=list)
    engram_num_embeddings: list[int] = field(default_factory=list)
    engram_max_ngram_size: int = 0
    engram_n_heads: int = 0
    engram_head_dim: int = 0
    engram_compressed_vocab_size: int = 0
    engram_pad_token_id: int = 0
    engram_vocab_size: int = 0

    # Sparse indexer top-k, one global constant (not per-layer).
    index_topk: int | None = None

    # RoPE: compressed layers use ``compress_rope_theta`` with
    # ``rope_scaling`` (YaRN); other layers use unscaled ``rope_theta``.
    rope_theta: float = 10000.0
    rope_scaling: dict[str, Any] | None = None
    compress_rope_theta: float = 160000.0

    # MLA attention/indexer shapes.
    num_attention_heads: int = 64
    head_dim: int = 512
    qk_rope_head_dim: int = 64
    q_lora_rank: int = 1024
    # Grouped low-rank output projection (``wo_a``/``wo_b``).
    o_groups: int = 8
    o_lora_rank: int = 1024
    index_n_heads: int = 64
    index_head_dim: int = 128

    # Hyper-connections (mHC); checkpoint-independent constants.
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6

    # Set after ``initialize``; select which logits and hidden states a
    # forward step returns.
    return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN
    return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE

    def validate_num_speculative_tokens(
        self, num_speculative_tokens: int
    ) -> None:
        """Checks a speculative config against the DSpark drafter's block.

        The drafter proposes one token per block position, and the block is
        non-causal at its trained width, so the two must be equal.

        Raises:
            ValueError: If ``num_speculative_tokens != dspark_block_size``.
        """
        if num_speculative_tokens != self.dspark_block_size:
            raise ValueError(
                "DeepSeek-V4.1 DSpark drafts exactly dspark_block_size="
                f"{self.dspark_block_size} tokens per step (the block size "
                "the checkpoint was trained with); got "
                f"num_speculative_tokens={num_speculative_tokens}"
            )

    def get_kv_params(self) -> KVCacheParamInterface:
        return self.kv_params

    def get_max_seq_len(self) -> int:
        return self.max_seq_len

    @staticmethod
    def get_text_config(huggingface_config: AutoConfig) -> AutoConfig:
        """DeepSeek-V4.1-Flash nests every text field under ``text_config``."""
        return huggingface_config.text_config

    @classmethod
    def estimate_vision_cache_entry_bytes(
        cls, huggingface_config: AutoConfig
    ) -> int:
        """One largest image span's bf16 embeddings; 0 without a tower."""
        vision_config = getattr(huggingface_config, "vision_config", None)
        if vision_config is None:
            return 0
        hidden_size = cls.get_text_config(huggingface_config).hidden_size
        return vision_config.max_image_tokens * hidden_size * 2

    @classmethod
    def get_vision_cache_row_spec(
        cls, huggingface_config: AutoConfig
    ) -> tuple[int, DType] | None:
        """One bf16 text-width row per image span position."""
        if getattr(huggingface_config, "vision_config", None) is None:
            return None
        hidden_size = cls.get_text_config(huggingface_config).hidden_size
        return hidden_size, DType.bfloat16

    @classmethod
    def construct_kv_params(
        cls,
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParamInterface:
        """Builds KV cache params from the nested ``text_config``.

        Callers pass the raw top-level ``huggingface_config``.
        """
        text_config = cls.get_text_config(huggingface_config)
        return kv_leaves.construct_kv_params(
            huggingface_config=text_config,
            pipeline_config=pipeline_config,
            devices=devices,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
        )

    @classmethod
    def calculate_max_seq_len(
        cls,
        huggingface_config: AutoConfig,
        model_config: MAXModelConfig,
    ) -> int:
        text_config = cls.get_text_config(huggingface_config)
        return upper_bounded_default(
            upper_bound=text_config.max_position_embeddings,
            default=model_config.max_length,
        )

    @classmethod
    def resolve_text_fields(cls, text_config: AutoConfig) -> dict[str, Any]:
        """Returns the config kwargs read from ``text_config``.

        :meth:`initialize` merges these with the fields it resolves itself.
        """
        return dict(
            vocab_size=text_config.vocab_size,
            hidden_size=text_config.hidden_size,
            num_hidden_layers=text_config.num_hidden_layers,
            rms_norm_eps=text_config.rms_norm_eps,
            sliding_window=text_config.sliding_window,
            n_routed_experts=text_config.n_routed_experts,
            n_shared_experts=text_config.n_shared_experts,
            num_hash_layers=getattr(text_config, "num_hash_layers", 0),
            num_dspark_layers=num_dspark_stages(text_config),
            dspark_noise_token_id=getattr(
                text_config, "dspark_noise_token_id", None
            ),
            dspark_target_layer_ids=list(
                getattr(text_config, "dspark_target_layer_ids", [])
            ),
            dspark_block_size=getattr(text_config, "dspark_block_size", 5),
            dspark_markov_rank=getattr(text_config, "dspark_markov_rank", 256),
            # `or`: 0 means "use the trunk's count"; the top-k key appears
            # under either name.
            dspark_n_routed_experts=(
                getattr(text_config, "dspark_n_routed_experts", 0)
                or text_config.n_routed_experts
            ),
            dspark_num_experts_per_tok=(
                getattr(text_config, "dspark_num_experts_per_tok", 0)
                or getattr(text_config, "dspark_n_activated_experts", 0)
                or text_config.num_experts_per_tok
            ),
            moe_intermediate_size=text_config.moe_intermediate_size,
            num_experts_per_tok=text_config.num_experts_per_tok,
            routed_scaling_factor=text_config.routed_scaling_factor,
            swiglu_limit=text_config.swiglu_limit,
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
            engram_max_ngram_size=text_config.engram_max_ngram_size,
            engram_n_heads=text_config.engram_n_heads,
            engram_head_dim=text_config.engram_head_dim,
            engram_compressed_vocab_size=text_config.engram_compressed_vocab_size,
            engram_pad_token_id=text_config.engram_pad_token_id,
            engram_vocab_size=text_config.engram_vocab_size,
            index_topk=text_config.index_topk,
            rope_theta=get_rope_theta(text_config),
            rope_scaling=text_config.rope_scaling,
            compress_rope_theta=text_config.compress_rope_theta,
            num_attention_heads=text_config.num_attention_heads,
            head_dim=text_config.head_dim,
            qk_rope_head_dim=text_config.qk_rope_head_dim,
            q_lora_rank=text_config.q_lora_rank,
            o_groups=text_config.o_groups,
            o_lora_rank=text_config.o_lora_rank,
            index_n_heads=text_config.index_n_heads,
            index_head_dim=text_config.index_head_dim,
        )

    @classmethod
    def resolve_quantization_fields(
        cls, huggingface_config: AutoConfig
    ) -> dict[str, Any]:
        """Returns the config kwargs read from ``quantization_config``.

        ``quantization_config`` sits at the top level, beside
        ``text_config``. Both fields are ``None`` for an unquantized
        checkpoint.
        """
        quantization_config = (
            getattr(huggingface_config, "quantization_config", None) or {}
        )
        return dict(
            expert_dtype=quantization_config.get("expert_dtype"),
            moe_quant_algo=quantization_config.get("moe_quant_algo"),
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
        text_config = cls.get_text_config(config)
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
        if model_config.data_parallel_degree not in (1, len(device_refs)):
            raise ValueError(
                "DeepSeek-V4.1 supports data_parallel_degree 1 (tensor-"
                "parallel attention) or the device count (data-parallel "
                f"attention), got {model_config.data_parallel_degree} for "
                f"{len(device_refs)} devices"
            )
        kv_params = cls.construct_kv_params(
            config,
            pipeline_config,
            device_refs,
            kv_cache_config,
            cache_dtype,
        )
        runtime = pipeline_config.runtime
        return cls(
            dtype=dtype,
            kv_params=kv_params,
            devices=device_refs,
            max_seq_len=max_seq_len,
            quantization_encoding=quantization_encoding,
            data_parallel_degree=model_config.data_parallel_degree,
            ep_config=data_parallel_ep_config(pipeline_config, text_config),
            forward_max_rows=forward_row_bound(
                runtime.max_num_input_tokens,
                runtime.max_batch_input_tokens,
                max_seq_len,
            ),
            **cls.resolve_text_fields(text_config),
            **cls.resolve_quantization_fields(config),
        )


def position_capacity(config: DeepseekV41Config) -> int:
    """Returns the exclusive bound on any position a forward step indexes.

    Speculative decoding can place a request's positions past
    ``max_seq_len``, up to the KV pool's speculative slack, so every
    position-indexed table (the rotary tables and the indexer's per-position
    scratch) is sized to ``max_seq_len`` plus that slack. Without speculative
    decoding the slack is zero.
    """
    return config.max_seq_len + spec_decode_cache_slack(config.kv_params)


MAX_IMAGES_PER_REQUEST = "max_images_per_request"
"""The one ``--vision-config-overrides`` key DeepSeek-V4.1 reads."""


def max_images_per_request(pipeline_config: PipelineConfig) -> int | None:
    """Returns the per-request image cap, ``None`` for no cap.

    Set with ``--vision-config-overrides '{"max_images_per_request": N}'``,
    as vLLM's ``--limit-mm-per-prompt``. ``0`` serves text only: the vision
    model is neither loaded nor planned, and the router rejects any request
    with an image.

    Raises:
        ValueError: On any other override key, or a cap that is not a
            non-negative integer.
    """
    overrides = dict(pipeline_config.model.vision_config_overrides)
    unknown = sorted(overrides.keys() - {MAX_IMAGES_PER_REQUEST})
    if unknown:
        raise ValueError(
            f"DeepSeek-V4.1 vision_config_overrides accepts only "
            f"{MAX_IMAGES_PER_REQUEST!r}, got {unknown}"
        )
    cap = overrides.get(MAX_IMAGES_PER_REQUEST)
    if cap is not None and (
        isinstance(cap, bool) or not isinstance(cap, int) or cap < 0
    ):
        raise ValueError(
            f"{MAX_IMAGES_PER_REQUEST} must be a non-negative integer, "
            f"got {cap!r}"
        )
    return cap


def forward_row_bound(
    max_num_input_tokens: int | None,
    max_batch_input_tokens: int,
    max_length: int,
) -> int:
    """Returns the most query rows one forward step carries.

    ``max_num_input_tokens`` when chunked prefill sets it; otherwise a
    prompt is encoded whole, up to ``max_length``.
    """
    return max_num_input_tokens or max(max_batch_input_tokens, max_length)


def dspark_draft_width(
    speculative: Any,
    target_huggingface_config: Any,
    draft_huggingface_config: Any,
) -> int:
    """Returns the DSpark draft width: the checkpoint's ``dspark_block_size``.

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
            f"was trained with); got num_speculative_tokens={requested}"
        )
    return block_size
