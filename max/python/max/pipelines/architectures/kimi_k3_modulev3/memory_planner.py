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
"""Memory planner for Kimi K3's ModuleV3 (Tensor API) text tower.

Reads straight off ``pipeline_config``/``huggingface_config``, like
``kimik2_5_modulev3``'s planner, with no dependency on ``model_config.py``.

Ports the non-cache reservations from ``kimi_k3/memory_planner.py``, with
three differences: the DSpark term is the drafter's own MLA prefill window
(the graph-API planner priced a decompressed KV cache instead, which this
drafter no longer has); no KDA live-state term, since the verify ring and
the live state ride the KV cache's ``state`` child as pages rather than
separate reservations; and an unconditional MXFP4 dequant reservation, since
this port has no packed-GEMM compute path yet to avoid it with --
over-reserving here, never an OOM.

One reservation the graph-API planner does not carry: on gfx950 the
``kda_decode`` op runs KDA prefill through the chunk pipeline, whose launch
allocates workspaces per call outside the planner, so
:func:`_amd_chunk_workspace_bytes` reserves them here instead.
"""

from __future__ import annotations

import logging
from typing import Any

from max.driver import accelerator_api
from max.dtype import DType
from max.nn._graph_ops.state_space.kimi_delta import (
    amd_kda_chunk_workspace_bytes,
)
from max.pipelines.kv_cache import cache_dtype_for_encoding
from max.pipelines.kv_cache.memory_planner import PagedMemoryPlanner
from max.pipelines.lib.config import PipelineConfig
from max.pipelines.lib.config.model_config import _select_quantization_encoding
from max.pipelines.lib.memory_estimation import FreeMemory
from max.pipelines.modeling.config_enums import (
    SupportedEncoding,
    supported_encoding_dtype,
)
from max.support.human_readable_formatter import to_human_readable_bytes
from transformers import AutoConfig

from .spec_mla import (
    kimi_k3_mla_buffer_size,
    kimi_k3_mla_folds_on_decode_branch,
    kimi_k3_mla_table_rows,
    kimi_k3_spec_num_draft_tokens,
    kimi_k3_verify_ring_len,
)
from .state_cache import kda_state_regions

logger = logging.getLogger(__name__)

_DEFAULT_ENCODING: SupportedEncoding = "bfloat16"
_MXFP4_ENCODING = "float4_e2m1fnx2"

# `flareMLA_decoding` asserts `depth == 576` and the fused MLA graph ops are
# bf16/fp8 only; real K3 (512 + 64) lands here, a from-scratch fixture may not.
_MLA_KERNEL_LATENT_DIM = 576


def _text_config(huggingface_config: AutoConfig) -> Any:
    """Kimi K3's ``config.json`` nests every text field under ``text_config``."""
    return getattr(huggingface_config, "text_config", huggingface_config)


def _resolved_dtype(encoding: SupportedEncoding) -> DType:
    """The dtype dense (non-expert) weights and activations run at.

    ``float4_e2m1fnx2`` resolves to uint8, which is how the routed experts are
    *stored*, not the width the rest of the model runs at -- the checkpoint's
    ``ignore`` list keeps attention, the latent entry/exit, the norms and the
    shared experts at bfloat16.
    """
    if encoding == _MXFP4_ENCODING:
        return DType.bfloat16
    return supported_encoding_dtype(encoding)


def _layer_counts(text_config: Any) -> tuple[int, int]:
    """Returns ``(num_kda_layers, num_full_attn_layers)``.

    Only the counts, not the (1-indexed, in the checkpoint) layer lists
    themselves -- every formula below needs a count, never a specific index.
    """
    linear_cfg = getattr(text_config, "linear_attn_config", None) or {}
    num_kda = len(linear_cfg.get("kda_layers", []))
    num_full_attn = len(linear_cfg.get("full_attn_layers", []))
    return num_kda, num_full_attn


def _mla_latent_dim(text_config: Any) -> int:
    return text_config.kv_lora_rank + text_config.qk_rope_head_dim


def _supports_mla_latent_cache(
    text_config: Any, cache_dtype: DType, num_devices: int
) -> bool:
    """Whether the fused MLA kernels can serve this shape at this TP degree.

    Mirrors ``KimiK3Config.supports_mla_latent_cache``: false sends the model
    down the decompressed cache, which opens no prefill window at all (see
    ``_mla_prefill_scratch_bytes``).
    """
    heads = text_config.num_attention_heads
    if num_devices == 0 or heads % num_devices:
        return False
    per_device = heads // num_devices
    return (
        cache_dtype in (DType.bfloat16, DType.float8_e4m3fn)
        and _mla_latent_dim(text_config) == _MLA_KERNEL_LATENT_DIM
        and (per_device <= 64 or per_device % 64 == 0)
    )


def _routed_expert_row_bound(
    pipeline_config: PipelineConfig, max_seq_len: int
) -> int:
    """Upper bound on the tokens one forward carries, for the EP dispatch.

    Direct port of ``KimiK3Config._routed_expert_row_bound``: chunked
    prefill makes ``max_batch_input_tokens`` hard; without it,
    ``max_batch_total_tokens`` (or else ``max_seq_len * max_batch_size``)
    bounds it instead. Speculative decoding would widen it further, though
    this stays zero overhead without a speculative config.

    Raises:
        ValueError: When no budget bounds a context-encoding batch.
    """
    runtime = pipeline_config.runtime
    floor = runtime.max_batch_input_tokens
    if runtime.enable_chunked_prefill:
        charged = floor
    elif runtime.max_batch_total_tokens is not None:
        charged = max(floor, runtime.max_batch_total_tokens)
    elif runtime.max_batch_size is not None:
        charged = max(floor, max_seq_len * runtime.max_batch_size)
    else:
        raise ValueError(
            "Kimi K3: chunked prefill is disabled and neither"
            " max-batch-total-tokens nor max-batch-size is set, so nothing"
            " bounds a context-encoding batch and the routed experts' matmul"
            " cannot size its M grid. Re-enable chunked prefill, or set"
            " --max-batch-total-tokens."
        )
    speculative = pipeline_config.speculative
    num_speculative_tokens = (
        (speculative.num_speculative_tokens or 0)
        if speculative is not None
        else 0
    )
    max_requests = runtime.max_batch_size or floor
    return charged + num_speculative_tokens * max_requests


def _replicated_weight_bytes(
    text_config: Any,
    *,
    dtype: DType,
    num_kda: int,
    num_full_attn: int,
    expert_parallel: bool,
) -> int:
    """Checkpoint bytes every device holds its own full copy of.

    Direct port of ``KimiK3Config.replicated_weight_bytes``: the attention
    latent entries, the MoE entry projection and the router gate are
    replicated under either split; the MoE exit projection and the shared
    experts are replicated only under expert parallelism, which has no
    allreduce in the block for a partial output of either to ride on.
    """
    hidden = text_config.hidden_size
    elem = dtype.size_in_bytes
    linear_cfg = getattr(text_config, "linear_attn_config", None) or {}
    kda_head_dim = linear_cfg.get("head_dim", 32)
    elems = num_kda * kda_head_dim * hidden
    elems += (
        num_full_attn
        * hidden
        * (
            text_config.q_lora_rank
            + text_config.kv_lora_rank
            + text_config.qk_rope_head_dim
        )
    )
    moe_layers = max(
        0, text_config.num_hidden_layers - text_config.first_k_dense_replace
    )
    per_moe_layer = (
        text_config.routed_expert_hidden_size * hidden
        + text_config.num_experts * (hidden + 1)
    )
    if expert_parallel:
        per_moe_layer += (
            hidden * text_config.routed_expert_hidden_size
            + 3
            * hidden
            * text_config.moe_intermediate_size
            * text_config.num_shared_experts
        )
    return (elems + moe_layers * per_moe_layer) * elem


def _weights_bytes_per_device(
    weights_size: int, *, replicated: int, n_devices: int
) -> int:
    """Splits a whole-checkpoint byte count into what ONE device holds.

    Clamped so a config that disagrees with the measured files cannot drive
    the divided remainder negative.
    """
    replicated = min(weights_size, replicated)
    split = weights_size - replicated
    return split // n_devices + replicated


def _mxfp4_dequant_bytes_per_device(
    text_config: Any, *, dtype: DType, n_devices: int
) -> int:
    """GPU bytes per device the MXFP4 routed experts need once dequantized.

    Unconditional for this port -- see the module docstring's third bullet.
    Two layers' worth, since nothing forces one layer's dequant buffer to be
    freed before the next layer's is allocated.
    """
    per_layer = (
        text_config.num_experts
        * text_config.routed_expert_hidden_size
        * 3
        * (text_config.moe_intermediate_size // n_devices)
        * dtype.size_in_bytes
    )
    return 2 * per_layer


def _ep_comm_buffer_bytes_per_device(
    text_config: Any,
    *,
    dtype: DType,
    routed_expert_row_bound: int,
    ep_size: int,
    use_allreduce: bool,
) -> int:
    """GPU bytes ONE device holds the EP dispatch/combine buffers in.

    Direct port of ``KimiK3Config.ep_comm_buffer_bytes``, deliberately not
    ``estimate_ep_memory_usage``: see that method's docstring for why it
    mirrors ``EPConfig.get_max_recv_tokens()`` instead. Single-node only,
    which is what drops the combine SEND buffer.
    """
    if ep_size <= 1:
        return 0
    top_k = text_config.num_experts_per_token
    max_tokens = routed_expert_row_bound
    if use_allreduce:
        recv_rows = max_tokens * min(text_config.num_experts // ep_size, top_k)
    else:
        recv_rows = max_tokens * min(text_config.num_experts, ep_size * top_k)
    latent = text_config.routed_expert_hidden_size
    # No packed EP dispatch path in this port yet either -- the dispatch
    # always carries `dtype`, never a 1-byte packed element.
    return (max_tokens + recv_rows) * latent * dtype.size_in_bytes + (
        top_k * max_tokens * latent * dtype.size_in_bytes
    )


def _mla_prefill_scratch_bytes(
    text_config: Any,
    *,
    mla_latent_cache: bool,
    num_full_attn: int,
    max_batch_context_length: int,
    dtype: DType,
) -> int:
    """GPU bytes ALL devices together hold the MLA prefill window in.

    Direct port of ``KimiK3Config.mla_prefill_scratch_bytes``. Zero without
    the latent cache: the decompressed parity form has the heads in the
    cache already and never opens a window.
    """
    if not mla_latent_cache or not num_full_attn:
        return 0
    return (
        max_batch_context_length
        * text_config.num_attention_heads
        * (text_config.qk_nope_head_dim + text_config.v_head_dim)
        * dtype.size_in_bytes
    )


def _dspark_mla_prefill_scratch_bytes(
    pipeline_config: PipelineConfig,
    *,
    n_devices: int,
    cache_dtype: DType,
    max_seq_len: int,
    page_size: int,
    dtype: DType,
    inferred_max_batch_size: int | None,
) -> int:
    """GPU bytes ALL devices together hold the DSpark drafter's window in.

    The drafter's latent leaf opens the same one-shot decompression window
    the target's MLA layers do whenever its block call does not fold
    (``kimi_k3_mla_folds_on_decode_branch``, keyed on the block's own
    query-row count: on AMD a BF16 cache at any head count, an FP8 one
    above 16 query heads per device, or a block wider than the fold's
    row bound; off AMD the head and row counts) -- and it opens it on
    every decode step, so no pipeline-role gate applies. Bounded by the
    same draft-aware cap ``dspark/model_config.py`` sizes the drafter's
    plan with -- batch slots and all -- so the reservation tracks what
    the graph allocates. Zero on the fold arm, which reads no plan and
    opens no window: reserving there would eat batch capacity for
    nothing.

    The drafter's geometry reads straight off its checkpoint config, the
    way the rest of this module reads the target's: the dspark parser lives
    in a package that already depends on this one, so importing it here
    would be a dependency cycle.

    Raises:
        ValueError: If the block call does not fold and no batch size is
            resolved to price the window's worst case for.
    """
    # `getattr`: this module reads straight off ``pipeline_config``, which
    # the planner tests stub out field by field.
    speculative = getattr(pipeline_config, "speculative", None)
    draft_model = getattr(pipeline_config, "draft_model", None)
    if speculative is None or draft_model is None:
        return 0
    hf_config = draft_model.huggingface_config
    if hf_config is None or getattr(hf_config, "model_type", None) != (
        "k3_dspark"
    ):
        return 0
    heads = int(getattr(hf_config, "num_attention_heads", 0))
    nope = int(getattr(hf_config, "qk_nope_head_dim", 0))
    rope = int(getattr(hf_config, "qk_rope_head_dim", 0))
    kv_lora = int(getattr(hf_config, "kv_lora_rank", 0))
    v_head = int(getattr(hf_config, "v_head_dim", 0))
    if (
        not heads
        or kv_lora + rope != _MLA_KERNEL_LATENT_DIM
        or not nope + v_head
    ):
        # The pairing refuses a draft this far from the fused MLA kernels at
        # config time; a stub that never got that far prices no window.
        return 0
    per_device = heads // max(1, n_devices)
    if kimi_k3_mla_folds_on_decode_branch(
        per_device,
        cache_dtype,
        accelerator_api(),
        num_query_rows=speculative.draft_width,
    ):
        # The decode fold: the block call reads no plan, so no window.
        return 0
    max_batch_size = (
        pipeline_config.runtime.max_batch_size or inferred_max_batch_size
    )
    if max_batch_size is None:
        raise ValueError(
            "Kimi K3: a speculative build prices the DSpark drafter's MLA"
            " plan window for a worst case over every batch slot, but"
            " max_batch_size is unresolved; pass --max-batch-size or let"
            " memory planning infer one."
        )
    cap = kimi_k3_mla_buffer_size(
        max_batch_context_length=(
            pipeline_config.runtime.max_batch_total_tokens or max_seq_len
        ),
        max_seq_len=max_seq_len,
        page_size=page_size,
        # The block call's window is its ``cache_length`` -- committed
        # context plus up to ``draft_width + 1`` accepted rows -- plus the
        # block's own ``draft_width`` rows: the same ``2 * draft_width + 1``
        # slack the pool reserves (``kv_cache/paged_kv_cache/_common.py``),
        # and the width ``dspark/model_config.py`` sizes the drafter's own
        # cap with (``draft_width`` is the draft checkpoint's
        # ``block_size``); the helper counts ``1 + num_draft_tokens``.
        num_draft_tokens=2 * speculative.draft_width,
        max_batch_size=max_batch_size,
        num_q_heads_per_device=per_device,
        cache_dtype=cache_dtype,
        device_api=accelerator_api(),
    )
    return cap * heads * (nope + v_head) * dtype.size_in_bytes


def _declared_state_dtypes(
    kv_cache_config: Any, encoding: SupportedEncoding
) -> tuple[DType, DType]:
    """Returns ``(conv, recurrent)`` storage dtypes of the KDA state.

    Direct port of ``KimiK3Config.declared_state_dtypes``.

    Raises:
        ValueError: If ``--state-pool-dtype`` names a dtype the pools cannot
            hold.
    """
    conv = DType.float32 if encoding == "float32" else DType.bfloat16
    pool_dtype_name = kv_cache_config.state_pool_dtype
    pool_dtypes = {"bfloat16": DType.bfloat16, "float32": DType.float32}
    if pool_dtype_name is not None and pool_dtype_name not in pool_dtypes:
        raise ValueError(
            "Kimi K3 state_pool_dtype must be 'bfloat16' or 'float32', got"
            f" {pool_dtype_name!r}"
        )
    recurrent = (
        DType.float32
        if pool_dtype_name is None
        else pool_dtypes[pool_dtype_name]
    )
    return conv, recurrent


def _per_request_state_bytes(
    text_config: Any,
    *,
    n_devices: int,
    ssm_dtype: DType,
    recurrent_state_dtype: DType,
    num_kda: int,
    ring_len: int = 0,
) -> int:
    """GPU bytes one request occupies in the KDA state leaves, per device.

    Direct port of ``KimiK3Config.per_request_state_bytes``, plus the
    speculative verify ring a speculative build declares: one scratch page
    per request, sized off the same region declaration the cache builds
    from. The pools follow the head split, so this is what ONE device pays.
    """
    if not num_kda:
        return 0
    linear_cfg = getattr(text_config, "linear_attn_config", None) or {}
    num_heads = linear_cfg.get("num_heads", 8)
    head_dim = linear_cfg.get("head_dim", 32)
    conv_kernel_size = linear_cfg.get("short_conv_kernel_size", 4)
    conv_dim = 3 * num_heads * head_dim
    conv = (
        (conv_dim // n_devices)
        * (conv_kernel_size - 1)
        * ssm_dtype.size_in_bytes
    )
    rec = (
        (num_heads // n_devices)
        * head_dim
        * head_dim
        * recurrent_state_dtype.size_in_bytes
    )
    live_bytes = num_kda * (conv + rec)
    if not ring_len:
        return live_bytes
    regions = kda_state_regions(
        num_kda_layers=num_kda,
        conv_dim=conv_dim,
        conv_kernel_size=conv_kernel_size,
        num_heads=num_heads,
        head_dim=head_dim,
        conv_dtype=ssm_dtype,
        recurrent_dtype=recurrent_state_dtype,
        num_devices=n_devices,
        ring_len=ring_len,
    )
    ring = regions[-1].bytes_per_page
    return live_bytes + ring


def _amd_chunk_workspace_bytes(
    pipeline_config: PipelineConfig,
    text_config: Any,
    *,
    n_devices: int,
    ep_size: int,
    num_kda: int,
    recurrent_state_dtype: DType,
    inferred_max_batch_size: int | None,
) -> int:
    """GPU bytes ALL devices together hold the gfx950 chunk workspaces in.

    KDA prefill runs the chunk pipeline inside the ``kda_decode`` op on
    gfx950 -- the bounded gate, a float32 pool -- and that launch allocates
    its workspaces per call, outside this planner. TP splits the KDA heads across devices while the pure-EP
    mesh replicates the layers, so one device holds ``num_heads //
    n_devices`` heads under TP and all of them under EP; the estimate this
    feeds is compared against free memory summed over devices, so the
    reservation is one device's bytes times the device count. Registration
    cannot be asked here -- it reads the graph under construction -- so a
    build without the op over-reserves, never OOMs.
    """
    if not num_kda:
        return 0
    linear_cfg = getattr(text_config, "linear_attn_config", None) or {}
    runtime = pipeline_config.runtime
    heads = linear_cfg.get("num_heads", 8)
    return n_devices * amd_kda_chunk_workspace_bytes(
        num_heads=heads // n_devices if ep_size == 1 else heads,
        head_dim=linear_cfg.get("head_dim", 32),
        lower_bound=linear_cfg.get("gate_lower_bound"),
        state_dtype=recurrent_state_dtype,
        decode_only=runtime.pipeline_role == "decode_only",
        max_batch_input_tokens=runtime.max_batch_input_tokens,
        enable_chunked_prefill=runtime.enable_chunked_prefill,
        max_batch_size=(runtime.max_batch_size or inferred_max_batch_size),
        max_length=pipeline_config.model.max_length,
    )


def _infer_optimal_batch_size(
    devices: list[Any],
    *,
    weights_per_device: int,
    device_memory_utilization: float,
    per_request_bytes: int,
) -> int:
    """Returns a memory-safe default ``max_batch_size``.

    Direct port of ``KimiK3Config.infer_optimal_batch_size``. The verify
    ring a speculative build declares rides ``per_request_bytes`` -- it
    is priced in :func:`_per_request_state_bytes` -- not a separate
    ``extra_per_request_bytes`` term. Falls back to 32 when the device
    query fails.
    """
    if per_request_bytes == 0:
        return 32
    try:
        free_bytes = FreeMemory.smallest(devices).hi
    except Exception:
        return 32
    budget = int(free_bytes * device_memory_utilization) - weights_per_device
    if budget <= 0:
        return 1
    return min(512, max(1, (budget // 2) // per_request_bytes))


# The tower's stored dtype, when the checkpoint declares none. Every
# published K3 checkpoint is bfloat16.
_DEFAULT_STORED_DTYPE = DType.bfloat16
_STORED_DTYPES = {
    "bfloat16": DType.bfloat16,
    "float32": DType.float32,
}


def _stored_dtype(huggingface_config: AutoConfig) -> DType:
    """Returns the dtype the checkpoint's tensors are stored at.

    NOT the resolved quantization encoding -- see
    ``kimi_k3.memory_planner._stored_dtype`` for why the discount has to be
    denominated in the dtype on disk.
    """
    declared = getattr(huggingface_config, "torch_dtype", None) or getattr(
        huggingface_config, "dtype", None
    )
    return _STORED_DTYPES.get(str(declared), _DEFAULT_STORED_DTYPE)


def vision_tower_bytes(huggingface_config: AutoConfig) -> int:
    """Checkpoint bytes for the vision tower this text-only slice never loads.

    Direct port of ``kimi_k3.memory_planner.vision_tower_bytes``; there is no
    vision code in this port at all (see ``k3_grounding.md``), so every tower
    tensor is dead weight in ``weights_size()``'s file total.

    Raises:
        ValueError: If ``vision_config`` is present but carries none of the
            names a MoonViT tower is sized by.
    """
    vision_config = getattr(huggingface_config, "vision_config", None)
    if vision_config is None:
        return 0
    if (
        getattr(vision_config, "vt_hidden_size", None) is None
        and getattr(vision_config, "hidden_size", None) is None
    ):
        # No tower to price. The kimi_k3 config shim synthesizes an empty
        # ``vision_config`` for a checkpoint that declares none (the
        # text-only releases, and the tiny test pair), which is a presence
        # signal for nothing. A deliberate plain-K3 planning change (such a
        # config previously fell through to the tower sizing below); the
        # PR description calls it out.
        return 0

    def _field(name: str, fallback: str) -> int:
        value = getattr(vision_config, name, None)
        if value is None:
            value = getattr(vision_config, fallback, None)
        if value is None:
            raise ValueError(
                f"Kimi K3 vision_config declares neither {name!r} nor"
                f" {fallback!r}; the tower discount cannot be sized."
            )
        return int(value)

    hidden = _field("vt_hidden_size", "hidden_size")
    intermediate = _field("vt_intermediate_size", "intermediate_size")
    num_layers = _field("vt_num_hidden_layers", "num_hidden_layers")
    patch_size = _field("patch_size", "vt_patch_size")

    # K3's `qkv_hidden_size` (1536) is decoupled from `vt_hidden_size` (256),
    # so attention is NOT K2.5's 4*h^2: it is wqkv [3*qkv_h, h] + wo [h, qkv_h].
    qkv_hidden = int(getattr(vision_config, "qkv_hidden_size", None) or hidden)
    per_layer = (
        4 * qkv_hidden * hidden  # wqkv (3x) + wo
        + 2 * hidden * intermediate  # mlp fc1 + fc2
        + 2 * hidden  # norm0 + norm1
    )
    in_channels = int(getattr(vision_config, "num_channels", 3))
    pos_emb_grid = int(getattr(vision_config, "init_pos_emb_height", 0)) * int(
        getattr(vision_config, "init_pos_emb_width", 0)
    )
    elems = (
        num_layers * per_layer
        + hidden * in_channels * patch_size**2  # patch_embed.proj
        + pos_emb_grid * hidden  # patch_embed.pos_emb
        + hidden  # final_layernorm
    )
    return elems * _stored_dtype(huggingface_config).size_in_bytes


class KimiK3ModuleV3MemoryPlanner(PagedMemoryPlanner):
    """Reserves what the KV cache does not hold, for Kimi K3's ModuleV3 port.

    See the module docstring for how this differs from the Graph-API
    ``KimiK3MemoryPlanner``.
    """

    _inferred_max_batch_size: int | None = None
    """The batch size :meth:`infer_max_batch_size` chose, which planning does
    not write back to ``runtime.max_batch_size`` before it asks
    :meth:`estimate_activation_memory`."""

    def estimate_weights_size(self, pipeline_config: PipelineConfig) -> int:
        """Counts only the language weights this architecture loads.

        Discounts the vision tower, then charges the replicated weights
        (attention latents, MoE entry/router, and -- under expert
        parallelism -- the exit projection and shared experts) to every
        device rather than dividing the checkpoint flat.
        """
        model_config = pipeline_config.model
        weights_size = model_config.weights_size()
        huggingface_config = model_config.huggingface_config
        if huggingface_config is None:
            return weights_size
        # Clamp: the discount is analytical while `weights_size` is measured,
        # so a config that disagrees with the files must not drive the
        # estimate negative.
        text_only = max(
            0, weights_size - vision_tower_bytes(huggingface_config)
        )

        text_config = _text_config(huggingface_config)
        n_devices = len(model_config.device_specs)
        encoding = _select_quantization_encoding(
            model_config, _DEFAULT_ENCODING
        )
        dtype = _resolved_dtype(encoding)
        num_kda, num_full_attn = _layer_counts(text_config)
        ep_size = max(pipeline_config.runtime.ep_size, 1)

        replicated = _replicated_weight_bytes(
            text_config,
            dtype=dtype,
            num_kda=num_kda,
            num_full_attn=num_full_attn,
            expert_parallel=ep_size > 1,
        )
        per_device = _weights_bytes_per_device(
            text_only, replicated=replicated, n_devices=n_devices
        )
        return n_devices * per_device

    def infer_max_batch_size(
        self,
        pipeline_config: PipelineConfig,
        devices: list[Any],
        weights_size: int,
    ) -> int | None:
        """Infers a batch size that leaves room for KDA state and MLA verify.

        ``weights_size`` is this planner's own cluster-wide
        :meth:`estimate_weights_size`, so dividing it by the device count
        recovers the same per-device figure that estimate used -- no need to
        recompute ``replicated_weight_bytes`` a second time.
        """
        model_config = pipeline_config.model
        huggingface_config = model_config.huggingface_config
        if huggingface_config is None:
            return None
        text_config = _text_config(huggingface_config)
        num_kda, num_full_attn = _layer_counts(text_config)
        if not num_kda:
            return None
        n_devices = len(devices)
        encoding = _select_quantization_encoding(
            model_config, _DEFAULT_ENCODING
        )
        ssm_dtype, recurrent_state_dtype = _declared_state_dtypes(
            model_config.kv_cache, encoding
        )
        per_request_bytes = _per_request_state_bytes(
            text_config,
            n_devices=n_devices,
            ssm_dtype=ssm_dtype,
            recurrent_state_dtype=recurrent_state_dtype,
            num_kda=num_kda,
            ring_len=kimi_k3_verify_ring_len(pipeline_config),
        )
        if pipeline_config.speculative is not None and num_full_attn:
            cache_dtype = cache_dtype_for_encoding(
                encoding, model_config.kv_cache.kv_cache_format
            )
            draft_tokens = kimi_k3_spec_num_draft_tokens(pipeline_config)
            if _supports_mla_latent_cache(
                text_config, cache_dtype, n_devices
            ) and not kimi_k3_mla_folds_on_decode_branch(
                text_config.num_attention_heads // n_devices,
                cache_dtype,
                accelerator_api(),
                draft_tokens + 1,
            ):
                per_request_bytes += (
                    kimi_k3_mla_table_rows(
                        max_seq_len=model_config.max_length or 0,
                        page_size=model_config.kv_cache.kv_cache_page_size,
                        num_draft_tokens=draft_tokens,
                    )
                    * (text_config.num_attention_heads // n_devices)
                    * (text_config.qk_nope_head_dim + text_config.v_head_dim)
                    * _resolved_dtype(encoding).size_in_bytes
                )
        weights_per_device = weights_size // n_devices if n_devices else 0
        inferred = _infer_optimal_batch_size(
            devices,
            weights_per_device=weights_per_device,
            device_memory_utilization=(
                model_config.kv_cache.device_memory_utilization
            ),
            per_request_bytes=per_request_bytes,
        )
        self._inferred_max_batch_size = inferred
        return inferred

    def estimate_activation_memory(
        self,
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
    ) -> int:
        """Reserves the MXFP4 dequant and EP buffers, and the MLA prefill window.

        Compared against free memory SUMMED over every device, so the total
        is what belongs here -- not one device's share. The KDA state is not
        one of these terms: it is pages of the KV pool, so the cache sizing
        already holds it. On gfx950 the KDA
        chunk-pipeline workspaces are activation too;
        :func:`_amd_chunk_workspace_bytes` reserves them.

        Raises:
            ValueError: Under speculation, when no batch size is resolved to
                price an MLA plan window's worst case for -- the target's,
                or the DSpark drafter's, which is priced at every pipeline
                role.
        """
        if huggingface_config is None:
            return 0
        model_config = pipeline_config.model
        text_config = _text_config(huggingface_config)
        n_devices = len(model_config.device_specs)
        encoding = _select_quantization_encoding(
            model_config, _DEFAULT_ENCODING
        )
        dtype = _resolved_dtype(encoding)
        num_kda, num_full_attn = _layer_counts(text_config)
        ep_size = max(pipeline_config.runtime.ep_size, 1)
        max_seq_len = model_config.max_length or 0
        routed_expert_row_bound = _routed_expert_row_bound(
            pipeline_config, max_seq_len
        )

        per_device = 0
        if encoding == _MXFP4_ENCODING:
            per_device += _mxfp4_dequant_bytes_per_device(
                text_config, dtype=dtype, n_devices=n_devices
            )
        per_device += _ep_comm_buffer_bytes_per_device(
            text_config,
            dtype=dtype,
            routed_expert_row_bound=routed_expert_row_bound,
            ep_size=ep_size,
            use_allreduce=pipeline_config.runtime.ep_use_allreduce,
        )
        total = n_devices * per_device

        cache_dtype = cache_dtype_for_encoding(
            encoding, model_config.kv_cache.kv_cache_format
        )
        draft_tokens = kimi_k3_spec_num_draft_tokens(pipeline_config)
        # A decode-only node never prefills, so the window is live there
        # only when a speculative verify misses the decode fold and takes
        # the prefill arm, which allocates it inside the op.
        verify_reads_plan = (
            draft_tokens > 0
            and not kimi_k3_mla_folds_on_decode_branch(
                text_config.num_attention_heads // max(1, n_devices),
                cache_dtype,
                accelerator_api(),
                draft_tokens + 1,
            )
        )
        target_window_bytes = 0
        if (
            pipeline_config.runtime.pipeline_role != "decode_only"
            or verify_reads_plan
        ):
            mla_latent_cache = _supports_mla_latent_cache(
                text_config, cache_dtype, n_devices
            )
            max_batch_context_length = (
                pipeline_config.runtime.max_batch_total_tokens
                or model_config.max_length
                or 0
            )
            if pipeline_config.speculative is not None:
                # The same draft-aware cap the pairing builds its plan and
                # rope table with (`pipeline_model._spec_mla_buffer_size`),
                # so the reservation tracks what the graph allocates. The
                # pairing sizes from the memory plan's clamped length
                # (`context.max_seq_len`); this estimate runs before the
                # plan exists and so reads the pre-clamp
                # ``model_config.max_length``, which the plan only ever
                # lowers (`memory_estimation.plan_from_sizes` clamps by
                # free memory and the draft's limit), so the reservation
                # always covers the graph's allocation.
                batch_size = (
                    pipeline_config.runtime.max_batch_size
                    or self._inferred_max_batch_size
                )
                if batch_size is None:
                    raise ValueError(
                        "Kimi K3: a speculative build prices the MLA plan"
                        " window for a worst case over every batch slot, but"
                        " max_batch_size is unresolved; pass"
                        " --max-batch-size or let memory planning infer"
                        " one."
                    )
                max_batch_context_length = kimi_k3_mla_buffer_size(
                    max_batch_context_length=max_batch_context_length,
                    max_seq_len=max_seq_len,
                    page_size=model_config.kv_cache.kv_cache_page_size,
                    num_draft_tokens=draft_tokens,
                    max_batch_size=batch_size,
                    num_q_heads_per_device=(
                        text_config.num_attention_heads // max(1, n_devices)
                    ),
                    cache_dtype=cache_dtype,
                    device_api=accelerator_api(),
                )
            target_window_bytes = _mla_prefill_scratch_bytes(
                text_config,
                mla_latent_cache=mla_latent_cache,
                num_full_attn=num_full_attn,
                max_batch_context_length=max_batch_context_length,
                dtype=dtype,
            )

        # The target and drafter windows run in separate layers.
        dspark_window_bytes = _dspark_mla_prefill_scratch_bytes(
            pipeline_config,
            n_devices=n_devices,
            cache_dtype=cache_dtype,
            max_seq_len=max_seq_len,
            page_size=model_config.kv_cache.kv_cache_page_size,
            dtype=dtype,
            inferred_max_batch_size=self._inferred_max_batch_size,
        )
        total += max(target_window_bytes, dspark_window_bytes)

        _, recurrent_state_dtype = _declared_state_dtypes(
            model_config.kv_cache, encoding
        )
        total += _amd_chunk_workspace_bytes(
            pipeline_config,
            text_config,
            n_devices=n_devices,
            ep_size=ep_size,
            num_kda=num_kda,
            recurrent_state_dtype=recurrent_state_dtype,
            inferred_max_batch_size=self._inferred_max_batch_size,
        )

        if total:
            logger.info(
                "Kimi K3 (ModuleV3) estimated activation memory: %s over %d"
                " device(s)",
                to_human_readable_bytes(total),
                n_devices,
            )
        return total
