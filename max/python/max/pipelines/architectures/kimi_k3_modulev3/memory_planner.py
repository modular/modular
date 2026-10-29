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
three differences: no DSpark term, since that machinery doesn't exist for
this port; no KDA live-state term, since it's a page of the KV cache's
``state`` child rather than a separate reservation; and an unconditional
MXFP4 dequant reservation, since this port has no packed-GEMM compute path
yet to avoid it with -- over-reserving here, never an OOM.
"""

from __future__ import annotations

import logging
from typing import Any

from max.dtype import DType
from max.pipelines.kv_cache import cache_dtype_for_encoding
from max.pipelines.kv_cache.memory_planner import PagedMemoryPlanner
from max.pipelines.lib.config import PipelineConfig
from max.pipelines.lib.config.model_config import _select_quantization_encoding
from max.pipelines.modeling.config_enums import (
    SupportedEncoding,
    supported_encoding_dtype,
)
from max.support.human_readable_formatter import to_human_readable_bytes
from transformers import AutoConfig

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
) -> int:
    """GPU bytes one request occupies in the KDA state leaves, per device.

    Direct port of ``KimiK3Config.per_request_state_bytes``. The pools follow
    the head split, so this is what ONE device pays.
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
    return num_kda * (conv + rec)


def _infer_optimal_batch_size(
    devices: list[Any],
    *,
    weights_per_device: int,
    device_memory_utilization: float,
    per_request_bytes: int,
) -> int:
    """Returns a memory-safe default ``max_batch_size``.

    Direct port of ``KimiK3Config.infer_optimal_batch_size`` (its
    ``extra_per_request_bytes`` is always 0 here -- there is no speculative
    shadow to add it for). Falls back to 32 when the device query fails.
    """
    if per_request_bytes == 0:
        return 32
    try:
        free_bytes = min(
            (int(d.stats.get("free_memory", 0)) for d in devices), default=0
        )
    except Exception:
        free_bytes = 0
    if free_bytes <= 0:
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
        """Infers a ``max_batch_size`` that leaves room for the KDA state.

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
        num_kda, _ = _layer_counts(text_config)
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
        )
        weights_per_device = weights_size // n_devices if n_devices else 0
        return _infer_optimal_batch_size(
            devices,
            weights_per_device=weights_per_device,
            device_memory_utilization=(
                model_config.kv_cache.device_memory_utilization
            ),
            per_request_bytes=per_request_bytes,
        )

    def estimate_activation_memory(
        self,
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
    ) -> int:
        """Reserves the MXFP4 dequant and EP buffers, and the MLA prefill window.

        Compared against free memory SUMMED over every device, so the total
        is what belongs here -- not one device's share. The KDA state is not
        one of these terms: it is pages of the KV pool, so the cache sizing
        already holds it.
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
        _, num_full_attn = _layer_counts(text_config)
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

        if pipeline_config.runtime.pipeline_role != "decode_only":
            cache_dtype = cache_dtype_for_encoding(
                encoding, model_config.kv_cache.kv_cache_format
            )
            mla_latent_cache = _supports_mla_latent_cache(
                text_config, cache_dtype, n_devices
            )
            max_batch_context_length = (
                pipeline_config.runtime.max_batch_total_tokens
                or model_config.max_length
                or 0
            )
            total += _mla_prefill_scratch_bytes(
                text_config,
                mla_latent_cache=mla_latent_cache,
                num_full_attn=num_full_attn,
                max_batch_context_length=max_batch_context_length,
                dtype=dtype,
            )

        if total:
            logger.info(
                "Kimi K3 (ModuleV3) estimated activation memory: %s over %d"
                " device(s)",
                to_human_readable_bytes(total),
                n_devices,
            )
        return total
