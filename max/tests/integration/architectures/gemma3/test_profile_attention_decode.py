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

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, TensorValue, ops
from max.interfaces import RequestID, TokenBuffer
from max.kv_cache import PagedKVCacheManager
from max.nn.attention import MHAMaskVariant
from max.nn.kernels import (
    KVCacheParams,
    flash_attention_ragged,
    fused_qk_ragged_rope,
    fused_qkv_ragged_matmul,
    k_rms_norm_rope_ragged,
    q_rms_norm_rope_ragged,
    rms_norm_key_cache,
    rope_k_cache_ragged,
    rope_ragged,
)
from max.nn.kv_cache import PagedCacheValues, unflatten_ragged_attention_inputs
from max.nn.rotary_embedding import Llama3RotaryEmbedding
from max.pipelines.architectures.gemma3.layers.attention import (
    Gemma3Attention as MaxGemma3Attention,
)
from max.pipelines.core import TextContext
from torch.utils.dlpack import from_dlpack


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value is not None else default


def _env_int_tuple(name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    value = os.environ.get(name)
    if value is None:
        return default
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


PAGE_SIZE = 128
EPS = 1e-6
DEFAULT_LAYER_IDX = 0
DEFAULT_LAYER_TYPE = "local"
DEFAULT_FUSED_VARIANT = "full"
DEFAULT_TIMING_ORDER_MODE = "baseline_then_fused"
FUSED_VARIANTS = ("full", "k_only", "q_only")
GRAPH_VARIANTS = (
    "baseline",
    "full",
    "k_only",
    "q_only_baseline",
    "q_only",
)
TIMING_ORDER_MODES = (
    DEFAULT_TIMING_ORDER_MODE,
    "both_orders",
)
Q_NORM_STD = 0.68
K_NORM_STD = 0.793
Q_PROJ_STD = 0.0284
K_PROJ_STD = 0.0309
V_PROJ_STD = 0.0309
O_PROJ_STD = 0.0237
DEFAULT_WARMUP_ITERS = 20
DEFAULT_TIMED_ITERS = 50
DEFAULT_CACHE_LEN_BASE = 1024
DEFAULT_CACHE_LEN_STEP = 7
DEFAULT_BATCH_SIZES = (64, 128)
WARMUP_ITERS = _env_int(
    "PROFILE_ATTENTION_DECODE_WARMUP_ITERS", DEFAULT_WARMUP_ITERS
)
TIMED_ITERS = _env_int(
    "PROFILE_ATTENTION_DECODE_TIMED_ITERS", DEFAULT_TIMED_ITERS
)
CACHE_LEN_BASE = _env_int(
    "PROFILE_ATTENTION_DECODE_CACHE_LEN_BASE", DEFAULT_CACHE_LEN_BASE
)
CACHE_LEN_STEP = _env_int(
    "PROFILE_ATTENTION_DECODE_CACHE_LEN_STEP", DEFAULT_CACHE_LEN_STEP
)
BATCH_SIZES = _env_int_tuple(
    "PROFILE_ATTENTION_DECODE_BATCH_SIZES", DEFAULT_BATCH_SIZES
)
if not BATCH_SIZES:
    raise ValueError(
        "PROFILE_ATTENTION_DECODE_BATCH_SIZES must define at least one batch size"
    )
if WARMUP_ITERS < 0:
    raise ValueError("PROFILE_ATTENTION_DECODE_WARMUP_ITERS must be >= 0")
if TIMED_ITERS <= 0:
    raise ValueError("PROFILE_ATTENTION_DECODE_TIMED_ITERS must be >= 1")
MAX_EXTRA_STEPS = WARMUP_ITERS + TIMED_ITERS


def _resolve_layer_metadata(config: dict[str, Any]) -> tuple[int, str]:
    layer_idx = int(
        os.environ.get("PROFILE_ATTENTION_LAYER_IDX", str(DEFAULT_LAYER_IDX))
    )
    num_hidden_layers = int(config["num_hidden_layers"])
    assert 0 <= layer_idx < num_hidden_layers, (
        f"layer_idx={layer_idx} must be in [0, {num_hidden_layers})"
    )
    layer_type = (
        "local"
        if bool((layer_idx + 1) % config["sliding_window_pattern"])
        else "global"
    )
    expected_layer_type = os.environ.get(
        "PROFILE_ATTENTION_LAYER_TYPE", DEFAULT_LAYER_TYPE
    )
    assert layer_type == expected_layer_type, (
        f"expected {expected_layer_type} layer, got {layer_type} for layer_idx={layer_idx}"
    )
    return layer_idx, layer_type


def _resolve_kv_num_layers(config: dict[str, Any], layer_idx: int) -> int:
    min_num_layers = layer_idx + 1
    kv_num_layers = int(
        os.environ.get("PROFILE_ATTENTION_KV_NUM_LAYERS", str(min_num_layers))
    )
    assert kv_num_layers >= min_num_layers, (
        f"kv_num_layers={kv_num_layers} must cover layer_idx={layer_idx}"
    )
    assert kv_num_layers <= int(config["num_hidden_layers"]), (
        "kv_num_layers cannot exceed the model layer count "
        f"({config['num_hidden_layers']})"
    )
    return kv_num_layers


def _resolve_fused_variant() -> str:
    fused_variant = os.environ.get(
        "PROFILE_ATTENTION_DECODE_FUSED_VARIANT",
        DEFAULT_FUSED_VARIANT,
    )
    assert fused_variant in FUSED_VARIANTS, (
        f"fused variant must be one of {FUSED_VARIANTS}, got {fused_variant!r}"
    )
    return fused_variant


def _resolve_graph_variants() -> tuple[str, str]:
    baseline_graph_variant = os.environ.get(
        "PROFILE_ATTENTION_DECODE_BASELINE_GRAPH_VARIANT"
    )
    fused_graph_variant = os.environ.get(
        "PROFILE_ATTENTION_DECODE_FUSED_GRAPH_VARIANT"
    )

    if baseline_graph_variant is None and fused_graph_variant is None:
        fused_variant = _resolve_fused_variant()
        baseline_graph_variant = (
            "q_only_baseline" if fused_variant == "q_only" else "baseline"
        )
        fused_graph_variant = {
            "full": "full",
            "k_only": "k_only",
            "q_only": "q_only",
        }[fused_variant]
    else:
        assert baseline_graph_variant is not None, (
            "PROFILE_ATTENTION_DECODE_BASELINE_GRAPH_VARIANT must be set "
            "whenever PROFILE_ATTENTION_DECODE_FUSED_GRAPH_VARIANT is set"
        )
        assert fused_graph_variant is not None, (
            "PROFILE_ATTENTION_DECODE_FUSED_GRAPH_VARIANT must be set "
            "whenever PROFILE_ATTENTION_DECODE_BASELINE_GRAPH_VARIANT is set"
        )

    assert baseline_graph_variant in GRAPH_VARIANTS, (
        "baseline graph variant must be one of "
        f"{GRAPH_VARIANTS}, got {baseline_graph_variant!r}"
    )
    assert fused_graph_variant in GRAPH_VARIANTS, (
        "fused graph variant must be one of "
        f"{GRAPH_VARIANTS}, got {fused_graph_variant!r}"
    )
    return baseline_graph_variant, fused_graph_variant


def _resolve_timing_order_mode() -> str:
    timing_order_mode = os.environ.get(
        "PROFILE_ATTENTION_DECODE_TIMING_ORDER",
        DEFAULT_TIMING_ORDER_MODE,
    ).strip()
    if timing_order_mode in ("", DEFAULT_TIMING_ORDER_MODE):
        return DEFAULT_TIMING_ORDER_MODE
    assert timing_order_mode in TIMING_ORDER_MODES, (
        "PROFILE_ATTENTION_DECODE_TIMING_ORDER must be one of "
        f"{TIMING_ORDER_MODES}, got {timing_order_mode!r}"
    )
    return timing_order_mode


def _comparison_metadata(
    baseline_graph_variant: str, fused_graph_variant: str
) -> tuple[str, str]:
    comparison = (baseline_graph_variant, fused_graph_variant)
    if comparison == ("baseline", "full"):
        return "decode-ragged-full-attention", ""
    if comparison == ("baseline", "k_only"):
        return "decode-ragged-full-attention-k-only", "_k_only"
    if comparison == ("q_only_baseline", "q_only"):
        return "decode-ragged-full-attention-q-only", "_q_only"
    if comparison == ("q_only", "full"):
        return "decode-ragged-full-attention-qk-incremental", "_qk_incremental"

    comparison_tag = (
        f"{baseline_graph_variant.replace('_', '-')}-vs-"
        f"{fused_graph_variant.replace('_', '-')}"
    )
    return (
        f"decode-ragged-full-attention-{comparison_tag}",
        f"_{comparison_tag.replace('-', '_')}",
    )


def _timing_order_metadata(
    comparison_mode: str,
    run_name_suffix: str,
    timing_order_mode: str,
) -> tuple[str, str]:
    if timing_order_mode == DEFAULT_TIMING_ORDER_MODE:
        return comparison_mode, run_name_suffix
    return f"{comparison_mode}-order-probe", f"{run_name_suffix}_order_probe"


def _load_text_config() -> dict[str, Any]:
    config_path = Path(os.environ["PIPELINES_TESTDATA"]) / "config.json"
    with open(config_path) as file:
        data = json.load(file)
    return data.get("text_config", data)


def _make_weight_registry(config: dict[str, Any]) -> dict[str, torch.Tensor]:
    torch.manual_seed(42)
    q_dim = config["head_dim"] * config["num_attention_heads"]
    kv_dim = config["head_dim"] * config["num_key_value_heads"]
    hidden_size = config["hidden_size"]
    return {
        "k_norm.weight": (
            torch.randn(config["head_dim"], dtype=torch.bfloat16) * K_NORM_STD
        ),
        "k_proj.weight": (
            torch.randn(kv_dim, hidden_size, dtype=torch.bfloat16) * K_PROJ_STD
        ),
        "o_proj.weight": (
            torch.randn(hidden_size, q_dim, dtype=torch.bfloat16) * O_PROJ_STD
        ),
        "q_norm.weight": (
            torch.randn(config["head_dim"], dtype=torch.bfloat16) * Q_NORM_STD
        ),
        "q_proj.weight": (
            torch.randn(q_dim, hidden_size, dtype=torch.bfloat16) * Q_PROJ_STD
        ),
        "v_proj.weight": (
            torch.randn(kv_dim, hidden_size, dtype=torch.bfloat16) * V_PROJ_STD
        ),
    }


class _Gemma3AttentionBaseline(MaxGemma3Attention):
    def __call__(
        self,
        x: TensorValue,
        kv_collection: PagedCacheValues,
        **kwargs,
    ) -> TensorValue:
        total_seq_len = x.shape[0]
        layer_idx = ops.constant(
            self.layer_idx, DType.uint32, device=DeviceRef.CPU()
        )

        xq = fused_qkv_ragged_matmul(
            self.kv_params,
            input=x,
            wqkv=self.wqkv,
            bias=self.wqkv_bias,
            input_row_offsets=kwargs["input_row_offsets"],
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            n_heads=self.n_heads,
        )
        xq = xq.reshape((-1, self.n_heads, self.kv_params.head_dim))

        use_local = bool((self.layer_idx + 1) % self.sliding_window_pattern)
        rope = self.rope_local if use_local else self.rope_global
        freqs_cis = ops.cast(rope.freqs_cis, xq.dtype).to(xq.device)

        rms_norm_key_cache(
            self.kv_params,
            kv_collection=kv_collection,
            gamma=self.k_norm.weight.cast(self.kv_params.dtype).to(
                self.devices[0]
            ),
            epsilon=self.qk_norm_eps,
            layer_idx=layer_idx,
            total_seq_len=total_seq_len,
            input_row_offsets=kwargs["input_row_offsets"],
            weight_offset=1.0,
        )
        xq = self.q_norm(xq)
        xq = fused_qk_ragged_rope(
            self.kv_params,
            xq,
            kwargs["input_row_offsets"],
            kv_collection,
            freqs_cis,
            layer_idx,
            interleaved=rope.interleaved,
        )

        mask_variant = (
            MHAMaskVariant.SLIDING_WINDOW_CAUSAL_MASK
            if use_local
            else MHAMaskVariant.CAUSAL_MASK
        )
        attn_out = flash_attention_ragged(
            self.kv_params,
            input=xq,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            input_row_offsets=kwargs["input_row_offsets"],
            mask_variant=mask_variant,
            scale=self.scale,
            local_window_size=self.local_window_size,
        )
        attn_out = ops.reshape(attn_out, shape=[total_seq_len, -1])
        return self.o_proj(attn_out)


class _Gemma3AttentionKOnlyFused(MaxGemma3Attention):
    def __call__(
        self,
        x: TensorValue,
        kv_collection: PagedCacheValues,
        **kwargs,
    ) -> TensorValue:
        total_seq_len = x.shape[0]
        layer_idx = ops.constant(
            self.layer_idx, DType.uint32, device=DeviceRef.CPU()
        )

        xq = fused_qkv_ragged_matmul(
            self.kv_params,
            input=x,
            wqkv=self.wqkv,
            bias=self.wqkv_bias,
            input_row_offsets=kwargs["input_row_offsets"],
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            n_heads=self.n_heads,
        )
        xq = xq.reshape((-1, self.n_heads, self.kv_params.head_dim))

        use_local = bool((self.layer_idx + 1) % self.sliding_window_pattern)
        rope = self.rope_local if use_local else self.rope_global
        freqs_cis = ops.cast(rope.freqs_cis, xq.dtype).to(xq.device)

        k_rms_norm_rope_ragged(
            self.kv_params,
            total_seq_len=total_seq_len,
            input_row_offsets=kwargs["input_row_offsets"],
            kv_collection=kv_collection,
            freqs_cis=freqs_cis,
            gamma=self.k_norm.weight.cast(self.kv_params.dtype).to(
                self.devices[0]
            ),
            epsilon=self.qk_norm_eps,
            layer_idx=layer_idx,
            weight_offset=1.0,
            interleaved=rope.interleaved,
        )
        xq = self.q_norm(xq)
        xq = rope_ragged(
            xq,
            kwargs["input_row_offsets"],
            kv_collection.cache_lengths,
            freqs_cis,
            interleaved=rope.interleaved,
        )

        mask_variant = (
            MHAMaskVariant.SLIDING_WINDOW_CAUSAL_MASK
            if use_local
            else MHAMaskVariant.CAUSAL_MASK
        )
        attn_out = flash_attention_ragged(
            self.kv_params,
            input=xq,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            input_row_offsets=kwargs["input_row_offsets"],
            mask_variant=mask_variant,
            scale=self.scale,
            local_window_size=self.local_window_size,
        )
        attn_out = ops.reshape(attn_out, shape=[total_seq_len, -1])
        return self.o_proj(attn_out)


class _Gemma3AttentionQOnlyBaseline(MaxGemma3Attention):
    def __call__(
        self,
        x: TensorValue,
        kv_collection: PagedCacheValues,
        **kwargs,
    ) -> TensorValue:
        total_seq_len = x.shape[0]
        layer_idx = ops.constant(
            self.layer_idx, DType.uint32, device=DeviceRef.CPU()
        )

        xq = fused_qkv_ragged_matmul(
            self.kv_params,
            input=x,
            wqkv=self.wqkv,
            bias=self.wqkv_bias,
            input_row_offsets=kwargs["input_row_offsets"],
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            n_heads=self.n_heads,
        )
        xq = xq.reshape((-1, self.n_heads, self.kv_params.head_dim))

        use_local = bool((self.layer_idx + 1) % self.sliding_window_pattern)
        rope = self.rope_local if use_local else self.rope_global
        freqs_cis = ops.cast(rope.freqs_cis, xq.dtype).to(xq.device)

        rms_norm_key_cache(
            self.kv_params,
            kv_collection=kv_collection,
            gamma=self.k_norm.weight.cast(self.kv_params.dtype).to(
                self.devices[0]
            ),
            epsilon=self.qk_norm_eps,
            layer_idx=layer_idx,
            total_seq_len=total_seq_len,
            input_row_offsets=kwargs["input_row_offsets"],
            weight_offset=1.0,
        )
        rope_k_cache_ragged(
            self.kv_params,
            total_seq_len=total_seq_len,
            input_row_offsets=kwargs["input_row_offsets"],
            kv_collection=kv_collection,
            freqs_cis=freqs_cis,
            layer_idx=layer_idx,
            interleaved=rope.interleaved,
        )
        xq = self.q_norm(xq)
        xq = rope_ragged(
            xq,
            kwargs["input_row_offsets"],
            kv_collection.cache_lengths,
            freqs_cis,
            interleaved=rope.interleaved,
        )

        mask_variant = (
            MHAMaskVariant.SLIDING_WINDOW_CAUSAL_MASK
            if use_local
            else MHAMaskVariant.CAUSAL_MASK
        )
        attn_out = flash_attention_ragged(
            self.kv_params,
            input=xq,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            input_row_offsets=kwargs["input_row_offsets"],
            mask_variant=mask_variant,
            scale=self.scale,
            local_window_size=self.local_window_size,
        )
        attn_out = ops.reshape(attn_out, shape=[total_seq_len, -1])
        return self.o_proj(attn_out)


class _Gemma3AttentionQOnlyFused(MaxGemma3Attention):
    def __call__(
        self,
        x: TensorValue,
        kv_collection: PagedCacheValues,
        **kwargs,
    ) -> TensorValue:
        total_seq_len = x.shape[0]
        layer_idx = ops.constant(
            self.layer_idx, DType.uint32, device=DeviceRef.CPU()
        )

        xq = fused_qkv_ragged_matmul(
            self.kv_params,
            input=x,
            wqkv=self.wqkv,
            bias=self.wqkv_bias,
            input_row_offsets=kwargs["input_row_offsets"],
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            n_heads=self.n_heads,
        )
        xq = xq.reshape((-1, self.n_heads, self.kv_params.head_dim))

        use_local = bool((self.layer_idx + 1) % self.sliding_window_pattern)
        rope = self.rope_local if use_local else self.rope_global
        freqs_cis = ops.cast(rope.freqs_cis, xq.dtype).to(xq.device)

        rms_norm_key_cache(
            self.kv_params,
            kv_collection=kv_collection,
            gamma=self.k_norm.weight.cast(self.kv_params.dtype).to(
                self.devices[0]
            ),
            epsilon=self.qk_norm_eps,
            layer_idx=layer_idx,
            total_seq_len=total_seq_len,
            input_row_offsets=kwargs["input_row_offsets"],
            weight_offset=1.0,
        )
        rope_k_cache_ragged(
            self.kv_params,
            total_seq_len=total_seq_len,
            input_row_offsets=kwargs["input_row_offsets"],
            kv_collection=kv_collection,
            freqs_cis=freqs_cis,
            layer_idx=layer_idx,
            interleaved=rope.interleaved,
        )
        xq = q_rms_norm_rope_ragged(
            xq,
            kwargs["input_row_offsets"],
            kv_collection.cache_lengths,
            freqs_cis,
            self.q_norm.weight.cast(self.kv_params.dtype).to(self.devices[0]),
            self.qk_norm_eps,
            weight_offset=1.0,
            interleaved=rope.interleaved,
        )

        mask_variant = (
            MHAMaskVariant.SLIDING_WINDOW_CAUSAL_MASK
            if use_local
            else MHAMaskVariant.CAUSAL_MASK
        )
        attn_out = flash_attention_ragged(
            self.kv_params,
            input=xq,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            input_row_offsets=kwargs["input_row_offsets"],
            mask_variant=mask_variant,
            scale=self.scale,
            local_window_size=self.local_window_size,
        )
        attn_out = ops.reshape(attn_out, shape=[total_seq_len, -1])
        return self.o_proj(attn_out)


def _build_graph(
    *,
    session: InferenceSession,
    kv_params: KVCacheParams,
    config: dict[str, Any],
    weight_registry: dict[str, torch.Tensor],
    graph_variant: str,
    layer_idx: int,
    layer_type: str,
) -> Any:
    device_ref = DeviceRef.GPU()
    max_seq_len = (
        CACHE_LEN_BASE
        + CACHE_LEN_STEP * (max(BATCH_SIZES) - 1)
        + MAX_EXTRA_STEPS
        + 256
    )
    attention_cls = {
        "baseline": _Gemma3AttentionBaseline,
        "full": MaxGemma3Attention,
        "k_only": _Gemma3AttentionKOnlyFused,
        "q_only_baseline": _Gemma3AttentionQOnlyBaseline,
        "q_only": _Gemma3AttentionQOnlyFused,
    }.get(graph_variant)
    if attention_cls is None:
        raise ValueError(
            f"unsupported attention decode graph variant {graph_variant!r}"
        )
    attention = attention_cls(
        rope_global=Llama3RotaryEmbedding(
            config["hidden_size"],
            config["num_attention_heads"],
            config["rope_theta"],
            max_seq_len,
            interleaved=False,
            head_dim=config["head_dim"],
        ),
        rope_local=Llama3RotaryEmbedding(
            config["hidden_size"],
            config["num_attention_heads"],
            config["rope_local_base_freq"],
            max_seq_len,
            interleaved=False,
            head_dim=config["head_dim"],
        ),
        num_attention_heads=config["num_attention_heads"],
        num_key_value_heads=config["num_key_value_heads"],
        hidden_size=config["hidden_size"],
        kv_params=kv_params,
        layer_idx=layer_idx,
        dtype=DType.bfloat16,
        devices=[device_ref],
        qk_norm_eps=EPS,
        sliding_window_pattern=config["sliding_window_pattern"],
        local_window_size=config["sliding_window"],
        has_bias=bool(config["attention_bias"]),
    )
    attention.load_state_dict(weight_registry)

    input_type = TensorType(
        DType.bfloat16,
        ["total_seq_len", config["hidden_size"]],
        device=device_ref,
    )
    input_row_offsets_type = TensorType(
        DType.uint32,
        ["input_row_offsets_len"],
        device=device_ref,
    )
    flattened_kv_types = kv_params.get_symbolic_inputs().flatten()

    graph_variant_name = {
        "baseline": "Baseline",
        "full": "Fused",
        "k_only": "KOnlyFused",
        "q_only_baseline": "QOnlyBaseline",
        "q_only": "QOnlyFused",
    }[graph_variant]
    graph_name = (
        f"Gemma3AttentionDecode{layer_type.title()}{graph_variant_name}"
    )

    with Graph(
        graph_name,
        input_types=(input_type, input_row_offsets_type, *flattened_kv_types),
    ) as graph:
        x, input_row_offsets, *kv_cache = graph.inputs
        kv_collection = unflatten_ragged_attention_inputs(
            kv_cache, n_devices=1
        )[0]
        graph.output(
            attention(
                x.tensor,
                kv_collection,
                input_row_offsets=input_row_offsets.tensor,
            )
        )

    return session.load(graph, weights_registry=attention.state_dict())


def _make_text_context(length: int, max_length: int) -> TextContext:
    return TextContext(
        request_id=RequestID(),
        max_length=max_length,
        tokens=TokenBuffer(np.zeros(length, dtype=np.int64)),
    )


def _make_runtime_inputs(
    *,
    session: InferenceSession,
    kv_params: KVCacheParams,
    batch_size: int,
) -> tuple[np.ndarray, Any]:
    cache_lengths = np.asarray(
        [
            CACHE_LEN_BASE + CACHE_LEN_STEP * request_idx
            for request_idx in range(batch_size)
        ],
        dtype=np.uint32,
    )
    max_cache_length = int(cache_lengths[-1]) + MAX_EXTRA_STEPS + 1
    total_num_pages = sum(
        math.ceil((int(cache_length) + MAX_EXTRA_STEPS + 1) / PAGE_SIZE)
        for cache_length in cache_lengths
    )

    kv_manager = PagedKVCacheManager(
        params=kv_params,
        total_num_pages=total_num_pages,
        session=session,
        max_batch_size=batch_size,
    )

    contexts: list[TextContext] = []
    for cache_length in cache_lengths:
        context = _make_text_context(int(cache_length), max_cache_length)
        kv_manager.claim(context.request_id, replica_idx=0)
        kv_manager.alloc(
            context,
            replica_idx=0,
            num_steps=MAX_EXTRA_STEPS + 1,
        )
        contexts.append(context)

    runtime_inputs = kv_manager.runtime_inputs([contexts], num_steps=1).inputs[
        0
    ]
    return cache_lengths, runtime_inputs


def _clone_kv_blocks(blocks: Buffer, seed: int) -> Buffer:
    torch.manual_seed(seed)
    shape = tuple(int(dim) for dim in blocks.shape)
    tensor = torch.randn(
        shape, dtype=torch.bfloat16, device="cuda"
    ).contiguous()
    return Buffer.from_dlpack(tensor)


def _device_uint32_buffer(array: np.ndarray, device: Accelerator) -> Buffer:
    return Buffer.from_numpy(array).to(device)


def _run_nvidia_smi_query(query_fields: str) -> list[list[str]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--query-{query_fields}",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return [
        [field.strip() for field in row]
        for row in csv.reader(result.stdout.splitlines())
        if row
    ]


def _build_gpu_isolation_guard() -> dict[str, Any] | None:
    raw_guard = os.environ.get(
        "PROFILE_ATTENTION_DECODE_ENFORCE_GPU_ISOLATION", ""
    ).strip()
    if raw_guard.lower() not in ("1", "true", "yes"):
        return None

    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    target_gpu_index = (
        0
        if cuda_visible_devices == ""
        else int(cuda_visible_devices.split(",")[0])
    )

    for index_text, gpu_uuid in _run_nvidia_smi_query("gpu=index,uuid"):
        if int(index_text) == target_gpu_index:
            return {
                "allowed_pid": os.getpid(),
                "target_gpu_index": target_gpu_index,
                "target_gpu_uuid": gpu_uuid,
            }

    raise AssertionError(
        "Could not resolve the target GPU UUID for "
        f"CUDA_VISIBLE_DEVICES={cuda_visible_devices!r}"
    )


def _write_results_json_if_requested(results: dict[str, Any]) -> None:
    path = os.environ.get("PROFILE_ATTENTION_DECODE_RESULTS_JSON")
    if not path:
        return
    Path(path).write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")


def _mark_results_progress(results: dict[str, Any], stage: str) -> None:
    results["progress"] = {
        "last_completed_stage": stage,
        "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write_results_json_if_requested(results)


def _assert_gpu_isolation(
    gpu_isolation_guard: dict[str, Any] | None,
    *,
    label: str,
) -> None:
    if gpu_isolation_guard is None:
        return

    resident_compute_apps = []
    unexpected_apps = []
    for (
        gpu_uuid,
        pid_text,
        process_name,
        used_gpu_memory_mib,
    ) in _run_nvidia_smi_query(
        "compute-apps=gpu_uuid,pid,process_name,used_gpu_memory"
    ):
        if gpu_uuid != gpu_isolation_guard["target_gpu_uuid"]:
            continue
        resident_app = {
            "gpu_uuid": gpu_uuid,
            "pid": int(pid_text),
            "process_name": process_name,
            "used_gpu_memory_mib": float(used_gpu_memory_mib),
        }
        resident_compute_apps.append(resident_app)
        if resident_app["pid"] != gpu_isolation_guard["allowed_pid"]:
            unexpected_apps.append(resident_app)

    if unexpected_apps:
        raise AssertionError(
            f"{label} at {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}: "
            "unexpected compute apps on physical GPU "
            f"{gpu_isolation_guard['target_gpu_index']} "
            f"({gpu_isolation_guard['target_gpu_uuid']}): "
            f"{unexpected_apps}; all resident apps: {resident_compute_apps}"
        )


def _make_benchmark_args(
    *,
    batch_size: int,
    hidden_size: int,
    cache_lengths: np.ndarray,
    runtime_inputs: Any,
    device: Accelerator,
    x_seed: int,
    blocks_seed: int,
) -> tuple[list[tuple[Any, ...]], Buffer]:
    torch.manual_seed(x_seed)
    x = torch.randn(
        (batch_size, hidden_size),
        dtype=torch.bfloat16,
        device="cuda",
    ).contiguous()
    x_buffer = Buffer.from_dlpack(x)
    row_offsets = _device_uint32_buffer(
        np.arange(batch_size + 1, dtype=np.uint32),
        device,
    )
    kv_blocks = _clone_kv_blocks(runtime_inputs.blocks, seed=blocks_seed)
    lookup_table = runtime_inputs.lookup_table.to(device)
    dispatch_metadata = runtime_inputs.attention_dispatch_metadata
    assert dispatch_metadata is not None

    args: list[tuple[Any, ...]] = []
    for step in range(MAX_EXTRA_STEPS + 1):
        step_cache_lengths = _device_uint32_buffer(
            cache_lengths + np.uint32(step), device
        )
        args.append(
            (
                x_buffer,
                row_offsets,
                kv_blocks,
                step_cache_lengths,
                lookup_table,
                runtime_inputs.max_lengths,
                dispatch_metadata,
            )
        )
    return args, kv_blocks


def _run_correctness_check(
    *,
    baseline: Any,
    fused: Any,
    baseline_args: tuple[Any, ...],
    fused_args: tuple[Any, ...],
    layer_idx: int,
) -> None:
    baseline_output = baseline.execute(*baseline_args)[0]
    fused_output = fused.execute(*fused_args)[0]
    torch.cuda.synchronize()

    torch.testing.assert_close(
        from_dlpack(baseline_output).to(torch.bfloat16),
        from_dlpack(fused_output).to(torch.bfloat16),
        rtol=2 * torch.finfo(torch.bfloat16).eps,
        atol=8 * torch.finfo(torch.bfloat16).eps,
    )

    # Only the addressed layer should mutate during decode; comparing that
    # layer keeps the global-layer harness under the PyTorch memory cap.
    baseline_blocks = from_dlpack(baseline_args[2]).to(torch.bfloat16)[
        :, :, layer_idx : layer_idx + 1, :, :, :
    ]
    fused_blocks = from_dlpack(fused_args[2]).to(torch.bfloat16)[
        :, :, layer_idx : layer_idx + 1, :, :, :
    ]
    torch.testing.assert_close(
        baseline_blocks,
        fused_blocks,
        rtol=2 * torch.finfo(torch.bfloat16).eps,
        atol=8 * torch.finfo(torch.bfloat16).eps,
    )


def _benchmark_us(compiled: Any, args: list[tuple[Any, ...]]) -> float:
    warmup_args = args[:WARMUP_ITERS]
    timed_args = args[WARMUP_ITERS : WARMUP_ITERS + TIMED_ITERS]

    for run_args in warmup_args:
        compiled.execute(*run_args)
    torch.cuda.synchronize()

    start_s = time.perf_counter()
    for run_args in timed_args:
        compiled.execute(*run_args)
    torch.cuda.synchronize()
    elapsed_s = time.perf_counter() - start_s
    return elapsed_s * 1e6 / len(timed_args)


def _benchmark_us_guarded(
    compiled: Any,
    args: list[tuple[Any, ...]],
    *,
    gpu_isolation_guard: dict[str, Any] | None,
    label: str,
) -> float:
    _assert_gpu_isolation(
        gpu_isolation_guard,
        label=f"{label}:before_warmup",
    )
    warmup_args = args[:WARMUP_ITERS]
    timed_args = args[WARMUP_ITERS : WARMUP_ITERS + TIMED_ITERS]

    for run_args in warmup_args:
        compiled.execute(*run_args)
    torch.cuda.synchronize()

    start_s = time.perf_counter()
    for run_args in timed_args:
        compiled.execute(*run_args)
    torch.cuda.synchronize()
    _assert_gpu_isolation(
        gpu_isolation_guard,
        label=f"{label}:after_timed",
    )
    elapsed_s = time.perf_counter() - start_s
    return elapsed_s * 1e6 / len(timed_args)


def _benchmark_pair_order(
    *,
    compiled_graphs: dict[str, Any],
    benchmark_args: dict[str, list[tuple[Any, ...]]],
    order: tuple[str, str],
    gpu_isolation_guard: dict[str, Any] | None,
    run_name: str,
) -> dict[str, float]:
    results: dict[str, float] = {}
    for graph_name in order:
        results[graph_name] = _benchmark_us_guarded(
            compiled_graphs[graph_name],
            benchmark_args[graph_name],
            gpu_isolation_guard=gpu_isolation_guard,
            label=f"{run_name}:{graph_name}",
        )
    return results


def test_profile_attention_decode() -> None:
    config = _load_text_config()
    layer_idx, layer_type = _resolve_layer_metadata(config)
    kv_num_layers = _resolve_kv_num_layers(config, layer_idx)
    baseline_graph_variant, fused_graph_variant = _resolve_graph_variants()
    timing_order_mode = _resolve_timing_order_mode()
    comparison_mode, run_name_suffix = _comparison_metadata(
        baseline_graph_variant, fused_graph_variant
    )
    comparison_mode, run_name_suffix = _timing_order_metadata(
        comparison_mode,
        run_name_suffix,
        timing_order_mode,
    )
    session = InferenceSession(devices=[Accelerator(0)])
    device = Accelerator(0)
    gpu_isolation_guard = _build_gpu_isolation_guard()
    kv_params = KVCacheParams(
        dtype=DType.bfloat16,
        devices=[DeviceRef.GPU()],
        n_kv_heads=config["num_key_value_heads"],
        head_dim=config["head_dim"],
        num_layers=kv_num_layers,
        page_size=PAGE_SIZE,
    )
    weight_registry = _make_weight_registry(config)
    baseline = _build_graph(
        session=session,
        kv_params=kv_params,
        config=config,
        weight_registry=weight_registry,
        graph_variant=baseline_graph_variant,
        layer_idx=layer_idx,
        layer_type=layer_type,
    )
    fused = _build_graph(
        session=session,
        kv_params=kv_params,
        config=config,
        weight_registry=weight_registry,
        graph_variant=fused_graph_variant,
        layer_idx=layer_idx,
        layer_type=layer_type,
    )

    benchmark_config: dict[str, Any] = {
        "dtype": "bfloat16",
        "hidden_size": config["hidden_size"],
        "head_dim": config["head_dim"],
        "num_q_heads": config["num_attention_heads"],
        "num_kv_heads": config["num_key_value_heads"],
        "page_size": PAGE_SIZE,
        "layer_idx": layer_idx,
        "layer_type": layer_type,
        "kv_num_layers": kv_num_layers,
        "mode": comparison_mode,
        "baseline_graph_variant": baseline_graph_variant,
        "fused_graph_variant": fused_graph_variant,
        "cache_len_base": CACHE_LEN_BASE,
        "cache_len_step": CACHE_LEN_STEP,
        "warmup_iters": WARMUP_ITERS,
        "timed_iters": TIMED_ITERS,
        "timing_order_mode": timing_order_mode,
        "selected_shapes": [
            f"decode_{layer_type}_layer{layer_idx}_bs{batch_size}_seq1_cache"
            f"{CACHE_LEN_BASE}_step{CACHE_LEN_STEP}{run_name_suffix}"
            for batch_size in BATCH_SIZES
        ],
    }
    if gpu_isolation_guard is not None:
        benchmark_config["gpu_isolation_guard"] = {
            "allowed_pid": gpu_isolation_guard["allowed_pid"],
            "target_gpu_index": gpu_isolation_guard["target_gpu_index"],
            "target_gpu_uuid": gpu_isolation_guard["target_gpu_uuid"],
        }
    results_json_path = os.environ.get("PROFILE_ATTENTION_DECODE_RESULTS_JSON")
    if results_json_path:
        benchmark_config["results_json_env"] = (
            "PROFILE_ATTENTION_DECODE_RESULTS_JSON"
        )
        benchmark_config["results_json_path"] = results_json_path

    if timing_order_mode == "both_orders":
        results: dict[str, Any] = {
            "benchmark_config": benchmark_config,
            "correctness": "pass",
            "progress": {
                "last_completed_stage": "initialized",
                "updated_at_utc": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                ),
            },
            "ordered_sweep_us": {},
            "order_position_us": {},
            "order_slowdown_ratio_when_second": {},
            "pair_speedup_ratio_by_order": {},
        }
    else:
        results = {
            "benchmark_config": benchmark_config,
            "correctness": "pass",
            "progress": {
                "last_completed_stage": "initialized",
                "updated_at_utc": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                ),
            },
            "first_sweep_us": {},
            "confirm_sweep_us": {},
            "average_us": {},
            "average_speedup_ratio_vs_baseline_graph": {},
        }
    _write_results_json_if_requested(results)

    for batch_size in BATCH_SIZES:
        cache_lengths, runtime_inputs = _make_runtime_inputs(
            session=session,
            kv_params=kv_params,
            batch_size=batch_size,
        )
        run_name = (
            f"decode_{layer_type}_layer{layer_idx}_bs{batch_size}_seq1_cache"
            f"{CACHE_LEN_BASE}_step{CACHE_LEN_STEP}{run_name_suffix}"
        )

        correctness_baseline_args, correctness_baseline_blocks = (
            _make_benchmark_args(
                batch_size=batch_size,
                hidden_size=config["hidden_size"],
                cache_lengths=cache_lengths,
                runtime_inputs=runtime_inputs,
                device=device,
                x_seed=100 + batch_size,
                blocks_seed=200 + batch_size,
            )
        )
        correctness_fused_args, correctness_fused_blocks = _make_benchmark_args(
            batch_size=batch_size,
            hidden_size=config["hidden_size"],
            cache_lengths=cache_lengths,
            runtime_inputs=runtime_inputs,
            device=device,
            x_seed=100 + batch_size,
            blocks_seed=200 + batch_size,
        )
        _run_correctness_check(
            baseline=baseline,
            fused=fused,
            baseline_args=correctness_baseline_args[0],
            fused_args=correctness_fused_args[0],
            layer_idx=layer_idx,
        )
        del correctness_baseline_blocks
        del correctness_fused_blocks
        del correctness_baseline_args
        del correctness_fused_args
        torch.cuda.empty_cache()

        if timing_order_mode == "both_orders":
            forward_baseline_args, _ = _make_benchmark_args(
                batch_size=batch_size,
                hidden_size=config["hidden_size"],
                cache_lengths=cache_lengths,
                runtime_inputs=runtime_inputs,
                device=device,
                x_seed=300 + batch_size,
                blocks_seed=400 + batch_size,
            )
            forward_fused_args, _ = _make_benchmark_args(
                batch_size=batch_size,
                hidden_size=config["hidden_size"],
                cache_lengths=cache_lengths,
                runtime_inputs=runtime_inputs,
                device=device,
                x_seed=300 + batch_size,
                blocks_seed=400 + batch_size,
            )
            forward_order = _benchmark_pair_order(
                compiled_graphs={
                    "baseline": baseline,
                    "fused": fused,
                },
                benchmark_args={
                    "baseline": forward_baseline_args,
                    "fused": forward_fused_args,
                },
                order=("baseline", "fused"),
                gpu_isolation_guard=gpu_isolation_guard,
                run_name=f"{run_name}:baseline_then_fused",
            )
            del forward_baseline_args
            del forward_fused_args
            torch.cuda.empty_cache()

            reverse_baseline_args, _ = _make_benchmark_args(
                batch_size=batch_size,
                hidden_size=config["hidden_size"],
                cache_lengths=cache_lengths,
                runtime_inputs=runtime_inputs,
                device=device,
                x_seed=500 + batch_size,
                blocks_seed=600 + batch_size,
            )
            reverse_fused_args, _ = _make_benchmark_args(
                batch_size=batch_size,
                hidden_size=config["hidden_size"],
                cache_lengths=cache_lengths,
                runtime_inputs=runtime_inputs,
                device=device,
                x_seed=500 + batch_size,
                blocks_seed=600 + batch_size,
            )
            reverse_order = _benchmark_pair_order(
                compiled_graphs={
                    "baseline": baseline,
                    "fused": fused,
                },
                benchmark_args={
                    "baseline": reverse_baseline_args,
                    "fused": reverse_fused_args,
                },
                order=("fused", "baseline"),
                gpu_isolation_guard=gpu_isolation_guard,
                run_name=f"{run_name}:fused_then_baseline",
            )
            del reverse_baseline_args
            del reverse_fused_args
            torch.cuda.empty_cache()

            results["ordered_sweep_us"][run_name] = {
                "baseline_then_fused": forward_order,
                "fused_then_baseline": reverse_order,
            }
            results["order_position_us"][run_name] = {
                "baseline_first": forward_order["baseline"],
                "baseline_second": reverse_order["baseline"],
                "fused_first": reverse_order["fused"],
                "fused_second": forward_order["fused"],
            }
            results["order_slowdown_ratio_when_second"][run_name] = {
                "baseline": (
                    reverse_order["baseline"] / forward_order["baseline"]
                ),
                "fused": forward_order["fused"] / reverse_order["fused"],
            }
            results["pair_speedup_ratio_by_order"][run_name] = {
                "baseline_then_fused": (
                    forward_order["baseline"] / forward_order["fused"]
                ),
                "fused_then_baseline": (
                    reverse_order["baseline"] / reverse_order["fused"]
                ),
            }
            _mark_results_progress(results, f"{run_name}:order_probe_complete")
            continue

        first_baseline_args, _ = _make_benchmark_args(
            batch_size=batch_size,
            hidden_size=config["hidden_size"],
            cache_lengths=cache_lengths,
            runtime_inputs=runtime_inputs,
            device=device,
            x_seed=300 + batch_size,
            blocks_seed=400 + batch_size,
        )
        first_fused_args, _ = _make_benchmark_args(
            batch_size=batch_size,
            hidden_size=config["hidden_size"],
            cache_lengths=cache_lengths,
            runtime_inputs=runtime_inputs,
            device=device,
            x_seed=300 + batch_size,
            blocks_seed=400 + batch_size,
        )
        first_baseline_us = _benchmark_us_guarded(
            baseline,
            first_baseline_args,
            gpu_isolation_guard=gpu_isolation_guard,
            label=f"{run_name}:first:baseline",
        )
        first_fused_us = _benchmark_us_guarded(
            fused,
            first_fused_args,
            gpu_isolation_guard=gpu_isolation_guard,
            label=f"{run_name}:first:fused",
        )
        results["first_sweep_us"][run_name] = {
            "baseline": first_baseline_us,
            "fused": first_fused_us,
        }
        _mark_results_progress(results, f"{run_name}:first_sweep")
        del first_baseline_args
        del first_fused_args
        torch.cuda.empty_cache()

        confirm_baseline_args, _ = _make_benchmark_args(
            batch_size=batch_size,
            hidden_size=config["hidden_size"],
            cache_lengths=cache_lengths,
            runtime_inputs=runtime_inputs,
            device=device,
            x_seed=500 + batch_size,
            blocks_seed=600 + batch_size,
        )
        confirm_fused_args, _ = _make_benchmark_args(
            batch_size=batch_size,
            hidden_size=config["hidden_size"],
            cache_lengths=cache_lengths,
            runtime_inputs=runtime_inputs,
            device=device,
            x_seed=500 + batch_size,
            blocks_seed=600 + batch_size,
        )
        confirm_baseline_us = _benchmark_us_guarded(
            baseline,
            confirm_baseline_args,
            gpu_isolation_guard=gpu_isolation_guard,
            label=f"{run_name}:confirm:baseline",
        )
        confirm_fused_us = _benchmark_us_guarded(
            fused,
            confirm_fused_args,
            gpu_isolation_guard=gpu_isolation_guard,
            label=f"{run_name}:confirm:fused",
        )
        results["confirm_sweep_us"][run_name] = {
            "baseline": confirm_baseline_us,
            "fused": confirm_fused_us,
        }
        _mark_results_progress(results, f"{run_name}:confirm_sweep")
        del confirm_baseline_args
        del confirm_fused_args
        torch.cuda.empty_cache()

        average_baseline_us = (first_baseline_us + confirm_baseline_us) / 2.0
        average_fused_us = (first_fused_us + confirm_fused_us) / 2.0
        results["average_us"][run_name] = {
            "baseline": average_baseline_us,
            "fused": average_fused_us,
        }
        results["average_speedup_ratio_vs_baseline_graph"][run_name] = (
            average_baseline_us / average_fused_us
        )

    if timing_order_mode == "both_orders":
        baseline_slowdowns = [
            run_results["baseline"]
            for run_results in results[
                "order_slowdown_ratio_when_second"
            ].values()
        ]
        fused_slowdowns = [
            run_results["fused"]
            for run_results in results[
                "order_slowdown_ratio_when_second"
            ].values()
        ]
        results["order_slowdown_geomean_when_second"] = {
            "baseline": float(np.exp(np.mean(np.log(baseline_slowdowns)))),
            "fused": float(np.exp(np.mean(np.log(fused_slowdowns)))),
        }
        baseline_then_fused_speedups = [
            run_results["baseline_then_fused"]
            for run_results in results["pair_speedup_ratio_by_order"].values()
        ]
        fused_then_baseline_speedups = [
            run_results["fused_then_baseline"]
            for run_results in results["pair_speedup_ratio_by_order"].values()
        ]
        results["pair_speedup_geomean_by_order"] = {
            "baseline_then_fused": float(
                np.exp(np.mean(np.log(baseline_then_fused_speedups)))
            ),
            "fused_then_baseline": float(
                np.exp(np.mean(np.log(fused_then_baseline_speedups)))
            ),
        }
    else:
        speedups = list(
            results["average_speedup_ratio_vs_baseline_graph"].values()
        )
        results["average_geomean_speedup_vs_baseline_graph"] = float(
            np.exp(np.mean(np.log(speedups)))
        )
    _mark_results_progress(results, "complete")

    print("GEMMA3_ATTENTION_DECODE_PROFILE_START")
    print(json.dumps(results, sort_keys=True))
    print("GEMMA3_ATTENTION_DECODE_PROFILE_END")
