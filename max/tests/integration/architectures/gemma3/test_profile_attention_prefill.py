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
    q_rms_norm_fused_qk_ragged_rope,
    rms_norm_key_cache,
    rope_ragged,
)
from max.nn.kv_cache import (
    AttentionDispatchResolver,
    KVCacheInputsPerDevice,
    PagedCacheValues,
    build_max_lengths_tensor,
    unflatten_ragged_attention_inputs,
)
from max.nn.rotary_embedding import Llama3RotaryEmbedding
from max.pipelines.architectures.gemma3.layers.attention import (
    Gemma3Attention as MaxGemma3Attention,
)
from max.pipelines.core import TextContext
from torch.utils.dlpack import from_dlpack


PAGE_SIZE = 128
EPS = 1e-6
DEFAULT_LAYER_IDX = 0
DEFAULT_LAYER_TYPE = "local"
DEFAULT_COMPARE_MODE = "baseline_vs_fused"
DEFAULT_FUSED_VARIANT = "full"
FUSED_VARIANTS = ("full", "k_only")
Q_NORM_STD = 0.68
K_NORM_STD = 0.793
Q_PROJ_STD = 0.0284
K_PROJ_STD = 0.0309
V_PROJ_STD = 0.0309
O_PROJ_STD = 0.0237
WARMUP_ITERS = 20
TIMED_ITERS = 50
PREFILL_SHAPES = (
    (1, 11),
    (1, 512),
    (1, 1024),
    (1, 2048),
    (2, 2048),
)
PLACEHOLDER_CACHE_LEN = 1
PAIR_MATRIX_COMPARE_MODE = "manual_and_attention_pair_matrix"
PAIR_MATRIX_GRAPH_SPECS = (
    (
        "manual_baseline_graph",
        "manual_baseline_graph_vs_attention_baseline",
        False,
    ),
    (
        "attention_baseline",
        "manual_baseline_graph_vs_attention_baseline",
        True,
    ),
    (
        "manual_fused_graph",
        "manual_fused_graph_vs_attention_fused",
        False,
    ),
    (
        "attention_fused",
        "manual_fused_graph_vs_attention_fused",
        True,
    ),
)
PAIR_MATRIX_PAIRS = (
    ("manual_baseline_graph", "attention_baseline"),
    ("manual_fused_graph", "attention_fused"),
    ("manual_baseline_graph", "manual_fused_graph"),
    ("attention_baseline", "attention_fused"),
    ("manual_baseline_graph", "attention_fused"),
    ("attention_baseline", "manual_fused_graph"),
)


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
    assert (
        layer_type == expected_layer_type
    ), f"expected {expected_layer_type} layer, got {layer_type} for layer_idx={layer_idx}"
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


def _resolve_compare_mode() -> str:
    compare_mode = os.environ.get(
        "PROFILE_ATTENTION_PREFILL_COMPARE_MODE", DEFAULT_COMPARE_MODE
    ).strip()
    if compare_mode in ("", DEFAULT_COMPARE_MODE):
        return DEFAULT_COMPARE_MODE
    if compare_mode == PAIR_MATRIX_COMPARE_MODE:
        return compare_mode
    if compare_mode == "manual_baseline_graph_vs_attention_baseline":
        return compare_mode
    if compare_mode == "manual_fused_graph_vs_attention_fused":
        return compare_mode
    raise ValueError(
        "PROFILE_ATTENTION_PREFILL_COMPARE_MODE must be one of "
        "{'baseline_vs_fused', 'manual_and_attention_pair_matrix', "
        "'manual_baseline_graph_vs_attention_baseline', "
        "'manual_fused_graph_vs_attention_fused'}, got "
        f"{compare_mode!r}"
    )


def _resolve_fused_variant() -> str:
    fused_variant = os.environ.get(
        "PROFILE_ATTENTION_PREFILL_FUSED_VARIANT",
        DEFAULT_FUSED_VARIANT,
    ).strip()
    if fused_variant in ("", DEFAULT_FUSED_VARIANT):
        return DEFAULT_FUSED_VARIANT
    if fused_variant in FUSED_VARIANTS:
        return fused_variant
    raise ValueError(
        "PROFILE_ATTENTION_PREFILL_FUSED_VARIANT must be one of "
        f"{FUSED_VARIANTS}, got {fused_variant!r}"
    )


def _resolve_prefill_shapes() -> tuple[tuple[int, int], ...]:
    raw_shapes = os.environ.get("PROFILE_ATTENTION_PREFILL_SHAPES", "").strip()
    if raw_shapes == "":
        return PREFILL_SHAPES

    resolved: list[tuple[int, int]] = []
    for raw_shape in raw_shapes.split(","):
        shape_text = raw_shape.strip().lower()
        if shape_text == "":
            continue
        if "x" not in shape_text:
            raise ValueError(
                "PROFILE_ATTENTION_PREFILL_SHAPES entries must look like "
                f"'batchxseq', got {raw_shape!r}"
            )
        batch_text, seq_text = shape_text.split("x", maxsplit=1)
        shape = (int(batch_text), int(seq_text))
        if shape not in PREFILL_SHAPES:
            raise ValueError(
                "PROFILE_ATTENTION_PREFILL_SHAPES only supports the existing "
                f"prefill grid {PREFILL_SHAPES}, got {shape}"
            )
        if shape not in resolved:
            resolved.append(shape)

    if not resolved:
        raise ValueError(
            "PROFILE_ATTENTION_PREFILL_SHAPES must select at least one shape"
        )
    return tuple(resolved)


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
        "k_norm.weight": torch.randn(
            config["head_dim"], dtype=torch.bfloat16
        )
        * K_NORM_STD,
        "k_proj.weight": torch.randn(
            kv_dim, hidden_size, dtype=torch.bfloat16
        )
        * K_PROJ_STD,
        "o_proj.weight": torch.randn(
            hidden_size, q_dim, dtype=torch.bfloat16
        )
        * O_PROJ_STD,
        "q_norm.weight": torch.randn(
            config["head_dim"], dtype=torch.bfloat16
        )
        * Q_NORM_STD,
        "q_proj.weight": torch.randn(
            q_dim, hidden_size, dtype=torch.bfloat16
        )
        * Q_PROJ_STD,
        "v_proj.weight": torch.randn(
            kv_dim, hidden_size, dtype=torch.bfloat16
        )
        * V_PROJ_STD,
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


def _manual_attention_output(
    *,
    attention: MaxGemma3Attention,
    kv_params: KVCacheParams,
    x: TensorValue,
    kv_collection: PagedCacheValues,
    input_row_offsets: TensorValue,
    use_fused_qk_rope: bool,
) -> TensorValue:
    total_seq_len = x.shape[0]
    layer_idx = ops.constant(
        attention.layer_idx, DType.uint32, device=DeviceRef.CPU()
    )

    xq = fused_qkv_ragged_matmul(
        kv_params,
        input=x,
        wqkv=attention.wqkv,
        bias=attention.wqkv_bias,
        input_row_offsets=input_row_offsets,
        kv_collection=kv_collection,
        layer_idx=layer_idx,
        n_heads=attention.n_heads,
    )
    xq = xq.reshape((-1, attention.n_heads, kv_params.head_dim))

    use_local = bool((attention.layer_idx + 1) % attention.sliding_window_pattern)
    rope = attention.rope_local if use_local else attention.rope_global
    freqs_cis = ops.cast(rope.freqs_cis, xq.dtype).to(xq.device)

    if use_fused_qk_rope:
        xq = q_rms_norm_fused_qk_ragged_rope(
            kv_params,
            xq,
            input_row_offsets,
            kv_collection,
            freqs_cis,
            attention.q_norm.weight.cast(kv_params.dtype).to(
                attention.devices[0]
            ),
            attention.k_norm.weight.cast(kv_params.dtype).to(
                attention.devices[0]
            ),
            attention.qk_norm_eps,
            layer_idx,
            weight_offset=1.0,
            interleaved=rope.interleaved,
        )
    else:
        rms_norm_key_cache(
            kv_params,
            kv_collection=kv_collection,
            gamma=attention.k_norm.weight.cast(kv_params.dtype).to(
                attention.devices[0]
            ),
            epsilon=attention.qk_norm_eps,
            layer_idx=layer_idx,
            total_seq_len=total_seq_len,
            input_row_offsets=input_row_offsets,
            weight_offset=1.0,
        )
        xq = attention.q_norm(xq)
        xq = fused_qk_ragged_rope(
            kv_params,
            xq,
            input_row_offsets,
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
        kv_params,
        input=xq,
        kv_collection=kv_collection,
        layer_idx=layer_idx,
        input_row_offsets=input_row_offsets,
        mask_variant=mask_variant,
        scale=attention.scale,
        local_window_size=attention.local_window_size,
    )
    attn_out = ops.reshape(attn_out, shape=[total_seq_len, -1])
    return attention.o_proj(attn_out)


def _manual_baseline_attention_output(
    *,
    attention: MaxGemma3Attention,
    kv_params: KVCacheParams,
    x: TensorValue,
    kv_collection: PagedCacheValues,
    input_row_offsets: TensorValue,
) -> TensorValue:
    return _manual_attention_output(
        attention=attention,
        kv_params=kv_params,
        x=x,
        kv_collection=kv_collection,
        input_row_offsets=input_row_offsets,
        use_fused_qk_rope=False,
    )


def _manual_fused_attention_output(
    *,
    attention: MaxGemma3Attention,
    kv_params: KVCacheParams,
    x: TensorValue,
    kv_collection: PagedCacheValues,
    input_row_offsets: TensorValue,
) -> TensorValue:
    return _manual_attention_output(
        attention=attention,
        kv_params=kv_params,
        x=x,
        kv_collection=kv_collection,
        input_row_offsets=input_row_offsets,
        use_fused_qk_rope=True,
    )


def _make_placeholder_text_context(max_length: int) -> TextContext:
    return TextContext(
        request_id=RequestID(),
        max_length=max_length,
        tokens=TokenBuffer(np.zeros(PLACEHOLDER_CACHE_LEN, dtype=np.int64)),
    )


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
        "PROFILE_ATTENTION_PREFILL_ENFORCE_GPU_ISOLATION", ""
    ).strip()
    if raw_guard.lower() not in ("1", "true", "yes"):
        return None

    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    target_gpu_index = (
        0 if cuda_visible_devices == "" else int(cuda_visible_devices.split(",")[0])
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


def _assert_gpu_isolation(
    gpu_isolation_guard: dict[str, Any] | None,
    *,
    label: str,
) -> None:
    if gpu_isolation_guard is None:
        return

    resident_compute_apps = []
    unexpected_apps = []
    for gpu_uuid, pid_text, process_name, used_gpu_memory_mib in (
        _run_nvidia_smi_query(
            "compute-apps=gpu_uuid,pid,process_name,used_gpu_memory"
        )
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


def _row_offsets_array(batch_size: int, seq_len: int) -> np.ndarray:
    return np.arange(
        0,
        (batch_size + 1) * seq_len,
        seq_len,
        dtype=np.uint32,
    )


def _prefill_run_name(
    layer_type: str,
    layer_idx: int,
    batch_size: int,
    seq_len: int,
    compare_mode: str,
    run_name_suffix: str = "",
) -> str:
    if compare_mode == PAIR_MATRIX_COMPARE_MODE:
        suffix = "_cache0_step0_attention_pair_matrix"
    elif compare_mode == "manual_baseline_graph_vs_attention_baseline":
        suffix = "_cache0_step0_attention_baseline_graph_parity"
    elif compare_mode == "manual_fused_graph_vs_attention_fused":
        suffix = "_cache0_step0_attention_fused_graph_parity"
    else:
        suffix = f"_cache0_step0_attention{run_name_suffix}"
    return (
        f"prefill_{layer_type}_layer{layer_idx}_bs{batch_size}_seq{seq_len}"
        f"{suffix}"
    )


def _build_graph(
    *,
    session: InferenceSession,
    kv_params: KVCacheParams,
    config: dict[str, Any],
    weight_registry: dict[str, torch.Tensor],
    batch_size: int,
    seq_len: int,
    use_fused: bool,
    fused_variant: str,
    layer_idx: int,
    layer_type: str,
    compare_mode: str,
) -> Any:
    device_ref = DeviceRef.GPU()
    max_seq_len = seq_len + PLACEHOLDER_CACHE_LEN + 256
    attention_cls: type[MaxGemma3Attention]
    if compare_mode == "manual_baseline_graph_vs_attention_baseline":
        attention_cls = _Gemma3AttentionBaseline
        graph_variant = (
            "AttentionBaseline" if use_fused else "ManualBaselineGraph"
        )
    elif compare_mode == "manual_fused_graph_vs_attention_fused":
        attention_cls = MaxGemma3Attention
        graph_variant = "AttentionFused" if use_fused else "ManualFusedGraph"
    else:
        if use_fused and fused_variant == "k_only":
            attention_cls = _Gemma3AttentionKOnlyFused
            graph_variant = "KOnlyFused"
        else:
            attention_cls = (
                MaxGemma3Attention
                if use_fused
                else _Gemma3AttentionBaseline
            )
            graph_variant = "Fused" if use_fused else "Baseline"
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
        [batch_size + 1],
        device=device_ref,
    )
    flattened_kv_types = kv_params.get_symbolic_inputs().flatten()

    graph_name = (
        f"Gemma3AttentionPrefill{layer_type.title()}"
        f"{graph_variant}BS{batch_size}Seq{seq_len}"
    )

    with Graph(
        graph_name,
        input_types=(input_type, input_row_offsets_type, *flattened_kv_types),
    ) as graph:
        x, input_row_offsets, *kv_cache = graph.inputs
        kv_collection = unflatten_ragged_attention_inputs(
            kv_cache, n_devices=1
        )[0]
        if (
            compare_mode == "manual_baseline_graph_vs_attention_baseline"
            and not use_fused
        ):
            graph.output(
                _manual_baseline_attention_output(
                    attention=attention,
                    kv_params=kv_params,
                    x=x.tensor,
                    kv_collection=kv_collection,
                    input_row_offsets=input_row_offsets.tensor,
                )
            )
        elif (
            compare_mode == "manual_fused_graph_vs_attention_fused"
            and not use_fused
        ):
            graph.output(
                _manual_fused_attention_output(
                    attention=attention,
                    kv_params=kv_params,
                    x=x.tensor,
                    kv_collection=kv_collection,
                    input_row_offsets=input_row_offsets.tensor,
                )
            )
        else:
            graph.output(
                attention(
                    x.tensor,
                    kv_collection,
                    input_row_offsets=input_row_offsets.tensor,
                )
            )

    return session.load(graph, weights_registry=attention.state_dict())


def _make_runtime_inputs(
    *,
    session: InferenceSession,
    kv_params: KVCacheParams,
    batch_size: int,
    seq_len: int,
    device: Accelerator,
) -> KVCacheInputsPerDevice:
    max_cache_length = PLACEHOLDER_CACHE_LEN + seq_len
    total_num_pages = batch_size * math.ceil(max_cache_length / PAGE_SIZE)
    kv_manager = PagedKVCacheManager(
        params=kv_params,
        total_num_pages=total_num_pages,
        session=session,
        max_batch_size=batch_size,
    )

    contexts: list[TextContext] = []
    for _ in range(batch_size):
        context = _make_placeholder_text_context(max_cache_length)
        kv_manager.claim(context.request_id, replica_idx=0)
        kv_manager.alloc(
            context,
            replica_idx=0,
            num_steps=seq_len,
        )
        contexts.append(context)

    runtime_inputs = kv_manager.runtime_inputs(
        [contexts],
        num_steps=seq_len,
    ).inputs[0]

    # The placeholder TextContext leaves KV metadata in decode shape
    # (`q_max_seq_len == 1`), but this harness is validating multi-token
    # prefill. Rebuild the packed metadata to match cache0 step0 prefill.
    dispatch_metadata = AttentionDispatchResolver(
        devices=kv_params.devices,
        is_mla=kv_params.is_mla,
        n_kv_heads_per_device=kv_params.n_kv_heads_per_device,
        num_q_heads_per_device=kv_params.num_q_heads_per_device,
        is_fp8_kv=kv_params.is_fp8_kv_dtype,
    ).resolve_for_replica(
        batch_size,
        seq_len,
        seq_len,
    )[0]

    return KVCacheInputsPerDevice(
        blocks=runtime_inputs.blocks,
        cache_lengths=_device_uint32_buffer(
            np.zeros(batch_size, dtype=np.uint32), device
        ),
        lookup_table=runtime_inputs.lookup_table,
        max_lengths=build_max_lengths_tensor(1, seq_len, seq_len),
        kv_scales=runtime_inputs.kv_scales,
        attention_dispatch_metadata=dispatch_metadata,
    )


def _clone_kv_blocks(blocks: Buffer, seed: int) -> Buffer:
    torch.manual_seed(seed)
    shape = tuple(int(dim) for dim in blocks.shape)
    tensor = torch.randn(shape, dtype=torch.bfloat16, device="cuda").contiguous()
    return Buffer.from_dlpack(tensor)


def _make_execution_args(
    *,
    session: InferenceSession,
    kv_params: KVCacheParams,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    device: Accelerator,
    x_seed: int,
    blocks_seed: int,
) -> tuple[Any, ...]:
    runtime_inputs = _make_runtime_inputs(
        session=session,
        kv_params=kv_params,
        batch_size=batch_size,
        seq_len=seq_len,
        device=device,
    )

    torch.manual_seed(x_seed)
    x = torch.randn(
        (batch_size * seq_len, hidden_size),
        dtype=torch.bfloat16,
        device="cuda",
    ).contiguous()

    return (
        Buffer.from_dlpack(x),
        _device_uint32_buffer(_row_offsets_array(batch_size, seq_len), device),
        _clone_kv_blocks(runtime_inputs.blocks, seed=blocks_seed),
        runtime_inputs.cache_lengths,
        runtime_inputs.lookup_table,
        runtime_inputs.max_lengths,
        runtime_inputs.attention_dispatch_metadata,
    )


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


def _to_bfloat16_tensor(value: Any) -> torch.Tensor:
    return from_dlpack(value).to(torch.bfloat16).contiguous()


def _pair_matrix_name(lhs_name: str, rhs_name: str) -> str:
    return f"{lhs_name}_vs_{rhs_name}"


def _pair_matrix_localization(
    failures: list[dict[str, str]],
) -> str:
    failed_pairs = {failure["pair"] for failure in failures}
    if failed_pairs == {"attention_baseline_vs_attention_fused"}:
        return "production_pair_only_fails_with_manual_pairings_passing"
    if "manual_baseline_graph_vs_attention_baseline" in failed_pairs:
        return "baseline_wrapper_diverges_from_manual_baseline_graph"
    if "manual_fused_graph_vs_attention_fused" in failed_pairs:
        return "fused_wrapper_diverges_from_manual_fused_graph"
    if "manual_baseline_graph_vs_manual_fused_graph" in failed_pairs:
        return "manual_baseline_and_manual_fused_graphs_diverge"
    if "attention_baseline_vs_attention_fused" in failed_pairs:
        return "production_baseline_and_fused_wrappers_diverge_in_pair_matrix"
    return "pair_matrix_found_mixed_divergence"


def _build_pair_matrix_graphs(
    *,
    session: InferenceSession,
    kv_params: KVCacheParams,
    config: dict[str, Any],
    weight_registry: dict[str, torch.Tensor],
    batch_size: int,
    seq_len: int,
    layer_idx: int,
    layer_type: str,
) -> dict[str, Any]:
    graphs: dict[str, Any] = {}
    for graph_name, compare_mode, use_fused in PAIR_MATRIX_GRAPH_SPECS:
        graphs[graph_name] = _build_graph(
            session=session,
            kv_params=kv_params,
            config=config,
            weight_registry=weight_registry,
            batch_size=batch_size,
            seq_len=seq_len,
            use_fused=use_fused,
            fused_variant=DEFAULT_FUSED_VARIANT,
            layer_idx=layer_idx,
            layer_type=layer_type,
            compare_mode=compare_mode,
        )
    return graphs


def _make_pair_matrix_execution_args(
    *,
    session: InferenceSession,
    kv_params: KVCacheParams,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    device: Accelerator,
    x_seed: int,
    blocks_seed: int,
) -> dict[str, tuple[Any, ...]]:
    return {
        graph_name: _make_execution_args(
            session=session,
            kv_params=kv_params,
            batch_size=batch_size,
            seq_len=seq_len,
            hidden_size=hidden_size,
            device=device,
            x_seed=x_seed,
            blocks_seed=blocks_seed,
        )
        for graph_name, _, _ in PAIR_MATRIX_GRAPH_SPECS
    }


def _compare_pair_matrix_tensors(
    lhs_name: str,
    rhs_name: str,
    stage: str,
    lhs_tensor: torch.Tensor,
    rhs_tensor: torch.Tensor,
) -> dict[str, str]:
    result = {"status": "pass"}
    try:
        torch.testing.assert_close(
            lhs_tensor,
            rhs_tensor,
            rtol=2 * torch.finfo(torch.bfloat16).eps,
            atol=8 * torch.finfo(torch.bfloat16).eps,
        )
    except AssertionError as exc:
        result = {
            "status": "failed",
            "details": str(exc),
            "stage": stage,
            "pair": _pair_matrix_name(lhs_name, rhs_name),
        }
    return result


def _run_pair_matrix_correctness_check(
    *,
    compiled: dict[str, Any],
    execution_args: dict[str, tuple[Any, ...]],
    layer_idx: int,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    outputs = {
        graph_name: _to_bfloat16_tensor(compiled_graph.execute(*execution_args[graph_name])[0])
        for graph_name, compiled_graph in compiled.items()
    }
    torch.cuda.synchronize()

    kv_blocks = {
        graph_name: _to_bfloat16_tensor(graph_args[2])[
            :, :, layer_idx : layer_idx + 1, :, :, :
        ]
        for graph_name, graph_args in execution_args.items()
    }

    pair_results: dict[str, Any] = {}
    failures: list[dict[str, str]] = []
    for lhs_name, rhs_name in PAIR_MATRIX_PAIRS:
        pair_name = _pair_matrix_name(lhs_name, rhs_name)
        attention_output_result = _compare_pair_matrix_tensors(
            lhs_name,
            rhs_name,
            "attention_output",
            outputs[lhs_name],
            outputs[rhs_name],
        )
        kv_result = _compare_pair_matrix_tensors(
            lhs_name,
            rhs_name,
            "kv_cache_layer_slice",
            kv_blocks[lhs_name],
            kv_blocks[rhs_name],
        )
        pair_results[pair_name] = {
            "attention_output": attention_output_result,
            "kv_cache_layer_slice": kv_result,
        }
        for result in (attention_output_result, kv_result):
            if result["status"] == "failed":
                failures.append(
                    {
                        "pair": result["pair"],
                        "stage": result["stage"],
                        "details": result["details"],
                    }
                )

    return pair_results, failures


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


def _geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def test_profile_attention_prefill() -> None:
    config = _load_text_config()
    layer_idx, layer_type = _resolve_layer_metadata(config)
    kv_num_layers = _resolve_kv_num_layers(config, layer_idx)
    compare_mode = _resolve_compare_mode()
    fused_variant = _resolve_fused_variant()
    prefill_shapes = _resolve_prefill_shapes()
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
    run_name_suffix = (
        "_k_only"
        if compare_mode == DEFAULT_COMPARE_MODE and fused_variant == "k_only"
        else ""
    )

    if (
        compare_mode != DEFAULT_COMPARE_MODE
        and fused_variant != DEFAULT_FUSED_VARIANT
    ):
        raise ValueError(
            "PROFILE_ATTENTION_PREFILL_FUSED_VARIANT only supports the "
            "baseline_vs_fused compare mode"
        )

    if compare_mode == "manual_baseline_graph_vs_attention_baseline":
        benchmark_mode = "prefill-ragged-full-attention-baseline-graph-parity"
        baseline_impl = "manual_baseline_graph"
        fused_impl = "attention_baseline"
        ratio_metric_key = (
            "average_runtime_ratio_manual_baseline_graph_over_"
            "attention_baseline"
        )
        average_geomean_key = (
            "average_geomean_runtime_ratio_manual_baseline_graph_over_"
            "attention_baseline"
        )
        average_large_geomean_key = (
            "average_large_shape_geomean_runtime_ratio_manual_baseline_graph_"
            "over_attention_baseline"
        )
        confirm_geomean_key = (
            "confirm_geomean_runtime_ratio_manual_baseline_graph_over_"
            "attention_baseline"
        )
        confirm_large_geomean_key = (
            "confirm_large_shape_geomean_runtime_ratio_manual_baseline_graph_"
            "over_attention_baseline"
        )
    elif compare_mode == "manual_fused_graph_vs_attention_fused":
        benchmark_mode = "prefill-ragged-full-attention-fused-graph-parity"
        baseline_impl = "manual_fused_graph"
        fused_impl = "attention_fused"
        ratio_metric_key = (
            "average_runtime_ratio_manual_fused_graph_over_attention_fused"
        )
        average_geomean_key = (
            "average_geomean_runtime_ratio_manual_fused_graph_over_"
            "attention_fused"
        )
        average_large_geomean_key = (
            "average_large_shape_geomean_runtime_ratio_manual_fused_graph_"
            "over_attention_fused"
        )
        confirm_geomean_key = (
            "confirm_geomean_runtime_ratio_manual_fused_graph_over_"
            "attention_fused"
        )
        confirm_large_geomean_key = (
            "confirm_large_shape_geomean_runtime_ratio_manual_fused_graph_"
            "over_attention_fused"
        )
    else:
        benchmark_mode = (
            "prefill-ragged-full-attention-k-only"
            if fused_variant == "k_only"
            else "prefill-ragged-full-attention"
        )
        baseline_impl = "attention_baseline"
        fused_impl = (
            "attention_k_only"
            if fused_variant == "k_only"
            else "attention_fused"
        )
        ratio_metric_key = "average_speedup_ratio_vs_attention_baseline"
        average_geomean_key = "average_geomean_speedup_vs_attention_baseline"
        average_large_geomean_key = (
            "average_large_shape_geomean_speedup_vs_attention_baseline"
        )
        confirm_geomean_key = "confirm_geomean_speedup_vs_attention_baseline"
        confirm_large_geomean_key = (
            "confirm_large_shape_geomean_speedup_vs_attention_baseline"
        )

    results: dict[str, Any] = {
        "benchmark_config": {
            "dtype": "bfloat16",
            "hidden_size": config["hidden_size"],
            "head_dim": config["head_dim"],
            "num_q_heads": config["num_attention_heads"],
            "num_kv_heads": config["num_key_value_heads"],
            "page_size": PAGE_SIZE,
            "layer_idx": layer_idx,
            "layer_type": layer_type,
            "kv_num_layers": kv_num_layers,
            "rope": "non-interleaved",
            "mode": benchmark_mode,
            "compare_mode": compare_mode,
            "fused_variant": fused_variant,
            "baseline_impl": baseline_impl,
            "fused_impl": fused_impl,
            "shapes": [
                _prefill_run_name(
                    layer_type,
                    layer_idx,
                    batch_size,
                    seq_len,
                    compare_mode,
                    run_name_suffix,
                )
                for batch_size, seq_len in prefill_shapes
            ],
            "warmup_iters": WARMUP_ITERS,
            "timed_iters": TIMED_ITERS,
            "placeholder_cache_len": PLACEHOLDER_CACHE_LEN,
        },
        "correctness": "pass",
        "first_sweep_us": {},
        "confirm_sweep_us": {},
        "average_us": {},
        ratio_metric_key: {},
    }

    if gpu_isolation_guard is not None:
        results["benchmark_config"]["gpu_isolation_guard"] = {
            "allowed_pid": gpu_isolation_guard["allowed_pid"],
            "target_gpu_index": gpu_isolation_guard["target_gpu_index"],
            "target_gpu_uuid": gpu_isolation_guard["target_gpu_uuid"],
        }

    if compare_mode == PAIR_MATRIX_COMPARE_MODE:
        results = {
            "benchmark_config": {
                "dtype": "bfloat16",
                "hidden_size": config["hidden_size"],
                "head_dim": config["head_dim"],
                "num_q_heads": config["num_attention_heads"],
                "num_kv_heads": config["num_key_value_heads"],
                "page_size": PAGE_SIZE,
                "layer_idx": layer_idx,
                "layer_type": layer_type,
                "kv_num_layers": kv_num_layers,
                "rope": "non-interleaved",
                "mode": "prefill-ragged-full-attention-pair-matrix",
                "compare_mode": compare_mode,
                "fused_variant": fused_variant,
                "graph_names": [graph_name for graph_name, _, _ in PAIR_MATRIX_GRAPH_SPECS],
                "pairs": [
                    _pair_matrix_name(lhs_name, rhs_name)
                    for lhs_name, rhs_name in PAIR_MATRIX_PAIRS
                ],
                "shapes": [
                    _prefill_run_name(
                        layer_type,
                        layer_idx,
                        batch_size,
                        seq_len,
                        compare_mode,
                        run_name_suffix,
                    )
                    for batch_size, seq_len in prefill_shapes
                ],
                "placeholder_cache_len": PLACEHOLDER_CACHE_LEN,
            },
            "correctness": "pass",
            "pair_results": {},
        }

        if gpu_isolation_guard is not None:
            results["benchmark_config"]["gpu_isolation_guard"] = {
                "allowed_pid": gpu_isolation_guard["allowed_pid"],
                "target_gpu_index": gpu_isolation_guard["target_gpu_index"],
                "target_gpu_uuid": gpu_isolation_guard["target_gpu_uuid"],
            }

        for batch_size, seq_len in prefill_shapes:
            run_name = _prefill_run_name(
                layer_type,
                layer_idx,
                batch_size,
                seq_len,
                compare_mode,
                run_name_suffix,
            )
            compiled = _build_pair_matrix_graphs(
                session=session,
                kv_params=kv_params,
                config=config,
                weight_registry=weight_registry,
                batch_size=batch_size,
                seq_len=seq_len,
                layer_idx=layer_idx,
                layer_type=layer_type,
            )
            correctness_args = _make_pair_matrix_execution_args(
                session=session,
                kv_params=kv_params,
                batch_size=batch_size,
                seq_len=seq_len,
                hidden_size=config["hidden_size"],
                device=device,
                x_seed=1000 + batch_size + seq_len,
                blocks_seed=2000 + batch_size + seq_len,
            )
            pair_results, failures = _run_pair_matrix_correctness_check(
                compiled=compiled,
                execution_args=correctness_args,
                layer_idx=layer_idx,
            )
            results["pair_results"][run_name] = pair_results
            if failures:
                results["correctness"] = "failed"
                results["failure_shape"] = run_name
                results["failures"] = failures
                results["localization"] = _pair_matrix_localization(
                    failures
                )
                print("GEMMA3_ATTENTION_PREFILL_PROFILE_START")
                print(json.dumps(results, indent=2, sort_keys=True))
                print("GEMMA3_ATTENTION_PREFILL_PROFILE_END")
                raise AssertionError(
                    f"{run_name} pair-matrix failures: {failures}"
                )

            del correctness_args
            del compiled
            torch.cuda.empty_cache()

        results["localization"] = "pair_matrix_passed_all_pairs"
        print("GEMMA3_ATTENTION_PREFILL_PROFILE_START")
        print(json.dumps(results, indent=2, sort_keys=True))
        print("GEMMA3_ATTENTION_PREFILL_PROFILE_END")
        return

    large_shape_names: list[str] = []

    for batch_size, seq_len in prefill_shapes:
        run_name = _prefill_run_name(
            layer_type,
            layer_idx,
            batch_size,
            seq_len,
            compare_mode,
            run_name_suffix,
        )
        if seq_len >= 512:
            large_shape_names.append(run_name)

        baseline = _build_graph(
            session=session,
            kv_params=kv_params,
            config=config,
            weight_registry=weight_registry,
            batch_size=batch_size,
            seq_len=seq_len,
            use_fused=False,
            fused_variant=fused_variant,
            layer_idx=layer_idx,
            layer_type=layer_type,
            compare_mode=compare_mode,
        )
        fused = _build_graph(
            session=session,
            kv_params=kv_params,
            config=config,
            weight_registry=weight_registry,
            batch_size=batch_size,
            seq_len=seq_len,
            use_fused=True,
            fused_variant=fused_variant,
            layer_idx=layer_idx,
            layer_type=layer_type,
            compare_mode=compare_mode,
        )

        baseline_correctness_args = _make_execution_args(
            session=session,
            kv_params=kv_params,
            batch_size=batch_size,
            seq_len=seq_len,
            hidden_size=config["hidden_size"],
            device=device,
            x_seed=1000 + batch_size + seq_len,
            blocks_seed=2000 + batch_size + seq_len,
        )
        fused_correctness_args = _make_execution_args(
            session=session,
            kv_params=kv_params,
            batch_size=batch_size,
            seq_len=seq_len,
            hidden_size=config["hidden_size"],
            device=device,
            x_seed=1000 + batch_size + seq_len,
            blocks_seed=2000 + batch_size + seq_len,
        )
        _run_correctness_check(
            baseline=baseline,
            fused=fused,
            baseline_args=baseline_correctness_args,
            fused_args=fused_correctness_args,
            layer_idx=layer_idx,
        )
        del baseline_correctness_args
        del fused_correctness_args
        torch.cuda.empty_cache()

        first_baseline_args = [
            _make_execution_args(
                session=session,
                kv_params=kv_params,
                batch_size=batch_size,
                seq_len=seq_len,
                hidden_size=config["hidden_size"],
                device=device,
                x_seed=3000 + batch_size + seq_len + iteration,
                blocks_seed=4000 + batch_size + seq_len + iteration,
            )
            for iteration in range(WARMUP_ITERS + TIMED_ITERS)
        ]
        first_fused_args = [
            _make_execution_args(
                session=session,
                kv_params=kv_params,
                batch_size=batch_size,
                seq_len=seq_len,
                hidden_size=config["hidden_size"],
                device=device,
                x_seed=3000 + batch_size + seq_len + iteration,
                blocks_seed=4000 + batch_size + seq_len + iteration,
            )
            for iteration in range(WARMUP_ITERS + TIMED_ITERS)
        ]
        first_baseline_us = _benchmark_us_guarded(
            baseline,
            first_baseline_args,
            gpu_isolation_guard=gpu_isolation_guard,
            label=f"{run_name}:baseline:first_sweep",
        )
        first_fused_us = _benchmark_us_guarded(
            fused,
            first_fused_args,
            gpu_isolation_guard=gpu_isolation_guard,
            label=f"{run_name}:fused:first_sweep",
        )
        results["first_sweep_us"][run_name] = {
            "baseline": first_baseline_us,
            "fused": first_fused_us,
        }
        del first_baseline_args
        del first_fused_args
        torch.cuda.empty_cache()

        confirm_baseline_args = [
            _make_execution_args(
                session=session,
                kv_params=kv_params,
                batch_size=batch_size,
                seq_len=seq_len,
                hidden_size=config["hidden_size"],
                device=device,
                x_seed=5000 + batch_size + seq_len + iteration,
                blocks_seed=6000 + batch_size + seq_len + iteration,
            )
            for iteration in range(WARMUP_ITERS + TIMED_ITERS)
        ]
        confirm_fused_args = [
            _make_execution_args(
                session=session,
                kv_params=kv_params,
                batch_size=batch_size,
                seq_len=seq_len,
                hidden_size=config["hidden_size"],
                device=device,
                x_seed=5000 + batch_size + seq_len + iteration,
                blocks_seed=6000 + batch_size + seq_len + iteration,
            )
            for iteration in range(WARMUP_ITERS + TIMED_ITERS)
        ]
        confirm_baseline_us = _benchmark_us_guarded(
            baseline,
            confirm_baseline_args,
            gpu_isolation_guard=gpu_isolation_guard,
            label=f"{run_name}:baseline:confirm_sweep",
        )
        confirm_fused_us = _benchmark_us_guarded(
            fused,
            confirm_fused_args,
            gpu_isolation_guard=gpu_isolation_guard,
            label=f"{run_name}:fused:confirm_sweep",
        )
        results["confirm_sweep_us"][run_name] = {
            "baseline": confirm_baseline_us,
            "fused": confirm_fused_us,
        }
        del confirm_baseline_args
        del confirm_fused_args
        torch.cuda.empty_cache()

        average_baseline_us = (first_baseline_us + confirm_baseline_us) / 2.0
        average_fused_us = (first_fused_us + confirm_fused_us) / 2.0
        results["average_us"][run_name] = {
            "baseline": average_baseline_us,
            "fused": average_fused_us,
        }
        results[ratio_metric_key][run_name] = average_baseline_us / average_fused_us

        del baseline
        del fused
        torch.cuda.empty_cache()

    ratios = list(results[ratio_metric_key].values())
    results[average_geomean_key] = _geomean(ratios)
    results[average_large_geomean_key] = _geomean(
        [
            results[ratio_metric_key][run_name]
            for run_name in large_shape_names
        ]
    )

    confirm_ratios = [
        results["confirm_sweep_us"][run_name]["baseline"]
        / results["confirm_sweep_us"][run_name]["fused"]
        for run_name in results["confirm_sweep_us"]
    ]
    results[confirm_geomean_key] = _geomean(confirm_ratios)
    results[confirm_large_geomean_key] = _geomean(
        [
            results["confirm_sweep_us"][run_name]["baseline"]
            / results["confirm_sweep_us"][run_name]["fused"]
            for run_name in large_shape_names
        ]
    )
    if gpu_isolation_guard is not None:
        results["gpu_isolation_guard"] = "pass"

    print("GEMMA3_ATTENTION_PREFILL_PROFILE_START")
    print(json.dumps(results, indent=2, sort_keys=True))
    print("GEMMA3_ATTENTION_PREFILL_PROFILE_END")
