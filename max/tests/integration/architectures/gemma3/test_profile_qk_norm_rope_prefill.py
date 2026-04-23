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
from max.graph import DeviceRef, Dim, Graph, TensorType, ops
from max.interfaces import RequestID, TokenBuffer
from max.kv_cache import PagedKVCacheManager
from max.nn.kernels import (
    KVCacheParams,
    fused_qk_ragged_rope,
    q_rms_norm_fused_qk_ragged_rope,
    rms_norm_key_cache,
)
from max.nn.kv_cache import (
    KVCacheInputsPerDevice,
    unflatten_ragged_attention_inputs,
)
from max.nn.rotary_embedding import Llama3RotaryEmbedding
from max.pipelines.architectures.gemma3.layers.rms_norm import Gemma3RMSNorm
from max.pipelines.core import TextContext
from torch.utils.dlpack import from_dlpack

PAGE_SIZE = 128
EPS = 1e-6
Q_NORM_STD = 0.68
K_NORM_STD = 0.793
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


def _resolve_prefill_shapes() -> tuple[tuple[int, int], ...]:
    raw_shapes = os.environ.get(
        "PROFILE_QK_NORM_ROPE_PREFILL_SHAPES", ""
    ).strip()
    if raw_shapes == "":
        return PREFILL_SHAPES

    resolved: list[tuple[int, int]] = []
    for raw_shape in raw_shapes.split(","):
        shape_text = raw_shape.strip().lower()
        if shape_text == "":
            continue
        if "x" not in shape_text:
            raise ValueError(
                "PROFILE_QK_NORM_ROPE_PREFILL_SHAPES entries must look like "
                f"'batchxseq', got {raw_shape!r}"
            )
        batch_text, seq_text = shape_text.split("x", maxsplit=1)
        shape = (int(batch_text), int(seq_text))
        if shape not in PREFILL_SHAPES:
            raise ValueError(
                "PROFILE_QK_NORM_ROPE_PREFILL_SHAPES only supports the existing "
                f"prefill grid {PREFILL_SHAPES}, got {shape}"
            )
        if shape not in resolved:
            resolved.append(shape)

    if not resolved:
        raise ValueError(
            "PROFILE_QK_NORM_ROPE_PREFILL_SHAPES must select at least one shape"
        )
    return tuple(resolved)


def _load_text_config() -> dict[str, Any]:
    config_path = Path(os.environ["PIPELINES_TESTDATA"]) / "config.json"
    with open(config_path) as file:
        data = json.load(file)
    return data.get("text_config", data)


def _make_weight_registry(head_dim: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(42)
    return {
        "q_gamma": torch.randn(head_dim, dtype=torch.bfloat16) * Q_NORM_STD,
        "k_gamma": torch.randn(head_dim, dtype=torch.bfloat16) * K_NORM_STD,
    }


def _make_placeholder_text_context(max_length: int) -> TextContext:
    return TextContext(
        request_id=RequestID(),
        max_length=max_length,
        tokens=TokenBuffer(np.zeros(PLACEHOLDER_CACHE_LEN, dtype=np.int64)),
    )


def _device_uint32_buffer(array: np.ndarray, device: Accelerator) -> Buffer:
    return Buffer.from_numpy(array).to(device)


def _row_offsets_array(batch_size: int, seq_len: int) -> np.ndarray:
    return np.arange(
        0,
        (batch_size + 1) * seq_len,
        seq_len,
        dtype=np.uint32,
    )


def _prefill_run_name(batch_size: int, seq_len: int) -> str:
    return f"prefill_bs{batch_size}_seq{seq_len}_cache0_step0_qk"


def _build_graph(
    *,
    session: InferenceSession,
    kv_params: KVCacheParams,
    config: dict[str, Any],
    weight_registry: dict[str, torch.Tensor],
    batch_size: int,
    seq_len: int,
    use_fused: bool,
) -> Any:
    device_ref = DeviceRef.GPU()
    total_seq_len = Dim(batch_size * seq_len)
    rope = Llama3RotaryEmbedding(
        config["hidden_size"],
        config["num_attention_heads"],
        config["rope_local_base_freq"],
        seq_len + PLACEHOLDER_CACHE_LEN + 256,
        interleaved=False,
        head_dim=config["head_dim"],
    )
    q_norm = Gemma3RMSNorm(config["head_dim"], DType.bfloat16, EPS)
    k_norm = Gemma3RMSNorm(config["head_dim"], DType.bfloat16, EPS)
    q_norm.weight.name = "q_gamma"
    k_norm.weight.name = "k_gamma"

    input_type = TensorType(
        DType.bfloat16,
        [
            batch_size * seq_len,
            config["num_attention_heads"],
            config["head_dim"],
        ],
        device=device_ref,
    )
    input_row_offsets_type = TensorType(
        DType.uint32,
        [batch_size + 1],
        device=device_ref,
    )
    flattened_kv_types = kv_params.get_symbolic_inputs().flatten()

    graph_name = (
        f"Gemma3WideQKNormRopePrefillFusedBS{batch_size}Seq{seq_len}"
        if use_fused
        else f"Gemma3WideQKNormRopePrefillBaselineBS{batch_size}Seq{seq_len}"
    )

    with Graph(
        graph_name,
        input_types=(input_type, input_row_offsets_type, *flattened_kv_types),
    ) as graph:
        xq, input_row_offsets, *kv_cache = graph.inputs
        kv_collection = unflatten_ragged_attention_inputs(
            kv_cache, n_devices=1
        )[0]
        layer_idx = ops.constant(0, DType.uint32, device=DeviceRef.CPU())
        freqs_cis = ops.cast(rope.freqs_cis, DType.bfloat16).to(device_ref)

        if use_fused:
            output = q_rms_norm_fused_qk_ragged_rope(
                kv_params,
                xq.tensor,
                input_row_offsets.tensor,
                kv_collection,
                freqs_cis,
                q_norm.weight.cast(kv_params.dtype).to(device_ref),
                k_norm.weight.cast(kv_params.dtype).to(device_ref),
                EPS,
                layer_idx,
                weight_offset=1.0,
                interleaved=False,
            )
        else:
            rms_norm_key_cache(
                kv_params,
                kv_collection=kv_collection,
                gamma=k_norm.weight.cast(kv_params.dtype).to(device_ref),
                epsilon=EPS,
                layer_idx=layer_idx,
                total_seq_len=total_seq_len,
                input_row_offsets=input_row_offsets.tensor,
                weight_offset=1.0,
            )
            output = q_norm(xq.tensor)
            output = fused_qk_ragged_rope(
                kv_params,
                output,
                input_row_offsets.tensor,
                kv_collection,
                freqs_cis,
                layer_idx,
                interleaved=False,
            )

        graph.output(output)

    return session.load(graph, weights_registry=weight_registry)


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

    return KVCacheInputsPerDevice(
        blocks=runtime_inputs.blocks,
        cache_lengths=_device_uint32_buffer(
            np.zeros(batch_size, dtype=np.uint32), device
        ),
        lookup_table=runtime_inputs.lookup_table,
        max_lengths=runtime_inputs.max_lengths,
        kv_scales=runtime_inputs.kv_scales,
        attention_dispatch_metadata=runtime_inputs.attention_dispatch_metadata,
    )


def _clone_kv_blocks(blocks: Buffer, seed: int) -> Buffer:
    torch.manual_seed(seed)
    shape = tuple(int(dim) for dim in blocks.shape)
    tensor = torch.randn(
        shape, dtype=torch.bfloat16, device="cuda"
    ).contiguous()
    return Buffer.from_dlpack(tensor)


def _make_execution_args(
    *,
    session: InferenceSession,
    kv_params: KVCacheParams,
    batch_size: int,
    seq_len: int,
    num_attention_heads: int,
    head_dim: int,
    device: Accelerator,
    xq_seed: int,
    blocks_seed: int,
) -> tuple[Any, ...]:
    runtime_inputs = _make_runtime_inputs(
        session=session,
        kv_params=kv_params,
        batch_size=batch_size,
        seq_len=seq_len,
        device=device,
    )

    torch.manual_seed(xq_seed)
    xq = torch.randn(
        (batch_size * seq_len, num_attention_heads, head_dim),
        dtype=torch.bfloat16,
        device="cuda",
    ).contiguous()

    return (
        Buffer.from_dlpack(xq),
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

    torch.testing.assert_close(
        from_dlpack(baseline_args[2]).to(torch.bfloat16),
        from_dlpack(fused_args[2]).to(torch.bfloat16),
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
        "PROFILE_QK_NORM_ROPE_PREFILL_ENFORCE_GPU_ISOLATION", ""
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


def test_profile_qk_norm_rope_prefill() -> None:
    config = _load_text_config()
    prefill_shapes = _resolve_prefill_shapes()
    session = InferenceSession(devices=[Accelerator(0)])
    device = Accelerator(0)
    gpu_isolation_guard = _build_gpu_isolation_guard()
    kv_params = KVCacheParams(
        dtype=DType.bfloat16,
        devices=[DeviceRef.GPU()],
        n_kv_heads=config["num_key_value_heads"],
        head_dim=config["head_dim"],
        num_layers=1,
        page_size=PAGE_SIZE,
    )
    weight_registry = _make_weight_registry(config["head_dim"])

    results: dict[str, Any] = {
        "benchmark_config": {
            "dtype": "bfloat16",
            "head_dim": config["head_dim"],
            "num_q_heads": config["num_attention_heads"],
            "num_kv_heads": config["num_key_value_heads"],
            "page_size": PAGE_SIZE,
            "rope": "non-interleaved",
            "mode": "prefill-ragged-live-graph-qk-one-shot",
            "shapes": [
                _prefill_run_name(batch_size, seq_len)
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
        "average_speedup_ratio_vs_graph_baseline": {},
    }

    large_shape_names: list[str] = []

    for batch_size, seq_len in prefill_shapes:
        run_name = _prefill_run_name(batch_size, seq_len)
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
        )
        fused = _build_graph(
            session=session,
            kv_params=kv_params,
            config=config,
            weight_registry=weight_registry,
            batch_size=batch_size,
            seq_len=seq_len,
            use_fused=True,
        )

        baseline_correctness_args = _make_execution_args(
            session=session,
            kv_params=kv_params,
            batch_size=batch_size,
            seq_len=seq_len,
            num_attention_heads=config["num_attention_heads"],
            head_dim=config["head_dim"],
            device=device,
            xq_seed=1000 + batch_size + seq_len,
            blocks_seed=2000 + batch_size + seq_len,
        )
        fused_correctness_args = _make_execution_args(
            session=session,
            kv_params=kv_params,
            batch_size=batch_size,
            seq_len=seq_len,
            num_attention_heads=config["num_attention_heads"],
            head_dim=config["head_dim"],
            device=device,
            xq_seed=1000 + batch_size + seq_len,
            blocks_seed=2000 + batch_size + seq_len,
        )
        _run_correctness_check(
            baseline=baseline,
            fused=fused,
            baseline_args=baseline_correctness_args,
            fused_args=fused_correctness_args,
        )

        first_baseline_args = [
            _make_execution_args(
                session=session,
                kv_params=kv_params,
                batch_size=batch_size,
                seq_len=seq_len,
                num_attention_heads=config["num_attention_heads"],
                head_dim=config["head_dim"],
                device=device,
                xq_seed=3000 + batch_size + seq_len + iteration,
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
                num_attention_heads=config["num_attention_heads"],
                head_dim=config["head_dim"],
                device=device,
                xq_seed=3000 + batch_size + seq_len + iteration,
                blocks_seed=4000 + batch_size + seq_len + iteration,
            )
            for iteration in range(WARMUP_ITERS + TIMED_ITERS)
        ]
        first_baseline_us = _benchmark_us_guarded(
            baseline,
            first_baseline_args,
            gpu_isolation_guard=gpu_isolation_guard,
            label=f"{run_name}:baseline:first",
        )
        first_fused_us = _benchmark_us_guarded(
            fused,
            first_fused_args,
            gpu_isolation_guard=gpu_isolation_guard,
            label=f"{run_name}:fused:first",
        )
        results["first_sweep_us"][run_name] = {
            "baseline": first_baseline_us,
            "fused": first_fused_us,
        }
        del first_baseline_args
        del first_fused_args

        confirm_baseline_args = [
            _make_execution_args(
                session=session,
                kv_params=kv_params,
                batch_size=batch_size,
                seq_len=seq_len,
                num_attention_heads=config["num_attention_heads"],
                head_dim=config["head_dim"],
                device=device,
                xq_seed=5000 + batch_size + seq_len + iteration,
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
                num_attention_heads=config["num_attention_heads"],
                head_dim=config["head_dim"],
                device=device,
                xq_seed=5000 + batch_size + seq_len + iteration,
                blocks_seed=6000 + batch_size + seq_len + iteration,
            )
            for iteration in range(WARMUP_ITERS + TIMED_ITERS)
        ]
        confirm_baseline_us = _benchmark_us_guarded(
            baseline,
            confirm_baseline_args,
            gpu_isolation_guard=gpu_isolation_guard,
            label=f"{run_name}:baseline:confirm",
        )
        confirm_fused_us = _benchmark_us_guarded(
            fused,
            confirm_fused_args,
            gpu_isolation_guard=gpu_isolation_guard,
            label=f"{run_name}:fused:confirm",
        )
        results["confirm_sweep_us"][run_name] = {
            "baseline": confirm_baseline_us,
            "fused": confirm_fused_us,
        }
        del confirm_baseline_args
        del confirm_fused_args
        del baseline_correctness_args
        del fused_correctness_args

        average_baseline_us = (first_baseline_us + confirm_baseline_us) / 2.0
        average_fused_us = (first_fused_us + confirm_fused_us) / 2.0
        results["average_us"][run_name] = {
            "baseline": average_baseline_us,
            "fused": average_fused_us,
        }
        results["average_speedup_ratio_vs_graph_baseline"][run_name] = (
            average_baseline_us / average_fused_us
        )

        del baseline
        del fused

    ratios = list(results["average_speedup_ratio_vs_graph_baseline"].values())
    results["average_geomean_speedup_vs_graph_baseline"] = _geomean(ratios)
    results["average_large_shape_geomean_speedup_vs_graph_baseline"] = _geomean(
        [
            results["average_speedup_ratio_vs_graph_baseline"][run_name]
            for run_name in large_shape_names
        ]
    )

    confirm_ratios = [
        results["confirm_sweep_us"][run_name]["baseline"]
        / results["confirm_sweep_us"][run_name]["fused"]
        for run_name in results["confirm_sweep_us"]
    ]
    results["confirm_geomean_speedup_vs_graph_baseline"] = _geomean(
        confirm_ratios
    )
    results["confirm_large_shape_geomean_speedup_vs_graph_baseline"] = _geomean(
        [
            results["confirm_sweep_us"][run_name]["baseline"]
            / results["confirm_sweep_us"][run_name]["fused"]
            for run_name in large_shape_names
        ]
    )

    print("GEMMA3_WIDE_QK_NORM_ROPE_PREFILL_PROFILE_START")
    print(json.dumps(results, indent=2, sort_keys=True))
    print("GEMMA3_WIDE_QK_NORM_ROPE_PREFILL_PROFILE_END")
