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

import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn.kernels import q_rms_norm_rope_ragged, rope_ragged
from max.nn.rotary_embedding import Llama3RotaryEmbedding
from max.pipelines.architectures.gemma3.layers.rms_norm import Gemma3RMSNorm
from torch.utils.dlpack import from_dlpack

EPS = 1e-6
Q_NORM_STD = 0.68
WARMUP_ITERS = 20
TIMED_ITERS = 50
CACHE_LEN_BASE = 1024
CACHE_LEN_STEP = 7
ITERATION_STEP = 1
BATCH_SIZES = (64, 128)
MAX_EXTRA_STEPS = WARMUP_ITERS + TIMED_ITERS


def _load_text_config() -> dict[str, Any]:
    config_path = Path(os.environ["PIPELINES_TESTDATA"]) / "config.json"
    with open(config_path) as file:
        data = json.load(file)
    return data.get("text_config", data)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value is not None else default


def _env_int_tuple(name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    value = os.environ.get(name)
    if value is None:
        return default
    return tuple(
        int(part.strip()) for part in value.split(",") if part.strip()
    )


def _profile_config(config: dict[str, Any]) -> dict[str, Any]:
    batch_sizes = _env_int_tuple(
        "PROFILE_Q_NORM_ROPE_DECODE_BATCH_SIZES", BATCH_SIZES
    )
    if not batch_sizes:
        raise ValueError(
            "PROFILE_Q_NORM_ROPE_DECODE_BATCH_SIZES must define at least one batch size"
        )

    num_attention_heads = _env_int(
        "PROFILE_Q_NORM_ROPE_NUM_Q_HEADS", config["num_attention_heads"]
    )
    head_dim = _env_int("PROFILE_Q_NORM_ROPE_HEAD_DIM", config["head_dim"])
    hidden_size = os.environ.get("PROFILE_Q_NORM_ROPE_HIDDEN_SIZE")
    if hidden_size is None and (
        num_attention_heads != config["num_attention_heads"]
        or head_dim != config["head_dim"]
    ):
        resolved_hidden_size = num_attention_heads * head_dim
    else:
        resolved_hidden_size = (
            int(hidden_size)
            if hidden_size is not None
            else config["hidden_size"]
        )

    return {
        "hidden_size": resolved_hidden_size,
        "num_attention_heads": num_attention_heads,
        "head_dim": head_dim,
        "rope_theta": _env_int(
            "PROFILE_Q_NORM_ROPE_THETA", config["rope_local_base_freq"]
        ),
        "cache_len_base": _env_int(
            "PROFILE_Q_NORM_ROPE_CACHE_LEN_BASE", CACHE_LEN_BASE
        ),
        "cache_len_step": _env_int(
            "PROFILE_Q_NORM_ROPE_CACHE_LEN_STEP", CACHE_LEN_STEP
        ),
        "iteration_step": _env_int(
            "PROFILE_Q_NORM_ROPE_DECODE_ITERATION_STEP", ITERATION_STEP
        ),
        "batch_sizes": batch_sizes,
    }


def _make_weight_registry(head_dim: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(42)
    return {
        "q_gamma": torch.randn(head_dim, dtype=torch.bfloat16) * Q_NORM_STD,
    }


def _device_uint32_buffer(array: np.ndarray, device: Accelerator) -> Buffer:
    return Buffer.from_numpy(array).to(device)


def _decode_run_name(
    batch_size: int,
    cache_len_base: int,
    cache_len_step: int,
    iteration_step: int,
) -> str:
    run_name = (
        f"decode_bs{batch_size}_seq1_cache{cache_len_base}"
        f"_step{cache_len_step}_q_only"
    )
    if iteration_step != ITERATION_STEP:
        return f"{run_name}_iterstep{iteration_step}"
    return run_name


def _build_graph(
    *,
    session: InferenceSession,
    config: dict[str, Any],
    batch_size: int,
    max_batch_size: int,
    use_fused: bool,
) -> Any:
    device_ref = DeviceRef.GPU()
    rope = Llama3RotaryEmbedding(
        config["hidden_size"],
        config["num_attention_heads"],
        config["rope_theta"],
        config["cache_len_base"]
        + config["cache_len_step"] * (max_batch_size - 1)
        + config["iteration_step"] * MAX_EXTRA_STEPS
        + 256,
        interleaved=False,
        head_dim=config["head_dim"],
    )
    q_norm = Gemma3RMSNorm(config["head_dim"], DType.bfloat16, EPS)
    q_norm.weight.name = "q_gamma"

    input_type = TensorType(
        DType.bfloat16,
        [batch_size, config["num_attention_heads"], config["head_dim"]],
        device=device_ref,
    )
    input_row_offsets_type = TensorType(
        DType.uint32,
        [batch_size + 1],
        device=device_ref,
    )
    start_pos_type = TensorType(
        DType.uint32,
        [batch_size],
        device=device_ref,
    )

    graph_name = (
        f"Gemma3WideQOnlyNormRopeDecodeFusedBS{batch_size}"
        if use_fused
        else f"Gemma3WideQOnlyNormRopeDecodeBaselineBS{batch_size}"
    )

    with Graph(
        graph_name,
        input_types=(input_type, input_row_offsets_type, start_pos_type),
    ) as graph:
        xq, input_row_offsets, start_pos = graph.inputs
        freqs_cis = ops.cast(rope.freqs_cis, DType.bfloat16).to(device_ref)
        gamma = q_norm.weight.cast(DType.bfloat16).to(device_ref)

        if use_fused:
            output = q_rms_norm_rope_ragged(
                xq.tensor,
                input_row_offsets.tensor,
                start_pos.tensor,
                freqs_cis,
                gamma,
                EPS,
                weight_offset=1.0,
                interleaved=False,
            )
        else:
            output = q_norm(xq.tensor)
            output = rope_ragged(
                output,
                input_row_offsets.tensor,
                start_pos.tensor,
                freqs_cis,
                interleaved=False,
            )

        graph.output(output)

    return session.load(
        graph,
        weights_registry=_make_weight_registry(config["head_dim"]),
    )


def _make_benchmark_args(
    *,
    batch_size: int,
    num_attention_heads: int,
    head_dim: int,
    cache_len_base: int,
    cache_len_step: int,
    iteration_step: int,
    device: Accelerator,
    xq_seed: int,
) -> tuple[list[tuple[Any, ...]], Buffer]:
    torch.manual_seed(xq_seed)
    xq = torch.randn(
        (batch_size, num_attention_heads, head_dim),
        dtype=torch.bfloat16,
        device="cuda",
    ).contiguous()
    xq_buffer = Buffer.from_dlpack(xq)
    row_offsets = _device_uint32_buffer(
        np.arange(batch_size + 1, dtype=np.uint32),
        device,
    )
    base_start_pos = np.asarray(
        [
            cache_len_base + cache_len_step * request_idx
            for request_idx in range(batch_size)
        ],
        dtype=np.uint32,
    )

    args: list[tuple[Any, ...]] = []
    for step in range(MAX_EXTRA_STEPS + 1):
        step_offset = np.uint32(step * iteration_step)
        start_pos = _device_uint32_buffer(
            base_start_pos + step_offset, device
        )
        args.append((xq_buffer, row_offsets, start_pos))

    return args, xq_buffer


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


def _geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def test_profile_q_norm_rope_decode() -> None:
    config = _profile_config(_load_text_config())
    session = InferenceSession(devices=[Accelerator(0)])
    device = Accelerator(0)

    results: dict[str, Any] = {
        "benchmark_config": {
            "dtype": "bfloat16",
            "hidden_size": config["hidden_size"],
            "head_dim": config["head_dim"],
            "num_q_heads": config["num_attention_heads"],
            "rope": "non-interleaved",
            "mode": "decode-ragged-live-graph-q-only",
            "selected_shapes": [
                _decode_run_name(
                    batch_size,
                    config["cache_len_base"],
                    config["cache_len_step"],
                    config["iteration_step"],
                )
                for batch_size in config["batch_sizes"]
            ],
            "cache_len_base": config["cache_len_base"],
            "cache_len_step": config["cache_len_step"],
            "iteration_step": config["iteration_step"],
            "warmup_iters": WARMUP_ITERS,
            "timed_iters": TIMED_ITERS,
        },
        "correctness": "pass",
        "first_sweep_us": {},
        "confirm_sweep_us": {},
        "average_us": {},
        "average_speedup_ratio_vs_graph_baseline": {},
    }

    speedups: list[float] = []

    for batch_size in config["batch_sizes"]:
        run_name = _decode_run_name(
            batch_size,
            config["cache_len_base"],
            config["cache_len_step"],
            config["iteration_step"],
        )
        baseline = _build_graph(
            session=session,
            config=config,
            batch_size=batch_size,
            max_batch_size=max(config["batch_sizes"]),
            use_fused=False,
        )
        fused = _build_graph(
            session=session,
            config=config,
            batch_size=batch_size,
            max_batch_size=max(config["batch_sizes"]),
            use_fused=True,
        )

        correctness_baseline_args, _ = _make_benchmark_args(
            batch_size=batch_size,
            num_attention_heads=config["num_attention_heads"],
            head_dim=config["head_dim"],
            cache_len_base=config["cache_len_base"],
            cache_len_step=config["cache_len_step"],
            iteration_step=config["iteration_step"],
            device=device,
            xq_seed=1000 + batch_size,
        )
        correctness_fused_args, _ = _make_benchmark_args(
            batch_size=batch_size,
            num_attention_heads=config["num_attention_heads"],
            head_dim=config["head_dim"],
            cache_len_base=config["cache_len_base"],
            cache_len_step=config["cache_len_step"],
            iteration_step=config["iteration_step"],
            device=device,
            xq_seed=1000 + batch_size,
        )
        _run_correctness_check(
            baseline=baseline,
            fused=fused,
            baseline_args=correctness_baseline_args[0],
            fused_args=correctness_fused_args[0],
        )
        del correctness_baseline_args
        del correctness_fused_args

        first_baseline_args, _ = _make_benchmark_args(
            batch_size=batch_size,
            num_attention_heads=config["num_attention_heads"],
            head_dim=config["head_dim"],
            cache_len_base=config["cache_len_base"],
            cache_len_step=config["cache_len_step"],
            iteration_step=config["iteration_step"],
            device=device,
            xq_seed=2000 + batch_size,
        )
        first_fused_args, _ = _make_benchmark_args(
            batch_size=batch_size,
            num_attention_heads=config["num_attention_heads"],
            head_dim=config["head_dim"],
            cache_len_base=config["cache_len_base"],
            cache_len_step=config["cache_len_step"],
            iteration_step=config["iteration_step"],
            device=device,
            xq_seed=2000 + batch_size,
        )
        first_baseline_us = _benchmark_us(baseline, first_baseline_args)
        first_fused_us = _benchmark_us(fused, first_fused_args)
        results["first_sweep_us"][run_name] = {
            "baseline": first_baseline_us,
            "fused": first_fused_us,
        }
        del first_baseline_args
        del first_fused_args

        confirm_baseline_args, _ = _make_benchmark_args(
            batch_size=batch_size,
            num_attention_heads=config["num_attention_heads"],
            head_dim=config["head_dim"],
            cache_len_base=config["cache_len_base"],
            cache_len_step=config["cache_len_step"],
            iteration_step=config["iteration_step"],
            device=device,
            xq_seed=3000 + batch_size,
        )
        confirm_fused_args, _ = _make_benchmark_args(
            batch_size=batch_size,
            num_attention_heads=config["num_attention_heads"],
            head_dim=config["head_dim"],
            cache_len_base=config["cache_len_base"],
            cache_len_step=config["cache_len_step"],
            iteration_step=config["iteration_step"],
            device=device,
            xq_seed=3000 + batch_size,
        )
        confirm_baseline_us = _benchmark_us(baseline, confirm_baseline_args)
        confirm_fused_us = _benchmark_us(fused, confirm_fused_args)
        results["confirm_sweep_us"][run_name] = {
            "baseline": confirm_baseline_us,
            "fused": confirm_fused_us,
        }
        del confirm_baseline_args
        del confirm_fused_args

        average_baseline_us = (first_baseline_us + confirm_baseline_us) / 2.0
        average_fused_us = (first_fused_us + confirm_fused_us) / 2.0
        speedup = average_baseline_us / average_fused_us

        results["average_us"][run_name] = {
            "baseline": average_baseline_us,
            "fused": average_fused_us,
        }
        results["average_speedup_ratio_vs_graph_baseline"][run_name] = speedup
        speedups.append(speedup)

    results["average_geomean_speedup_vs_graph_baseline"] = _geomean(speedups)
    print(json.dumps(results, indent=2))
