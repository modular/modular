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
PREFILL_SHAPES = (
    (1, 11),
    (1, 512),
    (1, 1024),
    (1, 2048),
    (2, 2048),
)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value is not None else default


def _resolve_prefill_shapes() -> tuple[tuple[int, int], ...]:
    raw_shapes = os.environ.get(
        "PROFILE_Q_NORM_ROPE_PREFILL_SHAPES", ""
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
                "PROFILE_Q_NORM_ROPE_PREFILL_SHAPES entries must look like "
                f"'batchxseq', got {raw_shape!r}"
            )
        batch_text, seq_text = shape_text.split("x", maxsplit=1)
        shape = (int(batch_text), int(seq_text))
        if shape[0] <= 0 or shape[1] <= 0:
            raise ValueError(
                "PROFILE_Q_NORM_ROPE_PREFILL_SHAPES entries must be positive, "
                f"got {shape}"
            )
        if shape not in resolved:
            resolved.append(shape)

    if not resolved:
        raise ValueError(
            "PROFILE_Q_NORM_ROPE_PREFILL_SHAPES must select at least one shape"
        )
    return tuple(resolved)


def _load_text_config() -> dict[str, Any]:
    config_path = Path(os.environ["PIPELINES_TESTDATA"]) / "config.json"
    with open(config_path) as file:
        data = json.load(file)
    return data.get("text_config", data)


def _profile_config(config: dict[str, Any]) -> dict[str, Any]:
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
        "rope_local_base_freq": _env_int(
            "PROFILE_Q_NORM_ROPE_THETA", config["rope_local_base_freq"]
        ),
        "prefill_shapes": _resolve_prefill_shapes(),
    }


def _make_weight_registry(head_dim: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(42)
    return {
        "q_gamma": torch.randn(head_dim, dtype=torch.bfloat16) * Q_NORM_STD,
    }


def _device_uint32_buffer(array: np.ndarray, device: Accelerator) -> Buffer:
    return Buffer.from_numpy(array).to(device)


def _row_offsets_array(batch_size: int, seq_len: int) -> np.ndarray:
    return np.arange(
        0,
        (batch_size + 1) * seq_len,
        seq_len,
        dtype=np.uint32,
    )


def _cache_lengths_array(batch_size: int) -> np.ndarray:
    return np.zeros(batch_size, dtype=np.uint32)


def _prefill_run_name(batch_size: int, seq_len: int) -> str:
    return f"prefill_bs{batch_size}_seq{seq_len}_cache0_step0_q_only"


def _build_graph(
    *,
    session: InferenceSession,
    config: dict[str, Any],
    weight_registry: dict[str, torch.Tensor],
    batch_size: int,
    seq_len: int,
    use_fused: bool,
) -> Any:
    device_ref = DeviceRef.GPU()
    rope = Llama3RotaryEmbedding(
        config["hidden_size"],
        config["num_attention_heads"],
        config["rope_local_base_freq"],
        seq_len + 256,
        interleaved=False,
        head_dim=config["head_dim"],
    )
    q_norm = Gemma3RMSNorm(config["head_dim"], DType.bfloat16, EPS)
    q_norm.weight.name = "q_gamma"

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
    start_pos_type = TensorType(
        DType.uint32,
        [batch_size],
        device=device_ref,
    )

    graph_name = (
        f"Gemma3WideQOnlyNormRopePrefillFusedBS{batch_size}Seq{seq_len}"
        if use_fused
        else f"Gemma3WideQOnlyNormRopePrefillBaselineBS{batch_size}Seq{seq_len}"
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

    return session.load(graph, weights_registry=weight_registry)


def _make_execution_args(
    *,
    batch_size: int,
    seq_len: int,
    num_attention_heads: int,
    head_dim: int,
    device: Accelerator,
    xq_seed: int,
) -> tuple[Any, ...]:
    torch.manual_seed(xq_seed)
    xq = torch.randn(
        (batch_size * seq_len, num_attention_heads, head_dim),
        dtype=torch.bfloat16,
        device="cuda",
    ).contiguous()
    return (
        Buffer.from_dlpack(xq),
        _device_uint32_buffer(_row_offsets_array(batch_size, seq_len), device),
        _device_uint32_buffer(_cache_lengths_array(batch_size), device),
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


def test_profile_q_norm_rope_prefill() -> None:
    config = _profile_config(_load_text_config())
    session = InferenceSession(devices=[Accelerator(0)])
    device = Accelerator(0)
    weight_registry = _make_weight_registry(config["head_dim"])
    prefill_shapes = config["prefill_shapes"]

    results: dict[str, Any] = {
        "benchmark_config": {
            "dtype": "bfloat16",
            "head_dim": config["head_dim"],
            "num_q_heads": config["num_attention_heads"],
            "rope": "non-interleaved",
            "mode": "prefill-ragged-live-graph-q-only",
            "shapes": [
                _prefill_run_name(batch_size, seq_len)
                for batch_size, seq_len in prefill_shapes
            ],
            "warmup_iters": WARMUP_ITERS,
            "timed_iters": TIMED_ITERS,
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
            config=config,
            weight_registry=weight_registry,
            batch_size=batch_size,
            seq_len=seq_len,
            use_fused=False,
        )
        fused = _build_graph(
            session=session,
            config=config,
            weight_registry=weight_registry,
            batch_size=batch_size,
            seq_len=seq_len,
            use_fused=True,
        )

        baseline_correctness_args = _make_execution_args(
            batch_size=batch_size,
            seq_len=seq_len,
            num_attention_heads=config["num_attention_heads"],
            head_dim=config["head_dim"],
            device=device,
            xq_seed=1000 + batch_size + seq_len,
        )
        fused_correctness_args = _make_execution_args(
            batch_size=batch_size,
            seq_len=seq_len,
            num_attention_heads=config["num_attention_heads"],
            head_dim=config["head_dim"],
            device=device,
            xq_seed=1000 + batch_size + seq_len,
        )
        _run_correctness_check(
            baseline=baseline,
            fused=fused,
            baseline_args=baseline_correctness_args,
            fused_args=fused_correctness_args,
        )

        first_baseline_args = [
            _make_execution_args(
                batch_size=batch_size,
                seq_len=seq_len,
                num_attention_heads=config["num_attention_heads"],
                head_dim=config["head_dim"],
                device=device,
                xq_seed=3000 + batch_size + seq_len + iteration,
            )
            for iteration in range(WARMUP_ITERS + TIMED_ITERS)
        ]
        first_fused_args = [
            _make_execution_args(
                batch_size=batch_size,
                seq_len=seq_len,
                num_attention_heads=config["num_attention_heads"],
                head_dim=config["head_dim"],
                device=device,
                xq_seed=3000 + batch_size + seq_len + iteration,
            )
            for iteration in range(WARMUP_ITERS + TIMED_ITERS)
        ]
        first_baseline_us = _benchmark_us(baseline, first_baseline_args)
        first_fused_us = _benchmark_us(fused, first_fused_args)
        results["first_sweep_us"][run_name] = {
            "baseline": first_baseline_us,
            "fused": first_fused_us,
        }
        del first_baseline_args
        del first_fused_args

        confirm_baseline_args = [
            _make_execution_args(
                batch_size=batch_size,
                seq_len=seq_len,
                num_attention_heads=config["num_attention_heads"],
                head_dim=config["head_dim"],
                device=device,
                xq_seed=5000 + batch_size + seq_len + iteration,
            )
            for iteration in range(WARMUP_ITERS + TIMED_ITERS)
        ]
        confirm_fused_args = [
            _make_execution_args(
                batch_size=batch_size,
                seq_len=seq_len,
                num_attention_heads=config["num_attention_heads"],
                head_dim=config["head_dim"],
                device=device,
                xq_seed=5000 + batch_size + seq_len + iteration,
            )
            for iteration in range(WARMUP_ITERS + TIMED_ITERS)
        ]
        confirm_baseline_us = _benchmark_us(baseline, confirm_baseline_args)
        confirm_fused_us = _benchmark_us(fused, confirm_fused_args)
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
    large_shape_ratio_names = large_shape_names or list(
        results["average_speedup_ratio_vs_graph_baseline"].keys()
    )
    results["average_large_shape_geomean_speedup_vs_graph_baseline"] = _geomean(
        [
            results["average_speedup_ratio_vs_graph_baseline"][run_name]
            for run_name in large_shape_ratio_names
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
    confirm_large_shape_names = large_shape_names or list(
        results["confirm_sweep_us"].keys()
    )
    results["confirm_large_shape_geomean_speedup_vs_graph_baseline"] = _geomean(
        [
            results["confirm_sweep_us"][run_name]["baseline"]
            / results["confirm_sweep_us"][run_name]["fused"]
            for run_name in confirm_large_shape_names
        ]
    )

    print("GEMMA3_WIDE_Q_NORM_ROPE_PREFILL_PROFILE_START")
    print(json.dumps(results, indent=2, sort_keys=True))
    print("GEMMA3_WIDE_Q_NORM_ROPE_PREFILL_PROFILE_END")
