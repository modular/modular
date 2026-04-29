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

import torch
from max.driver import Accelerator
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.pipelines.architectures.gemma3.layers.rms_norm import (
    Gemma3RMSNorm,
    gemma3_rms_norm_fused_residual_add,
)
from torch.utils.dlpack import from_dlpack

EPS = 1e-6
RMS_NORM_STD = 0.05
WARMUP_ITERS = 20
TIMED_ITERS = 50
BATCH_SIZES = (64, 128)


def _load_text_config() -> dict[str, Any]:
    config_path = Path(os.environ["PIPELINES_TESTDATA"]) / "config.json"
    with open(config_path) as file:
        data = json.load(file)
    return data.get("text_config", data)


def _make_weight_registry(hidden_size: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(42)
    return {
        "post_feedforward_layernorm.weight": (
            torch.randn(
                hidden_size,
                dtype=torch.bfloat16,
            )
            * RMS_NORM_STD
        ),
        "next_input_layernorm.weight": (
            torch.randn(
                hidden_size,
                dtype=torch.bfloat16,
            )
            * RMS_NORM_STD
        ),
    }


def _build_graph(
    *,
    session: InferenceSession,
    hidden_size: int,
    weight_registry: dict[str, torch.Tensor],
    use_fused_boundary: bool,
) -> Any:
    post_feedforward_layernorm = Gemma3RMSNorm(
        hidden_size,
        DType.bfloat16,
        EPS,
    )
    post_feedforward_layernorm.weight.name = "post_feedforward_layernorm.weight"
    next_input_layernorm = Gemma3RMSNorm(
        hidden_size,
        DType.bfloat16,
        EPS,
    )
    next_input_layernorm.weight.name = "next_input_layernorm.weight"

    input_type = TensorType(
        DType.bfloat16,
        ["rows", hidden_size],
        device=DeviceRef.GPU(),
    )

    graph_name = (
        "Gemma3PostMlpBoundaryFused"
        if use_fused_boundary
        else "Gemma3PostMlpBoundaryBaseline"
    )
    with Graph(graph_name, input_types=(input_type, input_type)) as graph:
        hidden_states, residual = graph.inputs
        if use_fused_boundary:
            next_norm, residual_out = gemma3_rms_norm_fused_residual_add(
                hidden_states.tensor,
                residual.tensor,
                post_feedforward_layernorm,
                next_input_layernorm,
            )
        else:
            post_mlp = post_feedforward_layernorm(hidden_states.tensor)
            residual_out = residual.tensor + post_mlp
            next_norm = next_input_layernorm(residual_out)
        graph.output(residual_out, next_norm)

    return session.load(graph, weights_registry=weight_registry)


def _make_inputs(
    *,
    batch_size: int,
    hidden_size: int,
    hidden_seed: int,
    residual_seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(hidden_seed)
    hidden_states = torch.randn(
        batch_size,
        hidden_size,
        dtype=torch.bfloat16,
        device="cuda",
    ).contiguous()
    torch.manual_seed(residual_seed)
    residual = torch.randn(
        batch_size,
        hidden_size,
        dtype=torch.bfloat16,
        device="cuda",
    ).contiguous()
    return hidden_states, residual


def _run_correctness_check(
    *,
    baseline: Any,
    fused: Any,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
) -> None:
    baseline_residual, baseline_next_norm = baseline.execute(
        hidden_states,
        residual,
    )
    fused_residual, fused_next_norm = fused.execute(hidden_states, residual)

    torch.testing.assert_close(
        from_dlpack(baseline_residual).to(torch.bfloat16),
        from_dlpack(fused_residual).to(torch.bfloat16),
        rtol=2 * torch.finfo(torch.bfloat16).eps,
        atol=8 * torch.finfo(torch.bfloat16).eps,
    )
    torch.testing.assert_close(
        from_dlpack(baseline_next_norm).to(torch.bfloat16),
        from_dlpack(fused_next_norm).to(torch.bfloat16),
        rtol=2 * torch.finfo(torch.bfloat16).eps,
        atol=8 * torch.finfo(torch.bfloat16).eps,
    )


def _benchmark_us(
    compiled: Any,
    args: list[tuple[torch.Tensor, torch.Tensor]],
) -> float:
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


def test_profile_post_mlp_boundary() -> None:
    config = _load_text_config()
    hidden_size = int(config["hidden_size"])
    session = InferenceSession(devices=[Accelerator(0)])
    weight_registry = _make_weight_registry(hidden_size)

    baseline = _build_graph(
        session=session,
        hidden_size=hidden_size,
        weight_registry=weight_registry,
        use_fused_boundary=False,
    )
    fused = _build_graph(
        session=session,
        hidden_size=hidden_size,
        weight_registry=weight_registry,
        use_fused_boundary=True,
    )

    results: dict[str, Any] = {
        "benchmark_config": {
            "dtype": "bfloat16",
            "hidden_size": hidden_size,
            "mode": "graph-post-mlp-boundary",
            "warmup_iters": WARMUP_ITERS,
            "timed_iters": TIMED_ITERS,
        },
        "correctness": "pass",
        "first_sweep_us": {},
        "confirm_sweep_us": {},
        "average_us": {},
        "average_speedup_ratio_vs_baseline": {},
    }

    for batch_size in BATCH_SIZES:
        run_name = f"post_mlp_boundary_bs{batch_size}_hidden{hidden_size}"

        baseline_args = [
            _make_inputs(
                batch_size=batch_size,
                hidden_size=hidden_size,
                hidden_seed=1000 + batch_size + iteration,
                residual_seed=2000 + batch_size + iteration,
            )
            for iteration in range(WARMUP_ITERS + TIMED_ITERS)
        ]
        fused_args = [
            _make_inputs(
                batch_size=batch_size,
                hidden_size=hidden_size,
                hidden_seed=1000 + batch_size + iteration,
                residual_seed=2000 + batch_size + iteration,
            )
            for iteration in range(WARMUP_ITERS + TIMED_ITERS)
        ]

        _run_correctness_check(
            baseline=baseline,
            fused=fused,
            hidden_states=baseline_args[0][0],
            residual=baseline_args[0][1],
        )

        first_baseline_us = _benchmark_us(baseline, baseline_args)
        first_fused_us = _benchmark_us(fused, fused_args)
        results["first_sweep_us"][run_name] = {
            "baseline": first_baseline_us,
            "fused": first_fused_us,
        }

        del baseline_args
        del fused_args

        baseline_args = [
            _make_inputs(
                batch_size=batch_size,
                hidden_size=hidden_size,
                hidden_seed=3000 + batch_size + iteration,
                residual_seed=4000 + batch_size + iteration,
            )
            for iteration in range(WARMUP_ITERS + TIMED_ITERS)
        ]
        fused_args = [
            _make_inputs(
                batch_size=batch_size,
                hidden_size=hidden_size,
                hidden_seed=3000 + batch_size + iteration,
                residual_seed=4000 + batch_size + iteration,
            )
            for iteration in range(WARMUP_ITERS + TIMED_ITERS)
        ]

        confirm_baseline_us = _benchmark_us(baseline, baseline_args)
        confirm_fused_us = _benchmark_us(fused, fused_args)
        results["confirm_sweep_us"][run_name] = {
            "baseline": confirm_baseline_us,
            "fused": confirm_fused_us,
        }

        average_baseline_us = 0.5 * (first_baseline_us + confirm_baseline_us)
        average_fused_us = 0.5 * (first_fused_us + confirm_fused_us)
        results["average_us"][run_name] = {
            "baseline": average_baseline_us,
            "fused": average_fused_us,
        }
        results["average_speedup_ratio_vs_baseline"][run_name] = (
            average_baseline_us / average_fused_us
        )

        del baseline_args
        del fused_args
        torch.cuda.empty_cache()

    speedups = list(results["average_speedup_ratio_vs_baseline"].values())
    results["average_geomean_speedup_vs_baseline"] = math.prod(speedups) ** (
        1.0 / len(speedups)
    )
    print(json.dumps(results, indent=2, sort_keys=True))
