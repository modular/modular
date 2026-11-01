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
"""FlashInfer's TRT-LLM NVFP4 MoE kernels under a CUDA graph at EP MoE shapes.

The competitor arm for `bench_mega_ffn_nvfp4_graph.py`. vLLM's
`trtllm_nvfp4_moe` experts run these kernels on SM100, and this script calls
them the same way, with the weight layout vLLM prepares at load time. It
imports vLLM's weight preparation and quantization, so run it with the Python
of a venv that has the `vllm` wheel installed. FlashInfer is then the version
vLLM pins.

Two boundaries:
- `routed`: `trtllm_fp4_block_scale_routed_moe` on pre-routed tokens without
  finalize. It computes the same (token, local expert) rows as MegaFFN.
- `monolithic`: vLLM's TP-attention/EP-MoE path. Each rank quantizes the whole
  batch, routes inside the kernel, skips non-local experts and finalizes.

Every call launches with PDL, which FlashInfer turns on by default on SM100.
The graph holds `--copies` calls, each with its own weights, so one call
cannot reuse the previous call's weights from L2. Token routing matches the
MAX arm: same generator, seed and busiest rank.

    <venv>/bin/python bench_flashinfer_trtllm_moe_nvfp4.py --tokens 48,8192
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

import flashinfer
import gpu_telemetry
import moe_routing
import numpy as np
import torch
from flashinfer.fused_moe import (
    RoutingMethodType,
    trtllm_fp4_block_scale_moe,
    trtllm_fp4_block_scale_routed_moe,
)
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.flashinfer_fp4_moe import (
    prepare_static_weights_for_trtllm_fp4_moe,
    reorder_w1w3_to_w3w1,
)


@dataclass
class Result:
    arm: str
    tokens: int
    routing: str
    rank: int
    rows: int
    active_experts: int
    copies: int
    round: int
    replays: int
    us_per_op: float
    us_per_op_all: list[float]
    sustained_us_per_op: float | None
    mj_per_op: float | None
    mean_w: float | None
    cold_us_per_op: float | None
    cold_us_per_op_all: list[float]
    kernels: list[str]
    wall_start: float
    wall_end: float


def _rand_u8(shape: tuple[int, ...]) -> torch.Tensor:
    return torch.randint(0, 256, shape, dtype=torch.uint8, device="cuda")


def _rand_e4m3(shape: tuple[int, ...]) -> torch.Tensor:
    return (
        (torch.rand(shape, device="cuda") * 1.5 + 0.5)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )


@dataclass
class Weights:
    w13: torch.Tensor
    w13_scales: torch.Tensor
    w2: torch.Tensor
    w2_scales: torch.Tensor


def make_weights(e: int, h: int, d: int) -> Weights:
    """Random checkpoint-layout weights, prepared the way vLLM loads them."""
    w13 = _rand_u8((e, 2 * d, h // 2))
    w13_scales = _rand_e4m3((e, 2 * d, h // 16))
    w2 = _rand_u8((e, h, d // 2))
    w2_scales = _rand_e4m3((e, h, d // 16))
    w13, w13_scales = reorder_w1w3_to_w3w1(w13, w13_scales)
    w13, w13_scales, w2, w2_scales = prepare_static_weights_for_trtllm_fp4_moe(
        w13,
        w2,
        w13_scales,
        w2_scales,
        hidden_size=h,
        intermediate_size=d,
        num_experts=e,
        is_gated_activation=True,
    )
    return Weights(w13, w13_scales, w2, w2_scales)


def forcing_logits(topk: np.ndarray, num_experts: int) -> torch.Tensor:
    """Router logits whose top-k (with zero bias) is exactly `topk`."""
    logits = torch.full((topk.shape[0], num_experts), -8.0)
    rows = torch.arange(topk.shape[0])[:, None]
    logits[rows, torch.from_numpy(topk).long()] = 8.0
    return logits.to(device="cuda", dtype=torch.float32)


def build_calls(
    args: argparse.Namespace,
    weights: list[Weights],
    topk: np.ndarray,
    rank: int,
) -> list[Callable[[], object]]:
    """Returns one closure per weight copy for the chosen boundary."""
    e_local = args.num_experts // args.ep_size
    tokens = topk.shape[0]
    h, d = args.hidden, args.moe_dim
    scalar = lambda v: torch.full((e_local,), v, device="cuda")
    common = dict(
        gemm1_bias=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_bias=None,
        output1_scale_scalar=scalar(1.0 / 1024),
        output1_scale_gate_scalar=scalar(1.0 / 1024),
        output2_scale_scalar=scalar(1.0 / 1024),
        num_experts=args.num_experts,
        top_k=args.top_k,
        intermediate_size=d,
        local_expert_offset=rank * e_local,
        local_num_experts=e_local,
        tune_max_num_tokens=args.chunk,
    )
    if args.boundary == "routed":
        x = _rand_u8((tokens, h // 2))
        x_scales = _rand_e4m3((tokens, h // 16)).view(torch.float8_e4m3fn)
        ids = torch.from_numpy(topk).to(device="cuda", dtype=torch.int32)
        topk_weights = torch.full(
            ids.shape, 1.0 / args.top_k, device="cuda", dtype=torch.bfloat16
        )
        return [
            lambda w=w: trtllm_fp4_block_scale_routed_moe(
                topk_ids=(ids, topk_weights),
                routing_bias=None,
                hidden_states=x,
                hidden_states_scale=x_scales,
                gemm1_weights=w.w13,
                gemm1_weights_scale=w.w13_scales.view(torch.float8_e4m3fn),
                gemm2_weights=w.w2,
                gemm2_weights_scale=w.w2_scales.view(torch.float8_e4m3fn),
                n_group=0,
                topk_group=0,
                routed_scaling_factor=None,
                routing_method_type=RoutingMethodType.Renormalize,
                do_finalize=False,
                **common,
            )
            for w in weights
        ]

    x_bf16 = torch.randn((tokens, h), device="cuda", dtype=torch.bfloat16)
    global_scale = torch.tensor([1.0], device="cuda")
    logits = forcing_logits(topk, args.num_experts)
    bias = torch.zeros(args.num_experts, device="cuda", dtype=torch.bfloat16)

    def call(w: Weights) -> object:
        x_fp4, x_sf = ops.scaled_fp4_quant(
            x_bf16, global_scale, is_sf_swizzled_layout=False
        )
        return trtllm_fp4_block_scale_moe(
            routing_logits=logits,
            routing_bias=bias,
            hidden_states=x_fp4,
            hidden_states_scale=x_sf.view(torch.float8_e4m3fn).reshape(
                tokens, -1
            ),
            gemm1_weights=w.w13,
            gemm1_weights_scale=w.w13_scales.view(torch.float8_e4m3fn),
            gemm2_weights=w.w2,
            gemm2_weights_scale=w.w2_scales.view(torch.float8_e4m3fn),
            n_group=1,
            topk_group=1,
            routed_scaling_factor=2.5,
            # Sigmoid scores, correction bias, grouped top-k.
            routing_method_type=RoutingMethodType.DeepSeekV3,
            do_finalize=True,
            **common,
        )

    return [lambda w=w: call(w) for w in weights]


@dataclass
class Captured:
    graph: torch.cuda.CUDAGraph
    outputs: list[object]
    copies: int


def capture(calls: list[Callable[[], object]]) -> Captured:
    """Captures the calls in one CUDA graph."""
    # Tactic selection happens outside capture, as vLLM does at startup.
    with flashinfer.autotune(True):
        for call in calls:
            call()
    for call in calls:
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = [call() for call in calls]
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    return Captured(graph, outputs, len(calls))


def time_graph(
    captured: Captured,
    replays: int,
    repeats: int,
    label: str = "",
    sleep_s: float = 0.0,
) -> list[float]:
    """Returns per-op us per repeat."""
    per_op = gpu_telemetry.time_bursts(
        captured.graph.replay,
        captured.copies,
        replays,
        repeats,
        sleep_s,
        label,
    )
    for out in captured.outputs:
        first = out[0] if isinstance(out, (list, tuple)) else out
        assert isinstance(first, torch.Tensor)
        if not torch.isfinite(first.float()).all():
            raise RuntimeError("non-finite output")
    return per_op


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--boundary", choices=["routed", "monolithic"])
    parser.add_argument("--tokens", default="48,96,192,8192")
    parser.add_argument("--routing", default="balanced,uniform,skewed")
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--ep-size", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=6144)
    parser.add_argument("--moe-dim", type=int, default=2048)
    parser.add_argument("--chunk", type=int, default=8192)
    parser.add_argument(
        "--copies",
        default="8",
        help=(
            "MoE calls per graph, comma-separated. `1,8` times one call alone"
            " and eight back to back in the same process."
        ),
    )
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--repeat-sleep-s",
        type=float,
        default=0.0,
        help=(
            "Idle the GPU this long before each timed repeat, so every repeat"
            " starts below the power limit."
        ),
    )
    parser.add_argument(
        "--sustain-s",
        type=float,
        default=0.0,
        help=(
            "Also replay each case for this many seconds (after as long a"
            " warm-up) and report steady-state time and energy per op."
        ),
    )
    parser.add_argument(
        "--cold-samples",
        type=int,
        default=0,
        help=(
            "Also time this many single replays per case, each after"
            " --cold-idle-s of idle, which finish before the power limit"
            " engages."
        ),
    )
    parser.add_argument("--cold-idle-s", type=float, default=0.25)
    parser.add_argument(
        "--log-kernels",
        action="store_true",
        help=(
            "Record the kernels one replay runs, since the autotuner can pick"
            " a different tactic in each process."
        ),
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Timing rounds per case; the order of --copies flips each round.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--rank",
        default="busiest",
        help="EP rank to benchmark: busiest, median, or an index.",
    )
    parser.add_argument(
        "--nvtx",
        action="store_true",
        help="Label each case's timed replays with an NVTX range.",
    )
    parser.add_argument("-o", "--output", default="")
    args = parser.parse_args()
    args.boundary = args.boundary or "routed"

    copies_list = [int(c) for c in args.copies.split(",")]
    e_local = args.num_experts // args.ep_size
    weights = [
        make_weights(e_local, args.hidden, args.moe_dim)
        for _ in range(max(copies_list))
    ]
    results = []
    for tokens in (int(t) for t in args.tokens.split(",")):
        for routing in args.routing.split(","):
            topk = moe_routing.routed_topk(
                tokens,
                args.top_k,
                args.num_experts,
                moe_routing.Routing(routing),
                ep_size=args.ep_size,
                seed=args.seed,
            )
            counts = moe_routing.routed_counts(topk, args.num_experts)
            rank = moe_routing.pick_rank(counts, args.ep_size, args.rank)
            local = moe_routing.rank_counts(counts, args.ep_size)[rank]
            graphs = {
                c: capture(build_calls(args, weights[:c], topk, rank))
                for c in copies_list
            }
            for rnd in range(args.rounds):
                order = copies_list if rnd % 2 == 0 else copies_list[::-1]
                for c in order:
                    label = f"t{tokens}_{routing}_c{c}_r{rnd}"
                    wall_start = time.time()
                    per_op = time_graph(
                        graphs[c],
                        args.replays,
                        args.repeats,
                        label=label if args.nvtx else "",
                        sleep_s=args.repeat_sleep_s,
                    )
                    steady = gpu_telemetry.sustained(
                        graphs[c].graph.replay, c, args.sustain_s
                    )
                    cold = gpu_telemetry.cold_bursts(
                        graphs[c].graph.replay,
                        c,
                        args.cold_samples,
                        args.cold_idle_s,
                    )
                    cold_median = statistics.median(cold) if cold else None
                    kernels = (
                        gpu_telemetry.replay_kernels(graphs[c].graph.replay)
                        if args.log_kernels
                        else []
                    )
                    result = Result(
                        arm=f"flashinfer_trtllm_{args.boundary}",
                        tokens=tokens,
                        routing=routing,
                        rank=rank,
                        rows=int(local.sum()),
                        active_experts=int((local > 0).sum()),
                        copies=c,
                        round=rnd,
                        replays=args.replays,
                        us_per_op=statistics.median(per_op),
                        us_per_op_all=per_op,
                        sustained_us_per_op=steady.us_per_op
                        if steady
                        else None,
                        mj_per_op=steady.mj_per_op if steady else None,
                        mean_w=steady.mean_w if steady else None,
                        cold_us_per_op=cold_median,
                        cold_us_per_op_all=cold,
                        kernels=kernels,
                        wall_start=wall_start,
                        wall_end=time.time(),
                    )
                    print(json.dumps(asdict(result)), flush=True)
                    results.append(asdict(result))
            del graphs
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
