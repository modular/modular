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
"""MegaFFN NVFP4 under a CUDA graph at per-rank EP MoE shapes.

Times the fused `mo.composite.mega_ffn_nvfp4` launch the way serving runs it:
captured into a device graph, with PDL chaining consecutive launches. The
graph holds `--copies` independent FFNs, each with its own weights, so one
launch cannot reuse the previous launch's weights from L2.

The routing tensors follow the EP dispatch layout for one rank: per-expert
row prefix sums over all local experts, each expert's scales starting on a
fresh 128-row block, and a token buffer sized for the whole chunk.

Run directly:
    br //max/kernels/benchmarks/misc/comparison:bench_mega_ffn_nvfp4_graph -- \
        --tokens 48,8192 --routing balanced,uniform
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from dataclasses import asdict, dataclass

import gpu_telemetry
import moe_routing
import numpy as np
import torch
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType
from max.nn.kernels import (
    grouped_matmul_block_scaled,
    grouped_matmul_blocked_swiglu,
)

NVFP4_SF_VECTOR_SIZE = 16
SF_ATOM_K = 4
SF_ATOM_M = (32, 4)


@dataclass(frozen=True)
class MoEConfig:
    """Per-rank MoE FFN dims."""

    num_experts: int
    top_k: int
    ep_size: int
    hidden: int
    moe_dim: int
    max_rows: int

    @property
    def local_experts(self) -> int:
        return self.num_experts // self.ep_size


@dataclass
class Result:
    arm: str
    tokens: int
    routing: str
    rank: int
    rows: int
    active_experts: int
    est_m: int
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
    wall_start: float
    wall_end: float
    max_abs_diff_vs_first: float
    rows_digest: str


def _scale_tile(rows: int, k: int) -> tuple[int, ...]:
    k_groups = k // NVFP4_SF_VECTOR_SIZE
    return (rows, k_groups // SF_ATOM_K, *SF_ATOM_M, SF_ATOM_K)


def _weight_scale_tile(e: int, n: int, k: int) -> tuple[int, ...]:
    return (e, n // (SF_ATOM_M[0] * SF_ATOM_M[1]), *_scale_tile(1, k)[1:])


def build_graph(cfg: MoEConfig, copies: int, device: DeviceRef) -> Graph:
    """Builds `copies` independent gate_up -> down FFNs over shared routing.

    Each pair fuses into one `mo.composite.mega_ffn_nvfp4`: both legs share
    the routing values and L1's outputs feed only L2.
    """
    cpu = DeviceRef.CPU()
    e, h, d = cfg.local_experts, cfg.hidden, cfg.moe_dim
    sf_rows = moe_routing.scale_rows_capacity(cfg.max_rows, e)
    shared = [
        TensorType(DType.uint8, (cfg.max_rows, h // 2), device=device),
        TensorType(DType.float8_e4m3fn, _scale_tile(sf_rows, h), device=device),
        TensorType(DType.uint32, (e + 1,), device=device),
        TensorType(DType.uint32, (e,), device=device),
        TensorType(DType.int32, (e,), device=device),
        TensorType(DType.float32, (e,), device=device),
        TensorType(DType.float32, (e,), device=device),
        TensorType(DType.float32, (e,), device=device),
        TensorType(DType.uint32, (2,), device=cpu),
        TensorType(DType.uint32, (), device=cpu),
    ]
    per_copy = [
        TensorType(DType.uint8, (e, 2 * d, h // 2), device=device),
        TensorType(
            DType.float8_e4m3fn, _weight_scale_tile(e, 2 * d, h), device=device
        ),
        TensorType(DType.uint8, (e, h, d // 2), device=device),
        TensorType(
            DType.float8_e4m3fn, _weight_scale_tile(e, h, d), device=device
        ),
    ]
    with Graph(
        "mega_ffn_nvfp4_graph_bench",
        input_types=shared + per_copy * copies,
    ) as graph:
        inputs = [v.tensor for v in graph.inputs]
        (
            hidden,
            a_scales,
            expert_start,
            a_scale_offsets,
            expert_ids,
            es13,
            es_down,
            c_input_scales,
            usage_stats,
            estimated_total_m,
        ) = inputs[: len(shared)]
        outputs = []
        for i in range(copies):
            base = len(shared) + i * len(per_copy)
            w13, w13_scales, w2, w2_scales = inputs[base : base + len(per_copy)]
            packed, sf = grouped_matmul_blocked_swiglu(
                hidden,
                w13,
                a_scales,
                w13_scales,
                expert_start,
                a_scale_offsets,
                expert_ids,
                usage_stats,
                expert_scales=es13,
                c_input_scales=c_input_scales,
                estimated_total_m=estimated_total_m,
            )
            outputs.append(
                grouped_matmul_block_scaled(
                    packed,
                    w2,
                    sf,
                    w2_scales,
                    expert_start,
                    a_scale_offsets,
                    expert_ids,
                    es_down,
                    usage_stats,
                    out_type=DType.bfloat16,
                    estimated_total_m=estimated_total_m,
                )
            )
        graph.output(*outputs)
    return graph


def _rand_u8(shape: tuple[int, ...]) -> torch.Tensor:
    return torch.randint(0, 256, shape, dtype=torch.uint8, device="cuda")


def _rand_e4m3(shape: tuple[int, ...]) -> torch.Tensor:
    # Scales in [0.5, 2) keep the random FP4 products finite.
    return (
        (torch.rand(shape, device="cuda") * 1.5 + 0.5)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )


def _gpu(t: torch.Tensor, dtype: DType | None = None) -> Buffer:
    buf = Buffer.from_dlpack(t)
    return buf.view(dtype) if dtype is not None else buf


def _cpu(a: np.ndarray) -> Buffer:
    return Buffer.from_dlpack(torch.from_numpy(a))


class Inputs:
    """Device tensors for the benchmark graph.

    The torch tensors own the memory the MAX buffers alias, so they live as
    long as the buffers do.
    """

    def __init__(self, cfg: MoEConfig, copies: int) -> None:
        e, h, d = cfg.local_experts, cfg.hidden, cfg.moe_dim
        sf_rows = moe_routing.scale_rows_capacity(cfg.max_rows, e)
        self.cfg = cfg
        self.hidden = _rand_u8((cfg.max_rows, h // 2))
        self.a_scales = _rand_e4m3(_scale_tile(sf_rows, h))
        self.es13 = torch.full((e,), 1.0 / 1024, device="cuda")
        self.es_down = torch.full((e,), 1.0 / 1024, device="cuda")
        self.c_input_scales = torch.full((e,), 2.0, device="cuda")
        self.weights = [
            (
                _rand_u8((e, 2 * d, h // 2)),
                _rand_e4m3(_weight_scale_tile(e, 2 * d, h)),
                _rand_u8((e, h, d // 2)),
                _rand_e4m3(_weight_scale_tile(e, h, d)),
            )
            for _ in range(copies)
        ]

    def buffers(
        self, layout: moe_routing.DispatchLayout, est_m: int, copies: int
    ) -> tuple[list[Buffer], list[torch.Tensor]]:
        """Returns the graph inputs for one routing and the tensors behind them."""
        cfg = self.cfg
        routing = [
            torch.from_numpy(layout.expert_start).cuda(),
            torch.from_numpy(layout.a_scale_offsets).cuda(),
            torch.from_numpy(layout.expert_ids).cuda(),
        ]
        buffers = [
            _gpu(self.hidden),
            _gpu(self.a_scales, DType.float8_e4m3fn),
            *[_gpu(t) for t in routing],
            _gpu(self.es13),
            _gpu(self.es_down),
            _gpu(self.c_input_scales),
            # Mirrors the host values serving binds: the usage stats
            # `max/python/max/nn/moe/quant_strategy.py` puts in place of the
            # dispatch's device ones, and the token estimate from
            # `expert_parallel.py` (tokens * top_k over the GPUs per node,
            # which is the EP degree here).
            _cpu(np.array([8192, cfg.local_experts], dtype=np.uint32)),
            _cpu(np.array(est_m, dtype=np.uint32)),
        ]
        for w13, w13_scales, w2, w2_scales in self.weights[:copies]:
            buffers += [
                _gpu(w13),
                _gpu(w13_scales, DType.float8_e4m3fn),
                _gpu(w2),
                _gpu(w2_scales, DType.float8_e4m3fn),
            ]
        return buffers, routing


@dataclass
class Captured:
    """A captured graph with the inputs and outputs it is bound to."""

    model: Model
    key: int
    inputs: list[Buffer]
    routing: list[torch.Tensor]
    outputs: list[Buffer]
    copies: int

    def replay(self) -> None:
        self.model.replay(self.key, *self.inputs)


def capture(
    model: Model,
    key: int,
    inputs: list[Buffer],
    routing: list[torch.Tensor],
    copies: int,
) -> Captured:
    outputs = model.capture(key, *inputs)
    for _ in range(3):
        model.replay(key, *inputs)
    torch.cuda.synchronize()
    return Captured(model, key, inputs, routing, outputs, copies)


def time_replays(
    graph: Captured,
    num_rows: int,
    replays: int,
    repeats: int,
    label: str = "",
    sleep_s: float = 0.0,
) -> tuple[list[float], torch.Tensor]:
    """Returns per-op microseconds per repeat and the first copy's rows."""
    per_op = gpu_telemetry.time_bursts(
        graph.replay, graph.copies, replays, repeats, sleep_s, label
    )
    # Only the routed rows are written.
    for out in graph.outputs:
        if not torch.isfinite(torch.from_dlpack(out)[:num_rows].float()).all():
            raise RuntimeError("non-finite output")
    rows = torch.from_dlpack(graph.outputs[0])[:num_rows].float().cpu()
    return per_op, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tokens", default="48,96,192,8192")
    parser.add_argument("--routing", default="balanced,uniform,skewed")
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--ep-size", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=6144)
    parser.add_argument("--moe-dim", type=int, default=2048)
    parser.add_argument(
        "--chunk",
        type=int,
        default=8192,
        help="Prefill chunk; the EP buffer holds chunk * ep_size rows.",
    )
    parser.add_argument(
        "--copies",
        default="8",
        help=(
            "FFNs per graph, comma-separated. `1,8` times one FFN alone and"
            " eight back to back in the same process."
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
        "--est-m",
        default="serving",
        help=(
            "Host token estimates to run, comma-separated: `serving`"
            " (tokens * top_k / ep_size) or integers. The estimate only"
            " selects the tile geometry."
        ),
    )
    parser.add_argument(
        "--nvtx",
        action="store_true",
        help="Label each case's timed replays with an NVTX range.",
    )
    # Tags kbench passes through from the shape YAML.
    parser.add_argument("--model", default="")
    parser.add_argument("--label", default="")
    parser.add_argument("-o", "--output", default="")
    args = parser.parse_args()

    cfg = MoEConfig(
        num_experts=args.num_experts,
        top_k=args.top_k,
        ep_size=args.ep_size,
        hidden=args.hidden,
        moe_dim=args.moe_dim,
        max_rows=args.chunk * args.ep_size,
    )
    copies_list = [int(c) for c in args.copies.split(",")]
    device = Accelerator()
    session = InferenceSession(devices=[device])
    models = {}
    for c in copies_list:
        t0 = time.perf_counter()
        models[c] = session.load(build_graph(cfg, c, DeviceRef.GPU()))
        # Compile time: run with an empty MODULAR_CACHE_DIR to measure it.
        print(f"# load_s copies={c} {time.perf_counter() - t0:.1f}", flush=True)

    # Seeded so output digests compare across processes and builds.
    torch.manual_seed(args.seed)
    data = Inputs(cfg, max(copies_list))
    results = []
    key = 0
    for tokens in (int(t) for t in args.tokens.split(",")):
        for routing in args.routing.split(","):
            topk = moe_routing.routed_topk(
                tokens,
                cfg.top_k,
                cfg.num_experts,
                moe_routing.Routing(routing),
                ep_size=cfg.ep_size,
                seed=args.seed,
            )
            counts = moe_routing.routed_counts(topk, cfg.num_experts)
            rank = moe_routing.pick_rank(counts, cfg.ep_size, args.rank)
            local = moe_routing.rank_counts(counts, cfg.ep_size)[rank]
            layout = moe_routing.dispatch_layout(local)
            first: torch.Tensor | None = None
            for est in args.est_m.split(","):
                est_m = (
                    tokens * cfg.top_k // cfg.ep_size
                    if est == "serving"
                    else int(est)
                )
                graphs = {}
                for c in copies_list:
                    key += 1
                    inputs, routing_tensors = data.buffers(layout, est_m, c)
                    graphs[c] = capture(
                        models[c], key, inputs, routing_tensors, c
                    )
                for rnd in range(args.rounds):
                    order = copies_list if rnd % 2 == 0 else copies_list[::-1]
                    for c in order:
                        label = f"t{tokens}_{routing}_est{est_m}_c{c}_r{rnd}"
                        wall_start = time.time()
                        per_op, rows_out = time_replays(
                            graphs[c],
                            layout.num_rows,
                            args.replays,
                            args.repeats,
                            label=label if args.nvtx else "",
                            sleep_s=args.repeat_sleep_s,
                        )
                        steady = gpu_telemetry.sustained(
                            graphs[c].replay, c, args.sustain_s
                        )
                        cold = gpu_telemetry.cold_bursts(
                            graphs[c].replay,
                            c,
                            args.cold_samples,
                            args.cold_idle_s,
                        )
                        cold_median = statistics.median(cold) if cold else None
                        if first is None:
                            first = rows_out
                        result = Result(
                            arm="max_mega_ffn",
                            tokens=tokens,
                            routing=routing,
                            rank=rank,
                            rows=layout.num_rows,
                            active_experts=int((local > 0).sum()),
                            est_m=est_m,
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
                            wall_start=wall_start,
                            wall_end=time.time(),
                            max_abs_diff_vs_first=float(
                                (rows_out - first).abs().max()
                            ),
                            # Compares output bytes across builds and modes.
                            rows_digest=hashlib.sha256(
                                rows_out.numpy().tobytes()
                            ).hexdigest()[:16],
                        )
                        print(json.dumps(asdict(result)), flush=True)
                        results.append(asdict(result))
                del graphs
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
