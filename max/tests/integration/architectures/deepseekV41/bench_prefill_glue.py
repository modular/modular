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
"""Times DeepSeek-V4.1's prefill glue with its switches off and on.

A manual A/B, not a CI test. Each case is the compressor's projection at
one ratio, or a forward's 40 CSA2 index lists, built as the model builds
it. Shapes:
``step``, the prefilling GPU's forward (an 8,168-token chunk after 122,880
cached tokens plus 24 requests decoding 6 tokens each near 127K, 8,312
rows), and ``decode``, those 24 requests alone (144 rows). Arms: ``old``
(switch off), ``new`` (on) and ``control``, a second compile of ``old``.
Before timing, each case checks ``new`` and ``control`` against ``old`` on
the same inputs: bit for bit where the math is the same, else within the
case's relative bound. Each arm is captured as a device graph and replayed
in host-timed bursts, the arms rotating every round. Prints one ``CSV,``
line per case, shape and arm: microseconds per call (median, p10, p90) and
the calls a forward makes.

Under ``--run_under=//utils/benchmarking/kepler/run_under:nsys_sqlite_runner``
a profile pass follows in one profiler region, each burst in an NVTX range
whose JSON label ``traces/nsys_arms.py`` pools into kernel time per call.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from max.driver import CPU, Accelerator, Buffer
from max.engine import InferenceSession, Model
from max.profiler.oneshot import cuda_profiler_region
from prefill_glue_cases import (
    compressor_model,
    compressor_weights,
    hidden,
    index_list_inputs,
    index_lists_model,
)

DECODE = [(127_000 + 37 * i, 6) for i in range(24)]
SHAPES = {"step": [(122_880, 8168), *DECODE], "decode": DECODE}
ARMS = {"old": False, "new": True, "control": False}


@dataclass
class Case:
    name: str
    per_forward: int
    build: Callable[[InferenceSession, bool], Model]
    inputs: Callable[[Sequence[tuple[int, int]]], list[Buffer]]
    rel_bound: float
    """Largest |new - old| relative to max |old|; 0 asks for equal bits."""


def _hidden(requests: Sequence[tuple[int, int]]) -> list[Buffer]:
    rows = sum(n for _, n in requests)
    return [Buffer.from_dlpack(hidden(rows, rows))]


def _lists(requests: Sequence[tuple[int, int]]) -> list[Buffer]:
    device = Accelerator()
    return [
        Buffer.from_numpy(a).to(device)
        for a in index_list_inputs(requests, seed=len(requests))
    ]


def cases() -> list[Case]:
    c1, c2 = compressor_weights(1), compressor_weights(2)
    # Accumulation order of exact bf16 products in float32.
    gemm_bound = 1e-4
    return [
        Case(
            "compressor_r2",
            3,
            lambda s, on: compressor_model(s, 2, on, c2),
            _hidden,
            gemm_bound,
        ),
        Case(
            "compressor_r1",
            1,
            lambda s, on: compressor_model(s, 1, on, c1),
            _hidden,
            gemm_bound,
        ),
        Case("index_lists", 1, index_lists_model, _lists, 0.0),
    ]


def _check(case: Case, old: Sequence[Buffer], new: Sequence[Buffer]) -> float:
    """Returns the worst relative difference, raising past the case's bound."""
    worst = 0.0
    for a, b in zip(old, new, strict=True):
        ta = torch.from_dlpack(a).double()
        tb = torch.from_dlpack(b).double()
        scale = ta.abs().max().item() or 1.0
        rel = (ta - tb).abs().max().item() / scale
        if not torch.isfinite(tb).all() or rel > case.rel_bound:
            raise AssertionError(f"{case.name}: relative difference {rel}")
        worst = max(worst, rel)
    return worst


def _bursts(
    replay: Callable[[], None], dev: Accelerator, replays: int, repeats: int
) -> list[float]:
    """Microseconds per replay, one value per burst of ``replays``."""
    out = []
    for _ in range(repeats):
        dev.synchronize()
        t0 = time.perf_counter()
        for _ in range(replays):
            replay()
        dev.synchronize()
        out.append((time.perf_counter() - t0) * 1e6 / replays)
    return out


def _profile(
    case: Case,
    rows: int,
    arms: dict[str, Callable[[], None]],
    dev: Accelerator,
    a: argparse.Namespace,
) -> None:
    """Replays each arm in NVTX ranges ``nsys_arms.py`` pools by label."""
    names = list(arms)
    with cuda_profiler_region():
        for p in range(a.profile_rounds):
            for pos, name in enumerate(names[p % 3 :] + names[: p % 3]):
                label = {
                    "op": case.name,
                    "t": rows,
                    "arm": name,
                    "replays": a.replays,
                    "pass": p,
                    "pos": pos,
                }
                dev.synchronize()
                torch.cuda.nvtx.range_push(json.dumps(label))
                for _ in range(a.replays):
                    arms[name]()
                dev.synchronize()
                torch.cuda.nvtx.range_pop()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shapes", default=",".join(SHAPES))
    ap.add_argument("--cases", default="")
    ap.add_argument("--rounds", type=int, default=9)
    ap.add_argument("--replays", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--profile-rounds", type=int, default=3)
    a = ap.parse_args()
    profiling = os.environ.get("MODULAR_PROFILE_RANGE") == "1"
    dev = Accelerator()
    session = InferenceSession(devices=[CPU(), dev])
    selected = [c for c in cases() if not a.cases or c.name in a.cases]
    print("CSV,case,shape,rows,arm,us,p10,p90,per_forward", flush=True)
    key = 0
    for case in selected:
        models = {arm: case.build(session, on) for arm, on in ARMS.items()}
        for shape in a.shapes.split(","):
            requests = SHAPES[shape]
            rows = sum(n for _, n in requests)
            buffers = case.inputs(requests)
            outs: dict[str, list[Buffer]] = {}
            arms: dict[str, Callable[[], None]] = {}
            for arm, model in models.items():
                key += 1
                outs[arm] = model.capture(key, *buffers)
                arms[arm] = functools.partial(model.replay, key, *buffers)
                arms[arm]()
            dev.synchronize()
            worst = _check(case, outs["old"], outs["new"])
            _check(case, outs["old"], outs["control"])
            names = list(arms)
            times: dict[str, list[float]] = {n: [] for n in names}
            for rnd in range(a.rounds):
                shift = rnd % len(names)
                for name in names[shift:] + names[:shift]:
                    times[name] += _bursts(
                        arms[name], dev, a.replays, a.repeats
                    )
            for name, ts in times.items():
                ts = sorted(ts)
                print(
                    f"CSV,{case.name},{shape},{rows},{name},"
                    f"{statistics.median(ts):.2f},{ts[len(ts) // 10]:.2f},"
                    f"{ts[(len(ts) * 9) // 10]:.2f},{case.per_forward}",
                    flush=True,
                )
            print(f"CHECK,{case.name},{shape},rel={worst:.2e}", flush=True)
            if profiling:
                _profile(case, rows, arms, dev, a)


if __name__ == "__main__":
    main()
