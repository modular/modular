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
"""EP2 kernel time of a top-6 MoE cell, CUDA-graph timed.

Two measurements, both on the two devices remote execution gives an EP2 job:

``top6``
    The production chain (``ep.dispatch_async`` -> MegaFFN ->
    ``ep.combine``) at 96 local experts per rank (192 routed over two ranks),
    top-6, hidden 5120, MoE width 2304, NVFP4, at T 48 and T 102 tokens per
    rank. The same experts at 192 local per rank exceed MegaFFN's scheduler
    (`MAX_EXPERTS` = 128) and the fused path (`fused_ep_wide_config`); that
    refusal is recorded, not worked around.

``fused``
    Both arms at a top-8 cell (hidden 6144, MoE 2048, 256 routed experts =
    128 local, capacity 32). Arm A is the
    shipping chain, arm B is ``mega_ffn.ep_fused``: dispatch, MegaFFN and the
    weighted combine in one launch per rank. The pair is what a megakernel
    has to beat.

``top6_pair``
    Both arms at the top-6 cell, in one session, at 8, 16, 24, 48, 102 and
    ragged [102, 48] tokens per rank.

Timing is one captured device graph per (arm, cell), replayed back to back
``iters`` times between two device-wide synchronizations, so a replay's
number is the wall time of the whole two-rank step, not a host enqueue. Both
ranks execute every replay; the figure is the slower rank's by construction
because the sync waits for both. Cells are timed in ABBA order and reported
as the median pair ratio.
"""

from __future__ import annotations

import dataclasses
import os
import statistics
import time

import pytest
from fused_megaffn_gates import (
    Geometry,
    build_arm,
    devices,
    ep_config,
    inputs,
    ship_combine_send,
    weights,
)
from max.driver import Accelerator, Buffer, accelerator_api, accelerator_count
from max.engine import InferenceSession, Model
from max.nn.legacy.comm.ep.ep_manager import _plan_fused_moe_workspace
from test_common.graph_utils import is_b100_b200

# A top-6 MoE at 96 local experts per rank.
TOP6 = Geometry(
    n_devices=2,
    num_experts=192,
    max_tokens_per_rank=128,
    hidden_dim=5120,
    moe_dim=2304,
    top_k=6,
    n_layers=1,
)
# The same experts at 192 local per rank: the cell the kernels refuse.
TOP6_EP2_384 = dataclasses.replace(TOP6, num_experts=384)
# A top-8 cell the fused path serves at 128 local experts.
FUSED = Geometry(
    n_devices=2,
    num_experts=256,
    max_tokens_per_rank=32,
    hidden_dim=6144,
    moe_dim=2048,
    top_k=8,
    n_layers=1,
)

TOP6_CELLS = {
    "t102": [102, 102],
    "t48": [48, 48],
    "ragged102_48": [102, 48],
}
# The pair covers small decode batches as well as the chain baseline's cells.
TOP6_PAIR_CELLS = {
    "t8": [8, 8],
    "t16": [16, 16],
    "t24": [24, 24],
    **TOP6_CELLS,
}
FUSED_CELLS = {
    "t32": [32, 32],
    "t24": [24, 24],
}

# The engine cannot capture a graph whose shared expert runs on its side
# stream (unjoined at capture), so every arm here runs it inline. Both arms
# keep the same shared expert, so the pair stays mode-matched.
os.environ["MODULAR_OVERLAP_SHARED_EXPERT"] = "0"

PAIRS = int(os.environ.get("EP2_MOE_PAIRS", "10"))
ITERS = int(os.environ.get("EP2_MOE_ITERS", "200"))
WARM = int(os.environ.get("EP2_MOE_WARM", "20"))


def _skip(geo: Geometry) -> None:
    if accelerator_api() == "hip":
        pytest.skip("NVIDIA-only kernels")
    if not is_b100_b200():
        pytest.skip("requires B100/B200 (SM100)")
    if accelerator_count() < geo.n_devices:
        pytest.skip(f"requires {geo.n_devices} GPUs (EP{geo.n_devices})")


def _median_replay_us(
    model: Model,
    key: int,
    bufs: list[Buffer],
    model_inputs: list[Buffer],
    accs: list[Accelerator],
    n_warm: int,
    n_iter: int,
) -> float:
    """Wall microseconds of one back-to-back graph replay, both ranks."""
    for _ in range(3):
        model.replay(key, *bufs, *model_inputs)
    for d in accs:
        d.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        model.replay(key, *bufs, *model_inputs)
    for d in accs:
        d.synchronize()
    return (time.perf_counter() - t0) / n_iter * 1e6


def _paired(
    arms: dict[str, tuple[Model, int, list[Buffer]]],
    bufs: list[Buffer],
    accs: list[Accelerator],
    pairs: int,
    iters: int,
    warm: int,
) -> dict[str, dict[str, float]]:
    """Times every arm in ABBA order over ``pairs`` pairs."""
    out: dict[str, list[float]] = {name: [] for name in arms}
    for model, key, model_inputs in arms.values():
        _median_replay_us(model, key, bufs, model_inputs, accs, warm, warm)
    for p in range(pairs):
        names = list(arms) if p % 2 == 0 else list(reversed(arms))
        for name in names:
            model, key, model_inputs = arms[name]
            out[name].append(
                _median_replay_us(
                    model, key, bufs, model_inputs, accs, warm, iters
                )
            )
    return {
        name: {
            "median_us": statistics.median(v),
            "min_us": min(v),
            "max_us": max(v),
            "n": len(v),
        }
        for name, v in out.items()
    }


def _refusals() -> dict[str, str]:
    """The fused path's answer for the top-6 EP2 cells, per rank count."""
    out: dict[str, str] = {}
    for name, geo in (
        ("top6_ep2_192local", TOP6_EP2_384),
        ("top6_ep2_96local", TOP6),
    ):
        _, _, reason = _plan_fused_moe_workspace(
            ep_config(geo, "nvfp4", True, shared=False)
        )
        out[name] = reason
    return out


@pytest.mark.parametrize("fmt", ["nvfp4"])
def test_top6_chain_timing(fmt: str) -> None:
    """The production chain at top-6 MoE shapes over two ranks, T 48 / 102."""
    _skip(TOP6)
    refused = _refusals()
    for name, reason in refused.items():
        print(f"REFUSED {name}: {reason}", flush=True)
    assert refused["top6_ep2_192local"], "fused path claimed 192 local experts"
    assert not refused["top6_ep2_96local"], refused["top6_ep2_96local"]

    accs, devices_ref = devices(TOP6)
    session = InferenceSession(devices=accs)
    wrapped = weights(TOP6, fmt)
    ship, ship_init, _ = build_arm(
        TOP6,
        fmt,
        False,
        wrapped,
        session,
        devices_ref,
        "SHIP",
        n_layers=1,
        shared=True,
    )
    for cell, lengths in TOP6_CELLS.items():
        bufs = [
            Buffer.from_dlpack(x).to(accs[i])
            for i, x in enumerate(inputs(TOP6, lengths, 0, cell))
        ]
        key = 200 + len(cell)
        ship.capture(key, *bufs, *ship_init.model_inputs())
        res = _paired(
            {"chain": (ship, key, ship_init.model_inputs())},
            bufs,
            accs,
            PAIRS,
            ITERS,
            WARM,
        )
        print(
            f"TIMING top6 {fmt} {cell} lengths={lengths}"
            f" local_experts={TOP6.num_experts // TOP6.n_devices}"
            f" top_k={TOP6.top_k} hidden={TOP6.hidden_dim}"
            f" moe_dim={TOP6.moe_dim}"
            f" chain_us={res['chain']['median_us']:.2f}"
            f" min={res['chain']['min_us']:.2f}"
            f" max={res['chain']['max_us']:.2f}"
            f" n={res['chain']['n']} iters={ITERS}",
            flush=True,
        )
        ship.release_captured_graph(key)


@pytest.mark.parametrize("fmt", ["nvfp4"])
def test_fused_vs_chain_timing(fmt: str) -> None:
    """Chain vs one-launch fused MoE at the EP2 geometry both arms serve."""
    _skip(FUSED)
    accs, devices_ref = devices(FUSED)
    session = InferenceSession(devices=accs)
    wrapped = weights(FUSED, fmt)
    ship, ship_init, _ = build_arm(
        FUSED,
        fmt,
        False,
        wrapped,
        session,
        devices_ref,
        "SHIP",
        n_layers=1,
        shared=True,
    )
    fused, init, _ = build_arm(
        FUSED,
        fmt,
        True,
        wrapped,
        session,
        devices_ref,
        "FULL",
        n_layers=1,
        shared=True,
    )
    assert init.config.fused_moe_ready, (
        "fused path not set up for this geometry"
    )
    assert ship_combine_send(fmt)
    for cell, lengths in FUSED_CELLS.items():
        bufs = [
            Buffer.from_dlpack(x).to(accs[i])
            for i, x in enumerate(inputs(FUSED, lengths, 0, cell))
        ]
        key_a, key_b = 400 + len(cell), 410 + len(cell)
        ship.capture(key_a, *bufs, *ship_init.model_inputs())
        fused.capture(key_b, *bufs, *init.model_inputs())
        res = _paired(
            {
                "chain": (ship, key_a, ship_init.model_inputs()),
                "fused": (fused, key_b, init.model_inputs()),
            },
            bufs,
            accs,
            PAIRS,
            ITERS,
            WARM,
        )
        ratio = res["fused"]["median_us"] / res["chain"]["median_us"]
        print(
            f"TIMING fused {fmt} {cell} lengths={lengths}"
            f" chain_us={res['chain']['median_us']:.2f}"
            f" fused_us={res['fused']['median_us']:.2f}"
            f" fused_over_chain={ratio:.3f}"
            f" n={res['chain']['n']} iters={ITERS}",
            flush=True,
        )
        ship.release_captured_graph(key_a)
        fused.release_captured_graph(key_b)


@pytest.mark.parametrize("fmt", ["nvfp4"])
def test_top6_fused_vs_chain_timing(fmt: str) -> None:
    """The top-6 pair: shipping chain vs one-launch fused at EP2."""
    _skip(TOP6)
    accs, devices_ref = devices(TOP6)
    session = InferenceSession(devices=accs)
    wrapped = weights(TOP6, fmt)
    ship, ship_init, _ = build_arm(
        TOP6,
        fmt,
        False,
        wrapped,
        session,
        devices_ref,
        "SHIP",
        n_layers=1,
        shared=True,
    )
    fused, init, _ = build_arm(
        TOP6,
        fmt,
        True,
        wrapped,
        session,
        devices_ref,
        "FULL",
        n_layers=1,
        shared=True,
    )
    assert init.config.fused_moe_ready, "fused path not set up for this cell"
    size, block, reason = _plan_fused_moe_workspace(
        ep_config(TOP6, fmt, True, shared=False)
    )
    assert not reason, reason
    print(
        f"PLAN top6 {fmt} workspace={size} B per rank block={block}",
        flush=True,
    )
    assert ship_combine_send(fmt)
    for cell, lengths in TOP6_PAIR_CELLS.items():
        bufs = [
            Buffer.from_dlpack(x).to(accs[i])
            for i, x in enumerate(inputs(TOP6, lengths, 0, cell))
        ]
        key_a, key_b = 600 + len(cell), 610 + len(cell)
        ship.capture(key_a, *bufs, *ship_init.model_inputs())
        fused.capture(key_b, *bufs, *init.model_inputs())
        res = _paired(
            {
                "chain": (ship, key_a, ship_init.model_inputs()),
                "fused": (fused, key_b, init.model_inputs()),
            },
            bufs,
            accs,
            PAIRS,
            ITERS,
            WARM,
        )
        ratio = res["fused"]["median_us"] / res["chain"]["median_us"]
        print(
            f"TIMING top6_pair {fmt} {cell} lengths={lengths}"
            f" local_experts={TOP6.num_experts // TOP6.n_devices}"
            f" top_k={TOP6.top_k} hidden={TOP6.hidden_dim}"
            f" moe_dim={TOP6.moe_dim} block={block}"
            f" chain_us={res['chain']['median_us']:.2f}"
            f" chain_min={res['chain']['min_us']:.2f}"
            f" chain_max={res['chain']['max_us']:.2f}"
            f" fused_us={res['fused']['median_us']:.2f}"
            f" fused_over_chain={ratio:.3f}"
            f" n={res['chain']['n']} iters={ITERS}",
            flush=True,
        )
        ship.release_captured_graph(key_a)
        fused.release_captured_graph(key_b)
