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
"""Scores MAX's MiMo-V2 against golden fixtures.

Runs an NVFP4-format checkpoint (the full export, a reduced layer prefix, or
a ``make_random_fixture.py`` fixture) through the weight adapter and the MAX
model, teacher-forced on the gold sequences: each batch is prefilled in
chunks through the paged KV cache, and each sequence's last ``--decode``
tokens go one decode step at a time. Every layer's output and every logit
row is scored against the gold F32 HuggingFace run:

* hidden states: per-row cosine distance and relative L2 per layer, and,
  with ``--floor``, each layer's pooled mean against 3x a BF16 reference's;
* logits: ``kl64`` (KL over the gold's top 64 plus one tail bucket, the
  fixtures' metric), exact KL where the gold stored full logits, top-1 and
  |delta NLL| of the gold token, split into generated (G) and prompt rows,
  and the BF16 tolerance of :func:`gate` over full-length prompts;
* on the beacon fixture, the window and router beacons.

``--isolated`` feeds each layer the gold output of the layer before it, so
every layer is checked on its own. ``--sabotage`` injects the defects the
gates must catch, and ``--reference`` scores another reference run instead
of MAX. The results record the count of expert weights the graph declares.
It needs the checkpoint, the fixtures and a GPU, so it runs by hand, never
in CI.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from mimo_runner import SABOTAGES, Runner
from safetensors.numpy import load_file, save_file

F64 = npt.NDArray[np.float64]


def _log_softmax(x: npt.NDArray[np.float32]) -> F64:
    x = x.astype(np.float64)
    m = x.max(-1, keepdims=True)
    return x - (m + np.log(np.exp(x - m).sum(-1, keepdims=True)))


def kl64(
    gold_lp: F64, gold_idx: npt.NDArray[np.int64], test_logits: np.ndarray
) -> F64:
    """The fixtures' ``kl64``: KL(gold || test) over gold's top 64 + tail."""
    out = np.empty(len(gold_lp))
    for t0 in range(0, len(gold_lp), 256):
        lp = _log_softmax(test_logits[t0 : t0 + 256])
        top = np.argpartition(-lp, 64, axis=-1)[:, :64]
        tl = np.take_along_axis(lp, top, -1)
        gi, gl = gold_idx[t0 : t0 + 256], gold_lp[t0 : t0 + 256]
        match = gi[:, :, None] == top[:, None, :]
        got = np.where(
            match.any(-1), (match * tl[:, None, :]).sum(-1), tl.min(-1)[:, None]
        )
        pa = np.exp(gl)
        ta = np.clip(1 - pa.sum(-1), 1e-30, None)
        tb = np.clip(1 - np.exp(got).sum(-1), 1e-30, None)
        out[t0 : t0 + 256] = (pa * (gl - got)).sum(-1) + ta * (
            np.log(ta) - np.log(tb)
        )
    return out


def _stats(x: np.ndarray) -> dict[str, float] | None:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "p50": float(np.quantile(x, 0.5)),
        "p99": float(np.quantile(x, 0.99)),
        "max": float(x.max()),
    }


def score(
    gold_dir: Path,
    seq: npt.NDArray[np.int64],
    prompt_len: int,
    logits: npt.NDArray[np.float32] | None,
    hidden: list[npt.NDArray[np.float32]],
) -> dict[str, Any]:
    t = len(seq)
    result: dict[str, Any] = {"T": t, "prompt_len": prompt_len}
    if logits is not None:
        result |= _score_logits(gold_dir, seq, prompt_len, logits)
    gold_hidden = load_file(str(gold_dir / "hidden.safetensors"))
    positions = gold_hidden["positions"].astype(np.int64)
    keep = positions < t
    result["hidden"] = {}
    for i, test in enumerate(hidden):
        ref = gold_hidden[f"layer_{i:02d}"][keep].astype(np.float64)
        got = test[positions[keep]].astype(np.float64)
        cos = _cosine_distance(ref, got)
        rel = np.linalg.norm(ref - got, axis=-1) / np.linalg.norm(ref, axis=-1)
        result["hidden"][f"layer_{i:02d}"] = {
            "cos_dist": _stats(cos),
            "cos_dist_generated": _stats(
                cos[positions[keep] >= prompt_len - 1]
            ),
            "cos_dist_lt128": _stats(cos[positions[keep] < 128]),
            "cos_dist_ge128": _stats(cos[positions[keep] >= 128]),
            "rel_l2": _stats(rel),
        }
    return result


def _score_logits(
    gold_dir: Path,
    seq: npt.NDArray[np.int64],
    prompt_len: int,
    logits: npt.NDArray[np.float32],
) -> dict[str, Any]:
    t = len(seq)
    rows = np.arange(t)
    gen = rows >= prompt_len - 1
    top = load_file(str(gold_dir / "logits_topk.safetensors"))
    gold_lp = (top["values"].astype(np.float64) - top["logsumexp"][:, None])[:t]
    gold_idx = top["indices"].astype(np.int64)[:t]
    kl = kl64(gold_lp, gold_idx, logits)
    lp_test = np.empty(t)
    top1 = np.empty(t, dtype=bool)
    for t0 in range(0, t, 256):
        lp = _log_softmax(logits[t0 : t0 + 256])
        nxt = seq[t0 + 1 : t0 + 257]
        lp_test[t0 : t0 + len(nxt)] = lp[np.arange(len(nxt)), nxt]
        top1[t0 : t0 + 256] = lp.argmax(-1) == gold_idx[t0 : t0 + 256, 0]
    lp_test[t - 1] = np.nan
    gold_tgt = top["target_logprob"].astype(np.float64)[:t]
    if t < len(top["target_logprob"]):
        # A prefix run's last row predicts a token the prefix does contain.
        lp_test[t - 1] = gold_tgt[t - 1] = np.nan
    dnll = np.abs(gold_tgt - lp_test)
    # Per-row values, for aggregating the gates over prompts.
    result: dict[str, Any] = {
        "_rows": {
            "kl64": kl,
            "top1": top1,
            "dnll": dnll,
            "generated": gen,
            "gold_nll": -gold_tgt,
            "test_nll": -lp_test,
        }
    }
    for region, mask in (("G", gen), ("P", ~gen)):
        if mask.any():
            result[region] = {
                "top1": float(top1[mask].mean()),
                "kl64": _stats(kl[mask]),
                "dnll": _stats(dnll[mask]),
            }
    if (~gen).sum() > 1:
        p = slice(0, prompt_len - 1)
        result["prompt_nll_absdiff"] = float(
            abs(np.nanmean(-gold_tgt[p]) - np.nanmean(-lp_test[p]))
        )

    full = load_file(str(gold_dir / "logits_full.safetensors"))
    keep = full["positions"] < t
    if keep.any():
        pos = full["positions"][keep]
        ref = _log_softmax(full["logits"][keep])
        got = _log_softmax(logits[pos])
        exact_kl = (np.exp(ref) * (ref - got)).sum(-1)
        result["full_kl"] = _stats(exact_kl)

    return result


def gate(rows: dict[str, dict[str, np.ndarray]]) -> dict[str, Any]:
    """The fixtures' BF16 gate over full-length, teacher-forced prompts.

    Generated rows (G) over all prompts: mean ``kl64`` at most 1e-2, p99 at
    most 0.15, top-1 at least 0.985 and mean |delta NLL| at most 0.03; per
    prompt, |delta mean prompt NLL| at most 0.4.
    """
    g = {
        k: np.concatenate([r[k][r["generated"]] for r in rows.values()])
        for k in ("kl64", "top1", "dnll")
    }
    dnll = g["dnll"][np.isfinite(g["dnll"])]
    prompt_nll = {
        name: float(
            abs(
                np.nanmean(r["gold_nll"][~r["generated"]])
                - np.nanmean(r["test_nll"][~r["generated"]])
            )
        )
        for name, r in rows.items()
        if (~r["generated"]).sum() > 1
    }
    result = {
        "prompts": len(rows),
        "G_rows": int(g["kl64"].size),
        "G_kl64_mean": float(g["kl64"].mean()),
        "G_kl64_p99": float(np.quantile(g["kl64"], 0.99)),
        "G_top1": float(g["top1"].mean()),
        "G_dnll_mean": float(dnll.mean()),
        "prompt_nll_absdiff_max": max(prompt_nll.values()),
    }
    result["pass"] = bool(
        result["G_kl64_mean"] <= 1e-2
        and result["G_kl64_p99"] <= 0.15
        and result["G_top1"] >= 0.985
        and result["G_dnll_mean"] <= 0.03
        and result["prompt_nll_absdiff_max"] <= 0.4
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument(
        "--reference",
        type=Path,
        help="Score this gold-format variant instead of running MAX.",
    )
    parser.add_argument(
        "--gold", type=Path, required=True, help="A gold variant directory."
    )
    parser.add_argument(
        "--prompts",
        required=True,
        help=(
            "Comma-separated NAME or NAME:LENGTH (a prefix of the sequence);"
            " with --batch, each ;-separated group runs as one batch."
        ),
    )
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--chunk", type=int, default=2048)
    parser.add_argument("--decode", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=16384)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--batch", action="store_true", help="Run each group as one batch."
    )
    parser.add_argument(
        "--isolated",
        action="store_true",
        help="Feed each layer the gold output of the layer before it.",
    )
    parser.add_argument(
        "--floor",
        type=Path,
        help=(
            "A gold-format BF16 reference run: each layer's mean cosine"
            " distance, pooled over the prompts, must stay within 3x its."
        ),
    )
    parser.add_argument(
        "--rows",
        default="",
        help="Comma-separated rows to report per-row cosine distances for.",
    )
    parser.add_argument(
        "--save", action="store_true", help="Also save logits and hiddens."
    )
    parser.add_argument(
        "--sabotage",
        choices=SABOTAGES,
        help=(
            "Inject a defect the gates must catch: a 129-key window, the"
            " router correction bias rounded to BF16, or StackedMoE's expert"
            " split, which pairs one device's gate rows with another's up"
            " rows."
        ),
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report_rows = [int(r) for r in args.rows.split(",") if r]
    beacons_path = args.gold.parent / "beacons.json"
    beacons = (
        json.loads(beacons_path.read_text()) if beacons_path.exists() else None
    )
    if beacons is not None:
        report_rows += beacons["window_edge_rows"]

    runner = (
        Runner(
            args.checkpoint,
            args.devices,
            args.max_seq_len,
            args.isolated,
            args.sabotage,
        )
        if args.checkpoint
        else None
    )
    groups = []
    for spec in args.prompts.split(";"):
        group = []
        for item in spec.split(","):
            name, _, length = item.partition(":")
            meta = json.loads((args.gold / name / "seq.json").read_text())
            seq = np.array(meta["ids"], dtype=np.int64)
            if length:
                seq = seq[: int(length)]
            group.append((name, seq, meta["prompt_len"], len(meta["ids"])))
        groups.extend([group] if args.batch else [[entry] for entry in group])
    results: dict[str, Any] = {}
    if runner is not None:
        results["expert_weights"] = _expert_weights(runner.registry)
    full_length_rows = {}
    for group in groups:
        t0 = time.time()
        if runner is None:
            outputs = [
                _load_reference(args.reference / name, len(seq))
                for name, seq, _, _ in group
            ]
        else:
            layer_inputs = None
            if args.isolated:
                layer_inputs = [
                    _gold_layer_inputs(
                        args.gold / name, len(seq), runner.num_layers
                    )
                    for name, seq, _, _ in group
                ]
            outputs = runner.teacher_forced(
                [seq for _, seq, _, _ in group],
                args.chunk,
                args.decode,
                layer_inputs,
            )
        elapsed = time.time() - t0
        for (name, seq, prompt_len, full_len), (logits, hidden) in zip(
            group, outputs, strict=True
        ):
            key = f"{name}:{len(seq)}"
            r = score(args.gold / name, seq, prompt_len, logits, hidden)
            r["seconds"] = elapsed
            if "_rows" in r:
                rows = r.pop("_rows")
                np.savez(args.out / f"{name}_{len(seq)}_rows.npz", **rows)
                if len(seq) == full_len:
                    full_length_rows[name] = rows
            if report_rows:
                gold_hidden = load_file(
                    str(args.gold / name / "hidden.safetensors")
                )
                r["rows"] = {
                    f"layer_{i:02d}": {
                        row: float(
                            _cosine_distance(
                                gold_hidden[f"layer_{i:02d}"][row], h[row]
                            )
                        )
                        for row in report_rows
                        if row < len(seq)
                    }
                    for i, h in enumerate(hidden)
                }
            results[key] = r
            if args.save:
                save_file(
                    ({} if logits is None else {"logits": logits})
                    | {f"layer_{i:02d}": h for i, h in enumerate(hidden)},
                    str(args.out / f"{name}_{len(seq)}.safetensors"),
                )
            last = r["hidden"][max(r["hidden"])]
            print(
                f"{key}: {elapsed:.1f} s, last-layer cos "
                f"{last['cos_dist']['mean']:.2e} / {last['cos_dist']['max']:.2e}, "
                + ", ".join(
                    f"{region} top1 {r[region]['top1']:.4f} kl64 "
                    f"{r[region]['kl64']['mean']:.2e}"
                    for region in ("G", "P")
                    if region in r
                ),
                flush=True,
            )
            (args.out / "results.json").write_text(
                json.dumps(results, indent=1)
            )
    if full_length_rows:
        results["gate"] = gate(full_length_rows)
        print("gate:", json.dumps(results["gate"]), flush=True)
    if args.floor is not None:
        floor = {}
        for group in groups:
            for name, seq, prompt_len, _ in group:
                _, hidden = _load_reference(args.floor / name, len(seq))
                floor[f"{name}:{len(seq)}"] = score(
                    args.gold / name, seq, prompt_len, None, hidden
                )
        results["floor_gate"] = floor_gate(results, floor)
        print("floor gate:", json.dumps(results["floor_gate"]), flush=True)
    if beacons is not None:
        results["beacon_gate"] = beacon_gate(beacons, results)
        print("beacon gate:", json.dumps(results["beacon_gate"]), flush=True)
    (args.out / "results.json").write_text(json.dumps(results, indent=1))


def floor_gate(
    results: dict[str, Any], floor: dict[str, Any]
) -> dict[str, Any]:
    """Each layer's pooled mean cosine distance against 3x a BF16 reference's.

    Pooled over every row of every prompt: on a single short prompt, one
    near-tied routing decision flipping either way dominates the mean.
    """

    def pooled(runs: dict[str, Any], layer: str) -> float:
        stats = [runs[key]["hidden"][layer]["cos_dist"] for key in floor]
        return sum(s["n"] * s["mean"] for s in stats) / sum(
            s["n"] for s in stats
        )

    layers = {}
    for layer in next(iter(floor.values()))["hidden"]:
        test, ref = pooled(results, layer), pooled(floor, layer)
        layers[layer] = {
            "cos_dist": test,
            "floor": ref,
            "ratio": test / ref,
            "pass": bool(test <= 3 * ref),
        }
    return {"layers": layers, "pass": all(v["pass"] for v in layers.values())}


BEACON_TOLERANCE = {"window": 5e-2, "router": 6e-3}
"""Cosine distance a correct run stays under at the beacons.

MAX quantizes activations to FP8 before every dense and expert GEMM, which
alone puts 6.6e-4 on layer 0; a clean run reaches 6.2e-3 at a window-edge
row and 1.9e-3 on the router median. A 129-key window takes the edge maximum
to 0.82, and a BF16 correction bias the router median to 1.9e-2."""


def beacon_gate(
    beacons: dict[str, Any], results: dict[str, Any]
) -> dict[str, Any]:
    """Scores the beacon fixture (see ``make_random_fixture.py``).

    The window beacon is read at the rows 128 past each beacon token in the
    first sliding layer, which must not see the beacon; the router beacon is
    the median over all rows of the layer whose routing only the correction
    bias decides.
    """
    r = results[f"{beacons['prompt']}:{beacons['length']}"]
    window = r["rows"][f"layer_{beacons['window_layer']:02d}"]
    edge = max(window[row] for row in beacons["window_edge_rows"])
    router = r["hidden"][f"layer_{beacons['router_layer']:02d}"]["cos_dist"][
        "p50"
    ]
    tolerance = BEACON_TOLERANCE
    return {
        "window_edge_max_cos_dist": edge,
        "router_median_cos_dist": router,
        "tolerance": tolerance,
        "pass": bool(
            edge <= tolerance["window"] and router <= tolerance["router"]
        ),
    }


_EXPERT_STACK = re.compile(
    r"layers\.\d+\.mlp\.experts\.(gate_up|down)_proj(_scale)?"
)
_PER_EXPERT = re.compile(r"\.experts\.\d+\.")


def _expert_weights(registry: dict[str, Any]) -> dict[str, Any]:
    """The expert weights the graph declares: the stacks, by rank, and any
    per-expert constant, which there must be none of."""
    stacks = [name for name in registry if _EXPERT_STACK.fullmatch(name)]
    return {
        "stacks": len(stacks),
        "ranks": {
            kind: sorted(
                {
                    len(registry[name].shape)
                    for name in stacks
                    if name.endswith("_scale") == (kind == "scales")
                }
            )
            for kind in ("codes", "scales")
        },
        "per_expert": sum(1 for name in registry if _PER_EXPERT.search(name)),
    }


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = a.astype(np.float64), b.astype(np.float64)
    return 1 - (a * b).sum(-1) / (
        np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    )


def _gold_layer_inputs(
    gold_dir: Path, length: int, num_layers: int
) -> list[npt.NDArray[np.float32]]:
    """Each layer's gold input: the embedding, then the layer before's output."""
    gold = load_file(str(gold_dir / "hidden.safetensors"))
    assert (gold["positions"][:length] == np.arange(length)).all()
    return [
        gold["embed" if k == 0 else f"layer_{k - 1:02d}"][:length]
        for k in range(num_layers)
    ]


def _load_reference(
    variant_dir: Path, length: int
) -> tuple[npt.NDArray[np.float32] | None, list[npt.NDArray[np.float32]]]:
    """A gold-format variant's logits (when stored at every row) and layers."""
    full = load_file(str(variant_dir / "logits_full.safetensors"))
    positions = full["positions"]
    logits = (
        full["logits"][:length]
        if len(positions) >= length
        and (positions[:length] == np.arange(length)).all()
        else None
    )
    hidden = load_file(str(variant_dir / "hidden.safetensors"))
    assert (hidden["positions"][:length] == np.arange(length)).all()
    layers = sorted(k for k in hidden if k.startswith("layer_"))
    return logits, [hidden[k][:length] for k in layers]


if __name__ == "__main__":
    main()
