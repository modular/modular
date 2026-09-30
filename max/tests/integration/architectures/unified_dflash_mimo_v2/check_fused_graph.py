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
"""Checks the fused MiMo-V2 DFlash graph on the real checkpoint.

Each subcommand loads one graph, so the target is on the GPUs once:

* ``selfcheck``: greedy speculative decoding of a workload through the fused
  graph. Every committed token must be the target's argmax, bitmask applied,
  in the forward that committed it. Reports the acceptance length, tokens
  per drafted verify step.
* ``specoff``: the plain base graph's one-token decode path, teacher-forced on
  what ``selfcheck`` committed: each committed token's gap below that path's
  argmax; and plain greedy decoding of the same
  prompts, for token equality.
* ``calibrate``: the verify path (``--path verify``, the fused graph) or the
  decode path (``--path decode``) teacher-forced on the reference fixtures'
  gold sequences and scored with their BF16 tolerance; ``pair`` compares the
  two.
* ``kvdump``: one sequence's drafter context from the fused graph, whole or
  in chunks, or from base-ctx in chunks and after a prefix hit on the fused
  graph's pages; ``kvcompare`` compares the dumps position by position.
* ``chunks``: base-ctx's logits and KV along one sequence prefilled whole and
  in chunks of each size: where the chunked run first departs, in which
  layers and by how much, and how far each run sits from the F32 gold.

Needs the checkpoint, the fixtures and the GPUs, so it runs by hand.
"""

from __future__ import annotations

import argparse
import functools
import importlib.util
import json
import time
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import numpy.typing as npt
from max.driver import Accelerator, Device
from max.engine import InferenceSession
from max.graph import DeviceRef
from max.graph.weights import SafetensorWeights, load_weights
from max.pipelines.architectures.dflash_mimo_v2 import (
    DFlashMiMoV2Config,
    load_mask_embedding,
)
from max.pipelines.architectures.dflash_mimo_v2 import (
    convert_safetensor_state_dict as convert_drafter,
)
from max.pipelines.architectures.dflash_mimo_v2.weight_adapters import (
    DRAFT_WEIGHTS_FILE,
    MASK_EMBEDDING,
    MASK_EMBEDDING_FILE,
)
from max.pipelines.architectures.mimo_v2.weight_adapters import (
    convert_safetensor_state_dict as convert_target,
)
from max.pipelines.architectures.unified_dflash_mimo_v2.model_config import (
    DRAFT,
)
from mimo_dflash_harness import (
    PAGE_SIZE,
    BaseRunner,
    FusedRunner,
    Model,
    PagedTree,
    Pages,
    Row,
    drafter_kv_params,
    generate,
    greedy,
)
from transformers import AutoConfig, AutoTokenizer

STOP_IDS = (151643, 151645, 151672)
GAP = 1.0
"""A committed token more than this below the decode path's argmax is an
exceedance."""
F32 = npt.NDArray[np.float32]


@functools.cache
def sampleable_vocab(checkpoint: Path) -> int:
    """The tokenizer's size; ``lm_head`` rows past it are padding."""
    return len(AutoTokenizer.from_pretrained(checkpoint))


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load(checkpoint: Path, max_seq_len: int) -> Model:
    """The real target and drafter, adapted to the serving layouts."""
    hf = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
    dflash = json.loads((checkpoint / "dflash" / "config.json").read_text())
    one = [DeviceRef.GPU()]
    shapes = DFlashMiMoV2Config.from_dflash_config(
        dflash,
        devices=one,
        kv_params=drafter_kv_params(dflash, one, speculative=False),
        max_seq_len=max_seq_len,
    )
    draft_state = convert_drafter(
        dict(
            load_weights([checkpoint / "dflash" / DRAFT_WEIGHTS_FILE]).items()
        ),
        shapes,
        load_mask_embedding(
            checkpoint / "dflash" / MASK_EMBEDDING_FILE, shapes
        ),
    )
    mask = draft_state.pop(MASK_EMBEDDING)
    t0 = time.time()
    weights = SafetensorWeights(sorted(checkpoint.glob("*.safetensors")))
    target_state = convert_target(dict(weights.items()), hf)
    _log(
        f"adapted {len(target_state)} target tensors in {time.time() - t0:.0f} s"
    )
    return Model(
        hf=hf,
        dflash=dflash,
        target_state=target_state,
        draft_state=draft_state,
        mask_embedding=mask,
        max_seq_len=max_seq_len,
    )


def _devices(n: int) -> tuple[list[Device], list[DeviceRef]]:
    devices: list[Device] = [Accelerator(i) for i in range(n)]
    return devices, [DeviceRef.from_device(d) for d in devices]


def _fused(
    args: argparse.Namespace, model: Model, max_batch: int
) -> tuple[FusedRunner, PagedTree]:
    devices, refs = _devices(args.devices)
    runner = FusedRunner(
        model.spec(refs, args.k, sampleable_vocab(args.checkpoint)),
        model.target_state,
        model.draft_state,
        model.mask_embedding,
        devices,
        InferenceSession(devices=devices),
        max_batch_size=max_batch,
        test_mutation=getattr(args, "test_mutation", None),
    )
    return runner, PagedTree(runner.groups, devices, args.pages)


def _base(
    args: argparse.Namespace, model: Model, *, writer: bool
) -> tuple[BaseRunner, PagedTree]:
    devices, refs = _devices(args.devices)
    runner = BaseRunner(
        model.target(refs, speculative=False),
        model.target_state,
        devices,
        InferenceSession(devices=devices),
        drafter=model.drafter(refs, speculative=False) if writer else None,
        draft_state=model.draft_state if writer else None,
    )
    return runner, PagedTree(runner.groups, devices, args.pages)


def workload(path: Path, checkpoint: Path, limit: int) -> list[list[int]]:
    """A recorded workload's chat-templated prompts, tokenized; a ``.jsonl``
    workload is one already-tokenized ``{"ids": [...]}`` prompt per line."""
    if path.suffix == ".jsonl":
        lines = path.read_text().splitlines()[:limit]
        return [json.loads(line)["ids"] for line in lines]
    items = json.loads(path.read_text())[:limit]
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    prompts = [
        tokenizer.encode(item["text"], add_special_tokens=False)
        for item in items
    ]
    wrong = [
        i
        for i, (ids, item) in enumerate(zip(prompts, items, strict=True))
        if len(ids) != item["num_tokens"]
    ]
    if wrong:
        raise ValueError(f"prompts {wrong[:5]} tokenize to another length")
    return prompts


def selfcheck(args: argparse.Namespace) -> dict[str, Any]:
    model = load(args.checkpoint, args.max_seq_len)
    sampleable = sampleable_vocab(args.checkpoint)
    runner, caches = _fused(args, model, args.batch)
    _log("compiled")
    prompts = workload(args.workload, args.checkpoint, args.limit)
    requests: list[dict[str, Any]] = []
    violations: list[str] = []
    for start in range(0, len(prompts), args.batch):
        batch = prompts[start : start + args.batch]
        t0 = time.time()
        gen = generate(
            runner, caches, Pages(), batch, args.max_new, stop_ids=STOP_IDS
        )
        violations += [f"prompt {start}+ {v}" for v in gen.violations]
        for i, prompt in enumerate(batch):
            requests.append(
                {
                    "prompt": prompt,
                    "tokens": gen.tokens[i],
                    "delivered": gen.delivered[i],
                    "accepted": gen.accepted[i],
                }
            )
        _log(
            f"prompts {start}-{start + len(batch) - 1}: acceptance"
            f" {gen.acceptance_length():.3f}, {len(gen.violations)} violations,"
            f" {time.time() - t0:.0f} s"
        )
    steps = [d for r in requests for d in r["delivered"]]
    accepted = np.array([a for r in requests for a in r["accepted"]])
    return {
        "requests": requests,
        "k": args.k,
        "test_mutation": args.test_mutation,
        "drafted_verify_steps": len(steps),
        "acceptance_length": sum(steps) / max(1, len(steps)),
        "per_position_acceptance": [
            float((accepted >= j).mean()) for j in range(1, args.k + 1)
        ],
        "violations": len(violations),
        "first_violations": violations[:20],
        "stopped": sum(r["tokens"][-1] in STOP_IDS for r in requests),
        "padding_ids_committed": sum(
            t >= sampleable for r in requests for t in r["tokens"]
        ),
    }


def specoff(args: argparse.Namespace) -> dict[str, Any]:
    run = json.loads(args.run.read_text())["requests"]
    model = load(args.checkpoint, args.max_seq_len)
    sampleable = sampleable_vocab(args.checkpoint)
    runner, caches = _base(args, model, writer=False)
    _log("compiled")
    gaps: list[float] = []
    per_request: list[dict[str, float]] = []
    equal, compared, first_divergence = 0, 0, []
    for start in range(0, len(run), args.batch):
        batch = run[start : start + args.batch]
        prompts = [r["prompt"] for r in batch]
        committed = [r["tokens"] for r in batch]
        longest = max(len(c) for c in committed)
        forced = [c + [c[-1]] * (longest - len(c)) for c in committed]
        _, logits = greedy(
            runner,
            caches,
            Pages(),
            prompts,
            longest,
            sampleable,
            forced=forced,
        )
        for i, seq in enumerate(committed):
            row = [
                float(z[:sampleable].max() - z[t])
                for t, z in zip(seq, logits[i], strict=False)
            ]
            gaps += row
            per_request.append(
                {
                    "positions": len(row),
                    "exceedances": sum(g > GAP for g in row),
                    "largest_gap": max(row, default=0.0),
                }
            )
        if args.no_plain:
            _log(f"requests {start}-{start + len(batch) - 1} done")
            continue
        plain, plain_logits = greedy(
            runner, caches, Pages(), prompts, longest, sampleable
        )
        for i, seq in enumerate(committed):
            n = len(seq)
            same = [a == b for a, b in zip(seq, plain[i][:n], strict=True)]
            equal += sum(same)
            compared += n
            if not all(same):
                j = same.index(False)
                top = np.sort(plain_logits[i][j][:sampleable])[-2:]
                first_divergence.append(
                    {
                        "request": start + i,
                        "index": j,
                        "margin": float(top[1] - top[0]),
                    }
                )
        _log(f"requests {start}-{start + len(batch) - 1} done")
    g = np.array(gaps)
    return {
        "positions": int(g.size),
        "exceedances": int((g > GAP).sum()),
        "padding_ids_committed": sum(
            t >= sampleable for r in run for t in r["tokens"]
        ),
        "gap_histogram": {
            str(edge): int((g > edge).sum()) for edge in (0.5, 1.0, 2.0, 4.0)
        },
        "largest_gap": float(g.max()),
        "gap_p999": float(np.quantile(g, 0.999)),
        "equal_to_plain_greedy": equal / max(1, compared),
        "identical_requests": len(run) - len(first_divergence),
        "first_divergences": first_divergence,
        "per_request": per_request,
    }


def _gold(gold: Path) -> list[tuple[str, npt.NDArray[np.int64], int]]:
    out = []
    for d in sorted(p for p in gold.iterdir() if (p / "seq.json").exists()):
        seq = json.loads((d / "seq.json").read_text())
        out.append(
            (d.name, np.array(seq["ids"], np.int64), int(seq["prompt_len"]))
        )
    return out


def verify_teacher_forced(
    runner: FusedRunner,
    caches: PagedTree,
    seqs: Sequence[npt.NDArray[np.int64]],
    prompt_lens: Sequence[int],
) -> tuple[list[F32], list[npt.NDArray[np.int64]]]:
    """Generated rows' logits along ``seqs`` on the fused graph's paths.

    Each prompt is prefilled whole, the path that commits the first output
    token, whose row is ``prompt_len - 1``. Then each step verifies the
    sequence's own next K tokens as drafts and advances by K + 1 whatever
    was accepted, so a later row ``p`` sits at verify offset
    ``(p - prompt_len) % (K + 1)``. Prompt rows are left NaN.
    """
    k = runner.k
    pages = Pages()
    logits = [np.full((len(s), runner.vocab), np.nan, np.float32) for s in seqs]
    offsets = [np.full(len(s), -1, np.int64) for s in seqs]
    rows = [Row([], 0, pages.take(len(s) + 4 * (k + 1))) for s in seqs]
    for i, (s, p) in enumerate(zip(seqs, prompt_lens, strict=True)):
        out = runner.step(caches, [Row(s[:p].tolist(), 0, rows[i].pages)])
        logits[i][p - 1] = out.logits[0][0]
    position = list(prompt_lens)
    live = [i for i, s in enumerate(seqs) if position[i] < len(s)]
    while live:
        step = []
        for i in live:
            p, s = position[i], seqs[i]
            drafts = s[p + 1 : p + 1 + k].tolist()
            drafts += [0] * (k - len(drafts))
            step.append(Row([int(s[p])], p, rows[i].pages, drafts))
        out = runner.step(caches, step)
        still = []
        for j, i in enumerate(live):
            p = position[i]
            n = min(k + 1, len(seqs[i]) - p)
            logits[i][p : p + n] = out.logits[j][:n]
            offsets[i][p : p + n] = np.arange(n)
            position[i] = p + k + 1
            if position[i] < len(seqs[i]):
                still.append(i)
        live = still
    return logits, offsets


def generated_gate(rows: dict[str, dict[str, np.ndarray]]) -> dict[str, Any]:
    """The reference fixtures' primary BF16 tolerance, on generated rows
    only."""
    gen = {
        k: np.concatenate([r[k][r["generated"]] for r in rows.values()])
        for k in ("kl64", "top1", "dnll")
    }
    dnll = gen["dnll"][np.isfinite(gen["dnll"])]
    result = {
        "G_rows": int(gen["kl64"].size),
        "G_kl64_mean": float(gen["kl64"].mean()),
        "G_kl64_p99": float(np.quantile(gen["kl64"], 0.99)),
        "G_top1": float(gen["top1"].mean()),
        "G_dnll_mean": float(dnll.mean()),
    }
    result["pass"] = bool(
        result["G_kl64_mean"] <= 1e-2
        and result["G_kl64_p99"] <= 0.15
        and result["G_top1"] >= 0.985
        and result["G_dnll_mean"] <= 0.03
    )
    return result


def decode_teacher_forced(
    runner: BaseRunner,
    caches: PagedTree,
    seqs: Sequence[npt.NDArray[np.int64]],
    prompt_lens: Sequence[int],
) -> list[F32]:
    """Every row's logits on the one-token decode path: each prompt
    prefilled in 2,048-token chunks, one sequence at a time, then the rest
    one decode step at a time, all sequences in one batch."""
    pages = Pages()
    logits = [np.empty((len(s), runner.vocab), np.float32) for s in seqs]
    rows = [Row([], 0, pages.take(len(s) + 1)) for s in seqs]
    for i, (s, p) in enumerate(zip(seqs, prompt_lens, strict=True)):
        for start in range(0, p, 2048):
            chunk = s[start : min(p, start + 2048)].tolist()
            logits[i][start : start + len(chunk)] = runner.step(
                caches, [Row(chunk, start, rows[i].pages)]
            )
    done = list(prompt_lens)
    while True:
        live = [i for i, s in enumerate(seqs) if done[i] < len(s)]
        if not live:
            return logits
        out = runner.step(
            caches,
            [
                Row([int(seqs[i][done[i]])], done[i], rows[i].pages)
                for i in live
            ],
        )
        for j, i in enumerate(live):
            logits[i][done[i]] = out[j]
            done[i] += 1


def _p2_scorer(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "check_reference_fixtures", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def calibrate(args: argparse.Namespace) -> dict[str, Any]:
    scorer = _p2_scorer(args.p2_harness)
    gold = _gold(args.gold)
    model = load(args.checkpoint, args.max_seq_len)
    seqs = [s for _, s, _ in gold]
    if args.path == "verify":
        runner, caches = _fused(args, model, len(gold))
        _log("compiled")
        logits, offsets = verify_teacher_forced(
            runner, caches, seqs, [p for *_, p in gold]
        )
    else:
        base, caches = _base(args, model, writer=False)
        _log("compiled")
        logits = decode_teacher_forced(
            base, caches, seqs, [p for *_, p in gold]
        )
        offsets = [np.zeros(len(s), np.int64) for s in seqs]
    rows, per_prompt = {}, {}
    generated = []
    for (tag, seq, prompt_len), z, off in zip(
        gold, logits, offsets, strict=True
    ):
        scored = scorer._score_logits(args.gold / tag, seq, prompt_len, z)
        rows[tag] = scored.pop("_rows")
        per_prompt[tag] = scored
        gen = np.arange(len(seq)) >= prompt_len - 1
        generated.append(z[gen].astype(np.float16))
        rows[tag]["offset"] = off
    np.save(args.out.with_suffix(".G_logits.npy"), np.concatenate(generated))
    result = {"path": args.path, "k": args.k, "gate": generated_gate(rows)}
    kl = np.concatenate([r["kl64"][r["generated"]] for r in rows.values()])
    off = np.concatenate([r["offset"][r["generated"]] for r in rows.values()])
    result["G_kl64_mean_by_offset"] = {
        int(o): float(kl[off == o].mean()) for o in np.unique(off)
    }
    result["per_prompt"] = per_prompt
    return result


def pair(args: argparse.Namespace) -> dict[str, Any]:
    """Full-vocabulary KL of the verify path's generated rows from the
    decode path's, to set beside the widest KL between two reference
    backends."""
    a = np.load(args.decode).astype(np.float32)
    b = np.load(args.verify).astype(np.float32)
    kl = []
    for t0 in range(0, len(a), 256):
        la = a[t0 : t0 + 256] - np.logaddexp.reduce(
            a[t0 : t0 + 256], -1, keepdims=True
        )
        lb = b[t0 : t0 + 256] - np.logaddexp.reduce(
            b[t0 : t0 + 256], -1, keepdims=True
        )
        kl.append((np.exp(la) * (la - lb)).sum(-1))
    k = np.concatenate(kl)
    top1 = (a.argmax(-1) == b.argmax(-1)).mean()
    return {
        "rows": int(k.size),
        "kl_mean": float(k.mean()),
        "kl_p99": float(np.quantile(k, 0.99)),
        "top1": float(top1),
    }


def kvdump(args: argparse.Namespace) -> dict[str, Any]:
    seq = _gold(args.gold)
    tokens = next(s for tag, s, _ in seq if tag == args.tag)[
        : args.length
    ].tolist()
    model = load(args.checkpoint, args.max_seq_len)
    hit = args.hit_pages * PAGE_SIZE
    block = int(model.dflash["block_size"])
    pages = Pages()
    dump: dict[str, Any] = {}
    if args.graph == "fused":
        runner, caches = _fused(args, model, 1)
        own = pages.take(len(tokens) + 2 * block)
        step = args.fused_chunk or len(tokens)
        for start in range(0, len(tokens), step):
            runner.step(caches, [Row(tokens[start : start + step], start, own)])
        dump["fused"] = caches.read(DRAFT, own, range(len(tokens)))
        exported: dict[str, Any] = {
            f"{g}.{d}": a
            for g, arrays in caches.export_pages(own[: args.hit_pages]).items()
            for d, a in enumerate(arrays)
        }
        np.savez(args.out.with_suffix(".pages.npz"), **exported)
    else:
        runner_b, caches = _base(args, model, writer=True)
        chunked = pages.take(len(tokens) + 2 * block)
        for start in range(0, len(tokens), args.chunk):
            runner_b.step(
                caches,
                [Row(tokens[start : start + args.chunk], start, chunked)],
            )
        dump["chunked"] = caches.read(DRAFT, chunked, range(len(tokens)))
        saved = np.load(args.hit_from)
        hit_pages = pages.take(hit)
        caches.import_pages(
            hit_pages,
            {
                g: [saved[f"{g}.{d}"] for d in range(args.devices)]
                for g in caches.blocks
            },
        )
        restored = hit_pages + pages.take(len(tokens) - hit + 2 * block)
        runner_b.step(caches, [Row(tokens[hit:], hit, restored)])
        dump["prefix_hit"] = caches.read(DRAFT, restored, range(len(tokens)))
    np.savez(
        args.out.with_suffix(".kv.npz"),
        **{k: v.astype(np.float16) for k, v in dump.items()},
    )
    return {"graph": args.graph, "positions": len(tokens), "hit": hit}


def kvcompare(args: argparse.Namespace) -> dict[str, Any]:
    fused = np.load(args.fused)["fused"].astype(np.float32)
    other = np.load(args.base_ctx)
    result: dict[str, Any] = {}
    for name in ("chunked", "prefix_hit"):
        got = other[name].astype(np.float32)
        for index, what in enumerate("KV"):
            a = np.moveaxis(got[index], 1, 0).reshape(got.shape[2], -1)
            b = np.moveaxis(fused[index], 1, 0).reshape(fused.shape[2], -1)
            err = np.linalg.norm(a - b, axis=1) / np.linalg.norm(b, axis=1)
            result[f"{name}_{what}"] = {
                "max": float(err.max()),
                "argmax": int(err.argmax()),
                "p99": float(np.quantile(err, 0.99)),
                "median": float(np.median(err)),
            }
    return result


def chunks(args: argparse.Namespace) -> dict[str, Any]:
    """Where a chunked prefill through base-ctx first departs from a whole
    one, in the logits and in each KV group, for each ``--sizes`` chunk; and
    how far each sits from the F32 gold past the first page boundary."""
    scorer = _p2_scorer(args.p2_harness)
    seq = _gold(args.gold)
    full, prompt_len = next((s, p) for tag, s, p in seq if tag == args.tag)
    tokens = full[: args.length].tolist()
    n = len(tokens)

    def against_gold(z: F32, start: int) -> dict[str, float]:
        rows = scorer._score_logits(
            args.gold / args.tag, np.array(tokens), prompt_len, z
        )["_rows"]
        return {
            "top1": float(rows["top1"][start:].mean()),
            "kl64_mean": float(rows["kl64"][start:].mean()),
        }

    model = load(args.checkpoint, args.max_seq_len)
    runner, caches = _base(args, model, writer=True)
    _log("compiled")
    pages = Pages()
    whole, own = pages.take(n), pages.take(n)
    ref = runner.step(caches, [Row(tokens, 0, whole)])
    ref_kv = {g: caches.read(g, whole, range(n)) for g in caches.blocks}
    result: dict[str, Any] = {"positions": n}
    for size in args.sizes:
        boundary = -(-size // PAGE_SIZE) * PAGE_SIZE
        logits = np.empty_like(ref)
        for start in range(0, n, size):
            chunk = tokens[start : start + size]
            logits[start : start + len(chunk)] = runner.step(
                caches, [Row(chunk, start, own)]
            )
        diff = np.abs(logits - ref).max(-1)
        top = logits.argmax(-1) == ref.argmax(-1)
        report: dict[str, Any] = {
            "logits_first_diff": int(np.argmax(diff > 0))
            if diff.any()
            else None,
            "logits_max_abs_by_chunk": [
                float(diff[s : s + size].max()) for s in range(0, n, size)
            ],
            "top1_by_chunk": [
                float(top[s : s + size].mean()) for s in range(0, n, size)
            ],
            "from": boundary,
            "whole_vs_gold": against_gold(ref, boundary),
            "chunked_vs_gold": against_gold(logits, boundary),
        }
        for group, want in ref_kv.items():
            got = caches.read(group, own, range(n))
            differs = got != want
            where = np.flatnonzero(differs.any(axis=(0, 1, 3, 4)))
            rel = np.linalg.norm(
                np.moveaxis(got - want, 2, 0).reshape(n, -1), axis=1
            ) / np.linalg.norm(np.moveaxis(want, 2, 0).reshape(n, -1), axis=1)
            entry: dict[str, Any] = {
                "first_diff": int(where[0]) if where.size else None,
                "max_rel_by_chunk": [
                    float(rel[s : s + size].max()) for s in range(0, n, size)
                ],
                "first_layers": [],
            }
            if where.size:
                # The first layers that differ at the first position that
                # does, and by how much: an onset of a few ulps is rounding.
                at = where[0]
                layers = np.flatnonzero(differs[:, :, at].any(axis=(0, 2, 3)))
                entry["first_layers"] = [
                    {
                        "layer": int(layer),
                        "differing": int(differs[:, layer, at].sum()),
                        "of": int(differs[:, layer, at].size),
                        "max_abs": float(
                            np.abs(got[:, layer, at] - want[:, layer, at]).max()
                        ),
                        "max_value": float(np.abs(want[:, layer, at]).max()),
                    }
                    for layer in layers[:3]
                ]
            report[group] = entry
        result[str(size)] = report
        _log(f"size {size}: {json.dumps(report)}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--checkpoint", type=Path, required=True)
        p.add_argument("--devices", type=int, default=2)
        p.add_argument("--k", type=int, default=7)
        p.add_argument("--pages", type=int, default=512)
        p.add_argument("--max-seq-len", type=int, default=32768)
        p.add_argument("--out", type=Path, required=True)

    p = sub.add_parser("selfcheck")
    common(p)
    p.add_argument("--workload", type=Path, required=True)
    p.add_argument("--limit", type=int, default=64)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--max-new", type=int, default=1024)
    p.add_argument("--test-mutation", choices=["accept_one_extra"])
    p = sub.add_parser("specoff")
    common(p)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument(
        "--no-plain",
        action="store_true",
        help="Score the committed tokens only, without plain greedy decoding.",
    )
    p = sub.add_parser("calibrate")
    common(p)
    p.add_argument("--gold", type=Path, required=True)
    p.add_argument("--p2-harness", type=Path, required=True)
    p.add_argument("--path", choices=["verify", "decode"], default="verify")
    p = sub.add_parser("pair")
    p.add_argument("--decode", type=Path, required=True)
    p.add_argument("--verify", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("kvdump")
    common(p)
    p.add_argument("--graph", choices=["fused", "base-ctx"], required=True)
    p.add_argument("--gold", type=Path, required=True)
    p.add_argument("--tag", default="p07_long_code_9k")
    p.add_argument("--length", type=int, default=3000)
    p.add_argument("--chunk", type=int, default=1000)
    p.add_argument("--hit-pages", type=int, default=15)
    p.add_argument("--hit-from", type=Path)
    p.add_argument(
        "--fused-chunk",
        type=int,
        default=0,
        help="Prefill the fused graph in chunks of this size, not whole.",
    )
    p = sub.add_parser("chunks")
    common(p)
    p.add_argument("--gold", type=Path, required=True)
    p.add_argument("--tag", default="p07_long_code_9k")
    p.add_argument("--length", type=int, default=3000)
    p.add_argument("--sizes", type=int, nargs="+", default=[1000, 1024])
    p.add_argument("--p2-harness", type=Path, required=True)
    p = sub.add_parser("kvcompare")
    p.add_argument("--fused", type=Path, required=True)
    p.add_argument("--base-ctx", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    command = {
        "selfcheck": selfcheck,
        "specoff": specoff,
        "calibrate": calibrate,
        "pair": pair,
        "kvdump": kvdump,
        "kvcompare": kvcompare,
        "chunks": chunks,
    }[args.command]
    result = command(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, default=str))
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("requests", "per_prompt", "per_request")
            },
            indent=1,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
