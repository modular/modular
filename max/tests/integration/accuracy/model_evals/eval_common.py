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
"""Shared scaffolding for the de-embedded dataset evals.

The individual eval modules (``aime_eval``, ``gpqa_eval``, ``hle_eval``,
``aa_lcr_eval``, ``aa_omniscience_eval``) were de-embedded from
``minimaxM3NonAgenticTextDatasetEval.yaml`` inline heredocs. This module
factors out the scaffolding they share — client construction, ``<think>``
stripping, chat-kwargs assembly, subsampling, parallel execution, exact-match
scoring, output writing, gated-dataset skipping, MCQ parsing, and LLM-judge
helpers — so each eval only holds its dataset-specific logic.

Everything here is pure or takes an injected client, so the evals unit-test
without a server or network.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import statistics
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Protocol

from openai import OpenAI
from tqdm import tqdm

# Root identity preamble that MiniMax prepends to every request in their own
# dataset evals (verified byte-identical across sampled production traces).
# Model-specific; kept as a CLI default so an eval can target other models by
# overriding it (or passing an empty string to drop the ``root`` turn).
DEFAULT_ROOT_PREAMBLE = (
    "Your model version is MiniMax-M3, developed by MiniMax. "
    "Knowledge cutoff: January 2026. Founded in early 2022, MiniMax is "
    "a global AI foundation model company committed to advancing the "
    "frontiers of AI towards AGI."
)
# MiniMax's own traces use the generic assistant system prompt.
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_YESNO_FIELD_RE = re.compile(r"correct\s*:\s*\**\s*(yes|no)")
_YESNO_BARE_RE = re.compile(r"\b(yes|no)\b")


class ChatClient(Protocol):
    """Minimal structural type for the OpenAI chat-completions client.

    Declared so the eval drivers can be exercised with a fake client in tests.
    """

    @property
    def chat(self) -> Any: ...


@dataclass
class GenParams:
    """Generation and sampling parameters for a single request.

    ``top_p`` and ``seed`` are ``Optional``: ``None`` omits the field entirely
    (server default). Judges pass ``top_p=None`` since they send no ``top_p``.
    """

    max_tokens: int = 98304
    temperature: float = 1.0
    top_p: float | None = 0.95
    seed: int | None = None


def make_client(base_url: str) -> OpenAI:
    """Builds the OpenAI-compatible client used by every eval.

    No client timeout: a request must only ever end by the server hitting
    ``max_tokens``, never a client-side deadline (which would drop the hardest,
    longest-reasoning problems and inflate accuracy). ``max_retries=0``: a retry
    opens a new request while the original generation keeps running and holding
    KV cache, piling the server up toward OOM.
    """
    return OpenAI(
        base_url=base_url.rstrip("/") + "/v1",
        api_key="dummy",
        timeout=None,
        max_retries=0,
    )


def strip_think(content: str | None) -> str:
    """Removes any ``<think>...</think>`` span and surrounding whitespace.

    Args:
        content: Raw assistant message content. ``None`` (returned by some
            backends when a response is truncated mid-reasoning) is treated as
            empty rather than raising.

    Returns:
        The think-stripped, stripped text.
    """
    return _THINK_RE.sub("", content or "").strip()


def build_chat_kwargs(
    model: str, messages: list[dict[str, str]], params: GenParams
) -> dict[str, Any]:
    """Assembles ``chat.completions.create`` kwargs.

    Omits ``top_p`` when ``None`` (judges send no ``top_p``) and ``seed`` when
    ``None`` (server default).
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": params.max_tokens,
        "temperature": params.temperature,
    }
    if params.top_p is not None:
        kwargs["top_p"] = params.top_p
    if params.seed is not None:
        kwargs["seed"] = params.seed
    return kwargs


def subsample(
    dataset: list[dict[str, Any]], sample_size: int | None
) -> list[dict[str, Any]]:
    """Takes ``sample_size`` items evenly across the dataset (``None`` = all)."""
    if not sample_size:
        return dataset
    step = max(1, len(dataset) // sample_size)
    return dataset[::step][:sample_size]


def make_repeat_samples(
    dataset: list[dict[str, Any]], repeats: int
) -> list[tuple[int, int, dict[str, Any]]]:
    """Expands ``dataset`` into ``(repeat_index, prompt_index, item)`` tuples.

    Every output row can then be tied back to its problem and repeat regardless
    of completion order.
    """
    return [
        (rep, i, item)
        for rep in range(repeats)
        for i, item in enumerate(dataset)
    ]


def run_parallel(
    items: list[Any],
    fn: Callable[[Any], dict[str, Any]],
    on_error: Callable[[Any, Exception], dict[str, Any]],
    workers: int,
    desc: str,
) -> tuple[list[dict[str, Any]], int]:
    """Runs ``fn`` over ``items`` on a thread pool with a tqdm progress bar.

    On error, ``on_error(item, exc)`` builds a row that is RECORDED (never
    dropped) so dropping the hardest items can't inflate accuracy.

    Returns:
        A ``(results, errors)`` tuple.
    """
    results: list[dict[str, Any]] = []
    errors = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(fn, it): it for it in items}
        for fut in tqdm(as_completed(futures), total=len(items), desc=desc):
            it = futures[fut]
            try:
                results.append(fut.result())
            except Exception as e:
                errors += 1
                print(f"Error: {e}")
                results.append(on_error(it, e))
    return results, errors


def token_stats(results: list[dict[str, Any]]) -> tuple[float, float]:
    """Computes ``(mean, p50)`` output tokens over non-error, tokened rows."""
    otoks = [
        r["completion_tokens"]
        for r in results
        if "completion_tokens" in r and "error" not in r
    ]
    mean_output_tokens = round(statistics.mean(otoks), 1) if otoks else 0.0
    p50_output_tokens = round(statistics.median(otoks), 1) if otoks else 0.0
    return mean_output_tokens, p50_output_tokens


def exact_match_score(
    results: list[dict[str, Any]], total: int, errors: int
) -> dict[str, Any]:
    """Computes the accuracy + token-stats summary over every submitted sample.

    Errors/timeouts count as incorrect (they stay in ``results`` as rows with an
    ``error`` key) so dropping the hardest problems cannot inflate accuracy.
    """
    correct = sum(1 for r in results if r.get("correct"))
    accuracy = correct / total if total else 0.0
    mean_output_tokens, p50_output_tokens = token_stats(results)
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "answered": total - errors,
        "errors": errors,
        "mean_output_tokens": mean_output_tokens,
        "p50_output_tokens": p50_output_tokens,
    }


def dump_jsonl(out_dir: str, results: list[dict[str, Any]]) -> None:
    """Writes ``results.jsonl`` (one JSON object per line) into ``out_dir``."""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.jsonl"), "w") as f:
        f.write("\n".join(json.dumps(r) for r in results))


def dump_score(out_dir: str, summary: dict[str, Any]) -> None:
    """Writes ``score.json`` (pretty-printed) into ``out_dir``."""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "score.json"), "w") as f:
        json.dump(summary, f, indent=2)


def append_github_env(
    metric_prefix: str, score: float, mean_tokens: float, p50_tokens: float
) -> None:
    """Appends the ``<PREFIX>_SCORE/_MEAN_TOKENS/_P50_TOKENS`` CI metric keys.

    A no-op when ``GITHUB_ENV`` is unset (local runs), matching the historical
    inline behavior.
    """
    env_file = os.environ.get("GITHUB_ENV")
    if not env_file:
        return
    with open(env_file, "a") as f:
        f.write(f"{metric_prefix}_SCORE={score:.4f}\n")
        f.write(f"{metric_prefix}_MEAN_TOKENS={mean_tokens:.0f}\n")
        f.write(f"{metric_prefix}_P50_TOKENS={p50_tokens:.0f}\n")


def write_outputs(
    out_dir: str,
    results: list[dict[str, Any]],
    summary: dict[str, Any],
    metric_prefix: str,
    label: str | None = None,
) -> None:
    """Writes ``results.jsonl`` + ``score.json`` and appends CI env metrics.

    The standard-eval convenience for exact-match-style summaries (produced by
    :func:`exact_match_score`). ``label`` defaults to ``metric_prefix`` and only
    affects the human-readable stdout line.
    """
    label = label or metric_prefix
    dump_jsonl(out_dir, results)
    dump_score(out_dir, summary)
    print(
        f"{label}: {summary['accuracy']:.4f} "
        f"({summary['correct']}/{summary['total']}, {summary['errors']} "
        f"errors/timeouts) mean_output_tokens={summary['mean_output_tokens']} "
        f"p50_output_tokens={summary['p50_output_tokens']}"
    )
    append_github_env(
        metric_prefix,
        summary["accuracy"],
        summary["mean_output_tokens"],
        summary["p50_output_tokens"],
    )


def is_hf_access_error(exc: BaseException) -> bool:
    """Reports whether ``exc`` (or its cause chain) is a gated-repo access error."""
    chain = ""
    e: BaseException | None = exc
    while e is not None:
        chain += str(e)
        e = getattr(e, "__cause__", None) or getattr(e, "__context__", None)
    return (
        "gated" in chain.lower()
        or "403" in chain
        or "enable access" in chain.lower()
    )


def load_gated(
    loader: Callable[[], list[dict[str, Any]]], *, label: str, dataset_id: str
) -> list[dict[str, Any]]:
    """Runs ``loader`` for a gated dataset, skipping the eval on access denial.

    On an HF gated-access error, prints a ``::warning::`` and raises
    ``SystemExit(0)`` so the CI step is skipped (not failed); other errors
    propagate.
    """
    try:
        return loader()
    except Exception as e:
        if is_hf_access_error(e):
            print(
                f"::warning::{label} skipped — HF token does not have access to "
                f"{dataset_id} (gated). Visit "
                f"https://huggingface.co/datasets/{dataset_id} "
                f"to request access and enable gated-repo permissions on your token."
            )
            raise SystemExit(0) from None
        raise


def stable_seed(key: str) -> int:
    """Derives a reproducible 32-bit seed from ``key`` via md5.

    Python's built-in ``hash()`` on ``str`` is salted by ``PYTHONHASHSEED`` and
    varies run-to-run, so it cannot drive a reproducible shuffle; md5 can.
    """
    return int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) & 0xFFFFFFFF


def parse_mcq_letter(raw: str, labels: str) -> str:
    """Extracts the chosen multiple-choice letter from a think-stripped answer.

    Takes the first standalone label stated after the last answer cue ("the
    answer is ...", "correct choice ..."); falls back to the last standalone
    label on the final non-empty line, then anywhere. Avoids grabbing a letter
    out of a word like "choices"/"definitely" or from a trailing distractor
    mention ("A, not B").

    Args:
        raw: Think-stripped response text.
        labels: The option letters, e.g. ``"ABCD"``.

    Returns:
        The uppercased chosen letter, or ``""`` when nothing parseable is found.
    """
    cls = f"[{''.join(labels)}]"
    cues = list(
        re.finditer(r"\b(?:answer|correct choice)\b", raw, re.IGNORECASE)
    )
    if cues:
        m = re.search(rf"\b({cls})\b", raw[cues[-1].end() :])
        if m:
            return m.group(1).upper()
    last_line = next((l for l in reversed(raw.splitlines()) if l.strip()), "")
    matches = re.findall(rf"\b({cls})\b", last_line) or re.findall(
        rf"\b({cls})\b", raw
    )
    return matches[-1].upper() if matches else ""


def parse_yes_no_verdict(text: str) -> bool:
    """Parses an LLM judge verdict into a boolean.

    Prefers the official ``correct: yes/no`` field; falls back to the last bare
    ``yes``/``no`` token. Defaults to ``False`` (incorrect) when nothing is
    parseable.
    """
    lowered = text.lower()
    m = _YESNO_FIELD_RE.search(lowered)
    if m:
        return m.group(1) == "yes"
    tokens = _YESNO_BARE_RE.findall(lowered)
    return bool(tokens) and tokens[-1] == "yes"


def judge(
    client: ChatClient,
    model: str,
    prompt: str,
    seed: int | None = None,
    max_tokens: int = 16384,
) -> str:
    """Runs a single-turn judge request and returns its think-stripped text.

    Uses ``temperature=0.0`` and no ``top_p``. ``max_tokens`` budgets a
    reasoning judge to finish thinking and still emit its verdict.
    """
    params = GenParams(
        max_tokens=max_tokens, temperature=0.0, top_p=None, seed=seed
    )
    resp = client.chat.completions.create(
        **build_chat_kwargs(
            model, [{"role": "user", "content": prompt}], params
        )
    )
    return strip_think(resp.choices[0].message.content or "")


def self_judge(
    client: ChatClient,
    model: str,
    prompt: str,
    seed: int | None = None,
    max_tokens: int = 16384,
) -> bool:
    """Runs :func:`judge` and parses a yes/no verdict from its response."""
    return parse_yes_no_verdict(
        judge(client, model, prompt, seed=seed, max_tokens=max_tokens)
    )
