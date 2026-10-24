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
"""AIME math-competition eval against an OpenAI-compatible server.

De-embedded from ``minimaxM3NonAgenticTextDatasetEval.yaml`` so the eval logic
lives in a tested, locally-runnable module instead of an inline YAML heredoc.
The scoring core (:func:`grade`) and the driver (:func:`run_eval`) are pure and
take an injected client, so they unit-test without a server or network. Shared
scaffolding lives in :mod:`eval_common`.

The year's problem set is parameterised by ``--dataset`` (default
``math-ai/aime25``). AIME26 uses the same grader, same prompt and same
``\\boxed{}`` extraction, so one binary serves both years; the caller just
picks the dataset ID. The metric prefix and out-dir default from the dataset
name (``math-ai/aime26`` -> ``AIME26`` and ``/tmp/aime26-results``).

Run locally against a server on ``localhost:8000``::

    ./bazelw run //max/tests/integration/accuracy/model_evals:aime_eval -- \\
        --base-url http://localhost:8000 --model MiniMaxAI/MiniMax-M3-MXFP8 \\
        --sample-size 2 --seed 0 --out-dir /tmp/aime25

To run AIME26 instead, swap the dataset::

    ./bazelw run //max/tests/integration/accuracy/model_evals:aime_eval -- \\
        --base-url http://localhost:8000 --model MiniMaxAI/MiniMax-M3-MXFP8 \\
        --dataset math-ai/aime26

Results stream to ``<out-dir>/results.jsonl`` as each request completes, so a
killed/crashed run keeps everything already finished. To debug specific
problems (e.g. re-run exactly the ones that errored or looked wrong last
time), pass ``--row-ids`` instead of ``--sample-size``::

    ./bazelw run //max/tests/integration/accuracy/model_evals:aime_eval -- \\
        --base-url http://localhost:8000 --model MiniMaxAI/MiniMax-M3-MXFP8 \\
        --row-ids 3,17,42 --out-dir /tmp/aime25-debug
"""

from __future__ import annotations

import re
from typing import Any

import click
from datasets import load_dataset
from eval_common import (
    DEFAULT_ROOT_PREAMBLE,
    DEFAULT_SYSTEM_PROMPT,
    ChatClient,
    GenParams,
    chat_completion,
    dump_config,
    exact_match_score,
    expand_repeats,
    make_client,
    prepare_out_dir,
    response_fields,
    resumable_config,
    run_parallel,
    select_rows,
    strip_think,
    write_outputs,
)

_BOXED_OPEN_RE = re.compile(r"\\boxed\s*\{")
_INT_RE = re.compile(r"-?\d+")
# A whole integer, signed when the minus is unary rather than a subtraction:
# not the tail of a decimal, not followed by a fraction part.
_ANSWER_INT_RE = re.compile(r"(?:(?<![\w}).\]])-)?(?<![\d.])\d+(?!\d|\.\d)")
# A box that ends on a fraction states a non-integer answer.
_TRAILING_FRAC_RE = re.compile(r"\\[dt]?frac\s*\{[^{}]*\}\s*\{[^{}]*\}\W*$")
# "\pmod{1000}", "\bmod 1000", "\mod{1000}", "\text{ mod } 1000".
_MODULUS_RE = re.compile(r"(?:\\[pb]?mod|\bmod)\b\W*\d+")
# Environment markers, so "\frac{662}{3}. \end{gathered}" still ends on the
# fraction.
_TEX_ENV_RE = re.compile(r"\\(?:begin|end)\s*\{[^{}]*\}")
# TeX lengths such as the "2pt" in "\\[2pt]" or "\hspace{1em}".
_TEX_LENGTH_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:pt|em|ex|mm|cm|mu)\b")


def boxed_contents(text: str) -> list[str]:
    """Returns the body of every ``\\boxed{...}`` in ``text``, in order.

    Counts braces rather than matching a regex, because a box whose body opens
    an environment (``\\boxed{\\begin{aligned} ... \\end{aligned}}``) has a
    closing brace long before its own. Escaped ``\\{`` and ``\\}`` do not
    count. An unterminated box yields everything after its opener, so a
    response truncated mid-box still offers its digits.

    Args:
        text: The response text to scan.

    Returns:
        The box bodies, outermost boxes only.
    """
    bodies: list[str] = []
    pos = 0
    while match := _BOXED_OPEN_RE.search(text, pos):
        start = end = match.end()
        depth = 1
        while end < len(text) and depth:
            char = text[end]
            if char == "\\":
                end += 2
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            end += 1
        bodies.append(text[start : end - 1] if depth == 0 else text[start:])
        pos = end
    return bodies


def extract_answer(text: str) -> str:
    """Returns the last integer in ``text``, the answer a derivation ends on.

    Strips modulus clauses and TeX lengths first, since an answer reduced
    "mod 1000" otherwise ends on the modulus. An answer outside AIME's 0-999
    range is returned as is: it is the model's (wrong) answer, not a parse
    failure, and preferring an in-range integer would read an intermediate
    term of ``1000 + 106 = 1106`` as the answer. A negative or fractional
    final answer is kept negative or yields nothing, so it cannot match an
    expected answer by its digits alone.

    Args:
        text: A box body or a whole response.

    Returns:
        The answer's digits, or ``""`` when ``text`` holds no integer.
    """
    text = _TEX_ENV_RE.sub(" ", _MODULUS_RE.sub(" ", text))
    text = _TEX_LENGTH_RE.sub(" ", text)
    if _TRAILING_FRAC_RE.search(text):
        return ""
    candidates = _ANSWER_INT_RE.findall(text)
    return candidates[-1] if candidates else ""


def grade(content: str | None, expected: object) -> tuple[bool, str]:
    """Grades one AIME response against its expected integer answer.

    Reads the answer from the last ``\\boxed{...}``, and from the whole text
    only when there is no box. A box with no integer in it, such as a boxed
    "no such value exists", extracts nothing rather than borrowing a number
    from the reasoning. Comparison is integer-valued so ``"042"`` matches
    ``42``.

    Args:
        content: Raw assistant message content (may include a ``<think>`` span
            or be ``None``).
        expected: The dataset's expected answer (stringified before parsing).

    Returns:
        A ``(correct, predicted)`` tuple, where ``predicted`` is the extracted
        answer string (empty when nothing parseable was found).
    """
    raw = strip_think(content)
    boxed = boxed_contents(raw)
    predicted = extract_answer(boxed[-1] if boxed else raw)
    exp_ints = _INT_RE.findall(str(expected))
    correct = (
        bool(predicted)
        and bool(exp_ints)
        and int(predicted) == int(exp_ints[-1])
    )
    return correct, predicted


def build_messages(
    problem: str, root_preamble: str, system_prompt: str
) -> list[dict[str, str]]:
    """Builds the chat messages for one problem.

    The ``root`` preamble turn is omitted when ``root_preamble`` is empty, so a
    non-MiniMax target can run without it.
    """
    messages: list[dict[str, str]] = []
    if root_preamble:
        messages.append({"role": "root", "content": root_preamble})
    messages.append({"role": "system", "content": system_prompt})
    messages.append(
        {
            "role": "user",
            "content": "Return your final response within \\boxed{}. "
            + problem,
        }
    )
    return messages


DEFAULT_DATASET = "math-ai/aime25"


def dataset_defaults(dataset_id: str) -> tuple[str, str]:
    """Derives the metric prefix and out-dir defaults from a dataset id.

    Keying both off the dataset keeps ``--dataset`` self-sufficient: a run of
    ``math-ai/aime26`` with no other flags exports ``AIME26_*`` metrics into
    ``/tmp/aime26-results`` instead of clobbering AIME25's.

    Args:
        dataset_id: HuggingFace dataset id, e.g. ``"math-ai/aime26"``.

    Returns:
        A ``(metric_prefix, out_dir)`` tuple, e.g.
        ``("AIME26", "/tmp/aime26-results")``.
    """
    name = dataset_id.rsplit("/", 1)[-1]
    assert name, f"cannot derive a dataset name from {dataset_id!r}"
    # The prefix keys GITHUB_ENV variables, so it must be identifier-safe.
    metric_prefix = re.sub(r"\W", "_", name).upper()
    return metric_prefix, f"/tmp/{name}-results"


def load_aime_dataset(
    dataset_id: str = DEFAULT_DATASET,
) -> list[dict[str, Any]]:
    """Loads an AIME problem set, falling back to the ``train`` split.

    The ``math-ai/aime25`` and ``math-ai/aime26`` sets share the same schema
    (``problem`` + ``answer``) and the same grader, so one loader covers both
    years. ``math-ai/aime25`` publishes its 30 problems under ``test``; a
    sibling that happens to put them under ``train`` is handled by the
    fallback instead of forcing every caller to know which split it ships.
    """
    try:
        return list(load_dataset(dataset_id, split="test"))
    except Exception:
        return list(load_dataset(dataset_id, split="train"))


def infer(
    client: ChatClient,
    model: str,
    item: tuple[int, int, dict[str, Any]],
    params: GenParams,
    root_preamble: str,
    system_prompt: str,
) -> dict[str, Any]:
    """Runs and grades a single (repeat, problem) sample."""
    repeat_index, prompt_index, sample = item
    problem = sample["problem"]
    expected = str(sample["answer"]).strip()
    messages = build_messages(problem, root_preamble, system_prompt)
    resp = chat_completion(client, model, messages, params)
    choice = resp.choices[0]
    correct, predicted = grade(choice.message.content, expected)
    return {
        "prompt_index": prompt_index,
        "repeat_index": repeat_index,
        "problem": problem[:120],
        "expected": expected,
        "predicted": predicted,
        "correct": correct,
        "finish_reason": choice.finish_reason,
        "response": strip_think(choice.message.content),
        "completion_tokens": (
            resp.usage.completion_tokens if resp.usage else 0
        ),
        **response_fields(choice),
    }


def run_eval(
    client: ChatClient,
    indexed_dataset: list[tuple[int, dict[str, Any]]],
    model: str,
    repeats: int,
    workers: int,
    params: GenParams,
    root_preamble: str,
    system_prompt: str,
    out_dir: str | None = None,
    resume: bool = False,
    label: str = "AIME",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Runs AIME over ``indexed_dataset`` with ``repeats``.

    Each ``(repeat, problem)`` pair is submitted independently so a row can be
    tied back to its problem and repeat regardless of completion order. A
    failed/timed-out request is recorded as an incorrect row, never dropped.
    When ``out_dir`` is set, results stream to ``results.jsonl`` as they
    complete, so a crash mid-run doesn't lose already-finished samples.

    Args:
        client: OpenAI-compatible chat client (real or fake).
        indexed_dataset: ``(original_index, problem)`` pairs, already narrowed
            by :func:`eval_common.select_rows`.
        model: Served model name to request.
        repeats: Repeats per problem (``temperature`` makes each independent).
        workers: Max concurrent in-flight requests.
        params: Generation/sampling parameters.
        root_preamble: MiniMax root turn (empty to omit).
        system_prompt: System turn content.
        out_dir: When set, stream results to ``<out_dir>/results.jsonl``.
        resume: Keep the finished rows already in ``out_dir`` and run only
            the rest (see :func:`eval_common.run_parallel`).
        label: Human-readable eval name used in the progress bar and the
            opening log line, e.g. ``"AIME25"`` or ``"AIME26"``.

    Returns:
        A ``(results, score)`` tuple.
    """
    samples = expand_repeats(indexed_dataset, repeats)
    print(
        f"{label}: evaluating {len(indexed_dataset)} problems x {repeats} "
        f"repeats = {len(samples)} total samples"
    )

    def fn(item: tuple[int, int, dict[str, Any]]) -> dict[str, Any]:
        return infer(client, model, item, params, root_preamble, system_prompt)

    def on_error(
        item: tuple[int, int, dict[str, Any]], exc: Exception
    ) -> dict[str, Any]:
        rep, pi, p = item
        return {
            "prompt_index": pi,
            "repeat_index": rep,
            "problem": p["problem"][:120],
            "expected": str(p["answer"]).strip(),
            "error": str(exc),
            "correct": False,
        }

    results, errors = run_parallel(
        samples, fn, on_error, workers, label, out_dir=out_dir, resume=resume
    )
    score = exact_match_score(results, len(samples), errors)
    score["unextracted"] = sum(
        1 for r in results if "error" not in r and not r.get("predicted")
    )
    return results, score


@click.command()
@click.option(
    "--base-url",
    required=True,
    help="Server base URL, e.g. http://localhost:8000",
)
@click.option("--model", required=True, help="Served model name to request.")
@click.option(
    "--sample-size",
    type=int,
    default=None,
    help="Max problems (evenly sampled). Empty = full dataset. Applied "
    "before repeats. Mutually exclusive with --row-ids.",
)
@click.option(
    "--row-ids",
    default=None,
    help="Explicit comma-separated dataset row indices to evaluate, e.g. "
    "'3,17,42' (indices into the full dataset, order/duplicates preserved). "
    "Mutually exclusive with --sample-size — use this to re-run exactly the "
    "problems that errored or looked wrong last time.",
)
@click.option("--repeats", type=int, default=16, help="Repeats per problem.")
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Per-request seed for reproducibility (omitted when unset).",
)
@click.option(
    "--workers", type=int, default=16, help="Max concurrent requests."
)
@click.option(
    "--out-dir",
    default=None,
    help="Output directory. Defaults to /tmp/<dataset name>-results, e.g. "
    "/tmp/aime26-results for --dataset math-ai/aime26.",
)
@click.option("--max-tokens", type=int, default=98304, show_default=True)
@click.option("--temperature", type=float, default=1.0, show_default=True)
@click.option("--top-p", type=float, default=0.95, show_default=True)
@click.option(
    "--root-preamble",
    default=DEFAULT_ROOT_PREAMBLE,
    help="Root identity turn (empty to omit; default is MiniMax-M3's).",
)
@click.option(
    "--system-prompt", default=DEFAULT_SYSTEM_PROMPT, show_default=True
)
@click.option(
    "--metric-prefix",
    default=None,
    help=(
        "Prefix for the GITHUB_ENV metric keys the job summary reads. Also "
        "used as the human-readable eval label in logs. Defaults to the "
        "upper-cased dataset name, e.g. AIME26 for --dataset math-ai/aime26."
    ),
)
@click.option(
    "--dataset",
    "dataset_id",
    default=DEFAULT_DATASET,
    show_default=True,
    help=(
        "HuggingFace dataset id of the AIME problem set. Both "
        "``math-ai/aime25`` and ``math-ai/aime26`` share the same schema "
        "and grader, so one binary serves both years."
    ),
)
@click.option(
    "--resume",
    is_flag=True,
    help="Keep the finished rows already in --out-dir and run only the rest. "
    "Refuses when they were produced with different options.",
)
def main(
    base_url: str,
    model: str,
    sample_size: int | None,
    row_ids: str | None,
    repeats: int,
    seed: int | None,
    workers: int,
    out_dir: str | None,
    max_tokens: int,
    temperature: float,
    top_p: float,
    root_preamble: str,
    system_prompt: str,
    metric_prefix: str | None,
    dataset_id: str,
    resume: bool,
) -> None:
    """Runs AIME against a running OpenAI-compatible server and scores it."""
    default_prefix, default_out_dir = dataset_defaults(dataset_id)
    metric_prefix = metric_prefix or default_prefix
    out_dir = out_dir or default_out_dir
    prepare_out_dir(
        out_dir,
        resumable_config(click.get_current_context().params),
        resume,
    )
    client = make_client(base_url)
    dataset = load_aime_dataset(dataset_id)
    indexed_dataset = select_rows(dataset, sample_size, row_ids)
    params = GenParams(
        max_tokens=max_tokens, temperature=temperature, top_p=top_p, seed=seed
    )
    dump_config(
        out_dir,
        base_url,
        params,
        indexed_dataset,
        dataset,
        system_prompt=system_prompt,
        root_preamble=root_preamble,
    )
    results, summary = run_eval(
        client,
        indexed_dataset,
        model,
        repeats,
        workers,
        params,
        root_preamble,
        system_prompt,
        out_dir=out_dir,
        resume=resume,
        label=metric_prefix,
    )
    write_outputs(out_dir, results, summary, metric_prefix)


if __name__ == "__main__":
    main()
