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
"""SciCode scientific-code-generation eval against an OpenAI-compatible server.

De-embedded from ``minimaxM3ScicodeEval.yaml`` so the eval logic lives in a
locally-runnable module instead of an inline YAML heredoc. Each problem is
solved step-by-step the way the official harness does it: the model sees the
earlier steps and the code it wrote for them, implements only the next step's
function, and the test script is assembled from the dependencies plus every
step's code so far. Each sub-step is executed against the official SciCode test
cases and scored pass@1 over the benchmark's 338 sub-steps (three steps have
no tests; the official harness substitutes reference code for them, see
:mod:`scicode_reference_steps`).

The generation half reuses :mod:`eval_common`. The test-execution half must run
under a Python that has the third-party ``scicode`` package and ``h5py``
installed (they are not bazel deps), so tests run as a subprocess against a
separate interpreter passed via ``--test-python`` (the CI eval venv). That
subprocess's environment strips ``PYTHONPATH``/``PYTHONHOME`` so bazel runfiles
on the parent's path cannot shadow the venv's ``scicode``/``h5py``.

Run locally against a server on ``localhost:8000`` (``--test-python`` must point
at a venv with ``scicode`` + ``h5py``)::

    ./bazelw run //max/tests/integration/accuracy/model_evals:scicode_eval -- \\
        --base-url http://localhost:8000 --model MiniMaxAI/MiniMax-M3-MXFP8 \\
        --test-python /tmp/eval-venv/bin/python \\
        --test-data /tmp/scicode-test-data.h5 --sample-size 2

Results stream to ``<out-dir>/results.jsonl`` as each problem completes, so a
killed/crashed run keeps everything already finished. To debug specific
problems (e.g. re-run exactly the ones that errored or looked wrong last
time), pass ``--row-ids`` instead of ``--sample-size``::

    ./bazelw run //max/tests/integration/accuracy/model_evals:scicode_eval -- \\
        --base-url http://localhost:8000 --model MiniMaxAI/MiniMax-M3-MXFP8 \\
        --test-python /tmp/eval-venv/bin/python \\
        --test-data /tmp/scicode-test-data.h5 --row-ids 3,17,42
"""

from __future__ import annotations

import json
import os
import re
import statistics
import subprocess
import tempfile
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

import click
from datasets import load_dataset
from eval_common import (
    ChatClient,
    GenParams,
    chat_completion,
    dump_config,
    make_client,
    response_fields,
    run_parallel,
    select_rows,
)
from scicode_reference_steps import REFERENCE_STEP_CODE

# The official harness (eval/scripts/test_generated_code.py) gives each step's
# test script 1800 s.
OFFICIAL_TEST_TIMEOUT_S = 1800.0

# The "with background" variant of the official prompt
# (eval/data/multistep_template.txt): earlier steps come with their gold
# background and the code the model wrote for them; the model implements only
# the next step and must not repeat earlier functions, which the test script
# supplies.
_PROMPT_TEMPLATE = """\
PROBLEM DESCRIPTION:
You will be provided with problem steps along with background knowledge necessary for solving the problem. Your task will be to develop a Python solution focused on the next step of the problem-solving process.

PROBLEM STEPS AND FUNCTION CODE:
Here, you'll find the Python code for the initial steps of the problem-solving process. This code is integral to building the solution.

{problem_steps_str}

NEXT STEP - PROBLEM STEP AND FUNCTION HEADER:
This part will describe the next step in the problem-solving process. A function header will be provided, and your task is to develop the Python code for this next step based on the provided description and function header.

{next_step_str}

DEPENDENCIES:
Use only the following dependencies in your solution. Do not include these dependencies at the beginning of your code.

{dependencies}

RESPONSE GUIDELINES:
Now, based on the instructions and information provided above, write the complete and executable Python program for the next step in a single block.
Your response should focus exclusively on implementing the solution for the next step, adhering closely to the specified function header and the context provided by the initial steps.
Your response should NOT include the dependencies and functions of all previous steps. If your next step function calls functions from previous steps, please make sure it uses the headers provided without modification.
DO NOT generate EXAMPLE USAGE OR TEST CODE in your response. Please make sure your response python code in format of ```python```."""


# The official extract_python_script (scicode/gen/models.py) deletes the model's
# imports, so the dependencies block is the only source of imports in both the
# prompts and the test scripts. Indented imports go too, as they do there.
_IMPORT_LINE = re.compile(
    r"^\s*(import .*|from .*\s+import\s+.*)", re.MULTILINE
)


def extract_code(text: str | None) -> str:
    """Extracts the first fenced code block, falling back to the raw text.

    Import lines are removed, as the official harness does.
    """
    text = text or ""
    if "```python" in text:
        code = text.split("```python", 1)[1].split("```", 1)[0]
    elif "```" in text:
        code = text.split("```", 1)[1].split("```", 1)[0]
    else:
        code = text
    return _IMPORT_LINE.sub("", code).strip()


def _describe_step(step: dict[str, Any]) -> str:
    description = step.get("step_description_prompt", "")
    background = step.get("step_background", "")
    return f"{description}\n{background}" if background else description


def build_step_prompt(
    dependencies: str,
    previous_steps: Sequence[tuple[dict[str, Any], str]],
    step: dict[str, Any],
) -> str:
    """Builds the prompt for one sub-step from the official template.

    Args:
        dependencies: The problem's ``required_dependencies`` import block.
        previous_steps: ``(sub_step, code)`` pairs for the steps already
            solved, in order; the code is what the model wrote for that step
            (or the reference code for a step the harness never generates).
        step: The sub-step to implement next.
    """
    problem_steps = "\n\n------\n\n".join(
        f"{_describe_step(prev)}\n\n{code}" for prev, code in previous_steps
    )
    next_step = (
        f"{_describe_step(step)}\n\n"
        f"{step.get('function_header', '')}\n\n{step.get('return_line', '')}"
    )
    return _PROMPT_TEMPLATE.format(
        problem_steps_str=problem_steps,
        next_step_str=next_step,
        dependencies=dependencies,
    )


def _subprocess_env() -> dict[str, str]:
    """Env for the test subprocess, minus ``PYTHONPATH``/``PYTHONHOME``.

    Under ``bazel run`` the parent process leaks a bazel-runfiles ``PYTHONPATH``;
    inheriting it into the ``--test-python`` interpreter would shadow that
    venv's ``scicode``/``h5py`` with the bazel environment's modules.
    """
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    return env


def run_step_tests(
    code: str,
    dependencies: str,
    step_number: object,
    test_cases: list[str],
    test_python: str,
    test_data: str,
    timeout: float = OFFICIAL_TEST_TIMEOUT_S,
) -> bool:
    """Runs one step's official test cases, returning whether they all pass.

    Builds one script like the official SciCode harness: dependencies + the
    solution so far (every step's code, not just this step's), then binds
    ``target`` from ``test_data.h5`` for each test case (the test strings
    assert against a bare ``target``). Runs it under ``test_python`` (which
    must have ``scicode`` + ``h5py``) with ``timeout`` seconds to finish.
    """
    parts = [
        dependencies,
        "",
        code,
        "",
        "from scicode.parse.parse import process_hdf5_to_tuple",
        f"targets = process_hdf5_to_tuple({step_number!r}, "
        f"{len(test_cases)}, {test_data!r})",
        "",
    ]
    for idx, tc in enumerate(test_cases):
        parts.append(f"target = targets[{idx}]")
        parts.append(tc)
        parts.append("")
    script = "\n".join(parts)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".py", mode="w", delete=False
        ) as f:
            f.write(script)
            tmp = f.name
        result = subprocess.run(
            [test_python, tmp],
            capture_output=True,
            timeout=timeout,
            text=True,
            env=_subprocess_env(),
        )
        return result.returncode == 0
    except Exception:
        return False
    finally:
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass


def step_index(step_number: str) -> int:
    """Encodes a step number such as ``"77.3"`` as ``7703``.

    Used as the step's ``prompt_index``: stable across subsampling and
    dataset reordering, and ``prompt_index // 100`` recovers the problem.
    """
    problem, _, step = step_number.partition(".")
    return int(problem) * 100 + int(step)


def infer(
    client: ChatClient,
    model: str,
    sample: dict[str, Any],
    params: GenParams,
    test_python: str,
    test_data: str,
    test_timeout: float = OFFICIAL_TEST_TIMEOUT_S,
) -> dict[str, Any]:
    """Solves one problem step-by-step and runs each step's tests.

    Returns ``{"step_results", "call_output_tokens", "problem_output_tokens"}``:
    one ``step_results`` entry per generated sub-step (with its pass/fail),
    one ``call_output_tokens`` entry per generation call, and the per-problem
    token sum. A step in :data:`REFERENCE_STEP_CODE` is neither generated nor
    scored; its reference code joins the solution for the steps after it.
    """
    problem = sample.get("problem_description_main", "")
    dependencies = sample.get("required_dependencies", "")
    sub_steps = sample.get("sub_steps", [])
    step_results: list[dict[str, Any]] = []
    solved: list[tuple[dict[str, Any], str]] = []
    call_output_tokens: list[int] = []
    problem_output_tokens = 0

    for step_idx, step in enumerate(sub_steps):
        step_number = str(step.get("step_number", ""))
        reference = REFERENCE_STEP_CODE.get(step_number)
        if reference is not None:
            solved.append((step, reference))
            continue
        test_cases = step.get("test_cases", [])
        prompt = build_step_prompt(dependencies, solved, step)
        sent_at = datetime.now(timezone.utc).isoformat()
        resp = chat_completion(
            client, model, [{"role": "user", "content": prompt}], params
        )
        call_tokens = resp.usage.completion_tokens if resp.usage else 0
        call_output_tokens.append(call_tokens)
        problem_output_tokens += call_tokens
        message = resp.choices[0].message
        solved.append((step, extract_code(message.content)))
        solution = "\n".join(code for _, code in solved)

        passed = (
            run_step_tests(
                solution,
                dependencies,
                step_number,
                test_cases,
                test_python,
                test_data,
                test_timeout,
            )
            if test_cases
            else False
        )
        step_results.append(
            {
                "prompt_index": step_index(step_number),
                "timestamp": sent_at,
                "problem": problem[:80],
                "step_number": step_number,
                "step_idx": step_idx,
                "passed": passed,
                "finish_reason": resp.choices[0].finish_reason,
                "completion_tokens": call_tokens,
                **response_fields(resp.choices[0]),
            }
        )

    return {
        "step_results": step_results,
        "call_output_tokens": call_output_tokens,
        "problem_output_tokens": problem_output_tokens,
    }


def load_scicode_dataset() -> list[dict[str, Any]]:
    """Loads the full 80-problem SciCode benchmark (65 test + 15 dev).

    Each JSONL is loaded separately: the two files have mismatched columns (dev
    carries ``general_solution``), so a combined load fails schema-casting under
    the bazel-pinned ``datasets`` 2.x. Loading one file at a time as ``train``
    sidesteps the cross-file cast.
    """
    ds: list[dict[str, Any]] = []
    for data_file in ("problems_test.jsonl", "problems_dev.jsonl"):
        ds.extend(
            load_dataset(
                "SciCode1/SciCode", data_files=data_file, split="train"
            )
        )
    return ds


def run_eval(
    client: ChatClient,
    indexed_dataset: list[tuple[int, dict[str, Any]]],
    model: str,
    workers: int,
    params: GenParams,
    test_python: str,
    test_data: str,
    out_dir: str | None = None,
    test_timeout: float = OFFICIAL_TEST_TIMEOUT_S,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Runs SciCode over ``indexed_dataset`` and returns ``(step_results, summary)``.

    A problem whose inference raises keeps one errored row per sub-step in
    ``step_results`` but is left out of the summary (its sub-steps are not
    scored), matching the historical inline behavior. When ``out_dir`` is set,
    results stream to ``<out_dir>/results.jsonl`` as they complete, so a crash
    mid-run doesn't lose already-finished problems.

    Args:
        client: OpenAI-compatible chat client.
        indexed_dataset: ``(original_index, problem)`` pairs, already narrowed
            by :func:`eval_common.select_rows`.
        model: Served model name to request.
        workers: Max concurrent in-flight requests.
        params: Generation/sampling parameters.
        test_python: Python interpreter with ``scicode`` + ``h5py`` installed.
        test_data: Path to the SciCode ``test_data.h5`` file.
        out_dir: When set, stream per-problem results to
            ``<out_dir>/results.jsonl`` as each future resolves.
        test_timeout: Seconds one step's test script may run.

    Returns:
        A ``(step_results, summary)`` tuple.
    """

    def scored_steps(
        sample: dict[str, Any],
    ) -> list[tuple[int, dict[str, Any]]]:
        return [
            (step_idx, step)
            for step_idx, step in enumerate(sample.get("sub_steps", []))
            if str(step.get("step_number", "")) not in REFERENCE_STEP_CODE
        ]

    total_substeps = sum(len(scored_steps(s)) for _, s in indexed_dataset)
    print(
        f"SciCode: evaluating {len(indexed_dataset)} problems "
        f"({total_substeps} sub-steps)"
    )

    def fn(item: tuple[int, dict[str, Any]]) -> dict[str, Any]:
        _, sample = item
        return infer(
            client, model, sample, params, test_python, test_data, test_timeout
        )

    def on_error(
        item: tuple[int, dict[str, Any]], exc: Exception
    ) -> dict[str, Any]:
        _, sample = item
        return {
            "error": str(exc),
            "step_results": [
                {
                    "prompt_index": step_index(step["step_number"]),
                    "step_number": step["step_number"],
                    "step_idx": step_idx,
                    "passed": False,
                    "error": str(exc),
                }
                for step_idx, step in scored_steps(sample)
            ],
        }

    results, _ = run_parallel(
        indexed_dataset, fn, on_error, workers, "SciCode", out_dir=out_dir
    )

    all_results: list[dict[str, Any]] = []
    scored: list[dict[str, Any]] = []
    call_output_tokens: list[int] = []
    problem_output_tokens: list[int] = []
    for r in results:
        # An errored problem's steps have no time of their own; they take
        # the time the problem started, which run_parallel stamped.
        all_results.extend(
            {"timestamp": r["timestamp"], **step} for step in r["step_results"]
        )
        if "error" in r:
            continue
        scored.extend(r["step_results"])
        call_output_tokens.extend(r["call_output_tokens"])
        problem_output_tokens.append(r["problem_output_tokens"])

    passed = sum(1 for r in scored if r["passed"])
    pass_at1 = passed / len(scored) if scored else 0.0
    # Primary metric: per-call completion tokens (one entry per sub-step
    # generation request), matching the per-call reference baseline.
    mean_output_tokens = (
        round(statistics.mean(call_output_tokens), 1)
        if call_output_tokens
        else 0.0
    )
    p50_output_tokens = (
        round(statistics.median(call_output_tokens), 1)
        if call_output_tokens
        else 0.0
    )
    # Secondary metric: per-problem summed completion tokens (reference only).
    mean_problem_tokens = (
        round(statistics.mean(problem_output_tokens), 1)
        if problem_output_tokens
        else 0.0
    )
    p50_problem_tokens = (
        round(statistics.median(problem_output_tokens), 1)
        if problem_output_tokens
        else 0.0
    )
    summary = {
        "pass_at_1": pass_at1,
        "passed": passed,
        "total": len(scored),
        "mean_output_tokens": mean_output_tokens,
        "p50_output_tokens": p50_output_tokens,
        "mean_problem_output_tokens": mean_problem_tokens,
        "p50_problem_output_tokens": p50_problem_tokens,
    }
    return all_results, summary


@click.command()
@click.option(
    "--base-url",
    required=True,
    help="Server base URL, e.g. http://localhost:8000",
)
@click.option("--model", required=True, help="Served model name to request.")
@click.option(
    "--test-python",
    default="/tmp/eval-venv/bin/python",
    show_default=True,
    help="Python interpreter (with scicode + h5py) used to run step tests.",
)
@click.option(
    "--test-data",
    default=lambda: os.environ.get("SCICODE_DATA", ""),
    help="Path to the SciCode test_data.h5 (defaults to $SCICODE_DATA).",
)
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
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Per-request seed for reproducibility (omitted when unset).",
)
@click.option("--workers", type=int, default=4, help="Max concurrent requests.")
@click.option(
    "--out-dir", default="/tmp/scicode-results", help="Output directory."
)
@click.option("--max-tokens", type=int, default=66566, show_default=True)
@click.option("--temperature", type=float, default=1.0, show_default=True)
@click.option("--top-p", type=float, default=0.95, show_default=True)
@click.option(
    "--test-timeout",
    type=float,
    default=OFFICIAL_TEST_TIMEOUT_S,
    show_default=True,
    help="Seconds one step's test script may run (the official harness's "
    "limit by default).",
)
@click.option(
    "--metric-prefix",
    default="SCICODE",
    help="Prefix for the GITHUB_ENV metric keys the job summary reads.",
)
def main(
    base_url: str,
    model: str,
    test_python: str,
    test_data: str,
    sample_size: int | None,
    row_ids: str | None,
    seed: int | None,
    workers: int,
    out_dir: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    test_timeout: float,
    metric_prefix: str,
) -> None:
    """Runs SciCode against a running OpenAI-compatible server and scores it."""
    if not test_data:
        raise click.UsageError(
            "--test-data (or $SCICODE_DATA) must point at test_data.h5"
        )
    client = make_client(base_url)
    dataset = load_scicode_dataset()
    indexed_dataset = select_rows(dataset, sample_size, row_ids)
    params = GenParams(
        max_tokens=max_tokens, temperature=temperature, top_p=top_p, seed=seed
    )
    dump_config(out_dir, base_url, params, indexed_dataset, dataset)
    all_results, summary = run_eval(
        client,
        indexed_dataset,
        model,
        workers,
        params,
        test_python,
        test_data,
        out_dir=out_dir,
        test_timeout=test_timeout,
    )

    with open(os.path.join(out_dir, "results.jsonl"), "w") as f:
        f.write("\n".join(json.dumps(r) for r in all_results))
    with open(os.path.join(out_dir, "score.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(
        f"SciCode pass@1: {summary['pass_at_1']:.4f} "
        f"({summary['passed']}/{summary['total']}) | "
        f"per-call mean output tokens: {summary['mean_output_tokens']:.1f} | "
        f"per-call p50 output tokens: {summary['p50_output_tokens']:.1f} | "
        f"per-problem mean: {summary['mean_problem_output_tokens']:.1f} | "
        f"per-problem p50: {summary['p50_problem_output_tokens']:.1f}"
    )

    env_file = os.environ.get("GITHUB_ENV")
    if env_file:
        with open(env_file, "a") as f:
            f.write(f"{metric_prefix}_SCORE={summary['pass_at_1']:.4f}\n")
            f.write(
                f"{metric_prefix}_MEAN_TOKENS={summary['mean_output_tokens']:.0f}\n"
            )
            f.write(
                f"{metric_prefix}_P50_TOKENS={summary['p50_output_tokens']:.0f}\n"
            )


if __name__ == "__main__":
    main()
