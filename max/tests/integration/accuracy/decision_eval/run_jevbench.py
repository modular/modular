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

"""Scores a ``/v1/systemone`` server on JevBench.

Sends each case as one System One request (the way the benchmark's own harness
does), turns the answer into per-option probabilities and writes
``predictions.jsonl`` and ``scores.json``. The server can be MAX, the reference
implementation's ``decider.serve`` or any compatible endpoint.

Get the cases with
``huggingface-cli download Leanmcp/jevbench --repo-type dataset --local-dir DIR``
and pass ``DIR/cases``.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import aiohttp
import click
from cases import (
    load_cases,
    make_prediction,
    probabilities_from_answer,
    systemone_request,
)
from metrics import format_table, summarize_by_slice
from predictions import Prediction, write_predictions

_RETRIES = 2


async def _ask(
    session: aiohttp.ClientSession,
    url: str,
    case: dict[str, Any],
    model: str,
) -> tuple[Prediction | None, int, str | None]:
    """Asks one case. Returns the prediction, input tokens and any error."""
    body = systemone_request(case, model)
    error = "no attempt"
    for _ in range(_RETRIES + 1):
        try:
            async with session.post(url, json=body) as response:
                payload = await response.json()
                if response.status != 200:
                    error = f"HTTP {response.status}: {payload}"
                    continue
            answer = payload["answers"]["decision"]
            probabilities = probabilities_from_answer(case, answer)
            tokens = int(payload.get("usage", {}).get("input_tokens", 0))
            return make_prediction(case, probabilities), tokens, None
        except (aiohttp.ClientError, KeyError, ValueError, AssertionError) as e:
            error = f"{type(e).__name__}: {e}"
    return None, 0, f"{case['case_id']}: {error}"


async def run_cases(
    base_url: str, model: str, cases: list[dict[str, Any]], concurrency: int
) -> tuple[list[Prediction], int, list[str]]:
    """Asks every case with ``concurrency`` requests in flight."""
    url = base_url.rstrip("/") + "/v1/systemone"
    gate = asyncio.Semaphore(concurrency)
    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(timeout=timeout) as session:

        async def bounded(case: dict[str, Any]) -> Any:
            async with gate:
                return await _ask(session, url, case, model)

        results = await asyncio.gather(*(bounded(case) for case in cases))
    predictions = [p for p, _, _ in results if p is not None]
    errors = [e for _, _, e in results if e is not None]
    return predictions, sum(tokens for _, tokens, _ in results), errors


@click.command()
@click.option(
    "--cases-dir", type=click.Path(path_type=Path, exists=True), required=True
)
@click.option("--base-url", default="http://localhost:8000", show_default=True)
@click.option("--model", required=True, help="Model name sent in each request.")
@click.option("--output-dir", type=click.Path(path_type=Path), required=True)
@click.option("--slices", default=None, help="Comma-separated slice names.")
@click.option("--limit-per-slice", type=click.IntRange(min=1), default=None)
@click.option(
    "--max-options", type=click.IntRange(min=1), default=10, show_default=True
)
@click.option(
    "--concurrency", type=click.IntRange(min=1), default=16, show_default=True
)
@click.option(
    "--allow-partial",
    is_flag=True,
    help="Exit 0 even if some cases failed. The scores then cover only the "
    "answered cases and are not comparable to a complete run.",
)
def main(
    cases_dir: Path,
    base_url: str,
    model: str,
    output_dir: Path,
    slices: str | None,
    limit_per_slice: int | None,
    max_options: int,
    concurrency: int,
    allow_partial: bool,
) -> None:
    cases = load_cases(
        cases_dir,
        slices=slices.split(",") if slices else None,
        max_options=max_options,
        limit_per_slice=limit_per_slice,
    )
    click.echo(f"{len(cases)} cases")
    started = time.monotonic()
    predictions, input_tokens, errors = asyncio.run(
        run_cases(base_url, model, cases, concurrency)
    )
    elapsed = time.monotonic() - started
    assert predictions, f"every request failed; first error: {errors[:1]}"

    scores = summarize_by_slice(predictions)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_predictions(output_dir / "predictions.jsonl", predictions)
    (output_dir / "scores.json").write_text(
        json.dumps(
            {
                "model": model,
                "cases": len(cases),
                "complete": not errors,
                "answered": len(predictions),
                "errors": errors[:20],
                "input_tokens": input_tokens,
                "seconds": elapsed,
                "scores": scores,
            },
            indent=2,
        )
    )
    click.echo(format_table(scores))
    click.echo(
        f"answered {len(predictions)}/{len(cases)} in {elapsed:.0f}s "
        f"({input_tokens} input tokens); {len(errors)} errors"
    )
    if errors:
        click.echo("first errors:\n" + "\n".join(errors[:5]))
        if not allow_partial:
            raise click.ClickException(
                f"{len(errors)} of {len(cases)} cases failed, so the scores "
                "above are not comparable; fix the server or pass "
                "--allow-partial for a diagnostic run"
            )


if __name__ == "__main__":
    main()
