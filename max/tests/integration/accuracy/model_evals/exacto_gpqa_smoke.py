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
"""Runs the frozen GPQA smoke cohort with explicit output and wall budgets."""

import argparse
import hashlib
import json
import os
import signal
import threading
import time
from pathlib import Path

import exacto_gpqa_task
import exacto_streaming as exacto_streaming
from inspect_ai import eval
from inspect_ai.dataset import MemoryDataset

SMOKE_SEED = "distinf674-mimo-gpqa-smoke-v1"
SMOKE_SIZE = 32
OUTPUT_TOKENS = 32768
WALL_SECONDS = 900


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-connections", type=int, default=8)
    args = parser.parse_args()
    if args.max_connections < 1:
        parser.error("--max-connections must be positive")
    ids = sorted(
        sorted(
            range(1, 199),
            key=lambda question: hashlib.sha256(
                f"{SMOKE_SEED}:{question}".encode()
            ).digest(),
        )[:SMOKE_SIZE]
    )
    source = Path(exacto_streaming.__file__).parent
    hashes = {
        name: hashlib.sha256((source / name).read_bytes()).hexdigest()
        for name in (
            "exacto_gpqa_task.py",
            "exacto_gpqa_openrouter.py",
            "exacto_mcq.py",
            "exacto_streaming.py",
            Path(__file__).name,
        )
    }
    task = exacto_gpqa_task.gpqa_diamond()
    assert len(task.dataset) == 198 and task.config.max_tokens is None
    # Filter after the canonical loader shuffles each original row's options.
    selected = [sample for sample in task.dataset if sample.id in ids]
    assert [sample.id for sample in selected] == ids
    task.dataset = MemoryDataset(selected, name=f"mimo_gpqa_smoke_{SMOKE_SIZE}")
    task.metadata = {
        **(task.metadata or {}),
        "smoke_seed": SMOKE_SEED,
        "scheduled_question_ids": ids,
        "harness_sha256": hashes,
        "output_token_cap": OUTPUT_TOKENS,
        "suite_wall_budget_seconds": WALL_SECONDS,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["LOCAL_BASE_URL"] = (
        args.url.rstrip("/").removesuffix("/v1") + "/v1"
    )
    os.environ.setdefault("LOCAL_API_KEY", "dummy")
    model = "openai-stream/local/" + args.model
    model_args = {"timeout": None, "max_retries": 0}
    expired = threading.Event()

    def expire() -> None:
        expired.set()
        # Inspect handles SIGINT by cancelling active samples and flushing logs.
        os.kill(os.getpid(), signal.SIGINT)

    timer = threading.Timer(WALL_SECONDS, expire)
    timer.daemon = True
    result = None
    started = time.monotonic()
    report = {
        "scope": f"Bounded {SMOKE_SIZE}-question smoke; not a full GPQA score or a release gate.",
        "scheduled": SMOKE_SIZE,
        "question_ids": ids,
        "smoke_seed": SMOKE_SEED,
        "harness_sha256": hashes,
        "served_model": args.model,
        "output_token_cap": OUTPUT_TOKENS,
        "suite_wall_budget_seconds": WALL_SECONDS,
        "temperature": 0.5,
        "reasoning_effort": "high",
        "epochs": 1,
    }
    try:
        timer.start()
        logs = eval(
            task,
            model=model,
            model_args=model_args,
            epochs=1,
            max_connections=args.max_connections,
            temperature=0.5,
            reasoning_effort="high",
            max_tokens=OUTPUT_TOKENS,
            log_dir=str(args.output / "logs"),
            log_format="json",
            log_buffer=1,
            log_realtime=True,
            log_samples=True,
            display="plain",
            fail_on_error=False,
            continue_on_fail=True,
            retry_on_error=0,
        )
        assert len(logs) == 1
        result = logs[0]
    finally:
        timer.cancel()
        samples = list(result.samples or []) if result is not None else []
        keys = {(sample.id, sample.epoch) for sample in samples}
        assert len(keys) == len(samples) and keys <= {(q, 1) for q in ids}
        scored = [
            sample
            for sample in samples
            if not sample.error
            and sample.scores
            and "mcq_scorer" in sample.scores
        ]
        scored_ids = {sample.id for sample in scored}
        correct = sum(
            sample.scores["mcq_scorer"].value == "C" for sample in scored
        )
        complete = len(scored) == SMOKE_SIZE
        report.update(
            status="COMPLETE_SMOKE" if complete else "PARTIAL_SMOKE",
            native_status=result.status if result is not None else None,
            wall_budget_expired=expired.is_set(),
            elapsed_seconds=time.monotonic() - started,
            scored=len(scored),
            correct=correct,
            incorrect=len(scored) - correct,
            unanswered=sum(
                not sample.scores["mcq_scorer"].answer for sample in scored
            ),
            token_capped=sum(
                bool(
                    sample.output
                    and sample.output.choices
                    and sample.output.choices[0].stop_reason
                    in ("max_tokens", "model_length")
                )
                for sample in scored
            ),
            unresolved=SMOKE_SIZE - len(scored),
            observed_correct_fraction_of_scheduled=correct / SMOKE_SIZE,
            possible_correct_fraction_range=[
                correct / SMOKE_SIZE,
                (correct + SMOKE_SIZE - len(scored)) / SMOKE_SIZE,
            ],
            errors=sum(bool(sample.error) for sample in samples),
            error_retries=sum(
                len(sample.error_retries or []) for sample in samples
            ),
            samples=[
                {
                    "id": sample.id,
                    "epoch": sample.epoch,
                    "grade": sample.scores["mcq_scorer"].value
                    if sample.id in scored_ids
                    else None,
                    "extracted_answer": sample.scores["mcq_scorer"].answer
                    if sample.id in scored_ids
                    else None,
                    "stop_reason": sample.output.choices[0].stop_reason
                    if sample.output and sample.output.choices
                    else None,
                    "output_tokens": sample.output.usage.output_tokens
                    if sample.output and sample.output.usage
                    else None,
                    "error": str(sample.error) if sample.error else None,
                }
                for sample in samples
            ],
        )
        (args.output / "smoke-summary.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print(
            json.dumps(
                {
                    key: value
                    for key, value in report.items()
                    if key != "samples"
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
