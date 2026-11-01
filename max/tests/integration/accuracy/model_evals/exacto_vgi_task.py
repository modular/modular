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
"""Inspect task for VGI-Bench as OpenRouter's benchmark harness runs it.

Samples come from :mod:`exacto_vgi_openrouter`, answers are extracted by
:mod:`exacto_mcq` (a port of the harness's own extractor), and the video goes
out by URL through the marker parts of :mod:`exacto_media`, so it needs the
``openai-stream`` or ``openai-chat`` provider from :mod:`exacto_streaming`.

A failed request is scored the way the harness scores it: a client error such
as a 400 for a video the server could not fetch scores 0, while a 429 or 5xx
that outlives the retries is left out of the score. Run with
``--no-fail-on-error`` so those skipped samples do not stop the eval.

Runs only inside the harness venv (``run_exacto_vgi_local.sh``), which has
inspect_ai and datasets; bazel never imports it.
"""

from __future__ import annotations

import exacto_mcq as mcq
import exacto_media as media
import exacto_vgi_openrouter as vgi
from datasets import load_dataset
from inspect_ai import Epochs, Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import (
    ChatMessageUser,
    ContentText,
    GenerateConfig,
    ModelOutput,
    get_model,
)
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Scorer,
    Target,
    accuracy,
    scorer,
    std,
    stderr,
)
from inspect_ai.solver import Generate, Solver, TaskState, solver
from openai import APIStatusError

#: Where a request that scored 0 because it failed records why.
MODEL_ERROR_KEY = "model_error"


def load_samples() -> list[Sample]:
    """Loads the pinned VGI-Bench split as harness-identical samples."""
    manifest = vgi.load_media_manifest()
    records = load_dataset(
        vgi.DATASET,
        vgi.DATASET_CONFIG,
        revision=vgi.DATASET_REVISION,
        split=vgi.SPLIT,
    )
    samples = []
    for index, record in enumerate(records):
        built = vgi.record_to_sample(record, manifest)
        # Ids are 1-based row numbers, as in the GPQA task, because
        # exacto_report reads each row's question index from them. The
        # harness names a question by its question_id, which the metadata
        # keeps.
        samples.append(
            Sample(
                id=index + 1,
                input=[
                    ChatMessageUser(
                        content=[
                            ContentText(
                                text=media.video_url_marker(built.video_url)
                            ),
                            ContentText(text=built.prompt),
                        ]
                    )
                ],
                target=built.target,
                metadata=dict(built.metadata),
            )
        )
    if len(samples) != vgi.NUM_QUESTIONS:
        raise ValueError(
            f"{vgi.DATASET}@{vgi.DATASET_REVISION} has {len(samples)} rows, "
            f"expected {vgi.NUM_QUESTIONS}"
        )
    return samples


def _status(ex: Exception) -> int | None:
    if isinstance(ex, APIStatusError):
        return ex.status_code
    return vgi.status_from_error_text(str(ex))


@solver
def generate_scoring_client_errors() -> Solver:
    """Generates once, scoring a client error as a wrong answer."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        api = get_model().api
        if not getattr(api, "expands_video_url_markers", False):
            raise RuntimeError(
                f"{type(api).__name__} would send the video URL as text; "
                "use the openai-stream or openai-chat provider"
            )
        try:
            return await generate(state)
        except Exception as ex:
            status = _status(ex)
            if (
                vgi.model_error_outcome(status)
                is not vgi.ErrorOutcome.INCORRECT
            ):
                raise
            message = f"HTTP {status}: {str(ex).splitlines()[0][:500]}"
            state.output = ModelOutput(
                model=str(state.model), choices=[], error=message
            )
            state.metadata[MODEL_ERROR_KEY] = message
            return state

    return solve


@scorer(metrics=[accuracy(), stderr(), std()])
def harness_mcq() -> Scorer:
    """The harness's ``mcqScorer``, with failed requests scored 0."""

    async def score(state: TaskState, target: Target) -> Score:
        error = state.metadata.get(MODEL_ERROR_KEY)
        if error:
            return Score(
                value=INCORRECT,
                answer=None,
                explanation=f"Model error: {error}",
            )
        result = mcq.score_mcq(state.output.completion, target.text)
        return Score(
            value=CORRECT if result.correct else INCORRECT,
            answer=result.answer,
            explanation=result.explanation,
        )

    return score


@task
def vgi_bench() -> Task:
    """VGI-Bench with the harness's prompt, media, sampling and epochs.

    The reasoning effort is left to the caller's generate config (the runner's
    ``--reasoning-effort``), so a run can drop or change it without a new task.
    """
    manifest = vgi.load_media_manifest()
    return Task(
        name=vgi.BENCHMARK_ID,
        dataset=MemoryDataset(load_samples(), name=vgi.DATASET),
        solver=[generate_scoring_client_errors()],
        scorer=harness_mcq(),
        config=GenerateConfig(temperature=vgi.TEMPERATURE),
        epochs=Epochs(vgi.EPOCHS),
        metadata={
            "harness": vgi.HARNESS,
            "harness_commit": vgi.HARNESS_COMMIT,
            "dataset_revision": vgi.DATASET_REVISION,
            "dataset_tag": vgi.DATASET_TAG,
            "media_manifest_hash": manifest.manifest_hash,
        },
    )
