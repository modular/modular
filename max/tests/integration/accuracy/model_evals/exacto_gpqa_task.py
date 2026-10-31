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
"""Inspect task for GPQA-Diamond as OpenRouter's benchmark harness runs it.

Samples come from :mod:`exacto_gpqa_openrouter` and answers are extracted by
:mod:`exacto_mcq`, a port of the harness's own ``extract.ts``. openbench
0.5.3's MCQ scorer, which this task used before, is what ``extract.ts``
started from, but the two have drifted: the harness's English patterns also
accept ``Answer: $C``, which openbench reads only through its lower-priority
multilingual fallback, and its multilingual patterns read ``[A-J]`` rather
than ``[A-D]``.

Runs only inside the harness venv (``run_exacto_gpqa_local.sh``), which has
inspect_ai and datasets.
"""

from __future__ import annotations

import os

import exacto_gpqa_openrouter as gpqa
import exacto_mcq as mcq
from datasets import load_dataset
from inspect_ai import Epochs, Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import GenerateConfig
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
from inspect_ai.solver import TaskState, generate, system_message


def load_samples() -> list[Sample]:
    """Loads the pinned GPQA-Diamond split as harness-identical samples.

    Reads ``$GPQA_DATASET_FILE`` instead of the HF dataset when it is set,
    after checking it is the pinned revision's CSV byte for byte.
    """
    dataset_file = os.environ.get("GPQA_DATASET_FILE")
    if dataset_file:
        gpqa.check_dataset_file(dataset_file)
        records = load_dataset("csv", data_files=dataset_file, split="train")
    else:
        records = load_dataset(
            gpqa.DATASET, revision=gpqa.DATASET_REVISION, split=gpqa.SPLIT
        )
    samples = []
    for index, record in enumerate(records):
        prompt, target = gpqa.record_to_prompt(record, index)
        # Ids stay 1-based like openbench's auto_id, so a question keeps its
        # id across logs from either task.
        samples.append(
            Sample(
                id=index + 1,
                input=prompt,
                target=target,
                metadata={"subdomain": record.get("Subdomain")},
            )
        )
    if len(samples) != gpqa.NUM_QUESTIONS:
        raise ValueError(
            f"{gpqa.DATASET}@{gpqa.DATASET_REVISION} has {len(samples)} rows, "
            f"expected {gpqa.NUM_QUESTIONS}"
        )
    return samples


# Named as openbench's scorer was, so logs from before and after the switch
# carry their score under the same key for the smoke runner and answer audit.
@scorer(metrics=[accuracy(), stderr(), std()], name="mcq_scorer")
def harness_mcq() -> Scorer:
    """The harness's ``mcqScorer``."""

    async def score(state: TaskState, target: Target) -> Score:
        result = mcq.score_mcq(state.output.completion, target.text)
        return Score(
            value=CORRECT if result.correct else INCORRECT,
            answer=result.answer,
            explanation=result.explanation,
        )

    return score


@task
def gpqa_diamond() -> Task:
    """GPQA-Diamond with the harness's prompt, shuffle, sampling and epochs.

    The reasoning effort is left to the caller's generate config (the runner's
    ``--reasoning-effort``), so a run can drop or change it without a new task.
    """
    return Task(
        name="gpqa_diamond",
        dataset=MemoryDataset(load_samples(), name=gpqa.DATASET),
        solver=[system_message(gpqa.SYSTEM_MESSAGE), generate()],
        scorer=harness_mcq(),
        config=GenerateConfig(temperature=gpqa.TEMPERATURE),
        epochs=Epochs(gpqa.EPOCHS),
        metadata={
            "harness": gpqa.HARNESS,
            "harness_commit": gpqa.HARNESS_COMMIT,
            "mcq_extractor_commit": mcq.HARNESS_COMMIT,
            "dataset_revision": gpqa.DATASET_REVISION,
        },
    )
