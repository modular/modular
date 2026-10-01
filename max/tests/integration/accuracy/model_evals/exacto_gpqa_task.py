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

Samples come from :mod:`exacto_gpqa_openrouter`. The scorer is openbench's MCQ
scorer, which the harness's ``src/benchmarks/scorers/mcq/extract.ts`` ports
pattern for pattern, so answers are extracted the same way on both sides.

Runs only inside the harness venv (``run_exacto_gpqa_local.sh``), which has
inspect_ai, openbench and datasets; bazel never imports it.
"""

from __future__ import annotations

import exacto_gpqa_openrouter as gpqa
from datasets import load_dataset
from inspect_ai import Epochs, Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate, system_message
from openbench.scorers.mcq import create_mcq_scorer


def load_samples() -> list[Sample]:
    """Loads the pinned GPQA-Diamond split as harness-identical samples."""
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
        scorer=create_mcq_scorer()(),
        config=GenerateConfig(temperature=gpqa.TEMPERATURE),
        epochs=Epochs(gpqa.EPOCHS),
        metadata={
            "harness": gpqa.HARNESS,
            "harness_commit": gpqa.HARNESS_COMMIT,
            "dataset_revision": gpqa.DATASET_REVISION,
        },
    )
