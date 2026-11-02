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
"""Checks ModuleV3 Gemma 4 image understanding through the generate API."""

from __future__ import annotations

import time

import numpy as np
import numpy.typing as npt
from _generate_api import (
    CONVERSATIONS,
    GEMMA4,
    TEXT_STEPS,
    Report,
    RowComparison,
    check_module_v3,
    check_near_tie,
    check_revision,
    check_worst,
    compare_steps,
    device_memory_used_gib,
    dump_logits,
    first_divergence,
    load_gemma4,
    load_reference,
    worst,
)
from max.pipelines import AutoProcessor
from max.pipelines.modeling.types import PipelineTask

RUN = "gemma4_multimodal"


def test_gemma4_multimodal() -> None:
    report = Report(RUN)
    try:
        _run(report)
    finally:
        report.save()


def _run(report: Report) -> None:
    reference = load_reference(f"{RUN}.json")
    revision = reference["revision"]
    check_revision(GEMMA4, revision)
    rows = reference["rows"]
    assert all(len(row["steps"]) == TEXT_STEPS for row in rows)
    processor = AutoProcessor(GEMMA4, revision=revision)
    comparisons: list[RowComparison] = []
    prompt_matches: list[bool] = []
    full: list[npt.NDArray[np.float32]] = []
    start = time.monotonic()
    model = load_gemma4(revision)
    try:
        report.time("load_seconds", start)
        report.data["architecture"] = check_module_v3(
            model, PipelineTask.TEXT_GENERATION
        )
        report.data["memory_used_gib"] = device_memory_used_gib()
        start = time.monotonic()
        for index, (conversation, row) in enumerate(
            zip(CONVERSATIONS, rows, strict=True)
        ):
            reference_tokens = [step["token"] for step in row["steps"]]
            result = model.generate(
                **processor(conversation.messages()),
                max_new_tokens=TEXT_STEPS,
                top_k=1,
                ignore_eos=True,
                output_logits=True,
            )
            assert result.logits is not None
            prompt_length = result.prompt_lengths[0]
            prompt_matches.append(
                result.sequences[0][:prompt_length] == row["prompt_ids"]
            )
            report.data["prompt_matches_reference"] = prompt_matches
            assert prompt_matches[-1], (
                f"Conversation {index}: MAX's chat template or image expansion"
                " differs from the reference's prompt."
            )
            new_tokens = result.sequences[0][prompt_length:]
            divergence = first_divergence(new_tokens, reference_tokens)
            # A conversation cannot carry the reference's tokens, so the
            # comparison stops at the first step that picked another token.
            compared = TEXT_STEPS
            if divergence is not None:
                check_near_tie(
                    row["steps"][divergence],
                    new_tokens[divergence],
                    f"conversation {index}",
                    divergence,
                )
                compared = divergence + 1
            logits = list(result.logits[0])[:compared]
            full.append(np.stack(logits))
            comparisons.append(
                RowComparison(
                    divergence,
                    processor.decode(new_tokens),
                    processor.decode(reference_tokens),
                    compare_steps(row["steps"][:compared], logits),
                )
            )
            report.data["rows"] = [row.summary() for row in comparisons]
        report.time("generate_seconds", start)
    finally:
        model.close()
        del model
    dump_logits(f"{RUN}_max_logits.npz", full)
    values = worst(comparisons)
    report.data["worst"] = values
    check_worst(RUN, values)
