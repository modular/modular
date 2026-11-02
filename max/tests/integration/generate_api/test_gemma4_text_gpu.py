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
"""Checks ModuleV3 Gemma 4 text generation through the generate API."""

from __future__ import annotations

import time

import numpy as np
import numpy.typing as npt
import transformers
from _generate_api import (
    GEMMA4,
    TEXT_PROMPTS,
    TEXT_STEPS,
    Report,
    Row,
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
    prefill_reference_steps,
    text_messages,
    worst,
)
from max.pipelines import AutoModel, AutoTokenizer
from max.pipelines.modeling.types import PipelineTask

RUN = "gemma4_text"


def _compare(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    reference: Row,
    new_tokens: list[int],
    logits: list[npt.NDArray[np.float32]],
    label: str,
) -> RowComparison:
    reference_tokens = [step["token"] for step in reference["steps"]]
    divergence = first_divergence(new_tokens, reference_tokens)
    if divergence is not None:
        check_near_tie(
            reference["steps"][divergence],
            new_tokens[divergence],
            label,
            divergence,
        )
        prefill_reference_steps(
            model, reference["prompt_ids"], reference_tokens, logits, divergence
        )
    return RowComparison(
        divergence,
        tokenizer.decode(new_tokens),
        tokenizer.decode(reference_tokens),
        compare_steps(reference["steps"], logits),
    )


def test_gemma4_text() -> None:
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
    tokenizer = AutoTokenizer(GEMMA4, revision=revision)
    # AutoTokenizer adds no chat template, so the prompt ids come from the
    # Hugging Face tokenizer, as a caller of generate(input_ids) would build
    # them.
    chat_template = transformers.AutoTokenizer.from_pretrained(
        GEMMA4, revision=revision
    )
    prompt_ids = [
        list(
            chat_template.apply_chat_template(
                text_messages(prompt),
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
            )["input_ids"]
        )
        for prompt in TEXT_PROMPTS
    ]
    for ids, row in zip(prompt_ids, rows, strict=True):
        assert ids == row["prompt_ids"], (
            "The tokenizer or chat template changed."
        )
    single: list[RowComparison] = []
    batched: list[RowComparison] = []
    full: list[npt.NDArray[np.float32]] = []
    start = time.monotonic()
    model = load_gemma4(revision, max_batch_size=len(TEXT_PROMPTS))
    try:
        report.time("load_seconds", start)
        report.data["architecture"] = check_module_v3(
            model, PipelineTask.TEXT_GENERATION
        )
        report.data["memory_used_gib"] = device_memory_used_gib()
        start = time.monotonic()
        for index, (ids, row) in enumerate(zip(prompt_ids, rows, strict=True)):
            result = model.generate(
                [ids],
                max_new_tokens=TEXT_STEPS,
                top_k=1,
                ignore_eos=True,
                output_logits=True,
            )
            assert result.logits is not None
            new_tokens = result.sequences[0][result.prompt_lengths[0] :]
            logits = list(result.logits[0])
            full.append(np.stack(logits))
            single.append(
                _compare(
                    model, tokenizer, row, new_tokens, logits, f"row {index}"
                )
            )
            report.data["rows"] = [row.summary() for row in single]
        report.time("generate_seconds", start)
        # The same prompts as one batch.
        result = model.generate(
            prompt_ids,
            max_new_tokens=TEXT_STEPS,
            top_k=1,
            ignore_eos=True,
            output_logits=True,
        )
        assert result.logits is not None
        for index, row in enumerate(rows):
            new_tokens = result.sequences[index][result.prompt_lengths[index] :]
            batched.append(
                _compare(
                    model,
                    tokenizer,
                    row,
                    new_tokens,
                    list(result.logits[index]),
                    f"batched row {index}",
                )
            )
            report.data["batched_rows"] = [row.summary() for row in batched]
    finally:
        model.close()
        del model
    dump_logits(f"{RUN}_max_logits.npz", full)
    values = worst(single + batched)
    report.data["worst"] = values
    check_worst(RUN, values)
