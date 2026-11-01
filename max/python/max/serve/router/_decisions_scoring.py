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

"""Shared encode-and-score steps behind ``/v1/decisions`` and ``/v1/systemone``.

A question is rendered into one or more prompts (rows) that each end at an
answer position. One prefill step per row returns the full-vocabulary
log-probability of the row's answer labels (see
:meth:`~max.serve.pipelines.llm.TokenGeneratorPipeline.score`); no token is
generated. The prompt format (:mod:`max.serve.router._decision_format`)
decides how many rows a question takes.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass

from fastapi import Request
from max.pipelines.context import SamplingParams
from max.pipelines.modeling.types import (
    LabelScoringTokenizer,
    PipelineTask,
    RequestID,
    TextGenerationRequest,
)
from max.serve.pipelines._label_scoring import LabelScoringOutput
from max.serve.pipelines.llm import TokenGeneratorPipeline
from max.serve.router._decisions_prompt import QuestionView


class DecisionRequestError(ValueError):
    """A request this server or model cannot answer, which is the caller's to fix.

    The routes map it to HTTP 400. Any other ``ValueError`` is a server fault
    and becomes a logged 500.
    """


@dataclass(frozen=True)
class EncodedRow:
    """One prompt to score and the label tokens read at its end."""

    prompt_ids: list[int]
    """The prompt, sent to the scorer as token ids so the labels stay valid
    next-token candidates after exactly this tokenization."""
    label_ids: list[int]


@dataclass(frozen=True)
class EncodedQuestion:
    """A question rendered to scoring rows."""

    question_id: str
    view: QuestionView
    rows: list[EncodedRow]


def require_scoring(
    request: Request, pipeline: TokenGeneratorPipeline, route: str
) -> LabelScoringTokenizer:
    """Refuses servers and models that cannot score labels.

    Args:
        request: The incoming request, for the app state.
        pipeline: The pipeline serving the request.
        route: The route path, named in the error.

    Returns:
        The pipeline's tokenizer, typed for what scoring needs.

    Raises:
        DecisionRequestError: If the server does not run text generation, uses
            LoRA, variable logits or speculative decoding (whose pipelines
            have no scoring path), or its tokenizer lacks a chat template or a
            context length.
    """
    state = request.app.state
    if state.task != PipelineTask.TEXT_GENERATION:
        raise DecisionRequestError(
            f"{route} requires a text generation model, but this server "
            f"runs {state.task.value}"
        )
    config = state.pipeline_config
    unsupported = [
        feature
        for feature, enabled in (
            ("speculative decoding", config.speculative is not None),
            ("LoRA", config.lora is not None),
            ("variable logits", config.sampling.enable_variable_logits),
        )
        if enabled
    ]
    if unsupported:
        raise DecisionRequestError(
            f"{route} does not support a server using {', '.join(unsupported)}"
        )
    tokenizer = pipeline.tokenizer
    if not isinstance(tokenizer, LabelScoringTokenizer):
        raise DecisionRequestError(
            f"{route} requires a tokenizer with a chat template and a "
            "context length"
        )
    return tokenizer


async def score_all(
    request: Request,
    pipeline: TokenGeneratorPipeline,
    model_name: str,
    encoded: Sequence[EncodedQuestion],
) -> list[list[LabelScoringOutput]]:
    """Scores every row concurrently, cancelling the rest on a failure.

    Returns:
        One list of row results per question, in row order.
    """
    request_id = request.state.request_id
    rows = [row for item in encoded for row in item.rows]
    tasks = [
        asyncio.ensure_future(
            pipeline.score(
                TextGenerationRequest(
                    request_id=RequestID(f"{request_id}_{index}"),
                    model_name=model_name,
                    prompt=row.prompt_ids,
                    sampling_params=SamplingParams(top_k=1, temperature=0),
                    timestamp_ns=request.state.request_timer.start_ns,
                    request_path=request.url.path,
                ),
                row.label_ids,
            )
        )
        for index, row in enumerate(rows)
    ]
    try:
        flat = list(await asyncio.gather(*tasks))
    except BaseException:
        for task in tasks:
            task.cancel()
        raise
    results: list[list[LabelScoringOutput]] = []
    start = 0
    for item in encoded:
        results.append(flat[start : start + len(item.rows)])
        start += len(item.rows)
    return results


def usage_totals(
    results: Sequence[Sequence[LabelScoringOutput]],
) -> tuple[int, int | None]:
    """Total prompt tokens, and prefix-cache hits if the pipeline reports them."""
    flat = [result for rows in results for result in rows]
    prompt_tokens = sum(result.prompt_token_count for result in flat)
    cached = [result.cached_token_count for result in flat]
    cached_tokens = (
        sum(count or 0 for count in cached)
        if any(count is not None for count in cached)
        else None
    )
    return prompt_tokens, cached_tokens
