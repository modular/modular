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

"""Fake tokenizer and scoring pipeline shared by the decisions router tests."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Any

from fastapi import APIRouter, FastAPI
from max.pipelines.modeling.types import (
    PipelineTask,
    TextGenerationRequest,
    TextGenerationRequestMessage,
)
from max.serve.pipelines._label_scoring import LabelScoringOutput
from max.serve.request import register_request
from max.serve.router._chat_format import ChatFormat
from max.serve.router._decision_format import DecisionFormat

MODEL = "test-decider"
THINK_START, THINK_END = 90_001, 90_002

_PIECE = re.compile(r"<think>|</think>|\w+|\s|.")


class FakeTokenizer:
    """Word-level tokenizer with a chat template that can open a think block."""

    eos_token_ids: set[int] = set()
    expects_content_wrapping = False

    def __init__(
        self,
        *,
        label_merges: bool = False,
        has_context_length: bool = True,
        has_reasoning: bool = True,
    ) -> None:
        self._label_merges = label_merges
        if has_reasoning:
            self.reasoning_start_token_id = THINK_START
            self.reasoning_end_token_id = THINK_END
        if has_context_length:
            self.max_length = 512
        self.template_options: list[dict[str, Any]] = []
        self.rendered_prompts: list[str] = []

    async def new_context(self, request: Any) -> Any:
        raise NotImplementedError("the fake pipeline scores without contexts")

    async def decode(self, encoded: Any, **kwargs: Any) -> str:
        raise NotImplementedError

    def apply_chat_template(
        self,
        messages: list[TextGenerationRequestMessage],
        tools: Any,
        **options: Any,
    ) -> str:
        self.template_options.append(options)
        think = (
            "<think>\n\n</think>\n\n"
            if options.get("enable_thinking") is False
            else "<think>\n"
        )
        prompt = f"<user>\n{messages[0].content}\n<assistant>\n{think}"
        self.rendered_prompts.append(prompt)
        return prompt

    async def encode(self, text: str, add_special_tokens: bool = False) -> Any:
        assert add_special_tokens is False
        if self._label_merges and text.endswith(("A", "B")):
            return [1, 2, 3]
        ids = []
        for piece in _PIECE.findall(text):
            ids.append(
                {"<think>": THINK_START, "</think>": THINK_END}.get(
                    piece, 1000 + sum(map(ord, piece))
                )
            )
        return ids


class FakePipeline:
    """Scores each request with logprobs from ``scorer(request, labels)``."""

    model_name = MODEL
    lora_queue = None

    def __init__(
        self,
        tokenizer: FakeTokenizer,
        scorer: Callable[[TextGenerationRequest, Sequence[int]], list[float]],
    ) -> None:
        self.tokenizer = tokenizer
        self._scorer = scorer
        self.requests: list[TextGenerationRequest] = []
        self.labels: list[list[int]] = []
        self.prompt_ids: list[list[int]] = []

    async def score(
        self,
        request: TextGenerationRequest,
        label_token_ids: Sequence[int],
    ) -> LabelScoringOutput:
        assert isinstance(request.prompt, list)
        self.requests.append(request)
        self.labels.append(list(label_token_ids))
        self.prompt_ids.append(list(request.prompt))
        return LabelScoringOutput(
            label_log_probabilities=self._scorer(request, label_token_ids),
            prompt_token_count=len(request.prompt),
            cached_token_count=2,
        )


def prefer_first(
    request: TextGenerationRequest, labels: Sequence[int]
) -> list[float]:
    probabilities = [0.6, 0.2, 0.1, 0.05][: len(labels)]
    return [math.log(p) for p in probabilities]


def make_app(
    pipeline: FakePipeline,
    router: APIRouter,
    *,
    decision_format: DecisionFormat | None = None,
    task: PipelineTask = PipelineTask.TEXT_GENERATION,
    speculative: bool = False,
) -> FastAPI:
    """An app serving ``router`` over ``pipeline``."""
    app = FastAPI(title="MAX Serve Test")
    register_request(app)
    app.include_router(router)
    app.state.pipeline = pipeline
    app.state.task = task
    app.state.decision_format = decision_format or ChatFormat()
    app.state.pipeline_config = SimpleNamespace(
        speculative=object() if speculative else None,
        lora=None,
        sampling=SimpleNamespace(enable_variable_logits=False),
    )
    return app
