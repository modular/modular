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
"""Test the ``/v1/decisions`` endpoint against a real model on the GPU."""

import asyncio
import math
from typing import Any

import pytest
from async_asgi_testclient import TestClient
from fastapi import FastAPI
from max.driver import DeviceSpec
from max.pipelines import PipelineArgs
from max.pipelines.lib import KVCacheConfig, PipelineRuntimeConfig
from max.serve.mocks.mock_api_requests import simple_openai_request

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
# One bf16 ulp of a logit near 16 moves a low-mass label's probability by
# about 0.06, so labels with a tiny share of the vocabulary mass are noisy.
BF16_PROBABILITY_TOLERANCE = 0.1

DECISION_REQUEST: dict[str, Any] = {
    "input": "I was charged twice for order A-104. Please refund one.",
    "return_prompt_token_ids": True,
    "questions": [
        {
            "id": "team",
            "type": "choice",
            "question": "Which team should handle this?",
            "options": [
                {"name": "billing", "description": "Charges and refunds"},
                {"name": "technical", "description": "Bugs and outages"},
                {"name": "other"},
            ],
        },
        {
            "id": "refund",
            "type": "yes_no",
            "question": "Is a refund requested?",
        },
        {
            "id": "urgency",
            "type": "score",
            "question": "How urgent is this?",
            "levels": ["no rush", "soon", "today", "immediately"],
        },
    ],
}


def _pipeline_args(**runtime: Any) -> PipelineArgs:
    return PipelineArgs(
        model_path=MODEL_NAME,
        device_specs=[DeviceSpec.accelerator()],
        quantization_encoding="bfloat16",
        kv_cache=KVCacheConfig(),
        max_length=1024,
        runtime=PipelineRuntimeConfig(max_batch_size=16, **runtime),
    )


def _assert_probabilities_close(
    answers: dict[str, Any], expected_answers: dict[str, Any]
) -> None:
    assert answers.keys() == expected_answers.keys()
    for question_id, answer in answers.items():
        expected = expected_answers[question_id]["probabilities"]
        for name, probability in answer["probabilities"].items():
            assert math.isclose(
                probability, expected[name], abs_tol=BF16_PROBABILITY_TOLERANCE
            )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pipeline_config",
    [
        # The default resolves to the overlap scheduler on this GPU.
        _pipeline_args(),
        _pipeline_args(force=True, enable_overlap_scheduler=False),
    ],
    ids=["overlap", "no-overlap"],
    indirect=True,
)
async def test_decisions_gpu(app: FastAPI) -> None:
    async with TestClient(app, timeout=180.0) as client:
        first = await client.post("/v1/decisions", json=DECISION_REQUEST)
        assert first.status_code == 200, first.text
        body = first.json()

        assert body["object"] == "decisions"
        assert body["usage"]["completion_tokens"] == 0
        assert body["usage"]["prompt_tokens"] > 0
        assert set(body["answers"]) == {"team", "refund", "urgency"}
        for answer in body["answers"].values():
            probabilities = answer["probabilities"]
            assert math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-6)
            assert 0.0 < answer["label_mass"] <= 1.0 + 1e-6
            assert len(answer["label_token_ids"]) == len(probabilities)
            assert answer["prompt_token_ids"]
        assert body["answers"]["team"]["choice"] in {
            "billing",
            "technical",
            "other",
        }

        # Repeats and requests batched together, including with text
        # generation, only differ by bf16 noise from batch-shape-dependent
        # kernels.
        repeat = await client.post("/v1/decisions", json=DECISION_REQUEST)
        _assert_probabilities_close(repeat.json()["answers"], body["answers"])
        generation = client.post(
            "/v1/chat/completions",
            json=simple_openai_request(model_name=MODEL_NAME),
        )
        concurrent = [
            client.post("/v1/decisions", json=DECISION_REQUEST)
            for _ in range(3)
        ]
        responses = await asyncio.gather(generation, *concurrent)
        assert responses[0].status_code == 200
        for response in responses[1:]:
            assert response.status_code == 200, response.text
            _assert_probabilities_close(
                response.json()["answers"], body["answers"]
            )
