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

"""A decider model served with the decider prompt format, on a GPU."""

from __future__ import annotations

import math
from typing import Any

import pytest
from async_asgi_testclient import TestClient
from fastapi import FastAPI
from max.driver import DeviceSpec
from max.pipelines import PipelineArgs
from max.pipelines.lib import KVCacheConfig, PipelineRuntimeConfig

MODEL_NAME = "Mapika/decider-0.8b"
TICKET = "I was charged twice for order A-104 and I am furious. Refund one."
MIN_CLEAR_PROBABILITY = 0.9


def _pipeline_args() -> PipelineArgs:
    return PipelineArgs(
        model_path=MODEL_NAME,
        device_specs=[DeviceSpec.accelerator()],
        quantization_encoding="bfloat16",
        kv_cache=KVCacheConfig(),
        max_length=2048,
        runtime=PipelineRuntimeConfig(max_batch_size=16),
    )


def _decisions_request(questions: list[dict[str, Any]]) -> dict[str, Any]:
    return {"model": MODEL_NAME, "input": TICKET, "questions": questions}


@pytest.mark.asyncio
@pytest.mark.parametrize("pipeline_config", [_pipeline_args()], indirect=True)
async def test_decider_decisions_gpu(app: FastAPI) -> None:
    async with TestClient(app, timeout=300.0) as client:
        response = await client.post(
            "/v1/decisions",
            json=_decisions_request(
                [
                    {
                        "id": "team",
                        "type": "choice",
                        "question": "Which team should handle this?",
                        "options": [
                            {"name": "billing"},
                            {"name": "technical"},
                            {"name": "sales"},
                        ],
                    },
                    {
                        "id": "angry",
                        "type": "yes_no",
                        "question": "Is the customer angry?",
                    },
                    {
                        "id": "happy",
                        "type": "yes_no",
                        "question": "Is the customer happy?",
                    },
                    {
                        "id": "anger",
                        "type": "score",
                        "question": "How angry is the customer?",
                        "levels": ["calm", "annoyed", "angry", "furious"],
                    },
                ]
            ),
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["prompt_format_version"] == 2
        assert body["usage"]["completion_tokens"] == 0

        answers = body["answers"]
        for answer in answers.values():
            total = sum(answer["probabilities"].values())
            assert math.isclose(total, 1.0, abs_tol=1e-6)
        # Clear cases a decision model must get right.
        assert answers["team"]["choice"] == "billing"
        assert (
            answers["team"]["probabilities"]["billing"] > MIN_CLEAR_PROBABILITY
        )
        assert answers["angry"]["probabilities"]["yes"] > MIN_CLEAR_PROBABILITY
        assert answers["happy"]["probabilities"]["no"] > MIN_CLEAR_PROBABILITY
        # A score is the normalized fit of each level, so it is a distribution.
        assert set(answers["anger"]["probabilities"]) == {"0", "1", "2", "3"}


@pytest.mark.asyncio
@pytest.mark.parametrize("pipeline_config", [_pipeline_args()], indirect=True)
async def test_decider_systemone_and_limits_gpu(app: FastAPI) -> None:
    async with TestClient(app, timeout=300.0) as client:
        # Everything in the instructions, nothing in the state.
        response = await client.post(
            "/v1/systemone",
            json={
                "model": MODEL_NAME,
                "state": {},
                "questions": {
                    "q": {
                        "type": "choice",
                        "instructions": f"{TICKET}\nWhich team handles this?",
                        "criteria": {"billing": None, "technical": None},
                    }
                },
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["answers"]["q"]["choice"] == "billing"

        # A decider model reads at most 10 options, and says so.
        wide = {name: None for name in "abcdefghijk"}
        response = await client.post(
            "/v1/systemone",
            json={
                "model": MODEL_NAME,
                "state": TICKET,
                "questions": {
                    "q": {
                        "type": "choice",
                        "instructions": "Pick one",
                        "criteria": wide,
                    }
                },
            },
        )
        assert response.status_code == 400
        assert "options per choice" in response.text
