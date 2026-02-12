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
"""Test serving a Mamba SSM model on the GPU."""

from __future__ import annotations

import pytest
from async_asgi_testclient import TestClient
from max.driver import DeviceSpec
from max.pipelines import PipelineConfig
from max.serve.mocks.mock_api_requests import simple_openai_request
from max.serve.schemas.openai import (
    CreateChatCompletionResponse,
    CreateCompletionResponse,
)

MAMBA_MODEL = "state-spaces/mamba-130m-hf"

MAMBA_PIPELINE_CONFIG = PipelineConfig(
    model_path=MAMBA_MODEL,
    max_length=512,
    device_specs=[DeviceSpec.accelerator()],
    max_batch_size=1,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pipeline_config",
    [MAMBA_PIPELINE_CONFIG],
    indirect=True,
)
async def test_mamba_serve_v1_chat_completions_gpu(
    app: FastAPI,  # type: ignore
) -> None:
    async with TestClient(app, timeout=180.0) as client:
        raw_response = await client.post(
            "/v1/chat/completions",
            json=simple_openai_request(model_name=MAMBA_MODEL)
            | {"max_tokens": 20},
        )

        response = CreateChatCompletionResponse.model_validate_json(
            raw_response.text
        )
        assert len(response.choices) == 1
        # Base model may not produce EOS within max_tokens.
        assert response.choices[0].finish_reason in ("stop", "length")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pipeline_config",
    [MAMBA_PIPELINE_CONFIG],
    indirect=True,
)
@pytest.mark.parametrize(
    "prompt,expected_choices",
    [
        ("Hello world", 1),
        (["Hello world"], 1),
        ([1, 2, 3], 1),
        ([[1, 2, 3]], 1),
    ],
)
async def test_mamba_serve_v1_completions_gpu(
    app: FastAPI,  # type: ignore
    prompt: str | list[str] | list[int] | list[list[int]],
    expected_choices: int,
) -> None:
    async with TestClient(app, timeout=180.0) as client:
        raw_response = await client.post(
            "/v1/completions",
            json={
                "model": MAMBA_MODEL,
                "prompt": prompt,
                "max_tokens": 20,
            },
        )
        response = CreateCompletionResponse.model_validate(raw_response.json())

        assert len(response.choices) == expected_choices
        # Base model may not produce EOS within max_tokens.
        assert response.choices[0].finish_reason in ("stop", "length")
