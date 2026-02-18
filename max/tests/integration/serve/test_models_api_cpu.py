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

import hf_repo_lock
import pytest
from async_asgi_testclient import TestClient
from fastapi import FastAPI
from max.driver import DeviceSpec
from max.pipelines import PipelineConfig, SupportedEncoding
from max.serve.schemas.openai import (
    CreateChatCompletionResponse,
    ListModelsResponse,
    Model,
)

SMOLLM_135M_REPO_ID = "HuggingFaceTB/SmolLM-135M"


@pytest.mark.asyncio()
@pytest.mark.parametrize(
    "pipeline_config",
    [
        PipelineConfig(
            model_path=SMOLLM_135M_REPO_ID,
            huggingface_model_revision=hf_repo_lock.revision_for_hf_repo(
                SMOLLM_135M_REPO_ID
            ),
            max_length=512,
            device_specs=[DeviceSpec.cpu()],
            quantization_encoding=SupportedEncoding.float32,
            cache_strategy="paged",
            max_batch_size=16,
        )
    ],
    indirect=True,
)
async def test_serve_models(app: FastAPI) -> None:
    async with TestClient(app, timeout=720.0) as client:
        raw_response = await client.get("/v1/models")

        response = ListModelsResponse.model_validate(raw_response.json())

        assert len(response.data) == 1
        assert response.data[0].id == SMOLLM_135M_REPO_ID

        raw_response = await client.get("/v1/models/SmolLM-135M")

        response2 = Model.model_validate(raw_response.json())

        assert response2.id == SMOLLM_135M_REPO_ID


MODEL_ALIAS = "foobar"
MODEL_NAME = "modularai/SmolLM-135M-Instruct-FP32"


@pytest.mark.asyncio()
@pytest.mark.parametrize(
    "pipeline_config",
    [
        PipelineConfig(
            model_path=MODEL_NAME,
            served_model_name=MODEL_ALIAS,
            max_length=512,
            device_specs=[DeviceSpec.cpu()],
            quantization_encoding=SupportedEncoding.float32,
            cache_strategy="paged",
            max_batch_size=16,
        )
    ],
    indirect=True,
)
async def test_served_model_name(app: FastAPI) -> None:
    async with TestClient(app, timeout=720.0) as client:
        # Request model list
        raw_response = await client.get("/v1/models")
        response = ListModelsResponse.model_validate(raw_response.json())

        # Assert alias in model list
        assert len(response.data) == 1
        assert response.data[0].id == MODEL_ALIAS

        # Make a request to the alias
        raw_response = await client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL_ALIAS,
                "messages": [{"role": "user", "content": "tell me a joke"}],
                "stream": False,
                "max_tokens": 3,
            },
        )
        # Validate response
        _ = CreateChatCompletionResponse.model_validate(raw_response.json())

        # Make a request to the actual model name
        raw_response = await client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "tell me a joke"}],
                "stream": False,
                "max_tokens": 3,
            },
        )
        # Validate request failed
        assert raw_response.status_code == 400
