# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import pytest
from async_asgi_testclient import TestClient
from fastapi import FastAPI
from max.driver import DeviceSpec
from max.nn.kv_cache import KVCacheStrategy
from max.pipelines import PipelineConfig, SupportedEncoding
from max.serve.schemas.openai import (
    CreateChatCompletionResponse,
    ListModelsResponse,
    Model,
)


@pytest.mark.asyncio()
@pytest.mark.parametrize(
    "pipeline_config",
    [
        PipelineConfig(
            model_path="HuggingFaceTB/SmolLM-135M",
            max_length=512,
            max_new_tokens=3,
            device_specs=[DeviceSpec.cpu()],
            quantization_encoding=SupportedEncoding.float32,
            cache_strategy=KVCacheStrategy.PAGED,
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
        assert response.data[0].id == "HuggingFaceTB/SmolLM-135M"

        raw_response = await client.get("/v1/models/SmolLM-135M")

        response2 = Model.model_validate(raw_response.json())

        assert response2.id == "HuggingFaceTB/SmolLM-135M"


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
            max_new_tokens=3,
            device_specs=[DeviceSpec.cpu()],
            quantization_encoding=SupportedEncoding.float32,
            cache_strategy=KVCacheStrategy.PAGED,
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
            },
        )
        # Validate response
        response = CreateChatCompletionResponse.model_validate(
            raw_response.json()
        )

        # Make a request to the actual model name
        raw_response = await client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "tell me a joke"}],
                "stream": False,
            },
        )
        # Validate request failed
        assert raw_response.status_code == 400
