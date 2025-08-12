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
from max.serve.schemas.openai import ListModelsResponse, Model


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
