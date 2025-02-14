# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import pytest
from async_asgi_testclient import TestClient
from max.driver import DeviceSpec
from max.pipelines import PipelineConfig, PipelineEngine, SupportedEncoding
from max.pipelines.kv_cache import KVCacheStrategy
from max.serve.schemas.openai import ListModelsResponse, Model  # type: ignore


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pipeline_config",
    [
        PipelineConfig(
            huggingface_repo_id="HuggingFaceTB/SmolLM-135M",
            max_length=512,
            max_new_tokens=3,
            device_specs=[DeviceSpec.cpu()],
            quantization_encoding=SupportedEncoding.float32,
            cache_strategy=KVCacheStrategy.CONTINUOUS,
            max_batch_size=16,
            engine=PipelineEngine.MAX,
        )
    ],
    indirect=True,
)
async def test_serve_models(app):
    async with TestClient(app, timeout=720.0) as client:
        raw_response = await client.get("/v1/models")

        response = ListModelsResponse.model_validate(raw_response.json())

        assert len(response.data) == 1
        assert response.data[0].id == "HuggingFaceTB/SmolLM-135M"

        raw_response = await client.get("/v1/models/SmolLM-135M")

        response = Model.model_validate(raw_response.json())

        assert response.id == "HuggingFaceTB/SmolLM-135M"
