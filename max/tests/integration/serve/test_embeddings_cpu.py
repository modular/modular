# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import pytest
from async_asgi_testclient import TestClient
from max.driver import DeviceSpec
from max.pipelines import PipelineConfig
from max.serve.schemas.openai import CreateEmbeddingResponse  # type: ignore


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pipeline_config",
    [
        PipelineConfig(
            huggingface_repo_id="sentence-transformers/all-mpnet-base-v2",
            max_length=256,
            device_specs=[DeviceSpec.cpu()],
        )
    ],
    indirect=True,
)
async def test_serve_embeddings(app):
    async with TestClient(app, timeout=720.0) as client:
        raw_response = await client.post(
            "/v1/embeddings",
            json={
                "input": "Turn this sentence into embeddings",
                "model": "sentence-transformers/all-mpnet-base-v2",
            },
        )

        response = CreateEmbeddingResponse.model_validate(raw_response.json())

        assert response.data[0].embedding is not None
