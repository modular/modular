# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

import pytest
from async_asgi_testclient import TestClient
from fastapi import FastAPI
from max.driver import DeviceSpec
from max.pipelines import PipelineConfig
from max.serve.schemas.openai import CreateEmbeddingResponse


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pipeline_config",
    [
        PipelineConfig(
            model_path="sentence-transformers/all-mpnet-base-v2",
            max_length=256,
            device_specs=[DeviceSpec.cpu()],
        )
    ],
    indirect=True,
)
async def test_serve_embeddings(app: FastAPI) -> None:
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
