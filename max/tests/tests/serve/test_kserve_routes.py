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


import pytest
from fastapi.testclient import TestClient
from max.pipelines.lib import PipelineConfig
from max.serve.api_server import ServingTokenGeneratorSettings, fastapi_app
from max.serve.config import APIType, Settings
from max.serve.mocks.mock_api_requests import (
    simple_kserve_request,
    simple_kserve_response,
)
from max.serve.pipelines.echo_gen import (
    EchoPipelineTokenizer,
    EchoTokenGenerator,
)


@pytest.fixture
def app(mock_pipeline_config: PipelineConfig):  # noqa: ANN201
    settings = Settings(api_types=[APIType.KSERVE], use_heartbeat=False)
    serving_settings = ServingTokenGeneratorSettings(
        model_factory=EchoTokenGenerator,
        pipeline_config=mock_pipeline_config,
        tokenizer=EchoPipelineTokenizer(),
    )
    return fastapi_app(settings, serving_settings)


@pytest.mark.skip(reason="Implementing infer/ for real is a WIP.")
def test_kserve_basic_infer(app) -> None:  # noqa: ANN001
    with TestClient(app) as client:
        response = client.post(
            "/v2/models/Add/versions/0/infer",
            json=simple_kserve_request(),
        )
        assert response.json() == simple_kserve_response()
