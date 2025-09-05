# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
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
def app(mock_pipeline_config: PipelineConfig):
    settings = Settings(
        api_types=[APIType.KSERVE], MAX_SERVE_USE_HEARTBEAT=False
    )
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
