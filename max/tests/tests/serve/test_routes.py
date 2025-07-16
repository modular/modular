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
from max.serve.pipelines.echo_gen import (
    EchoPipelineTokenizer,
    EchoTokenGenerator,
)
from max.serve.router import openai_routes
from max.serve.schemas.openai import CreateCompletionRequest  # type: ignore


class MockPipelineConfig(PipelineConfig):
    def __init__(self):
        self.max_batch_size = 1


@pytest.fixture
def app():
    settings = Settings(
        api_types=[APIType.KSERVE], MAX_SERVE_USE_HEARTBEAT=False
    )

    pipeline_settings = ServingTokenGeneratorSettings(
        model_name="echo",
        model_factory=EchoTokenGenerator,
        pipeline_config=MockPipelineConfig(),
        tokenizer=EchoPipelineTokenizer(),
    )
    return fastapi_app(settings, pipeline_settings)


# In Bazel, we throw PackageNotFoundError since we're not running as a proper package.
def test_version_endpoint_exists(app) -> None:  # noqa: ANN001
    with TestClient(app) as client:
        response = client.get("/version")
        assert response.status_code == 200
        assert response.json()
        assert response.json()["version"]


def test_health_endpoint_exists(app):  # noqa: ANN001
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200


def test_prompts() -> None:
    completion_req = CreateCompletionRequest(
        model="whatev",
        prompt="Why is the sky blue?",
    )
    prompts = openai_routes.openai_get_prompts_from_completion_request(
        completion_req
    )
    assert len(prompts) == 1

    completion_req = CreateCompletionRequest(
        model="whatev",
        prompt=["Why is the sky blue?", "what time is it?"],
    )
    prompts = openai_routes.openai_get_prompts_from_completion_request(
        completion_req
    )
    assert len(prompts) == 2

    completion_req = CreateCompletionRequest(
        model="whatev",
        prompt=[[1, 2, 3]],
    )
    prompts = openai_routes.openai_get_prompts_from_completion_request(
        completion_req
    )
    assert len(prompts) == 1

    completion_req = CreateCompletionRequest(
        model="whatev",
        prompt=[1, 2, 3],
    )
    prompts = openai_routes.openai_get_prompts_from_completion_request(
        completion_req
    )
    assert len(prompts) == 1

    # prompt item
    completion_req = CreateCompletionRequest(
        model="whatev",
        prompt=[[1, 2, 3]],
    )
    prompts = openai_routes.openai_get_prompts_from_completion_request(
        completion_req
    )
    assert len(prompts) == 1

    completion_req = CreateCompletionRequest(
        model="whatev",
        prompt=[[1, 2, 3], [4, 5, 6]],
    )
    prompts = openai_routes.openai_get_prompts_from_completion_request(
        completion_req
    )
    assert len(prompts) == 2
