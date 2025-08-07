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
from max.serve.schemas.openai import InputItem, PromptItem  # type: ignore


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


def test_health_endpoint_exists(app) -> None:  # noqa: ANN001
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200


def test_prompts() -> None:
    prompts = openai_routes.get_prompts_from_openai_request(
        "Why is the sky blue?"
    )
    assert len(prompts) == 1

    prompts = openai_routes.get_prompts_from_openai_request(
        ["Why is the sky blue?", "what time is it?"]
    )
    assert len(prompts) == 2

    prompts = openai_routes.get_prompts_from_openai_request([[1, 2, 3]])
    assert len(prompts) == 1

    prompts = openai_routes.get_prompts_from_openai_request([1, 2, 3])
    assert len(prompts) == 1

    # prompt item
    prompts = openai_routes.get_prompts_from_openai_request([[1, 2, 3]])
    assert len(prompts) == 1

    prompts = openai_routes.get_prompts_from_openai_request(
        [[1, 2, 3], [4, 5, 6]]
    )
    assert len(prompts) == 2

    # prompt item (explicit)
    prompts = openai_routes.get_prompts_from_openai_request(
        [PromptItem(root=[1, 2, 3])]
    )
    assert len(prompts) == 1

    prompts = openai_routes.get_prompts_from_openai_request(
        [PromptItem(root=[1, 2, 3]), PromptItem(root=[4, 5, 6])]
    )
    assert len(prompts) == 2

    # input item (explicit)
    prompts = openai_routes.get_prompts_from_openai_request(
        [InputItem(root=[1, 2, 3])]
    )
    assert len(prompts) == 1

    prompts = openai_routes.get_prompts_from_openai_request(
        [InputItem(root=[1, 2, 3]), InputItem(root=[4, 5, 6])]
    )
    assert len(prompts) == 2
