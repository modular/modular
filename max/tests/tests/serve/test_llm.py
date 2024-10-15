# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Unit tests for serve/pipelines/llm.py."""

from dataclasses import dataclass
from typing import Optional

from async_asgi_testclient import TestClient
import pytest

from max.pipelines import TokenGeneratorContext as Context
from max.serve.api_server import fastapi_app
from max.serve.config import APIType, Settings
from max.serve.debug import DebugSettings
from max.serve.mocks.mock_api_requests import simple_openai_request
from max.serve.pipelines.deps import token_pipeline
from max.serve.pipelines.llm import (
    TokenGeneratorRequest,
    TokenGeneratorPipeline,
    TokenGeneratorPipelineConfig,
)


@dataclass
class MockValueErrorGenerator:
    """A mock generator that throws a value error when a new context is created.
    """

    async def new_context(
        self, prompt: str, max_new_tokens: Optional[int] = None
    ) -> Context:
        raise ValueError()

    async def next_token(self, batch: dict[str, Context]) -> dict[str, str]:
        return {}

    async def release(self, context: Context):
        pass


@pytest.mark.parametrize("url", ["/v1/chat/completions"])
@pytest.mark.asyncio
async def test_routes_catches_new_context_value_error(url):
    """Test the server's response to a value error when calling new context."""

    # NOTE(matt): The config here _shouldn't_ impact anything.
    config = TokenGeneratorPipelineConfig.dynamic_homogenous(batch_size=1)
    pipeline = TokenGeneratorPipeline(config, MockValueErrorGenerator())
    fast_app = fastapi_app(
        Settings(api_types=[APIType.OPENAI]), DebugSettings(), [pipeline]
    )
    fast_app.dependency_overrides[token_pipeline] = lambda: pipeline

    async with TestClient(fast_app) as client:
        json = simple_openai_request("test")
        response = await client.post(url, json=json)
        assert response.status_code == 400
