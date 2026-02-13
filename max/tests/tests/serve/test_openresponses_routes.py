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
"""Tests for OpenResponses API routes."""

import logging
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from max.interfaces import (
    BaseContext,
    GenerationStatus,
    PipelineTokenizer,
    RequestID,
)
from max.interfaces.generation import GenerationOutput
from max.interfaces.request import OpenResponsesRequest
from max.interfaces.request.open_responses import OutputImageContent
from max.pipelines.lib import PIPELINE_REGISTRY, PipelineConfig
from max.serve.pipelines.general_handler import GeneralPipelineHandler
from max.serve.request import register_request
from max.serve.router import openresponses_routes

logger = logging.getLogger(__name__)


@dataclass
class MockOpenResponsesTokenizer(
    PipelineTokenizer[BaseContext, Any, OpenResponsesRequest]
):
    """Mock tokenizer for OpenResponses requests."""

    @property
    def eos(self) -> int:
        """Return a dummy EOS token ID."""
        return 0

    @property
    def expects_content_wrapping(self) -> bool:
        """Mock tokenizer doesn't require content wrapping."""
        return False

    async def encode(
        self, prompt: str, add_special_tokens: bool = False
    ) -> Any:
        """Mock encode method."""
        return np.array([ord(char) for char in prompt], dtype=np.int64)

    async def decode(self, encoded: Any, **kwargs) -> str:
        """Mock decode method."""
        return "mock decoded text"

    async def new_context(self, request: OpenResponsesRequest) -> BaseContext:
        """Creates a mock BaseContext for OpenResponses requests."""

        # Create a minimal mock BaseContext
        @dataclass
        class MockContext(BaseContext):
            """Minimal mock context for testing."""

            request_id: RequestID
            status: GenerationStatus = GenerationStatus.ACTIVE

        return MockContext(request_id=request.request_id)


@pytest.fixture(autouse=True)
def patch_pipeline_registry_context_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Patch PIPELINE_REGISTRY.retrieve_context_type to return BaseContext."""

    def _mock_retrieve_context_type(
        pipeline_config: PipelineConfig,
        override_architecture: str | None = None,
    ) -> type[BaseContext]:
        return BaseContext

    monkeypatch.setattr(
        PIPELINE_REGISTRY,
        "retrieve_context_type",
        _mock_retrieve_context_type,
    )


class MockGeneralPipelineHandler(GeneralPipelineHandler):
    """Mock implementation of GeneralPipelineHandler for testing."""

    def __init__(self) -> None:
        # Skip the parent constructor that requires real dependencies
        self.model_name = "test-model"
        self.logger = Mock()
        self.debug_logging = False

    async def next(
        self, request: OpenResponsesRequest
    ) -> AsyncGenerator[GenerationOutput, None]:
        """Mock implementation that yields a simple text response."""
        # Create a simple mock response
        pixel_data = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        yield GenerationOutput(
            request_id=request.request_id,
            final_status=GenerationStatus.END_OF_SEQUENCE,
            output=[OutputImageContent.from_numpy(pixel_data, format="png")],
        )


@pytest.fixture(scope="function")
def app(fixture_tokenizer, mock_pipeline_config: PipelineConfig):  # noqa: ANN001, ANN201
    """Create a test app with OpenResponses API enabled."""
    # Create a minimal FastAPI app without the full lifespan
    app = FastAPI(title="MAX Serve Test")

    # Register request middleware to add request_id to request.state
    register_request(app)

    # Register the OpenResponses routes
    app.include_router(openresponses_routes.router)

    # Inject a mock handler into app.state
    app.state.handler = MockGeneralPipelineHandler()

    return app


def test_openresponses_simple_request(app) -> None:  # noqa: ANN001
    """Test a simple OpenResponses request with string input."""
    with TestClient(app) as client:
        request_data = {
            "model": "test-model",
            "input": "Generate an image of a cat",
        }
        response = client.post("/v1/responses", json=request_data)

        assert response.status_code == 200
        response_json = response.json()

        # Check response structure
        assert "id" in response_json
        assert response_json["id"].startswith("resp_")
        assert response_json["object"] == "response"
        assert response_json["status"] == "completed"
        assert response_json["model"] == "test-model"
        assert "output" in response_json
        assert len(response_json["output"]) > 0

        # Check message structure
        message = response_json["output"][0]
        assert "id" in message
        assert message["id"].startswith("msg_")
        assert message["role"] == "assistant"
        assert "content" in message


def test_openresponses_message_list_input(app) -> None:  # noqa: ANN001
    """Test OpenResponses request with message list input."""
    with TestClient(app) as client:
        request_data = {
            "model": "test-model",
            "input": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
                {"role": "user", "content": "What about 3+3?"},
            ],
        }
        response = client.post("/v1/responses", json=request_data)

        assert response.status_code == 200
        response_json = response.json()
        assert response_json["status"] == "completed"


def test_openresponses_streaming_not_supported(app) -> None:  # noqa: ANN001
    """Test that streaming requests are rejected during validation."""
    with TestClient(app) as client:
        request_data = {
            "model": "test-model",
            "input": "Test",
            "stream": True,
        }
        response = client.post("/v1/responses", json=request_data)

        # Streaming validation happens during Pydantic validation, so returns 422
        assert response.status_code == 422  # UNPROCESSABLE_ENTITY
        assert "not currently supported" in response.json()["detail"].lower()


def test_openresponses_invalid_request(app) -> None:  # noqa: ANN001
    """Test that invalid requests return appropriate errors."""
    with TestClient(app) as client:
        # Missing required 'model' field
        request_data = {
            "input": "Test",
        }
        response = client.post("/v1/responses", json=request_data)

        assert response.status_code == 422  # UNPROCESSABLE_ENTITY
