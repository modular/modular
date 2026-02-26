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
# Unit tests for model_worker
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from max.interfaces import (
    GenerationStatus,
    Pipeline,
    PipelineTask,
    PipelineTokenizer,
    RequestID,
    TextGenerationInputs,
    TextGenerationOutput,
)
from max.pipelines.core import TextContext
from max.pipelines.lib import PIPELINE_REGISTRY, MAXModelConfig, PipelineConfig
from max.serve import api_server
from max.serve.config import Settings
from max.serve.pipelines.echo_gen import EchoTokenGenerator
from max.serve.pipelines.model_worker import start_model_worker
from max.serve.telemetry.metrics import NoopClient
from max.serve.worker_interface.zmq_interface import ZmqModelWorkerInterface


@pytest.fixture
def mock_pipeline_config() -> PipelineConfig:
    pipeline_config = PipelineConfig.model_construct(max_batch_size=1)

    model_config = MAXModelConfig.model_construct(served_model_name="echo")
    pipeline_config.model = model_config
    return pipeline_config


@pytest.fixture(autouse=True)
def patch_pipeline_registry_context_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Patch PIPELINE_REGISTRY.retrieve_context_type to always return TextContext.

    The tests in this module use simple mock pipeline configs that do not
    correspond to a registered architecture. The default implementation of
    `retrieve_context_type` would raise in this case, but for these tests we
    only care that a valid context type is provided, not which one.
    """

    def _mock_retrieve_context_type(
        pipeline_config: PipelineConfig,
        override_architecture: str | None = None,
        task: PipelineTask | None = None,
    ) -> type[TextContext]:
        return TextContext

    monkeypatch.setattr(
        PIPELINE_REGISTRY,
        "retrieve_context_type",
        _mock_retrieve_context_type,
    )


@dataclass(frozen=True)
class MockContext(Mock):
    """Mock context that implements BaseContext protocol."""

    request_id: RequestID
    status: GenerationStatus = GenerationStatus.ACTIVE

    @property
    def is_done(self) -> bool:
        """Whether the request has completed generation."""
        return self.status.is_done


@pytest.mark.asyncio
async def test_model_worker_propagates_exception(
    mock_pipeline_config: PipelineConfig,
) -> None:
    """Tests raising in the model worker context manager."""
    settings = Settings()

    with pytest.raises(ValueError, match="kaboom"):
        async with start_model_worker(
            EchoTokenGenerator,
            mock_pipeline_config,
            settings=settings,
            metric_client=NoopClient(),
            model_worker_interface=ZmqModelWorkerInterface(
                PipelineTask.TEXT_GENERATION,
                context_type=TextContext,
            ),
        ):
            raise ValueError("kaboom")


class MockInvalidTokenGenerator(
    Pipeline[TextGenerationInputs[MockContext], TextGenerationOutput]
):
    ERROR_MESSAGE = "CRASH TEST DUMMY"

    def __init__(self) -> None:
        raise ValueError(MockInvalidTokenGenerator.ERROR_MESSAGE)

    def execute(
        self, inputs: TextGenerationInputs[MockContext]
    ) -> dict[RequestID, TextGenerationOutput]:
        raise ValueError()

    def release(self, request_id: RequestID) -> None:
        pass


@pytest.mark.asyncio
async def test_model_worker_propagates_construction_exception(
    mock_pipeline_config: PipelineConfig,
) -> None:
    """Tests raising in the model worker task."""
    settings = Settings()

    # The MockTokenGenerator crashes the remote subprocess
    # then ProcessMonitor checks throw TimeoutError here
    with pytest.raises(
        ValueError, match=MockInvalidTokenGenerator.ERROR_MESSAGE
    ):
        async with start_model_worker(
            MockInvalidTokenGenerator,
            mock_pipeline_config,
            settings=settings,
            model_worker_interface=ZmqModelWorkerInterface(
                PipelineTask.TEXT_GENERATION,
                context_type=TextContext,
            ),
            metric_client=NoopClient(),
        ):
            pass


class MockSlowTokenGenerator(
    Pipeline[TextGenerationInputs[MockContext], TextGenerationOutput]
):
    def __init__(self) -> None:
        time.sleep(0.2)

    def execute(
        self, inputs: TextGenerationInputs[MockContext]
    ) -> dict[RequestID, TextGenerationOutput]:
        raise ValueError()

    def release(self, request_id: RequestID) -> None:
        pass


@pytest.mark.asyncio
async def test_model_worker_start_timeout(
    mock_pipeline_config: PipelineConfig,
) -> None:
    """Tests raising in the model worker task."""
    settings = Settings(mw_timeout_s=0.1)

    with pytest.raises(
        TimeoutError, match="Model Worker failed to become ready"
    ):
        async with start_model_worker(
            MockSlowTokenGenerator,
            mock_pipeline_config,
            settings=settings,
            metric_client=NoopClient(),
            model_worker_interface=ZmqModelWorkerInterface(
                PipelineTask.TEXT_GENERATION, context_type=TextContext
            ),
        ):
            pass


class MockTokenizer(PipelineTokenizer):  # type: ignore
    @property
    def eos(self) -> int:
        return 0

    @property
    def expects_content_wrapping(self) -> bool:
        return False

    async def new_context(self, req: Any) -> Any:
        return None

    async def encode(self, text: str, spectok: bool) -> list[int]:
        return []

    async def decode(self, encoded: Any, **kwargs) -> str:
        return ""


@pytest.mark.asyncio
async def test_lifespan_propagates_worker_exception(
    mock_pipeline_config: PipelineConfig,
) -> None:
    """Tests raising in the model worker task."""
    settings = Settings()
    serving_settings = api_server.ServingTokenGeneratorSettings(
        model_factory=MockInvalidTokenGenerator,
        pipeline_config=mock_pipeline_config,
        tokenizer=MockTokenizer(),
    )

    # The MockTokenGenerator crashes the remote subprocess
    # then ProcessMonitor checks throw TimeoutError here
    with pytest.raises(ValueError, match="CRASH TEST DUMMY"):
        async with api_server.lifespan(
            FastAPI(),
            settings,
            serving_settings,
        ):
            pass
