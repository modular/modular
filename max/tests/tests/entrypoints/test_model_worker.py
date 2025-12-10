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
from max.pipelines.lib import (
    MAXModelConfig,
    PipelineConfig,
)
from max.serve import api_server
from max.serve.config import Settings
from max.serve.pipelines.echo_gen import EchoTokenGenerator
from max.serve.pipelines.model_worker import start_model_worker
from max.serve.scheduler.queues import SchedulerZmqConfigs
from max.serve.telemetry.metrics import NoopClient


class MockModelConfig(MAXModelConfig):
    def __init__(self):
        self.served_model_name = "echo"


class MockPipelineConfig(PipelineConfig):
    def __init__(self):
        self.max_batch_size = 1
        self._model_config = MockModelConfig()


@pytest.fixture
def mock_pipeline_config() -> PipelineConfig:
    return MockPipelineConfig()


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

    with pytest.raises(AssertionError):
        async with start_model_worker(
            EchoTokenGenerator,
            mock_pipeline_config,
            settings=settings,
            metric_client=NoopClient(),
            scheduler_zmq_configs=SchedulerZmqConfigs(
                PipelineTask.TEXT_GENERATION
            ),
        ):
            raise AssertionError


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
    with pytest.raises(TimeoutError, match="Worker died"):
        async with start_model_worker(
            MockInvalidTokenGenerator,
            mock_pipeline_config,
            settings=settings,
            scheduler_zmq_configs=SchedulerZmqConfigs(
                PipelineTask.TEXT_GENERATION
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
    settings = Settings(MAX_SERVE_MW_TIMEOUT=0.1)

    with pytest.raises(TimeoutError):
        async with start_model_worker(
            MockSlowTokenGenerator,
            mock_pipeline_config,
            settings=settings,
            metric_client=NoopClient(),
            scheduler_zmq_configs=SchedulerZmqConfigs(
                PipelineTask.TEXT_GENERATION
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
    with pytest.raises(TimeoutError, match="Worker died"):
        async with api_server.lifespan(
            FastAPI(title="Crash Test Dummy"),
            settings,
            serving_settings,
        ):
            pass
