# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# Unit tests for model_worker
from __future__ import annotations

import time
from dataclasses import dataclass
from unittest.mock import Mock

import pytest
from max.interfaces import (
    GenerationStatus,
    Pipeline,
    PipelineTask,
    RequestID,
    TextGenerationInputs,
    TextGenerationOutput,
)
from max.pipelines.lib import PipelineConfig
from max.serve.config import Settings
from max.serve.pipelines.echo_gen import EchoTokenGenerator
from max.serve.pipelines.model_worker import start_model_worker
from max.serve.telemetry.metrics import NoopClient


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
            pipeline_task=PipelineTask.TEXT_GENERATION,
        ):
            raise AssertionError


class MockInvalidTokenGenerator(
    Pipeline[TextGenerationInputs[MockContext], TextGenerationOutput]
):
    ERROR_MESSAGE = "I am invalid"

    def __init__(self) -> None:
        raise ValueError(MockInvalidTokenGenerator.ERROR_MESSAGE)

    def execute(
        self, inputs: TextGenerationInputs[MockContext]
    ) -> dict[RequestID, TextGenerationOutput]:
        raise ValueError()

    def release(self, request_id: RequestID) -> None:
        pass


@pytest.mark.skip("RESTORE-THIS")
@pytest.mark.asyncio
async def test_model_worker_propagates_construction_exception(
    mock_pipeline_config: PipelineConfig,
) -> None:
    """Tests raising in the model worker task."""
    settings = Settings()

    with pytest.raises(
        ValueError, match=MockInvalidTokenGenerator.ERROR_MESSAGE
    ):
        async with start_model_worker(
            MockInvalidTokenGenerator,
            mock_pipeline_config,
            settings=settings,
            metric_client=NoopClient(),
            pipeline_task=PipelineTask.TEXT_GENERATION,
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
            pipeline_task=PipelineTask.TEXT_GENERATION,
        ):
            pass
