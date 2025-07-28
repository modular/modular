# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# Unit tests for model_worker

import time

import pytest
from max.interfaces import PipelineTask, RequestID, TokenGenerator
from max.pipelines.lib import PipelineConfig
from max.serve.config import Settings
from max.serve.pipelines.echo_gen import EchoTokenGenerator
from max.serve.pipelines.model_worker import start_model_worker
from max.serve.telemetry.metrics import NoopClient


class MockPipelineConfig(PipelineConfig):
    def __init__(self):
        self.max_batch_size = 1


@pytest.mark.asyncio
async def test_model_worker_propagates_exception() -> None:
    """Tests raising in the model worker context manager."""
    settings = Settings()

    with pytest.raises(AssertionError):
        async with start_model_worker(
            EchoTokenGenerator,
            MockPipelineConfig(),
            settings=settings,
            metric_client=NoopClient(),
            pipeline_task=PipelineTask.TEXT_GENERATION,
        ):
            raise AssertionError


class MockInvalidTokenGenerator(TokenGenerator[str]):
    ERROR_MESSAGE = "I am invalid"

    def __init__(self) -> None:
        raise ValueError(MockInvalidTokenGenerator.ERROR_MESSAGE)

    def next_token(self, batch: dict[str, str]) -> dict[str, str]:  # type: ignore
        raise ValueError()

    def release(self, request_id: RequestID) -> None:
        pass


@pytest.mark.skip("RESTORE-THIS")
@pytest.mark.asyncio
async def test_model_worker_propagates_construction_exception() -> None:
    """Tests raising in the model worker task."""
    settings = Settings()

    with pytest.raises(
        ValueError, match=MockInvalidTokenGenerator.ERROR_MESSAGE
    ):
        async with start_model_worker(
            MockInvalidTokenGenerator,
            MockPipelineConfig(),
            settings=settings,
            metric_client=NoopClient(),
            pipeline_task=PipelineTask.TEXT_GENERATION,
        ):
            pass


class MockSlowTokenGenerator(TokenGenerator[str]):
    def __init__(self) -> None:
        time.sleep(0.2)

    def next_token(self, batch: dict[str, str]) -> dict[str, str]:  # type: ignore
        raise ValueError()

    def release(self, request_id: RequestID) -> None:
        pass


@pytest.mark.asyncio
async def test_model_worker_start_timeout() -> None:
    """Tests raising in the model worker task."""
    settings = Settings(MAX_SERVE_MW_TIMEOUT=0.1)

    with pytest.raises(TimeoutError):
        async with start_model_worker(
            MockSlowTokenGenerator,
            MockPipelineConfig(),
            settings=settings,
            metric_client=NoopClient(),
            pipeline_task=PipelineTask.TEXT_GENERATION,
        ):
            pass
