# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# Unit tests for model_worker

import time

import pytest
from max.pipelines.interfaces import TokenGenerator
from max.serve.pipelines.deps import echo_generator_pipeline
from max.serve.pipelines.echo_gen import EchoTokenGenerator
from max.serve.pipelines.model_worker import start_model_worker


@pytest.mark.asyncio
async def test_model_worker_propagates_exception() -> None:
    """Tests raising in the model worker context manager."""
    with pytest.raises(AssertionError):
        async with start_model_worker(
            {"test": EchoTokenGenerator}, {"test": echo_generator_pipeline()}
        ):
            raise AssertionError


class MockInvalidTokenGenerator(TokenGenerator[str]):
    ERROR_MESSAGE = "I am invalid"

    def __init__(self):
        raise ValueError(MockInvalidTokenGenerator.ERROR_MESSAGE)

    def next_token(self, batch: dict[str, str]) -> dict[str, str]:
        raise ValueError()

    def release(self, context: str):
        pass


@pytest.mark.asyncio
async def test_model_worker_propagates_construction_exception() -> None:
    """Tests raising in the model worker task."""
    with pytest.raises(
        ValueError, match=MockInvalidTokenGenerator.ERROR_MESSAGE
    ):
        async with start_model_worker(
            {"test": MockInvalidTokenGenerator},
            {"test": echo_generator_pipeline()},
        ):
            pass


class MockSlowTokenGenerator(TokenGenerator[str]):
    def __init__(self):
        time.sleep(0.2)

    def next_token(self, batch: dict[str, str]) -> dict[str, str]:
        raise ValueError()

    def release(self, context: str):
        pass


@pytest.mark.asyncio
async def test_model_worker_start_timeout() -> None:
    """Tests raising in the model worker task."""
    with pytest.raises(TimeoutError):
        async with start_model_worker(
            {"test": MockSlowTokenGenerator},
            {"test": echo_generator_pipeline()},
            timeout_secs=0.1,
        ):
            pass
