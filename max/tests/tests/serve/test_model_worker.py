# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# Unit tests for model_worker

import time

import pytest
from max.interfaces import TokenGenerator
from max.serve.config import Settings
from max.serve.kvcache_agent.dispatcher_factory import DispatcherFactory
from max.serve.kvcache_agent.dispatcher_transport import TransportMessage
from max.serve.pipelines.echo_gen import EchoTokenGenerator
from max.serve.pipelines.model_worker import start_model_worker
from max.serve.scheduler import (
    TokenGeneratorSchedulerConfig,
)
from max.serve.telemetry.metrics import NoopClient


@pytest.mark.asyncio
async def test_model_worker_propagates_exception() -> None:
    """Tests raising in the model worker context manager."""
    settings = Settings()
    dispatcher_factory = DispatcherFactory[str](
        settings.dispatcher_config, transport_payload_type=TransportMessage[str]
    )

    with pytest.raises(AssertionError):
        async with start_model_worker(
            EchoTokenGenerator,
            TokenGeneratorSchedulerConfig.continuous_heterogenous(
                tg_batch_size=1, ce_batch_size=1
            ),
            settings=settings,
            metric_client=NoopClient(),
            dispatcher_factory=dispatcher_factory,
        ):
            raise AssertionError


class MockInvalidTokenGenerator(TokenGenerator[str]):
    ERROR_MESSAGE = "I am invalid"

    def __init__(self) -> None:
        raise ValueError(MockInvalidTokenGenerator.ERROR_MESSAGE)

    def next_token(self, batch: dict[str, str]) -> dict[str, str]:  # type: ignore
        raise ValueError()

    def release(self, context: str) -> None:
        pass


@pytest.mark.skip("RESTORE-THIS")
@pytest.mark.asyncio
async def test_model_worker_propagates_construction_exception() -> None:
    """Tests raising in the model worker task."""
    settings = Settings()
    dispatcher_factory = DispatcherFactory[str](
        settings.dispatcher_config, transport_payload_type=TransportMessage[str]
    )

    with pytest.raises(
        ValueError, match=MockInvalidTokenGenerator.ERROR_MESSAGE
    ):
        async with start_model_worker(
            MockInvalidTokenGenerator,
            TokenGeneratorSchedulerConfig.continuous_heterogenous(
                tg_batch_size=1, ce_batch_size=1
            ),
            settings=settings,
            metric_client=NoopClient(),
            dispatcher_factory=dispatcher_factory,
        ):
            pass


class MockSlowTokenGenerator(TokenGenerator[str]):
    def __init__(self) -> None:
        time.sleep(0.2)

    def next_token(self, batch: dict[str, str]) -> dict[str, str]:  # type: ignore
        raise ValueError()

    def release(self, context: str) -> None:
        pass


@pytest.mark.asyncio
async def test_model_worker_start_timeout() -> None:
    """Tests raising in the model worker task."""
    settings = Settings(MAX_SERVE_MW_TIMEOUT=0.1)
    dispatcher_factory = DispatcherFactory[str](
        settings.dispatcher_config, transport_payload_type=TransportMessage[str]
    )

    with pytest.raises(TimeoutError):
        async with start_model_worker(
            MockSlowTokenGenerator,
            TokenGeneratorSchedulerConfig.continuous_heterogenous(
                tg_batch_size=1, ce_batch_size=1
            ),
            settings=settings,
            metric_client=NoopClient(),
            dispatcher_factory=dispatcher_factory,
        ):
            pass
