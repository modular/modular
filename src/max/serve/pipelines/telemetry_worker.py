# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from max.serve.config import MetricRecordingMethod, Settings
from max.serve.telemetry.asyncio_controller import start_asyncio_consumer
from max.serve.telemetry.metrics import (
    MetricClient,
    NoopClient,
    SyncClient,
)
from max.serve.telemetry.process_controller import start_process_consumer


@asynccontextmanager
async def start_telemetry_consumer(
    settings: Settings,
) -> AsyncGenerator[MetricClient, None]:
    method = settings.metric_recording
    if method == MetricRecordingMethod.NOOP:
        yield NoopClient()

    elif method == MetricRecordingMethod.SYNC:
        yield SyncClient(settings)

    elif method == MetricRecordingMethod.ASYNCIO:
        async with start_asyncio_consumer(settings) as controller:
            yield controller

    elif method == MetricRecordingMethod.PROCESS:
        async with start_process_consumer(settings) as controller:
            yield controller.Client(settings)

    else:
        raise Exception(f"Unrecognized metric_recording: {method}")
