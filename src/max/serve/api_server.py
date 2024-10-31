# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""MAX serving in Python prototype. Main API server thing."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any, Mapping, Optional

from max.serve.telemetry.logger import configureLogging

console_level: int = logging.INFO
file_path: str = ""
file_level: Optional[int] = None
otlp_level: Optional[int] = logging.INFO
if "MAX_SERVE_LOGS_CONSOLE_LEVEL" in os.environ:
    console_level = logging.getLevelName(
        os.environ["MAX_SERVE_LOGS_CONSOLE_LEVEL"]
    )
if "MAX_SERVE_LOGS_OTLP_LEVEL" in os.environ:
    otlp_level = logging.getLevelName(os.environ["MAX_SERVE_LOGS_OTLP_LEVEL"])
if "MAX_SERVE_LOGS_FILE_PATH" in os.environ:
    file_path = os.environ["MAX_SERVE_LOGS_FILE_PATH"]
    file_level = logging.getLevelName(
        os.environ.get("MAX_SERVE_LOGS_FILE_LEVEL", "DEBUG")
    )
if "MAX_SERVE_DISABLE_TELEMETRY" in os.environ:
    otlp_level = None
configureLogging(console_level, file_path, file_level, otlp_level)

from fastapi import FastAPI
from max.serve.config import APIType, Settings, api_prefix
from max.serve.debug import DebugSettings, register_debug
from max.serve.pipelines.deps import (
    BatchedTokenGeneratorState,
    all_pipeline_states,
)
from max.serve.pipelines.model_worker import start_model_worker
from max.serve.request import register_request
from max.serve.router import kserve_routes, openai_routes
from prometheus_client import (
    CollectorRegistry,
    make_asgi_app,
    multiprocess,
    disable_created_metrics,
)
from pydantic_settings import CliSettingsSource
from uvicorn import Config, Server

ROUTES = {
    APIType.KSERVE: kserve_routes,
    APIType.OPENAI: openai_routes,
}

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(
    app: FastAPI,
):
    async with AsyncExitStack() as stack:
        contexts = []
        model_factories = {}
        for name, state in all_pipeline_states().items():
            model_factories[name] = state.model_factory
            if isinstance(state, BatchedTokenGeneratorState):
                contexts.append(state.batched_generator)

        await asyncio.gather(*map(stack.enter_async_context, contexts))
        async with start_model_worker(model_factories):
            yield


def fastapi_app(
    settings: Settings,
    debug_settings: DebugSettings,
    pipeline_states_override: Mapping[str, Any] = {},
) -> FastAPI:
    pipeline_states = all_pipeline_states()
    if pipeline_states_override:
        pipeline_states.clear()
        pipeline_states.update(pipeline_states_override)

    app = FastAPI(lifespan=lifespan)
    for api_type in settings.api_types:
        app.include_router(
            ROUTES[api_type].router, prefix=api_prefix(settings, api_type)
        )

    app.mount("/metrics", make_metrics_app())

    request_limiter: Optional[asyncio.BoundedSemaphore] = None
    if settings.request_limit > 0:
        request_limiter = asyncio.BoundedSemaphore(settings.request_limit)
        logger.info("Configured request limiter to %d", settings.request_limit)
    app.state.request_limiter = request_limiter

    app.state.settings = settings
    register_request(app)

    app.state.debug_settings = debug_settings
    register_debug(app, debug_settings)
    return app


def fastapi_config(app: FastAPI) -> Config:
    config = Config(app=app, host="0.0.0.0", log_config=None)
    for route in app.routes:
        logger.info("Route enabled : %s", route)
    return config


def parse_settings(parser: argparse.ArgumentParser) -> Settings:
    cli_settings = CliSettingsSource(Settings, root_parser=parser)  # type: ignore
    return Settings(_cli_settings_source=cli_settings(args=True))  # type: ignore


def parse_debug_settings(parser: argparse.ArgumentParser) -> DebugSettings:
    cli_settings = CliSettingsSource(DebugSettings, root_parser=parser)  # type: ignore
    return DebugSettings(_cli_settings_source=cli_settings(args=True))  # type: ignore


def make_metrics_app():
    disable_created_metrics()
    # if PROMETHEUS_MULTIPROC_DIR is set we use multiprocess setup
    if os.getenv("PROMETHEUS_MULTIPROC_DIR") is not None:
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        return make_asgi_app(registry=registry)
    else:
        return make_asgi_app()


async def main() -> None:
    parser = argparse.ArgumentParser()
    app = fastapi_app(parse_settings(parser), parse_debug_settings(parser))  # type: ignore
    config = fastapi_config(app=app)
    server = Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
