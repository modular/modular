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
from functools import partial
from typing import AsyncContextManager, Optional, Sequence

from max.serve.telemetry.logger import configureLogging

console_level: int = logging.INFO
file_path: str = ""
file_level: Optional[int] = None
otlp_level: Optional[int] = logging.INFO
if "MAX_SERVE_LOGS_CONSOLE_LEVEL" in os.environ:
    console_level = int(os.environ["MAX_SERVE_LOGS_CONSOLE_LEVEL"])
if "MAX_SERVE_LOGS_OTLP_LEVEL" in os.environ:
    otlp_level = int(os.environ["MAX_SERVE_LOGS_OTLP_LEVEL"])
if "MAX_SERVE_LOGS_FILE_PATH" in os.environ:
    file_path = os.environ["MAX_SERVE_LOGS_FILE_PATH"]
    file_level = int(os.environ.get("MAX_SERVE_LOGS_FILE_LEVEL", logging.DEBUG))
if "MAX_SERVE_DISABLE_TELEMETRY" in os.environ:
    otlp_level = None
configureLogging(console_level, file_path, file_level, otlp_level)

from fastapi import FastAPI
from max.serve.config import APIType, Settings, api_prefix
from max.serve.debug import DebugSettings, register_debug
from max.serve.request import register_request
from max.serve.router import kserve_routes, openai_routes
from pydantic_settings import CliSettingsSource
from uvicorn import Config, Server

ROUTES = {
    APIType.KSERVE: kserve_routes,
    APIType.OPENAI: openai_routes,
}

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(pipelines: Sequence[AsyncContextManager], app: FastAPI):
    async with AsyncExitStack() as stack:
        await asyncio.gather(*(stack.enter_async_context(p) for p in pipelines))
        logger.info("Pipelines loaded!")
        yield
        logger.info("Pipelines unloaded!")


def fastapi_app(
    settings: Settings,
    debug_settings: DebugSettings,
    pipelines: Sequence[AsyncContextManager],
) -> FastAPI:
    app = FastAPI(lifespan=partial(lifespan, pipelines))
    for api_type in settings.api_types:
        app.include_router(
            ROUTES[api_type].router, prefix=api_prefix(settings, api_type)
        )

    register_debug(app, debug_settings)
    register_request(app)
    app.state.settings = settings
    app.state.debug_settings = debug_settings
    return app


def fastapi_config(app: FastAPI) -> Config:
    config = Config(app=app, host="0.0.0.0", log_config=None)
    return config


def parse_settings(parser: argparse.ArgumentParser) -> Settings:
    cli_settings = CliSettingsSource(Settings, root_parser=parser)  # type: ignore
    return Settings(_cli_settings_source=cli_settings(args=True))  # type: ignore


def parse_debug_settings(parser: argparse.ArgumentParser) -> DebugSettings:
    cli_settings = CliSettingsSource(DebugSettings, root_parser=parser)  # type: ignore
    return DebugSettings(_cli_settings_source=cli_settings(args=True))  # type: ignore


async def main() -> None:
    parser = argparse.ArgumentParser()
    app = fastapi_app(parse_settings(parser), parse_debug_settings(parser))  # type: ignore
    config = fastapi_config(app=app)
    server = Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
