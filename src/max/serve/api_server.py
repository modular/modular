# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


"""
MAX serving in Python prototype. Main API server thing.
"""

import argparse
import asyncio
import logging
from contextlib import AsyncExitStack, asynccontextmanager
from functools import partial
from typing import AsyncContextManager, Sequence

from fastapi import FastAPI
from max.serve.config import APIType, Settings, api_prefix
from max.serve.pipelines.deps import all_pipelines
from max.serve.router import kserve_routes, openai_routes
from pydantic_settings import CliSettingsSource
from uvicorn import Config, Server

ROUTES = {
    APIType.KSERVE: kserve_routes,
    APIType.OPENAI: openai_routes,
}

logging.basicConfig(
    level=logging.INFO,
    encoding="utf-8",
    format="%(asctime)s - %(levelname)s - %(message)s",
)
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
    pipelines: Sequence[AsyncContextManager] = all_pipelines(),
) -> FastAPI:
    app = FastAPI(lifespan=partial(lifespan, pipelines))
    for api_type in settings.api_types:
        app.include_router(
            ROUTES[api_type].router, prefix=api_prefix(settings, api_type)
        )
    return app


def fastapi_config(app: FastAPI) -> Config:
    config = Config(app=app)
    return config


def parse_settings(parser: argparse.ArgumentParser) -> Settings:
    cli_settings = CliSettingsSource(Settings, root_parser=parser)
    return Settings(_cli_settings_source=cli_settings(args=True))  # type: ignore


async def main() -> None:
    parser = argparse.ArgumentParser()
    app = fastapi_app(parse_settings(parser))
    config = fastapi_config(app=app)
    server = Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
