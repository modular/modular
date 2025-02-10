# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""MAX serving in Python prototype. Main API server thing."""

from __future__ import annotations

from max.serve.telemetry.logger import configureLogging

configureLogging()

import argparse
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import partial

import uvloop
from fastapi import FastAPI
from max.pipelines import PipelinesFactory, PipelineTokenizer
from max.serve.config import APIType, Settings, api_prefix
from max.serve.debug import DebugSettings, register_debug
from max.serve.pipelines.echo_gen import (
    EchoPipelineTokenizer,
    EchoTokenGenerator,
)
from max.serve.pipelines.llm import (
    TokenGeneratorPipeline,
    TokenGeneratorPipelineConfig,
)
from max.serve.pipelines.model_worker import start_model_worker
from max.serve.pipelines.telemetry_worker import start_telemetry_worker
from max.serve.request import register_request
from max.serve.router import kserve_routes, openai_routes
from max.serve.telemetry.metrics import METRICS, configure_metrics
from prometheus_client import disable_created_metrics, make_asgi_app
from pydantic_settings import CliSettingsSource
from uvicorn import Config, Server

ROUTES = {
    APIType.KSERVE: kserve_routes,
    APIType.OPENAI: openai_routes,
}

logger = logging.getLogger("max.serve")


@dataclass(frozen=True)
class ServingTokenGeneratorSettings:
    model_name: str
    model_factory: PipelinesFactory
    pipeline_config: TokenGeneratorPipelineConfig
    tokenizer: PipelineTokenizer


@asynccontextmanager
async def lifespan(
    app: FastAPI, serving_settings: ServingTokenGeneratorSettings
):
    host = app.extra["host"]
    port = app.extra["port"]
    logger.info(f"Launching server on http://{host}:{port}")
    configure_metrics()
    await METRICS.configure()

    try:
        async with (
            start_telemetry_worker() as tel_worker,
            start_model_worker(
                serving_settings.model_factory,
                serving_settings.pipeline_config,
            ) as engine_queue,
        ):
            METRICS.pipeline_load(serving_settings.model_name)
            pipeline: TokenGeneratorPipeline = TokenGeneratorPipeline(
                model_name=serving_settings.model_name,
                tokenizer=serving_settings.tokenizer,
                engine_queue=engine_queue,
            )
            app.state.pipeline = pipeline
            async with pipeline:
                logger.info(
                    f"Server ready on http://{host}:{port} (Press CTRL+C to quit)"
                )
                yield
    except KeyboardInterrupt as e:
        # Exit gracefully if user used Ctrl+C
        logger.info("Workers have shut down successfully (keyboard interrupt)")
    except Exception as e:
        logger.exception("Error occurred in model worker. %s", e)
    finally:
        logger.debug("start_model_worker has completed")
        await METRICS.shutdown()


def fastapi_app(
    settings: Settings,
    debug_settings: DebugSettings,
    serving_settings: ServingTokenGeneratorSettings,
) -> FastAPI:
    host = os.getenv("MAX_SERVE_HOST", "0.0.0.0")
    port = int(os.getenv("MAX_SERVE_PORT", "8000"))
    app = FastAPI(
        title="MAX Serve",
        lifespan=partial(lifespan, serving_settings=serving_settings),
        host=host,
        port=port,
    )

    app.mount("/metrics", make_metrics_app())

    for api_type in settings.api_types:
        app.include_router(
            ROUTES[api_type].router, prefix=api_prefix(settings, api_type)
        )

    app.state.settings = settings
    register_request(app)

    app.state.debug_settings = debug_settings
    register_debug(app, debug_settings)

    return app


def fastapi_config(app: FastAPI) -> Config:
    config = Config(
        app=app,
        host=app.extra["host"],
        port=app.extra["port"],
        log_config=None,
        loop="uvloop",
    )
    for route in app.routes:
        logger.debug("Route enabled : %s", route)
    return config


def parse_settings(parser: argparse.ArgumentParser) -> Settings:
    cli_settings = CliSettingsSource(Settings, root_parser=parser)  # type: ignore
    return Settings(_cli_settings_source=cli_settings(args=True))  # type: ignore


def parse_debug_settings(parser: argparse.ArgumentParser) -> DebugSettings:
    cli_settings = CliSettingsSource(DebugSettings, root_parser=parser)  # type: ignore
    return DebugSettings(_cli_settings_source=cli_settings(args=True))  # type: ignore


def make_metrics_app():
    disable_created_metrics()
    return make_asgi_app()


async def main() -> None:
    parser = argparse.ArgumentParser()
    serving_settings = ServingTokenGeneratorSettings(
        model_name="echo",
        model_factory=EchoTokenGenerator,
        pipeline_config=TokenGeneratorPipelineConfig.dynamic_homogenous(
            batch_size=1
        ),
        tokenizer=EchoPipelineTokenizer(),
    )
    app = fastapi_app(
        parse_settings(parser), parse_debug_settings(parser), serving_settings
    )
    config = fastapi_config(app=app)
    server = Server(config)
    await server.serve()


if __name__ == "__main__":
    uvloop.run(main())
