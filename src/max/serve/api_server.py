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
from dataclasses import dataclass
from typing import Union

from max.serve.telemetry.logger import configureLogging
from max.serve.telemetry.metrics import METRICS

console_level: Union[int, str] = logging.INFO
file_path: str = ""
file_level: Union[int, str, None] = None
otlp_level: Union[int, str, None] = None
metrics_egress_enabled = True
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
    metrics_egress_enabled = False
configureLogging(console_level, file_path, file_level, otlp_level)
METRICS.configure(metrics_egress_enabled)


from contextlib import asynccontextmanager
from functools import partial

from fastapi import FastAPI
from max.pipelines.interfaces import PipelineTokenizer, TokenGeneratorFactory
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
from max.serve.request import register_request
from max.serve.router import kserve_routes, openai_routes
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from prometheus_client import disable_created_metrics, make_asgi_app
from pydantic_settings import CliSettingsSource
from uvicorn import Config, Server

ROUTES = {
    APIType.KSERVE: kserve_routes,
    APIType.OPENAI: openai_routes,
}

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ServingTokenGeneratorSettings:
    model_name: str
    model_factory: TokenGeneratorFactory
    pipeline_config: TokenGeneratorPipelineConfig
    tokenizer: PipelineTokenizer


@asynccontextmanager
async def lifespan(
    app: FastAPI, serving_settings: ServingTokenGeneratorSettings
):
    async with start_model_worker(
        serving_settings.model_factory, serving_settings.pipeline_config
    ) as engine_queue:
        pipeline: TokenGeneratorPipeline = TokenGeneratorPipeline(
            model_name=serving_settings.model_name,
            tokenizer=serving_settings.tokenizer,
            engine_queue=engine_queue,
        )
        app.state.pipeline = pipeline
        async with pipeline:
            yield


def fastapi_app(
    settings: Settings,
    debug_settings: DebugSettings,
    serving_settings: ServingTokenGeneratorSettings,
) -> FastAPI:
    app = FastAPI(lifespan=partial(lifespan, serving_settings=serving_settings))

    app.mount("/metrics", make_metrics_app())

    for api_type in settings.api_types:
        app.include_router(
            ROUTES[api_type].router, prefix=api_prefix(settings, api_type)
        )

    app.state.settings = settings
    register_request(app)

    app.state.debug_settings = debug_settings
    register_debug(app, debug_settings)

    # Instrument application with traces
    FastAPIInstrumentor.instrument_app(app, excluded_urls="metrics/.*")
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
    asyncio.run(main())
