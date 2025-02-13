# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""MAX serving in Python prototype. Main API server thing."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import partial

import uvloop
from fastapi import FastAPI
from max.loggers import get_logger
from max.pipelines import PipelinesFactory, PipelineTokenizer
from max.serve.config import (
    APIType,
    Settings,
    api_prefix,
)
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
from max.serve.telemetry.metrics import METRICS
from prometheus_client import disable_created_metrics, make_asgi_app
from uvicorn import Config, Server

ROUTES = {
    APIType.KSERVE: kserve_routes,
    APIType.OPENAI: openai_routes,
}

logger = get_logger(__name__)


@dataclass(frozen=True)
class ServingTokenGeneratorSettings:
    # Pipeline config
    model_name: str
    model_factory: PipelinesFactory
    pipeline_config: TokenGeneratorPipelineConfig
    tokenizer: PipelineTokenizer


@asynccontextmanager
async def lifespan(
    app: FastAPI,
    server_settings: Settings,
    serving_settings: ServingTokenGeneratorSettings,
):
    await METRICS.configure(server_settings)
    logger.info(
        f"Launching server on http://{server_settings.host}:{server_settings.port}"
    )
    try:
        async with (
            start_telemetry_worker(
                worker_spawn_timeout=server_settings.telemetry_worker_spawn_timeout
            ) as tel_worker,
            start_model_worker(
                serving_settings.model_factory,
                serving_settings.pipeline_config,
                settings=server_settings,
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
                    f"Server ready on http://{server_settings.host}:{server_settings.port} (Press CTRL+C to quit)"
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
    serving_settings: ServingTokenGeneratorSettings,
) -> FastAPI:
    logger.info(f"Settings: {settings}")
    app = FastAPI(
        title="MAX Serve",
        lifespan=partial(
            lifespan,
            server_settings=settings,
            serving_settings=serving_settings,
        ),
    )

    app.mount("/metrics", make_metrics_app())

    for api_type in settings.api_types:
        app.include_router(
            ROUTES[api_type].router, prefix=api_prefix(settings, api_type)
        )

    app.state.settings = settings
    register_request(app)

    return app


def fastapi_config(app: FastAPI, server_settings: Settings) -> Config:
    config = Config(
        app=app,
        log_config=None,
        loop="uvloop",
        host=server_settings.host,
        port=server_settings.port,
    )

    for route in app.routes:
        logger.debug("Route enabled : %s", route)
    return config


def make_metrics_app():
    disable_created_metrics()
    return make_asgi_app()


async def main() -> None:
    server_settings = Settings()

    pipeline_settings = ServingTokenGeneratorSettings(
        model_name="echo",
        model_factory=EchoTokenGenerator,
        pipeline_config=TokenGeneratorPipelineConfig.dynamic_homogenous(
            batch_size=1
        ),
        tokenizer=EchoPipelineTokenizer(),
    )

    app = fastapi_app(server_settings, pipeline_settings)

    config = fastapi_config(app=app, server_settings=server_settings)
    server = Server(config)
    await server.serve()


if __name__ == "__main__":
    uvloop.run(main())
