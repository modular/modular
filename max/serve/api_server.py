# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""MAX serving in Python prototype. Main API server thing."""

from __future__ import annotations

import logging
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from functools import partial

from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
from max.interfaces import PipelinesFactory, PipelineTask, PipelineTokenizer
from max.pipelines.lib import PipelineConfig
from max.serve.config import APIType, MetricRecordingMethod, Settings
from max.serve.kvcache_agent import DispatcherFactory, TransportMessage
from max.serve.pipelines.kvcache_worker import start_kv_cache_service
from max.serve.pipelines.llm import (
    AudioGeneratorPipeline,
    TokenGeneratorPipeline,
)
from max.serve.pipelines.model_worker import start_model_worker
from max.serve.pipelines.telemetry_worker import start_telemetry_consumer
from max.serve.queue.lora_queue import LoRAQueue
from max.serve.recordreplay.jsonl import JSONLFileRecorder
from max.serve.recordreplay.middleware import RecorderMiddleware
from max.serve.request import register_request
from max.serve.router import kserve_routes, openai_routes, sagemaker_routes
from max.serve.scheduler.base import PayloadType
from max.serve.telemetry.common import send_telemetry_log
from max.serve.telemetry.metrics import METRICS
from uvicorn import Config

ROUTES = {
    APIType.KSERVE: kserve_routes,
    APIType.OPENAI: openai_routes,
    APIType.SAGEMAKER: sagemaker_routes,
}

logger = logging.getLogger("max.serve")


@dataclass(frozen=True)
class ServingTokenGeneratorSettings:
    # Pipeline config
    model_name: str
    model_factory: PipelinesFactory
    pipeline_config: PipelineConfig
    tokenizer: PipelineTokenizer
    pipeline_task: PipelineTask = PipelineTask.TEXT_GENERATION


@asynccontextmanager
async def lifespan(
    app: FastAPI,
    settings: Settings,
    serving_settings: ServingTokenGeneratorSettings,
):
    try:
        if not settings.disable_telemetry:
            send_telemetry_log(serving_settings.model_name)
    except Exception as e:
        logger.warning("Failed to send telemetry log: %s", e)

    if settings.offline_inference:
        raise ValueError(
            "It is not valid to start the API Server if the server is in offline inference mode"
        )

    dispatcher_factory = None
    if serving_settings.pipeline_config.pipeline_role.uses_dispatch_service:
        dispatcher_factory = DispatcherFactory[PayloadType](
            settings.dispatcher_config,
            transport_payload_type=TransportMessage[PayloadType],
        )

    logger.info("Starting server...")
    try:
        async with AsyncExitStack() as exit_stack:
            if dispatcher_factory is not None:
                logger.info("Starting Dispatch Service...")
                await exit_stack.enter_async_context(
                    start_kv_cache_service(settings, dispatcher_factory)
                )
                logger.info("Dispatch Service started.")

            # start telemetry worker and configure Metrics to use it
            metric_client = await exit_stack.enter_async_context(
                start_telemetry_consumer(settings)
            )

            METRICS.configure(client=metric_client)
            # start model worker
            engine_queue = await exit_stack.enter_async_context(
                start_model_worker(
                    serving_settings.model_factory,
                    serving_settings.pipeline_config,
                    settings,
                    metric_client,
                    serving_settings.pipeline_task,
                    dispatcher_factory,
                )
            )

            lora_queue: LoRAQueue | None = (
                LoRAQueue(
                    serving_settings.pipeline_config.lora_config.lora_request_endpoint,
                    serving_settings.pipeline_config.lora_config.lora_response_endpoint,
                )
                if serving_settings.pipeline_config.lora_config
                else None
            )

            METRICS.pipeline_load(serving_settings.model_name)
            pipeline: TokenGeneratorPipeline | AudioGeneratorPipeline
            if serving_settings.pipeline_task in (
                PipelineTask.TEXT_GENERATION,
                PipelineTask.EMBEDDINGS_GENERATION,
            ):
                pipeline = TokenGeneratorPipeline(
                    model_name=serving_settings.model_name,
                    tokenizer=serving_settings.tokenizer,
                    engine_queue=engine_queue,
                    lora_queue=lora_queue,
                )
            elif (
                serving_settings.pipeline_task == PipelineTask.AUDIO_GENERATION
            ):
                pipeline = AudioGeneratorPipeline(
                    model_name=serving_settings.model_name,
                    tokenizer=serving_settings.tokenizer,
                    engine_queue=engine_queue,
                    lora_queue=lora_queue,
                )
            else:
                raise ValueError(
                    f"Unsupported pipeline task: {serving_settings.pipeline_task}"
                )

            app.state.pipeline = pipeline

            await exit_stack.enter_async_context(pipeline)
            logger.info(
                f"\n\n**********\nServer ready on http://{settings.host}:{settings.port} (Press CTRL+C to quit)\n**********\n"
            )
            yield
    # TODO: Will we ever get here? KeyboardInterrupt is handled in the serve.py entrypoint.
    except KeyboardInterrupt as e:
        # Exit gracefully if user used Ctrl+C
        logger.info("Workers have shut down successfully (keyboard interrupt)")
    except Exception as e:
        logger.exception("Error occurred in model worker. %s", e)
    finally:
        logger.debug("start_model_worker has completed")


def version():
    """Returns max-serve version information."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        package_version = version("max")
        return JSONResponse({"version": package_version})
    except PackageNotFoundError:
        logger.debug("Version could not be reported for max.")
        return JSONResponse({"version": "unknown"})


async def health() -> Response:
    """Health check, tools like lm-eval use this to check for readiness."""
    return Response(status_code=200)


def make_metrics_app():
    from prometheus_client import disable_created_metrics, make_asgi_app

    disable_created_metrics()
    return make_asgi_app()


def fastapi_app(
    settings: Settings,
    serving_settings: ServingTokenGeneratorSettings,
) -> FastAPI:
    logger.info(f"Settings: {settings}")

    app = FastAPI(
        title="MAX Serve",
        lifespan=partial(
            lifespan, settings=settings, serving_settings=serving_settings
        ),
    )

    if settings.transaction_recording_file is not None:
        transaction_recording_file = settings.transaction_recording_file
        app.add_middleware(
            RecorderMiddleware,  # type: ignore
            recorder_factory=(
                lambda: JSONLFileRecorder(transaction_recording_file)
            ),
            include_responses=settings.transaction_recording_include_responses,
        )

    if (
        not settings.disable_telemetry
        and settings.metric_recording == MetricRecordingMethod.ASYNCIO
    ):
        app.mount("/metrics", make_metrics_app())

    app.add_api_route("/version", version)
    app.add_api_route("/health", health)

    for api_type in settings.api_types:
        app.include_router(ROUTES[api_type].router)

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
