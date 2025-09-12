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
from typing import Any

from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
from max.interfaces import (
    PipelinesFactory,
    PipelineTask,
    PipelineTokenizer,
    RequestID,
)
from max.pipelines.core import get_request_payload_from_pipeline_task
from max.pipelines.lib import PipelineConfig
from max.serve.config import APIType, MetricRecordingMethod, Settings
from max.serve.pipelines.llm import (
    AudioGeneratorPipeline,
    TokenGeneratorPipeline,
)
from max.serve.pipelines.model_worker import start_model_worker
from max.serve.pipelines.telemetry_worker import start_telemetry_consumer
from max.serve.queue.lora_queue import LoRAQueue
from max.serve.queue.zmq_queue import (
    ZmqPullSocket,
    ZmqPushSocket,
    create_zmq_push_pull_queues,
)
from max.serve.recordreplay.jsonl import JSONLFileRecorder
from max.serve.recordreplay.middleware import RecorderMiddleware
from max.serve.request import register_request
from max.serve.router import kserve_routes, openai_routes, sagemaker_routes
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
    model_factory: PipelinesFactory
    pipeline_config: PipelineConfig
    tokenizer: PipelineTokenizer[Any, Any, Any]
    pipeline_task: PipelineTask = PipelineTask.TEXT_GENERATION


@asynccontextmanager
async def lifespan(
    app: FastAPI,
    settings: Settings,
    serving_settings: ServingTokenGeneratorSettings,
):
    try:
        if not settings.disable_telemetry:
            send_telemetry_log(
                serving_settings.pipeline_config.model_config.model_name
            )
    except Exception as e:
        logger.warning("Failed to send telemetry log: %s", e)

    if settings.offline_inference:
        raise ValueError(
            "It is not valid to start the API Server if the server is in offline inference mode"
        )

    logger.info("Starting server...")
    try:
        async with AsyncExitStack() as exit_stack:
            # start telemetry worker and configure Metrics to use it
            metric_client = await exit_stack.enter_async_context(
                start_telemetry_consumer(settings)
            )
            METRICS.configure(client=metric_client)

            request_push_queue, request_pull_queue = (
                create_zmq_push_pull_queues(
                    payload_type=get_request_payload_from_pipeline_task(
                        serving_settings.pipeline_task
                    ),
                )
            )

            response_push_queue, response_pull_queue = (
                create_zmq_push_pull_queues(  # type: ignore
                    payload_type=serving_settings.pipeline_task.output_type,
                )
            )

            cancel_pull_queue: ZmqPullSocket[list[RequestID]]
            cancel_push_queue: ZmqPushSocket[list[RequestID]]
            cancel_push_queue, cancel_pull_queue = create_zmq_push_pull_queues(
                payload_type=list[RequestID]
            )

            # start model worker
            worker_monitor = await exit_stack.enter_async_context(
                start_model_worker(
                    serving_settings.model_factory,
                    serving_settings.pipeline_config,
                    settings,
                    metric_client,
                    request_queue=request_pull_queue,
                    response_queue=response_push_queue,
                    cancel_queue=cancel_pull_queue,
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

            METRICS.pipeline_load(
                serving_settings.pipeline_config.model_config.model_name
            )
            pipeline: TokenGeneratorPipeline | AudioGeneratorPipeline[Any]
            if serving_settings.pipeline_task in (
                PipelineTask.TEXT_GENERATION,
                PipelineTask.EMBEDDINGS_GENERATION,
            ):
                pipeline = TokenGeneratorPipeline(
                    model_name=serving_settings.pipeline_config.model_config.model_name,
                    tokenizer=serving_settings.tokenizer,
                    lora_queue=lora_queue,
                    request_queue=request_push_queue,
                    response_queue=response_pull_queue,
                    cancel_queue=cancel_push_queue,
                    worker_monitor=worker_monitor,
                )
            elif (
                serving_settings.pipeline_task == PipelineTask.AUDIO_GENERATION
            ):
                pipeline = AudioGeneratorPipeline(
                    model_name=serving_settings.pipeline_config.model_config.model_name,
                    tokenizer=serving_settings.tokenizer,
                    lora_queue=lora_queue,
                    request_queue=request_push_queue,
                    response_queue=response_pull_queue,
                    cancel_queue=cancel_push_queue,
                    worker_monitor=worker_monitor,
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
