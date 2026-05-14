# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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
import os
import signal
import socket
import tempfile
from collections.abc import AsyncGenerator, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
from max.interfaces import (
    AudioGenerationOutput,
    BaseContext,
    PipelineOutput,
    PipelinesFactory,
    PipelineTask,
    PipelineTokenizer,
)
from max.pipelines.core import TTSContext
from max.pipelines.lib import (
    PIPELINE_REGISTRY,
    AudioGenerationConfig,
    PipelineConfig,
)
from max.serve.config import APIType, MetricRecordingMethod, Settings
from max.serve.media import GeneratedMediaStore
from max.serve.pipelines.general_handler import GeneralPipelineHandler
from max.serve.pipelines.llm import (
    AudioGeneratorPipeline,
    TokenGeneratorPipeline,
)
from max.serve.pipelines.model_worker import start_model_worker
from max.serve.pipelines.reset_prefix_cache import ResetPrefixCacheFrontend
from max.serve.pipelines.telemetry_worker import start_telemetry_consumer
from max.serve.recordreplay.jsonl import JSONLFileRecorder
from max.serve.recordreplay.middleware import RecorderMiddleware
from max.serve.request import register_request
from max.serve.router import (
    kserve_routes,
    openai_routes,
    openresponses_routes,
    sagemaker_routes,
)
from max.serve.telemetry.common import send_telemetry_log
from max.serve.telemetry.metrics import METRICS
from max.serve.worker_interface import ModelWorkerProxy
from max.serve.worker_interface.lora_queue import LoRAQueue
from max.serve.worker_interface.zmq_interface import ZmqModelWorkerInterface
from uvicorn import Config

ROUTES = {
    APIType.KSERVE: kserve_routes,
    APIType.OPENAI: openai_routes,
    APIType.SAGEMAKER: sagemaker_routes,
    APIType.OPENRESPONSES: openresponses_routes,
}

logger = logging.getLogger("max.serve")


def validate_port_is_free(port: int) -> int:
    # check if port is already in use
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("", port))
            return port
        except OSError as e:
            raise ValueError(
                f"The network port {port} is already in use"
            ) from e


@dataclass(frozen=True)
class ServingTokenGeneratorSettings:
    # Pipeline config
    model_factory: PipelinesFactory  # type: ignore
    pipeline_config: PipelineConfig
    tokenizer: PipelineTokenizer[Any, Any, Any]
    pipeline_task: PipelineTask = PipelineTask.TEXT_GENERATION
    reasoning_parser_name: str | None = None
    temperature: float | None = None
    thinking_temperature: float | None = None


@asynccontextmanager
async def lifespan(
    app: FastAPI,
    settings: Settings,
    serving_settings: ServingTokenGeneratorSettings,
) -> AsyncGenerator[None]:
    try:
        if not settings.disable_telemetry:
            send_telemetry_log(
                serving_settings.pipeline_config.models.model_name
            )
    except Exception as e:
        logger.warning("Failed to send telemetry log: %s", e)

    if settings.offline_inference:
        raise ValueError(
            "It is not valid to start the API Server if the server is in offline inference mode"
        )

    logger.info("Starting server...")

    async with AsyncExitStack() as exit_stack:
        media_root = Path(
            exit_stack.enter_context(
                tempfile.TemporaryDirectory(prefix="max_serve_media_")
            )
        )
        app.state.media_store = GeneratedMediaStore(
            media_root,
            max_storage_bytes=(
                settings.generated_media_storage_mb * 1024 * 1024
            ),
        )

        # start telemetry worker and configure Metrics to use it

        metric_client = await exit_stack.enter_async_context(
            start_telemetry_consumer(settings)
        )
        METRICS.configure(client=metric_client)

        # start model worker

        override_architecture: str | None = None
        if serving_settings.pipeline_task == PipelineTask.AUDIO_GENERATION:
            if isinstance(
                serving_settings.pipeline_config, AudioGenerationConfig
            ):
                override_architecture = (
                    serving_settings.pipeline_config.audio_decoder
                )

        model_worker_interface = ZmqModelWorkerInterface[
            BaseContext, PipelineOutput
        ](
            serving_settings.pipeline_task,
            PIPELINE_REGISTRY.retrieve_context_type(
                serving_settings.pipeline_config,
                override_architecture=override_architecture,
                task=serving_settings.pipeline_task,
            ),
        )
        model_worker = await exit_stack.enter_async_context(
            start_model_worker(
                serving_settings.model_factory,
                serving_settings.pipeline_config,
                settings,
                metric_client,
                model_worker_interface=model_worker_interface,
            )
        )

        lora_queue: LoRAQueue | None = (
            LoRAQueue(
                serving_settings.pipeline_config.runtime.zmq_endpoint_base,
                serving_settings.pipeline_config.lora.lora_paths,
            )
            if serving_settings.pipeline_config.lora
            else None
        )

        METRICS.pipeline_load(
            serving_settings.pipeline_config.models.model_name
        )

        pipeline = {
            PipelineTask.TEXT_GENERATION: lambda: TokenGeneratorPipeline(
                model_name=serving_settings.pipeline_config.models.model_name,
                tokenizer=serving_settings.tokenizer,
                lora_queue=lora_queue,
                model_worker=model_worker,
                reasoning_parser_name=serving_settings.reasoning_parser_name,
            ),
            PipelineTask.EMBEDDINGS_GENERATION: lambda: TokenGeneratorPipeline(
                model_name=serving_settings.pipeline_config.models.model_name,
                tokenizer=serving_settings.tokenizer,
                lora_queue=lora_queue,
                model_worker=model_worker,
            ),
            PipelineTask.AUDIO_GENERATION: lambda: AudioGeneratorPipeline(
                model_name=serving_settings.pipeline_config.models.model_name,
                tokenizer=serving_settings.tokenizer,
                lora_queue=lora_queue,
                model_worker=cast(
                    ModelWorkerProxy[TTSContext, AudioGenerationOutput],
                    model_worker,
                ),
            ),
            # Pixel generation uses only the OpenResponses API via GeneralPipelineHandler
            PipelineTask.PIXEL_GENERATION: lambda: GeneralPipelineHandler(
                model_name=serving_settings.pipeline_config.models.model_name,
                tokenizer=serving_settings.tokenizer,
                model_worker=model_worker,
                lora_queue=lora_queue,
            ),
        }[serving_settings.pipeline_task]()

        # Store pipeline (may be GeneralPipelineHandler or modality-specific wrapper)
        # Legacy API routes (OpenAI, KServe, SageMaker) use modality-specific wrappers
        # OpenResponses API uses GeneralPipelineHandler
        app.state.pipeline = pipeline
        app.state.pipeline_config = serving_settings.pipeline_config

        # Also store as handler for OpenResponses API route compatibility
        # For pixel generation, this is the same as pipeline
        # For other tasks, we also create a separate handler instance
        if serving_settings.pipeline_task == PipelineTask.PIXEL_GENERATION:
            app.state.handler = pipeline
        else:
            app.state.handler = GeneralPipelineHandler(
                model_name=serving_settings.pipeline_config.models.model_name,
                tokenizer=serving_settings.tokenizer,
                model_worker=model_worker,
                lora_queue=lora_queue,
            )

        logger.info(
            f"\n\n{'*' * 80}\n\n"
            f"{f'🚀 Server ready on http://{settings.host}:{settings.port} (Press CTRL+C to quit)'.center(80)}\n\n"
            f"{'*' * 80}\n"
        )

        yield

        logger.info("Shutting down workers...")


def version() -> JSONResponse:
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


def make_metrics_app() -> Callable[..., Any]:
    from prometheus_client import disable_created_metrics, make_asgi_app

    disable_created_metrics()
    return make_asgi_app()


def fastapi_app(
    settings: Settings,
    serving_settings: ServingTokenGeneratorSettings,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan_wrap(app: FastAPI) -> AsyncGenerator[None, None]:
        try:
            async with lifespan(app, settings, serving_settings):
                yield
        except BaseException as e:
            # Worker already logs the detailed traceback, so we use
            # error (not exception) here to avoid duplicating it.
            logger.error("Worker exception, shutting down: %s", e)
            # Caught by uvicorn to shutdown the server
            os.kill(os.getpid(), signal.SIGINT)
            # After first SIGINT uvicorn waits for pending requests to complete
            # In our case, they would hang forever due to waiting on worker queues
            # Uvicorn listens for a second SIGINT to cancel this waiting phase and
            # close all remaining connections with "Internal Server Error" status
            os.kill(os.getpid(), signal.SIGINT)
            # Ideally we'd just rethrow here, which is caught by
            # starlette Router.lifespan() and converted into ASGI
            # lifespan.shutdown.failed event. However uvicorn only
            # listens for this event if it's already initiated the
            # shutdown sequence.
            # See https://github.com/Kludex/uvicorn/discussions/2298

    app = FastAPI(title="MAX Serve", lifespan=lifespan_wrap)

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

    reset_prefix_cache_frontend = ResetPrefixCacheFrontend(
        serving_settings.pipeline_config.runtime.zmq_endpoint_base
    )

    async def reset_prefix_cache() -> Response:
        """Reset the prefix cache."""
        if not serving_settings.pipeline_config.model.kv_cache.enable_prefix_caching:
            return Response(
                status_code=400,
                content="Prefix caching is not enabled. Ignoring request",
            )

        reset_prefix_cache_frontend.enqueue_reset_prefix_cache()
        return Response(status_code=200, content="Success")

    app.add_api_route(
        "/reset_prefix_cache", reset_prefix_cache, methods=["POST"]
    )

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
        timeout_graceful_shutdown=5,
    )

    for route in app.routes:
        logger.debug("Route enabled : %s", route)
    return config
