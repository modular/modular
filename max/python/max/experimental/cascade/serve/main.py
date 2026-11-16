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
"""OpenAI-compatible HTTP server for cascade-deployed pipelines.

Run with bazel:

.. code-block:: bash

    br //max/python/max/experimental/cascade/serve:main -- --help

Examples:

.. code-block:: bash

    # Dummy text generation (HTTP worker subprocesses)
    br //max/python/max/experimental/cascade/serve:main -- \
      --model-path dummy_textgen

    # Dummy text generation over gRPC worker subprocesses
    br //max/python/max/experimental/cascade/serve:main -- \
      --model-path dummy_textgen \
      --transport grpc

    # Four local CPU worker processes plus two remote gRPC workers
    br //max/python/max/experimental/cascade/serve:main -- \
      --model-path dummy_textgen \
      --transport grpc \
      --local-cpu-workers 4 \
      --remote-cpu-workers '["host-a:9001", "host-b:9001"]'

    # Echo mode: real tokenizer, no model -- measures cascade overhead
    br //max/python/max/experimental/cascade/serve:main -- \
      --model-path echo:meta-llama/Llama-3.1-8B-Instruct
"""

from __future__ import annotations

import logging
import os
from typing import Annotated

import cyclopts
from cyclopts import Parameter
from max.experimental.cascade.core.fd_limits import raise_nofile_soft_limit
from max.experimental.cascade.deployment.context_config import ContextConfig
from max.experimental.cascade.pipelines.all_pipelines import (
    build_pipeline,
    count_unique_device_specs,
)
from max.experimental.cascade.serve.all_routes import build_router
from max.experimental.cascade.serve.cascade_fastapi import CascadeFastAPI
from max.experimental.cascade.telemetry import configure_telemetry
from max.pipelines.lib import PipelineArgs
from max.pipelines.lib.pipeline_args import _flatten_set_fields

logger = logging.getLogger(__name__)
cli = cyclopts.App()


@cli.default
async def serve(
    pipeline_args: Annotated[
        PipelineArgs, Parameter(name="*")
    ] = PipelineArgs(),  # noqa: B008
    context_config: Annotated[
        ContextConfig, Parameter(name="*")
    ] = ContextConfig(),  # noqa: B008
    *,
    host: str = "localhost",
    port: int = 8000,
    task_arg: Annotated[tuple[str, ...] | None, Parameter(negative="")] = None,
    pretty_print_config: bool = False,
    headless: bool = False,
    max_queue_size: Annotated[int | None, Parameter(negative="")] = None,
    max_pending_requests: Annotated[int | None, Parameter(negative="")] = None,
    eplb_stats: Annotated[str | None, Parameter(negative="")] = None,
    log_prefix: Annotated[str | None, Parameter(negative="")] = None,
    allow_cold_interpreter_cache: bool = False,
    config_file: Annotated[str | None, Parameter(negative="")] = None,
    section_name: Annotated[str | None, Parameter(negative="")] = None,
) -> None:
    """Launch the experimental cascade inference server.

    Model selection is controlled through flat flags derived from
    :class:`~max.pipelines.lib.PipelineArgs` (``--model-path``,
    ``--quantization-encoding``, ...), the same argument surface as
    ``max.serve``. The server auto-detects the pipeline type
    (text-generation, image generation, …) from the model path and routes to
    the appropriate cascade pipeline builder.

    The deployment context is controlled through :class:`ContextConfig` flags:
    ``--transport`` (``http`` or ``grpc``), the per-device worker counts
    (``--local-cpu-workers`` / ``--local-gpu-workers``), and remote worker
    endpoints (``--remote-cpu-workers`` / ``--remote-gpu-workers``). Workers
    run in dedicated child processes; the pipeline's workers are routed by
    their ``deploy_hints`` to the matching device pool and load-balanced
    round-robin within it. With no CPU workers, local or remote, every
    worker runs in the server process.

    Args:
        pipeline_args: Flat user-facing pipeline arguments. All fields are
            exposed as top-level CLI flags via ``Parameter(name="*")``.
            Pass at minimum ``--model-path <repo-id>``.
        context_config: Deployment context configuration. All fields are
            exposed as top-level CLI flags via ``Parameter(name="*")``.
        host: Interface to bind uvicorn to.
        port: TCP port to bind uvicorn to.
        task_arg: Accepted for ``max serve`` compatibility; ignored.
        pretty_print_config: Accepted for ``max serve`` compatibility;
            ignored.
        headless: Accepted for ``max serve`` compatibility; ignored: the
            cascade server has no dispatcher/worker split.
        max_queue_size: Rejected: the cascade server has no request queue
            cap.
        max_pending_requests: Rejected: the cascade server has no
            scheduler pending-queue cap.
        eplb_stats: Rejected: the cascade server has no EPLB rebalance
            step.
        log_prefix: Published as ``MAX_SERVE_LOG_PREFIX`` for worker
            subprocesses that read it; the cascade server's own log
            lines are not prefixed.
        allow_cold_interpreter_cache: Permit compiling eager interpreter
            ops on demand. Like ``max serve``, the server refuses on-demand
            compiles by default, so it does not stall on a compile
            mid-request.
        config_file: YAML recipe, merged by ``from_flat_kwargs`` (CLI flags
            win per field). Not routed into the PipelineArgs constructor:
            a PipelineConfig-shaped recipe would fail validation there.
        section_name: Section of the config file to read values from.
    """
    # Rebuild through from_flat_kwargs so the parsed flags get ``max serve``
    # semantics: draft-model inheritance, config-file merge, and manifest
    # probing. Cyclopts constructs PipelineArgs directly, which skips them.
    # ``cascade=True`` skips the HuggingFace manifest probe, which the
    # dummy and echo pipelines must not need.
    pipeline_args = PipelineArgs.from_flat_kwargs(
        **{
            **_flatten_set_fields(pipeline_args),
            "config_file": config_file,
            "section_name": section_name,
            "cascade": True,
        }
    )

    # Before any worker subprocess inherits the environment, matching the
    # click path (max._entrypoints.pipelines._apply_interpreter_cache_policy
    # and the MAX_SERVE_LOG_PREFIX plumbing). Like ``max serve``, the
    # cascade server refuses on-demand eager compiles by default, so it
    # does not stall on a compile mid-request. The flag opts back in.
    if allow_cold_interpreter_cache:
        os.environ["MAX_EAGER_ALLOW_LAZY_COMPILE"] = "1"
    else:
        os.environ.setdefault("MAX_EAGER_ALLOW_LAZY_COMPILE", "0")
    if log_prefix is not None:
        # Published for any subprocess that reads it, but the cascade
        # server's own logs are unprefixed: nothing in this process calls
        # configure_logging.
        os.environ["MAX_SERVE_LOG_PREFIX"] = log_prefix
        logger.warning(
            "--log-prefix is published to MAX_SERVE_LOG_PREFIX for worker "
            "subprocesses, but the cascade server's own log lines are "
            "not prefixed."
        )

    if task_arg:
        logger.warning(
            "--task-arg is accepted for max serve compatibility but "
            "ignored by the cascade server."
        )
    if pretty_print_config:
        logger.warning(
            "--pretty-print-config is accepted for max serve compatibility "
            "but ignored by the cascade server."
        )
    if headless:
        logger.warning(
            "--headless is accepted for max serve compatibility but "
            "ignored: the cascade server has no dispatcher/worker split."
        )
    if max_queue_size is not None:
        raise ValueError(
            "--max-queue-size is not supported by the cascade server: it "
            "has no request queue cap. Remove the flag to serve."
        )
    if max_pending_requests is not None:
        raise ValueError(
            "--max-pending-requests is not supported by the cascade "
            "server: it has no scheduler pending-queue cap. Remove the "
            "flag to serve."
        )
    if eplb_stats is not None:
        raise ValueError(
            "--eplb-stats is not supported by the cascade server: it has "
            "no EPLB rebalance step. Remove the flag to serve."
        )

    await _serve_deployed(pipeline_args, context_config, host, port)


async def _serve_deployed(
    pipeline_args: PipelineArgs,
    context_config: ContextConfig,
    host: str,
    port: int,
) -> None:
    """Runs the cascade server from an already-built ``PipelineArgs``.

    ``max serve --cascade`` calls this directly. Its own
    ``from_flat_kwargs`` call built the instance. The cyclopts
    entrypoint calls this through :func:`serve`, which rebuilds the
    parsed flags first. Both paths share one body, so their serving
    behavior cannot drift.
    """
    raise_nofile_soft_limit()

    # Size the GPU worker pool from the model arguments unless the user
    # set it explicitly or every worker runs in this process.
    if context_config.local_gpu_workers is None and (
        context_config.local_cpu_workers or context_config.remote_cpu_workers
    ):
        n_gpu_workers = count_unique_device_specs(pipeline_args)
        logger.info("Setting local_gpu_workers to %d", n_gpu_workers)
        context_config = context_config.model_copy(
            update={"local_gpu_workers": n_gpu_workers}
        )

    # Before any worker subprocess inherits the environment: the OTLP
    # exporters read the endpoint from it themselves, in every process.
    context_config.export_otlp_endpoint()

    logger.info("Building cascade pipeline ...")
    pipeline = await build_pipeline(pipeline_args)
    rust_pool = any(
        "rust" in worker.deploy_hints
        for worker in pipeline._get_workers().values()
    )

    logger.info("Opening %s context ...", context_config.transport)
    async with context_config.open_context(rust_pool) as runtime:
        logger.info("Deploying cascade pipeline ...")
        await pipeline.deploy(runtime)

        logger.info("Serving OpenAI-compatible endpoint on %s:%d", host, port)
        app = CascadeFastAPI()
        app.include_router(
            await build_router(
                pipeline,
                runtime,
                pipeline_args.runtime.emit_reasoning_content,
            )
        )
        configure_telemetry("cascade.frontend", f"{host}:{port}")

        await app.serve(host, port)
