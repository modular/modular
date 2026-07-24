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
      --models.main.model-path dummy_textgen

    # Dummy text generation over gRPC worker subprocesses
    br //max/python/max/experimental/cascade/serve:main -- \
      --models.main.model-path dummy_textgen \
      --transport grpc

    # Four local CPU worker processes plus two remote gRPC workers
    br //max/python/max/experimental/cascade/serve:main -- \
      --models.main.model-path dummy_textgen \
      --transport grpc \
      --local-cpu-workers 4 \
      --remote-cpu-workers '["host-a:9001", "host-b:9001"]'

    # Echo mode: real tokenizer, no model -- measures cascade overhead
    br //max/python/max/experimental/cascade/serve:main -- \
      --models.main.model-path echo:meta-llama/Llama-3.1-8B-Instruct
"""

from __future__ import annotations

import logging
from typing import Annotated

import cyclopts
from cyclopts import Parameter
from max.experimental.cascade.deployment.context_config import ContextConfig
from max.experimental.cascade.pipelines.all_pipelines import (
    build_pipeline,
    count_unique_device_specs,
)
from max.experimental.cascade.serve.all_routes import build_router
from max.experimental.cascade.serve.cascade_fastapi import CascadeFastAPI
from max.pipelines.lib.config import PipelineConfig

logger = logging.getLogger(__name__)
cli = cyclopts.App()


@cli.default
async def serve(
    pipeline_config: Annotated[
        PipelineConfig, Parameter(name="*")
    ] = PipelineConfig(),
    context_config: Annotated[
        ContextConfig, Parameter(name="*")
    ] = ContextConfig(),
    *,
    host: str = "localhost",
    port: int = 8000,
) -> None:
    """Launch the experimental cascade inference server.

    Model selection is controlled through ``--models.*`` flags derived from
    :class:`~max.pipelines.lib.config.PipelineConfig`. The server auto-detects
    the pipeline type (text-generation, image generation, …) from the model
    path and routes to the appropriate cascade pipeline builder.

    The deployment context is controlled through :class:`ContextConfig` flags:
    ``--transport`` (``http`` or ``grpc``), the per-device worker counts
    (``--local-cpu-workers`` / ``--local-gpu-workers``), and remote worker
    endpoints (``--remote-cpu-workers`` / ``--remote-gpu-workers``). Workers
    run in dedicated child processes; the pipeline's workers are routed by
    their ``deploy_hints`` to the matching device pool and load-balanced
    round-robin within it.

    Args:
        pipeline_config: Full pipeline configuration. All fields are exposed
            as top-level CLI flags via ``Parameter(name="*")``.
            Pass at minimum ``--models.main.model-path <repo-id>``.
        context_config: Deployment context configuration. All fields are
            exposed as top-level CLI flags via ``Parameter(name="*")``.
        host: Interface to bind uvicorn to.
        port: TCP port to bind uvicorn to.
    """
    # Size the GPU worker pool from the model configuration unless the user
    # set it explicitly.
    if context_config.local_gpu_workers == 0:
        n_gpu_workers = count_unique_device_specs(pipeline_config)
        logger.info("Setting local_gpu_workers to %d", n_gpu_workers)
        context_config = context_config.model_copy(
            update={"local_gpu_workers": n_gpu_workers}
        )

    logger.info("Building cascade pipeline ...")
    pipeline = await build_pipeline(pipeline_config)

    logger.info("Opening %s context ...", context_config.transport)
    async with context_config.open_context() as runtime:
        logger.info("Deploying cascade pipeline ...")
        await pipeline.deploy(runtime)

        logger.info("Serving OpenAI-compatible endpoint on %s:%d", host, port)
        app = CascadeFastAPI()
        app.include_router(build_router(pipeline))

        await app.serve(host, port)


if __name__ == "__main__":
    cli()
