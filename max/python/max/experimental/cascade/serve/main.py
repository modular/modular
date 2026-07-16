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

    # Dummy text generation
    br //max/python/max/experimental/cascade/serve:main -- \
      --models.main.model-path dummy_textgen

    # Dummy image generation
    br //max/python/max/experimental/cascade/serve:main -- \
      --models.main.model-path dummy_imgen
"""

from __future__ import annotations

import logging
from typing import Annotated

import cyclopts
from cyclopts import Parameter
from max.experimental.cascade import LocalRuntime
from max.experimental.cascade.pipelines.all_pipelines import build_pipeline
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
    *,
    host: str = "localhost",
    port: int = 8000,
) -> None:
    """Launch the experimental cascade inference server.

    Model selection is controlled through ``--models.*`` flags derived from
    :class:`~max.pipelines.lib.config.PipelineConfig`. The server auto-detects
    the pipeline type (text-generation, image generation, …) from the model
    path and routes to the appropriate cascade pipeline builder.

    Args:
        pipeline_config: Full pipeline configuration. All fields are exposed
            as top-level CLI flags via ``Parameter(name="*")``.
            Pass at minimum ``--models.main.model-path <repo-id>``.
        host: Interface to bind uvicorn to.
        port: TCP port to bind uvicorn to.
    """
    logger.info("Building cascade pipeline ...")
    pipeline = await build_pipeline(pipeline_config)

    logger.info("Opening local runtime ...")
    async with LocalRuntime() as runtime:
        logger.info("Deploying cascade pipeline ...")
        await pipeline.deploy(runtime)

        logger.info("Serving OpenAI-compatible endpoint on %s:%d", host, port)
        app = CascadeFastAPI()
        app.include_router(build_router(pipeline))

        await app.serve(host, port)


if __name__ == "__main__":
    cli()
