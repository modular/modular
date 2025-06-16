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

from __future__ import annotations

import functools
import logging
import os
from typing import Any

import click

logger = logging.getLogger(__name__)


class WithLazyPipelineOptions(click.Command):
    """Command wrapper that defers loading pipeline configuration options

    Lazily applies pipeline_config_options to the callback only when
    command help or execution is actually requested, improving startup time.
    This is somewhat of a hack,
    and should be removed when the pipeline_config_options decorator is fast.
    """

    def __init__(self, *args, **kwargs):
        self._options_loaded = False
        super().__init__(*args, **kwargs)

    def _ensure_options_loaded(self):
        if not self._options_loaded:
            # Lazily load and apply pipeline_config_options decorator
            from max.entrypoints.cli import pipeline_config_options

            # In Click, each command has a callback function that's executed when the command runs.
            # The callback contains the actual implementation of the command.
            # Here, we're applying the pipeline_config_options decorator to add CLI parameters
            # to our callback function dynamically, rather than statically at import time.
            self.callback = pipeline_config_options(self.callback)
            self._options_loaded = True

            # When Click decorators (like @click.option) are applied to a function,
            # they attach Parameter objects to the function via a __click_params__ attribute.
            # We need to extract these parameters and add them to the command's params list
            # so Click knows about them for argument parsing, help text generation, etc.
            # Create a copy to avoid modifying the original list
            self.params = self.params.copy()
            for param in getattr(self.callback, "__click_params__", []):
                self.params.append(param)

    def get_help(self, ctx):
        self._ensure_options_loaded()
        return super().get_help(ctx)

    def invoke(self, ctx):
        self._ensure_options_loaded()
        return super().invoke(ctx)

    def parse_args(self, ctx, args):
        self._ensure_options_loaded()
        return super().parse_args(ctx, args)

    def get_params(self, ctx):
        self._ensure_options_loaded()
        return super().get_params(ctx)

    def shell_complete(self, ctx, incomplete):
        self._ensure_options_loaded()
        return super().shell_complete(ctx, incomplete)


class ModelGroup(click.Group):
    def get_command(self, ctx, cmd_name):
        rv = click.Group.get_command(self, ctx, cmd_name)
        if rv is not None:
            if any(param.name == "task_flags" for param in rv.params):
                rv.ignore_unknown_options = True
            return rv
        supported = ", ".join(self.list_commands(ctx))
        ctx.fail(
            f"Command not supported: {cmd_name}\nSupported commands:"
            f" {supported}"
        )


@click.command(cls=ModelGroup)
@click.option(
    "--version",
    is_flag=True,
    callback=lambda ctx, param, value: print_version(ctx, param, value),
    expose_value=False,
    is_eager=True,  # Eager ensures this runs before other options/commands
    help="Show the MAX version and exit.",
)
def main() -> None:
    pass


def configure_telemetry() -> None:
    from max.serve.config import Settings
    from max.serve.telemetry.common import configure_logging, configure_metrics

    settings = Settings()
    configure_logging(settings)
    configure_metrics(settings)


def common_server_options(func):
    @click.option(
        "--profile-serve",
        is_flag=True,
        show_default=True,
        default=False,
        help=(
            "Whether to enable pyinstrument profiling on the serving endpoint."
        ),
    )
    @click.option(
        "--performance-fake",
        type=click.Choice(["none", "no-op", "speed-of-light", "vllm"]),
        default="none",
        help="Fake the engine performance (for benchmarking)",
    )
    @click.option(
        "--model-name",
        type=str,
        help="Deprecated, please use `model_path` instead. Optional model alias for serving the model.",
    )
    @click.option(
        "--sim-failure",
        type=int,
        default=0,
        help="Simulate fake-perf with failure percentage",
    )
    @click.option(
        "--experimental-enable-kvcache-agent",
        is_flag=True,
        show_default=True,
        default=False,
        help="Experimental: Enable KV Cache Agent support.",
    )
    @click.option("--port", type=int, help="Port to run the server on.")
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


@main.command(name="serve", cls=WithLazyPipelineOptions)
@common_server_options
@click.option(
    "--task", type=str, default="text_generation", help="The task to run."
)
@click.argument("task_flags", nargs=-1, type=click.UNPROCESSED)
def cli_serve(
    profile_serve: bool,
    performance_fake: str,
    model_name: str | None,
    sim_failure: int,
    experimental_enable_kvcache_agent: bool,
    port: int,
    task: str,
    task_flags: list[str],
    **config_kwargs: Any,
) -> None:
    """Start a model serving endpoint for inference.

    This command launches a server that can handle inference requests for the
    specified model. The server supports various performance optimization
    options and monitoring capabilities.
    """
    from max.entrypoints.cli import serve_pipeline
    from max.entrypoints.cli.config import parse_task_flags
    from max.pipelines import (
        AudioGenerationConfig,
        PipelineConfig,
        PipelineTask,
    )

    # Initialize config, and serve.

    # Load tokenizer & pipeline.
    pipeline_config: PipelineConfig
    if task == PipelineTask.AUDIO_GENERATION:
        pipeline_config = AudioGenerationConfig.from_flags(
            parse_task_flags(task_flags), **config_kwargs
        )
    else:
        pipeline_config = PipelineConfig(**config_kwargs)
    failure_percentage = None
    if sim_failure > 0:
        failure_percentage = sim_failure
    serve_pipeline(
        pipeline_config=pipeline_config,
        profile=profile_serve,
        performance_fake=performance_fake,
        model_name=model_name,
        failure_percentage=failure_percentage,
        experimental_enable_kvcache_agent=experimental_enable_kvcache_agent,
        port=port,
        pipeline_task=PipelineTask(task),
    )


@main.command(name="generate", cls=WithLazyPipelineOptions)
@click.option(
    "--prompt",
    type=str,
    default="I believe the meaning of life is",
    help="The text prompt to use for further generation.",
)
@click.option(
    "--image_url",
    type=str,
    multiple=True,
    default=[],
    help=(
        "Images to include along with prompt, specified as URLs."
        " The images are ignored if the model does not support"
        " image inputs."
    ),
)
@click.option(
    "--num-warmups",
    type=int,
    default=0,
    show_default=True,
    help="# of warmup iterations to run before the final timed run.",
)
def cli_pipeline(
    prompt: str,
    image_url: list[str],
    num_warmups: int,
    **config_kwargs: Any,
) -> None:
    """Generate text using the specified model.

    This command runs text generation using the loaded model, optionally
    accepting image inputs for multimodal models.
    """
    from max.entrypoints.cli import generate_text_for_pipeline
    from max.pipelines import PipelineConfig

    if config_kwargs["max_new_tokens"] == -1:
        # Limit generate default max_new_tokens to 100.
        config_kwargs["max_new_tokens"] = 100

    # Load tokenizer & pipeline.
    pipeline_config = PipelineConfig(**config_kwargs)
    generate_text_for_pipeline(
        pipeline_config,
        prompt=prompt,
        image_urls=image_url,
        num_warmups=num_warmups,
    )


@main.command(name="encode", cls=WithLazyPipelineOptions)
@click.option(
    "--prompt",
    type=str,
    default="I believe the meaning of life is",
    help="The text prompt to use for further generation.",
)
@click.option(
    "--num-warmups",
    type=int,
    default=0,
    show_default=True,
    help="# of warmup iterations to run before the final timed run.",
)
def encode(prompt: str, num_warmups: int, **config_kwargs: Any) -> None:
    """Encode text input into model embeddings.

    This command processes the input text through the model's encoder, producing
    embeddings that can be used for various downstream tasks.
    """
    from max.entrypoints.cli import pipeline_encode
    from max.pipelines import PipelineConfig

    # Load tokenizer & pipeline.
    pipeline_config = PipelineConfig(**config_kwargs)
    pipeline_encode(pipeline_config, prompt=prompt, num_warmups=num_warmups)


@main.command(name="warm-cache", cls=WithLazyPipelineOptions)
def cli_warm_cache(**config_kwargs) -> None:
    """Load and compile the model to prepare caches."""
    from max.pipelines import PIPELINE_REGISTRY, PipelineConfig

    pipeline_config = PipelineConfig(**config_kwargs)
    _ = PIPELINE_REGISTRY.retrieve(pipeline_config)


@main.command(name="list")
@click.option(
    "--json",
    is_flag=True,
    show_default=True,
    default=False,
    help="Print the list of pipelines options in JSON format.",
)
def cli_list(json: bool) -> None:
    """List available pipeline configurations and models.

    This command displays information about all registered pipelines and their
    configurations. Output can be formatted as human-readable text or JSON.
    """
    from max.entrypoints.cli import (
        list_pipelines_to_console,
        list_pipelines_to_json,
    )

    if json:
        list_pipelines_to_json()
    else:
        list_pipelines_to_console()


def print_version(
    ctx: click.Context, param: click.Parameter, value: bool
) -> None:
    if not value or ctx.resilient_parsing:
        return
    from max import _core

    click.echo(f"MAX {_core.__version__}")
    ctx.exit()


if __name__ == "__main__":
    if directory := os.getenv("BUILD_WORKSPACE_DIRECTORY"):
        os.chdir(directory)

    configure_telemetry()
    main()
