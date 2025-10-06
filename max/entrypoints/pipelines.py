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
import sys
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

import click
from click import shell_completion
from typing_extensions import ParamSpec

# Please keep all max imports inside their respective functions.
# This is best practive to keep the CLI invocation fast


logger = logging.getLogger("max.entrypoints")

_P = ParamSpec("_P")
_R = TypeVar("_R")


class WithLazyPipelineOptions(click.Command):
    """Command wrapper that defers loading pipeline configuration options

    Lazily applies pipeline_config_options to the callback only when
    command help or execution is actually requested, improving startup time.
    This is somewhat of a hack,
    and should be removed when the pipeline_config_options decorator is fast.
    """

    def __init__(self, *args, **kwargs) -> None:
        self._options_loaded = False
        super().__init__(*args, **kwargs)

    @staticmethod
    def _add_options(callback: Callable[_P, _R]) -> Callable[_P, _R]:
        from max.entrypoints.cli import pipeline_config_options

        return pipeline_config_options(callback)

    def _ensure_options_loaded(self) -> None:
        if self._options_loaded:
            return

        # Lazily load and apply pipeline_config_options decorator

        # In Click, each command has a callback function that's executed when the command runs.
        # The callback contains the actual implementation of the command.
        # Here, we're applying the pipeline_config_options decorator to add CLI parameters
        # to our callback function dynamically, rather than statically at import time.
        assert self.callback is not None
        self.callback = self._add_options(self.callback)
        self._options_loaded = True

        # When Click decorators (like @click.option) are applied to a function,
        # they attach Parameter objects to the function via a __click_params__ attribute.
        # We need to extract these parameters and add them to the command's params list
        # so Click knows about them for argument parsing, help text generation, etc.
        # Create a copy to avoid modifying the original list
        self.params = self.params.copy()
        for param in getattr(self.callback, "__click_params__", []):
            self.params.append(param)

    def get_help(self, ctx: click.Context) -> str:
        self._ensure_options_loaded()
        return super().get_help(ctx)

    def invoke(self, ctx: click.Context) -> Any:
        self._ensure_options_loaded()
        return super().invoke(ctx)

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        self._ensure_options_loaded()
        return super().parse_args(ctx, args)

    def get_params(self, ctx: click.Context) -> list[click.Parameter]:
        self._ensure_options_loaded()
        return super().get_params(ctx)

    def shell_complete(
        self, ctx: click.Context, incomplete: str
    ) -> list[shell_completion.CompletionItem]:
        self._ensure_options_loaded()
        return super().shell_complete(ctx, incomplete)


class WithLazySamplingAndPipelineOptions(WithLazyPipelineOptions):
    @staticmethod
    def _add_options(callback: Callable[_P, _R]) -> Callable[_P, _R]:
        from max.entrypoints.cli import (
            pipeline_config_options,
            sampling_params_options,
        )

        return sampling_params_options(pipeline_config_options(callback))


class ModelGroup(click.Group):
    def get_command(
        self, ctx: click.Context, cmd_name: str
    ) -> click.Command | None:
        rv = click.Group.get_command(self, ctx, cmd_name)
        if rv is not None:
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
@click.option(
    "--log-level",
    type=click.Choice(
        ["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False
    ),
    default="INFO",
    help="Set logging level explicitly (ignored if --verbose or --quiet is used).",
)
def main(log_level: str = "INFO") -> None:
    from max.entrypoints.cli.entrypoint import configure_cli_logging

    # Configure logging first, before any other initialization
    configure_cli_logging(level=log_level)
    configure_telemetry()


def configure_telemetry(color: str | None = None) -> None:
    from max.serve.config import Settings
    from max.serve.telemetry.common import configure_metrics

    settings = Settings()
    configure_metrics(settings)


def common_server_options(func: Callable[_P, _R]) -> Callable[_P, _R]:
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
        "--sim-failure",
        type=int,
        default=0,
        help="Simulate fake-perf with failure percentage",
    )
    @click.option("--port", type=int, help="Port to run the server on.")
    @click.option(
        "--headless",
        is_flag=True,
        show_default=True,
        default=False,
        help="Run only the dispatcher service and model worker without the API server.",
    )
    @click.option(
        "--log-prefix",
        type=str,
        help="Optional prefix to add to all log messages for this server instance.",
    )
    @functools.wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        return func(*args, **kwargs)

    return wrapper


@main.command(name="serve", cls=WithLazyPipelineOptions)
@common_server_options
@click.option(
    "--task", type=str, default="text_generation", help="The task to run."
)
@click.option(
    "--task-arg",
    multiple=True,
    type=str,  # Take them all in as strings
    help="Task-specific arguments to pass to the underlying model (can be used multiple times).",
)
@click.option(
    "--pretty-print-config",
    is_flag=True,
    default=False,
    help="Pretty Print Entire Config",
)
def cli_serve(
    profile_serve: bool,
    sim_failure: int,
    port: int,
    headless: bool,
    log_prefix: str | None,
    task: str,
    task_arg: tuple[str, ...],
    pretty_print_config: bool,
    **config_kwargs: Any,
) -> None:
    """Start a model serving endpoint for inference.

    This command launches a server that can handle inference requests for the
    specified model. The server supports various performance optimization
    options and monitoring capabilities.
    """
    from max.entrypoints.cli import serve_api_server_and_model_worker
    from max.entrypoints.cli.config import parse_task_flags
    from max.entrypoints.workers import start_workers
    from max.interfaces import PipelineTask, SamplingParams
    from max.pipelines import AudioGenerationConfig, PipelineConfig
    from max.serve.config import Settings
    from max.serve.telemetry.common import configure_logging

    # Initialize Settings for API Server
    setting_kwargs: dict[str, Any] = {}
    if port is not None:
        setting_kwargs["MAX_SERVE_PORT"] = port

    if log_prefix is not None:
        setting_kwargs["MAX_SERVE_LOG_PREFIX"] = log_prefix

    if headless is not None:
        setting_kwargs["MAX_SERVE_HEADLESS"] = headless

    settings = Settings(**setting_kwargs)

    # Initialize config, and serve.
    # Load tokenizer & pipeline.
    pipeline_config: PipelineConfig
    if task == PipelineTask.AUDIO_GENERATION:
        pipeline_config = AudioGenerationConfig.from_flags(
            parse_task_flags(task_arg), **config_kwargs
        )
    else:
        pipeline_config = PipelineConfig(**config_kwargs)

    # Log Pipeline and Sampling Configuration
    if pretty_print_config:
        # Log Pipeline Related Info
        pipeline_config.log_pipeline_info()

        # Log Default Sampling Configuration
        sampling_params = SamplingParams()
        sampling_params.log_sampling_info()

        # Log API Server Related Info
        settings.log_server_info()
    else:
        pipeline_config.log_basic_config()

    failure_percentage = None
    if sim_failure > 0:
        failure_percentage = sim_failure

    # Configure Logging Globally
    configure_logging(settings)

    if headless:
        start_workers(
            settings=settings,
            pipeline_config=pipeline_config,
        )
    else:
        serve_api_server_and_model_worker(
            settings=settings,
            pipeline_config=pipeline_config,
            pipeline_task=PipelineTask(task),
        )


@main.command(name="generate", cls=WithLazySamplingAndPipelineOptions)
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
    top_k: int,
    top_p: float,
    min_p: float,
    temperature: float,
    frequency_penalty: float,
    presence_penalty: float,
    repetition_penalty: float,
    max_new_tokens: int,
    min_new_tokens: int,
    ignore_eos: bool,
    stop: list[str],
    stop_token_ids: list[int],
    detokenize: bool,
    seed: int,
    **config_kwargs: Any,
) -> None:
    """Generate text using the specified model.

    This command runs text generation using the loaded model, optionally
    accepting image inputs for multimodal models.
    """
    from max.entrypoints.cli import generate_text_for_pipeline
    from max.interfaces import SamplingParams, SamplingParamsInput
    from max.pipelines import PipelineConfig

    params = SamplingParamsInput(
        top_k=top_k,
        top_p=top_p,
        min_p=min_p,
        temperature=temperature,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        repetition_penalty=repetition_penalty,
        # Limit generate default max_new_tokens to 100.
        max_new_tokens=max_new_tokens or 100,
        min_new_tokens=min_new_tokens,
        ignore_eos=ignore_eos,
        stop=stop,
        stop_token_ids=stop_token_ids,
        detokenize=detokenize,
        seed=seed,
    )

    # Load tokenizer & pipeline.
    pipeline_config = PipelineConfig(**config_kwargs)
    generate_text_for_pipeline(
        pipeline_config,
        sampling_params=SamplingParams.from_input(params),
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


# Because we already have an argparser for benchmark_serving.py, we shouldn't have
# to maintain a whole list of benchmark_serving CLI arg options here. This makes
# it harder to keep them in sync and is error prone. We unroll all the args
# instead and let BenchmarkCommand (which wraps benchmark_serving.py) handle them.
@main.command(
    name="benchmark",
    context_settings={
        "ignore_unknown_options": True,
        "allow_extra_args": True,
        "help_option_names": [],
    },
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def cli_benchmark(args: Sequence[str]) -> None:
    """Run benchmark tests on a serving model.

    This command runs comprehensive benchmark tests on a model server to measure
    performance metrics including throughput, latency, and resource utilization.
    Make sure that the MAX server is running and hosting a model before running
    this command.
    """
    # For benchmark command, we want to handle all arguments directly
    # and bypass Click's argument processing
    # args = ctx.params.get("args", [])

    from max.benchmark.benchmark_serving import main as benchmark_main
    from max.benchmark.benchmark_serving import (
        parse_args as benchmark_parse_args,
    )

    logger.debug("Running benchmark subcommand with args: %s", args)
    try:
        argparse_namespace = benchmark_parse_args(args=args)

        # Run the benchmark
        click.echo("Starting benchmark...")
        benchmark_main(argparse_namespace)
        click.echo("Benchmark completed successfully!")
    except SystemExit as e:
        # argparse calls sys.exit() for help and errors, we need to handle this
        if e.code == 0:
            # Help was requested and printed, just return
            return
        else:
            # There was an error, exit with the same code
            sys.exit(e.code)
    except Exception as e:
        click.echo(f"Benchmark failed: {e}", err=True)
        sys.exit(1)


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

    main()
