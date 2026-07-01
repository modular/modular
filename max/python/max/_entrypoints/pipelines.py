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
# This is best practice to keep the CLI invocation fast


logger = logging.getLogger("max._entrypoints")

_P = ParamSpec("_P")
_R = TypeVar("_R")

# Subcommands that should not configure/emit telemetry.
_TELEMETRY_OPT_OUT_COMMANDS = {"benchmark", "list"}


def check_model_flag_conflict(args: list[str]) -> None:
    """Raises if both --model and --model-path are specified.

    Raises:
        ValueError: If both --model and --model-path are specified.
    """
    saw_model = False
    saw_model_path = False
    for arg in args:
        if arg == "--model" or arg.startswith("--model="):
            saw_model = True
        elif arg == "--model-path" or arg.startswith("--model-path="):
            saw_model_path = True
        if saw_model and saw_model_path:
            raise ValueError("model_path and model cannot both be specified")


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
        from max._entrypoints.cli.config import pipeline_config_options

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
        check_model_flag_conflict(args)
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
        from max._entrypoints.cli.config import (
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
@click.pass_context
def main(ctx: click.Context, log_level: str = "INFO") -> None:
    from max._entrypoints.cli.entrypoint import configure_cli_logging

    # Configure logging first, before any other initialization
    configure_cli_logging(
        level=log_level, log_prefix=os.getenv("MAX_SERVE_LOG_PREFIX")
    )

    # Some subcommands opt out of telemetry.
    if ctx.invoked_subcommand not in _TELEMETRY_OPT_OUT_COMMANDS:
        configure_telemetry()


def configure_telemetry(color: str | None = None) -> None:
    from max.serve.config import Settings
    from max.serve.telemetry.common import configure_metrics

    settings = Settings()
    configure_metrics(settings)


def common_server_options(func: Callable[_P, _R]) -> Callable[_P, _R]:
    @click.option(
        "--port",
        type=int,
        help="Port for the HTTP API. Defaults to ``8000``.",
    )
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
    @click.option(
        "--eplb-stats",
        type=click.Path(exists=True, dir_okay=False, readable=True),
        default=None,
        help="Path to a snapshot JSON from /max_internal/eplb_stats. "
        "Triggers an EPLB rebalance at startup.",
    )
    @functools.wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        return func(*args, **kwargs)

    return wrapper


@main.command(name="serve", cls=WithLazyPipelineOptions)
@common_server_options
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
    port: int,
    headless: bool,
    log_prefix: str | None,
    eplb_stats: str | None,
    task_arg: tuple[str, ...],
    pretty_print_config: bool,
    **config_kwargs: Any,
) -> None:
    """Start a model serving endpoint for inference.

    Loads a model from a Hugging Face model ID or local path and
    exposes OpenAI-compatible HTTP endpoints for inference requests.
    """
    from max._entrypoints.cli.serve import serve_api_server_and_model_worker
    from max._entrypoints.workers import start_workers
    from max.pipelines import PipelineConfig
    from max.pipelines.context import SamplingParams, SamplingParamsInput
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

    if eplb_stats is not None:
        os.environ["MAX_SERVE_EPLB_STATS"] = eplb_stats
        setting_kwargs["MAX_SERVE_EPLB_STATS"] = eplb_stats

    settings = Settings(**setting_kwargs)

    # Initialize config, and serve.
    # Load tokenizer & pipeline.
    pipeline_config = PipelineConfig.from_flat_kwargs(**config_kwargs)

    # Log Pipeline and Sampling Configuration
    if pretty_print_config:
        # Log Default Sampling Configuration (only for single-model pipelines)
        if "main" in pipeline_config.models:
            sampling_params = SamplingParams.from_input_and_generation_config(
                SamplingParamsInput(),
                sampling_params_defaults=pipeline_config.model.sampling_params_defaults,
            )
            sampling_params.log_sampling_info()

        # Log API Server Related Info
        settings.log_server_info()

    # Configure Logging Globally
    configure_logging(settings)

    if headless:
        start_workers(
            settings=settings,
            pipeline_config=pipeline_config,
        )
    else:
        serve_api_server_and_model_worker(
            settings=settings, pipeline_config=pipeline_config
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
    help="Number of warmup iterations to run before the final timed run.",
)
@click.option(
    "--profile",
    is_flag=True,
    default=False,
    show_default=True,
    help=(
        "Capture a rudimentary profile of the timed run. If Nsight Systems "
        "(`nsys`) and an NVIDIA GPU are available, captures the GPU kernel "
        "trace into an .nsys-rep file and prints the top kernels. Always "
        "captures a Python/CPU profile via cProfile."
    ),
)
@click.option(
    "--profile-output",
    type=str,
    default=None,
    help=(
        "Path for the .nsys-rep file when `--profile` is on. Default: "
        "$BUILD_WORKSPACE_DIRECTORY/max-profile.nsys-rep, or "
        "./max-profile.nsys-rep."
    ),
)
@click.option(
    "--profile-top-n",
    type=int,
    default=15,
    show_default=True,
    help="Number of rows to show in the GPU kernel and Python profile tables.",
)
def cli_pipeline(
    prompt: str,
    image_url: list[str],
    num_warmups: int,
    profile: bool,
    profile_output: str | None,
    profile_top_n: int,
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
    from max._entrypoints.cli.generate import generate_text_for_pipeline
    from max.pipelines import PipelineConfig
    from max.pipelines.context import SamplingParams, SamplingParamsInput
    from max.profiler import maybe_reexec_under_nsys

    # When --profile is set and we have a usable nsys + NVIDIA GPU, re-exec
    # under nsys *before* loading any model state. The child process picks
    # up the same argv, detects it is running under nsys, and runs normally.
    # This keeps the parent process's wasted work to ~zero.
    if profile:
        maybe_reexec_under_nsys(profile_output, top_n=profile_top_n)

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
    pipeline_config = PipelineConfig.from_flat_kwargs(**config_kwargs)
    generate_text_for_pipeline(
        pipeline_config,
        sampling_params=SamplingParams.from_input_and_generation_config(
            params,
            sampling_params_defaults=pipeline_config.model.sampling_params_defaults,
        ),
        prompt=prompt,
        image_urls=image_url,
        num_warmups=num_warmups,
        profile=profile,
        profile_top_n=profile_top_n,
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
    help="Number of warmup iterations to run before the final timed run.",
)
def encode(prompt: str, num_warmups: int, **config_kwargs: Any) -> None:
    """Encode text input into model embeddings.

    This command processes the input text through the model's encoder, producing
    embeddings that can be used for various downstream tasks.
    """
    from max._entrypoints.cli.encode import pipeline_encode
    from max.pipelines import PipelineConfig

    # Load tokenizer & pipeline.
    pipeline_config = PipelineConfig.from_flat_kwargs(**config_kwargs)
    pipeline_encode(pipeline_config, prompt=prompt, num_warmups=num_warmups)


@main.command(name="warm-cache", cls=WithLazyPipelineOptions)
@click.option(
    "--target",
    type=str,
    default=None,
    help=(
        "Target API and architecture to compile for (e.g., cuda, cuda:sm_90, "
        "hip:gfx942). When specified, uses virtual devices for compilation "
        "without requiring physical hardware."
    ),
)
def cli_warm_cache(target: str | None, **config_kwargs) -> None:
    """Load and compile the model to prepare caches."""
    from max.pipelines import PIPELINE_REGISTRY, PipelineConfig

    # Log what we're doing if target mode is enabled
    if target:
        from max.serve.config import parse_api_and_target_arch

        api, target_arch = parse_api_and_target_arch(target)
        logging.info(
            f"Compiling for target: {api} ({target_arch}) using virtual devices"
        )

    pipeline_config = PipelineConfig.from_flat_kwargs(**config_kwargs)
    _ = PIPELINE_REGISTRY.retrieve(pipeline_config)


@main.command(name="warm-interpreter-cache")
def cli_warm_interpreter_cache() -> None:
    """Compile the eager interpreter's graph-compiler models to prepare caches.

    Batch-compiles the full matmul and unary-elementwise matrix for this
    machine's devices into the on-disk cache, then drops a stamp. A later lazy
    eager process on the same device set adopts the warm (one batched cache
    load) instead of compiling each target on first use. Run it as a
    provisioning step on the target hardware. Pure optimization: if skipped, or
    on a different device set, dispatch compiles each target lazily.
    """
    import importlib

    # Dynamic import: _interpreter_ops is an optional Mojo-backed package, kept
    # out of this target's static deps (see its BUILD).
    matmul_gc = importlib.import_module("max._interpreter_ops.matmul_gc")
    unary_gc = importlib.import_module(
        "max._interpreter_ops.unary_elementwise_gc"
    )
    gc_compile = importlib.import_module("max._interpreter_ops.gc_compile")

    logger.info("Warming eager interpreter graph-compiler model cache...")
    matmul_gc.compile_matmul_sweep()
    unary_gc.compile_unary_sweep()
    if gc_compile.write_warm_stamp():
        logger.info("Done. Compiled models cached for this machine's devices.")
    else:
        logger.warning(
            "Compiled models into the cache, but MODULAR_DERIVED_PATH is unset, "
            "so the warm wasn't pinned to a shared location and later lazy "
            "processes won't adopt it. Set MODULAR_DERIVED_PATH (to the same "
            "value the consumers use) when warming."
        )


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
    from max._entrypoints.cli.list import (
        list_pipelines_to_console,
        list_pipelines_to_json,
    )

    if json:
        list_pipelines_to_json()
    else:
        list_pipelines_to_console()


# Argument parsing is handled by benchmark_serving.parse_args.
# All CLI args other than the explicit ``--profile*`` options below are
# forwarded as-is so the two entry points stay in sync.
@main.command(
    name="benchmark",
    context_settings={
        "ignore_unknown_options": True,
        "allow_extra_args": True,
        "help_option_names": [],
    },
)
@click.option(
    "--profile",
    "profile",
    is_flag=True,
    default=False,
    help=(
        "Translate to `--trace`, then render a ranked top-N GPU kernel "
        "table from the resulting `.nsys-rep` after the run completes. "
        "Unlike `max generate --profile`, this targets the *server* "
        "process (which must already be running under `nsys launch`) — "
        "the benchmark client just sends start/stop signals to nsys."
    ),
)
@click.option(
    "--profile-output",
    "profile_output",
    type=str,
    default=None,
    help=(
        "Path for the .nsys-rep file when `--profile` is on. Default: "
        "$BUILD_WORKSPACE_DIRECTORY/max-profile.nsys-rep, or "
        "./max-profile.nsys-rep."
    ),
)
@click.option(
    "--profile-top-n",
    "profile_top_n",
    type=int,
    default=15,
    show_default=True,
    help="Number of rows to show in the GPU kernel summary table.",
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def cli_benchmark(
    profile: bool,
    profile_output: str | None,
    profile_top_n: int,
    args: Sequence[str],
) -> None:
    """Run benchmark tests on a serving model.

    This command runs comprehensive benchmark tests on a model server to measure
    performance metrics including throughput, latency, and resource utilization.
    Make sure that the MAX server is running and hosting a model before running
    this command.
    """
    # ``--profile`` here is fundamentally different from ``max generate
    # --profile``. Generate runs in-process and re-execs under
    # ``nsys profile``; benchmark is a client that sends requests to a
    # separately running server, so the server must be launched under
    # ``nsys launch`` and we just signal start/stop on the existing nsys
    # session via the existing ``--trace`` flow.
    #
    # Import lazily to avoid importing benchmark modules at module load
    # time.
    from max.benchmark.sweep_benchmark_serving import main as sweep_main
    from max.profiler import oneshot

    args = list(args)
    profile_path: str | None = None
    if profile:
        profile_path = profile_output or oneshot.default_profile_output()
        args = oneshot.inject_trace_flags(args, profile_path)

    logger.debug("Running benchmark subcommand with args: %s", args)

    benchmark_succeeded = False
    try:
        click.echo("Starting benchmark...")
        sweep_main(args, app_name="max-benchmark")
        click.echo("Benchmark completed successfully!")
        benchmark_succeeded = True
    except SystemExit as e:
        # cyclopts calls sys.exit() for help and errors, we need to handle this
        if e.code == 0:
            # Help was requested and printed, just return
            return
        else:
            # There was an error, exit with the same code
            sys.exit(e.code)
    except Exception as e:
        click.echo(f"Benchmark failed: {e}", err=True)
        sys.exit(1)

    # Render outside the try so a renderer error doesn't get reported as
    # "Benchmark failed".
    if benchmark_succeeded and profile_path is not None:
        if os.path.exists(profile_path):
            oneshot.render_nsys_kernel_summary(
                profile_path, top_n=profile_top_n, out=sys.stdout
            )
            sys.stdout.write(f"\nFull profile saved to: {profile_path}\n")
            sys.stdout.write(f"Open with: nsys-ui {profile_path}\n")
        else:
            sys.stdout.write(
                f"\n--profile: no .nsys-rep found at {profile_path}. The "
                f"server must be launched under `nsys launch` for "
                f"--profile to produce a GPU trace.\n"
            )


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
