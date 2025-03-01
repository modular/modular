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


import functools
import logging
import os
import signal

# We need this unused, unsorted import to avoid an issue on
# graviton related to finding an incorrect GLIBCXX version.
from max.entrypoints.llm import LLM  # noqa: F401 # isort: skip

import click
from max.entrypoints.cli import (
    generate_text_for_pipeline,
    list_pipelines_to_console,
    list_pipelines_to_json,
    pipeline_config_options,
    pipeline_encode,
    serve_pipeline,
)
from max.pipelines import PIPELINE_REGISTRY, PipelineConfig
from max.pipelines.architectures import register_all_models
from max.serve.config import Settings
from max.serve.telemetry.common import configure_logging, configure_metrics

logger = logging.getLogger(__name__)


class ModelGroup(click.Group):
    def get_command(self, ctx, cmd_name):
        rv = click.Group.get_command(self, ctx, cmd_name)
        if rv is not None:
            return rv
        supported = ", ".join(self.list_commands(ctx))
        ctx.fail(
            f"Command not supported: {cmd_name}\nSupported commands:"
            f" {supported}"
        )


@click.command(cls=ModelGroup)
def main():
    settings = Settings()
    configure_logging(settings)
    configure_metrics(settings)
    register_all_models()


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
        "--batch-timeout",
        type=float,
        default=0.0,
        help="Custom timeout for any particular batch.",
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
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


@main.command(name="serve")
@pipeline_config_options
@common_server_options
def cli_serve(
    profile_serve,
    performance_fake,
    batch_timeout,
    model_name,
    sim_failure,
    **config_kwargs,
):
    """Start a model serving endpoint for inference.

    This command launches a server that can handle inference requests for the
    specified model. The server supports various performance optimization
    options and monitoring capabilities.
    """
    # Initialize config, and serve.
    pipeline_config = PipelineConfig(**config_kwargs)
    failure_percentage = None
    if sim_failure > 0:
        failure_percentage = sim_failure
    serve_pipeline(
        pipeline_config=pipeline_config,
        profile=profile_serve,
        performance_fake=performance_fake,
        batch_timeout=batch_timeout,
        model_name=model_name,
        failure_percentage=failure_percentage,
    )


@main.command(name="generate")
@pipeline_config_options
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
def cli_pipeline(prompt, image_url, num_warmups, **config_kwargs):
    """Generate text using the specified model.

    This command runs text generation using the loaded model, optionally
    accepting image inputs for multimodal models.
    """
    # Replit model_paths are kinda broken due to transformers
    # version mismatch. We manually update trust_remote_code to True
    # because the modularai version does not have the custom Python code needed
    # Without this, we get:
    #     ValueError: `attn_type` has to be either `multihead_attention` or
    #     `multiquery_attention`. Received: grouped_query_attention
    # Another reason why we override this flag here is because at PipelineConfig
    # instantiation below, we'll call AutoConfig.from_pretrained, which will
    # trigger the error above if not set to True.
    if "replit" in config_kwargs["model_path"]:
        config_kwargs["trust_remote_code"] = True

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


@main.command(name="encode")
@pipeline_config_options
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
def encode(prompt, num_warmups, **config_kwargs):
    """Encode text input into model embeddings.

    This command processes the input text through the model's encoder, producing
    embeddings that can be used for various downstream tasks.
    """
    # Load tokenizer & pipeline.
    pipeline_config = PipelineConfig(**config_kwargs)
    pipeline_encode(
        pipeline_config,
        prompt=prompt,
        num_warmups=num_warmups,
    )


@main.command(name="warm-cache")
@pipeline_config_options
def cli_warm_cache(**config_kwargs) -> None:
    """Load and compile the model to prepare caches.

    This command is particularly useful in combination with
    --save-to-serialized-model-path. Providing that option to this command
    will result in a compiled model being stored to that path. Subsequent
    invocations of other commands can then use --serialized-model-path to
    reuse the previously-compiled model.

    Even without --save-to-serialized-model-path, this command will as a side
    effect warm the Hugging Face cache and in some cases, MAX compilation
    caches.
    """
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
def cli_list(json):
    """List available pipeline configurations and models.

    This command displays information about all registered pipelines and their
    configurations. Output can be formatted as human-readable text or JSON.
    """
    if json:
        list_pipelines_to_json()
    else:
        list_pipelines_to_console()


if __name__ == "__main__":
    if directory := os.getenv("BUILD_WORKSPACE_DIRECTORY"):
        os.chdir(directory)

    # Workaround for https://github.com/buildbuddy-io/buildbuddy/issues/8326
    if (
        signal.getsignal(signal.SIGINT) == signal.SIG_IGN
        and signal.getsignal(signal.SIGTERM) == signal.SIG_IGN
        and signal.getsignal(signal.SIGQUIT) == signal.SIG_IGN
    ):
        # For SIGINT, Python remaps SIG_DFL to default_int_handler on startup.
        # We do the same here to retain the same behavior we would get if we
        # started normally.  (SIG_DFL terminates the process immediately;
        # default_int_handler raises KeyboardInterrupt.)
        signal.signal(signal.SIGINT, signal.default_int_handler)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGQUIT, signal.SIG_DFL)

    main()
