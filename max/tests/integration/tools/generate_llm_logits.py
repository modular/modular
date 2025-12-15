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

import os
import sys

# Standard library
from collections.abc import Callable, Generator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

# 3rd-party
import click
import torch
from create_pipelines import PIPELINE_ORACLES, GenericOracle
from max import driver
from max.engine import InferenceSession
from max.engine.api import PrintStyle
from max.entrypoints.cli import DevicesOptionType
from max.entrypoints.cli.entrypoint import configure_cli_logging
from max.nn.hooks import PrintHook
from max.nn.layer import Module
from run_models import (
    Flake,
    _detect_hf_flakes,
    get_max_default_encoding,
    get_torch_device,
    maybe_log_hf_downloads,
    run_max_model,
    run_torch_model,
)

# Tests
from test_common import (
    numpy_encoder,
    torch_print_hook,
)
from test_common.evaluate import NUM_STEPS, ModelOutput
from test_common.github_utils import github_log_group

# This is far from a universal standard, but this is the closest to a standard
# that I could find: BSD-derived programs sometimes use exit codes from
# "sysexits.h", which defines this exit code as "temp failure; user is invited
# to retry".  generate_llm_logits will emit this if it detects a failure is
# likely caused by a network flake and could be resolved by a retry.
EX_TEMPFAIL = 75


@contextmanager
def add_max_hooks(
    output_directory: Path | None = None,
) -> Generator[None, None, None]:
    """Context manager that adds tensor printing hooks by patching the model class."""

    # Save original InferenceSession initializer.
    original_inference_init = InferenceSession.__init__
    hook = PrintHook()
    original_inference_init = InferenceSession.__init__

    def get_wrapped_load_state_dict(
        original_load_state_dict: Callable[..., Any],
    ) -> Callable[..., Any]:
        def wrapped_load_state_dict(
            self: Any, *args: Any, **kwargs: Any
        ) -> Any:
            result = original_load_state_dict(self, *args, **kwargs)
            hook.name_layers(self)
            return result

        return wrapped_load_state_dict

    # If an output directory is provided, patch InferenceSession to enable debug prints.
    if output_directory is not None:

        def _patched_inference_init(
            session_self: InferenceSession, *args: Any, **kwargs: Any
        ) -> None:
            original_inference_init(session_self, *args, **kwargs)
            # Enable debug printing to file-style output when an output directory is specified.
            # If additional parameters (like output path) are supported, they can be added here.
            session_self.set_debug_print_options(
                style=PrintStyle.BINARY_MAX_CHECKPOINT,
                output_directory=output_directory,
            )

        InferenceSession.__init__ = _patched_inference_init  # type: ignore[assignment]

    original_load_state_dict = Module.load_state_dict
    Module.load_state_dict = get_wrapped_load_state_dict(  # type: ignore[method-assign]
        original_load_state_dict
    )

    try:
        yield
    finally:
        hook.remove()
        Module.load_state_dict = original_load_state_dict  # type: ignore[method-assign]
        # Restore original InferenceSession initializer if we patched it.
        InferenceSession.__init__ = original_inference_init  # type: ignore[method-assign]


@click.command()
@click.option(
    "--framework",
    "framework_name",
    type=click.Choice(["max", "torch"]),
    required=True,
    help="Framework to run pipeline with",
)
@click.option(
    "--pipeline",
    "pipeline_name",
    type=str,
    required=True,
    help="Pipeline to run. Must be a valid Transformers model path or the key to an existing pipeline oracle.",
)
@click.option(
    "--device",
    "--devices",
    "device_type",
    type=DevicesOptionType(),
    default="default",
    help="Type of device to run pipeline with. Default is to use the first available GPU.",
)
@click.option(
    "--encoding",
    "encoding_name",
    required=False,
    help="Quantization encoding to run pipeline with.",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    type=click.Path(path_type=Path),
    required=True,
    help="Path to output resulting goldens JSON to",
)
@click.option(
    "-r",
    "--reference",
    "reference_path",
    type=click.Path(path_type=Path),
    required=False,
    help="Path to reference golden JSON to compare to",
)
@click.option(
    "--print/--no-print",
    "print_output",
    type=bool,
    default=False,
    help="Dump goldens in non-JSON format to stdout",
)
@click.option(
    "--print-intermediates",
    "print_intermediates",
    is_flag=True,
    default=False,
    help="Outputs intermediate tensors from both frameworks to the console.",
)
@click.option(
    "--intermediates-dir",
    "intermediates_dir",
    type=click.Path(
        path_type=Path, dir_okay=True, file_okay=False, writable=True
    ),
    default=None,
    help="Directory to write intermediate tensors. If omitted, no files are written.",
)
@click.option(
    "--max-batch-size",
    "max_batch_size",
    type=int,
    default=None,
    help="The maximum batch size to use when evaluating the model.",
)
@click.option(
    "--log-hf-downloads",
    "log_hf_downloads",
    is_flag=True,
    default=False,
    help="Log HuggingFace file downloads for MAX and Torch models.",
)
@click.option(
    "--mini",
    "mini",
    is_flag=True,
    default=False,
    help="Run only a single prompt for a single step.",
)
def main(
    device_type: str | list[int],
    framework_name: str,
    pipeline_name: str,
    encoding_name: str | None,
    output_path: Path,
    reference_path: Path | None,
    print_output: bool,
    max_batch_size: int | None,
    log_hf_downloads: bool,
    print_intermediates: bool,
    intermediates_dir: Path | None,
    mini: bool,
) -> None:
    """Click command entry point that delegates to the implementation function.

    This wrapper exists because Click command functions aren't easily picklable,
    which causes issues when called from multiprocessing.
    """
    if pipeline_name == "gemma3-27b":
        # Running into dynamo error:
        # https://huggingface.co/google/gemma-3-4b-it/discussions/51
        torch._dynamo.config.disable = True

    if reference_path is not None:
        reference_logits = numpy_encoder.NumpyDecoder().decode(
            reference_path.read_text()
        )
    else:
        reference_logits = None

    try:
        generate_llm_logits(
            device_specs=DevicesOptionType.device_specs(device_type),
            framework_name=framework_name,
            pipeline_name=pipeline_name,
            encoding_name=encoding_name,
            output_path=output_path,
            reference=reference_logits,
            print_output=print_output,
            max_batch_size=max_batch_size,
            log_hf_downloads=log_hf_downloads,
            print_intermediates=print_intermediates,
            intermediates_dir=intermediates_dir,
            mini=mini,
        )
    except Flake:
        sys.exit(EX_TEMPFAIL)


@_detect_hf_flakes
def generate_llm_logits(
    device_specs: list[driver.DeviceSpec],
    framework_name: str,
    pipeline_name: str,
    output_path: Path,
    print_output: bool,
    encoding_name: str | None = None,
    max_batch_size: int | None = None,
    reference: list[ModelOutput] | None = None,
    log_hf_downloads: bool = False,
    print_intermediates: bool = False,
    intermediates_dir: Path | None = None,
    mini: bool = False,
) -> None:
    """Output logits to a file for a model based on a fixed set of prompts.

    The resulting logit golden files for two different frameworks can be used
    with //max/tests/integration/pipelines/python/llama3/verify to check their
    similarity.

    """
    if workspace_dir := os.getenv("BUILD_WORKSPACE_DIRECTORY"):
        os.chdir(workspace_dir)
    configure_cli_logging(level="INFO")

    if pipeline_name in PIPELINE_ORACLES:
        pipeline_oracle = PIPELINE_ORACLES[pipeline_name]
    else:
        pipeline_oracle = GenericOracle(
            model_path=pipeline_name,
        )

    if mini:
        inputs = pipeline_oracle.inputs[:1]
        num_steps = 1
    else:
        inputs = pipeline_oracle.inputs
        num_steps = NUM_STEPS

    evaluation_batch_size: int | list[int]
    if max_batch_size is None:
        if pipeline_oracle.default_batch_size is None:
            evaluation_batch_size = 1
        else:
            evaluation_batch_size = pipeline_oracle.default_batch_size
    else:
        evaluation_batch_size = max_batch_size

    title = f"{pipeline_name} - {framework_name.upper()} - {encoding_name or 'Default Encoding'}"
    with github_log_group(title):
        if framework_name == "max":
            if encoding_name is None:
                max_encoding_name = get_max_default_encoding(
                    pipeline_oracle, pipeline_name, device_specs
                )
            else:
                max_encoding_name = encoding_name

            hooks_ctx = (
                add_max_hooks(output_directory=intermediates_dir)
                if print_intermediates or intermediates_dir
                else nullcontext()
            )

            with maybe_log_hf_downloads(log_hf_downloads), hooks_ctx:
                max_pipeline_and_tokenizer = (
                    pipeline_oracle.create_max_pipeline(
                        encoding=max_encoding_name,
                        device_specs=device_specs,
                    )
                )

            print(f"Running {pipeline_name} model on MAX")
            results = run_max_model(
                task=pipeline_oracle.task,
                max_pipeline_and_tokenizer=max_pipeline_and_tokenizer,
                inputs=inputs,
                num_steps=num_steps,
                evaluation_batch_size=evaluation_batch_size,
                reference=reference,
            )
        elif framework_name == "torch":
            torch_device = get_torch_device(device_specs)
            # For multi-gpu, use auto to handle mapping automatically.
            device: Any = "auto" if len(device_specs) > 1 else torch_device

            with maybe_log_hf_downloads(log_hf_downloads):
                torch_pipeline_and_tokenizer = (
                    pipeline_oracle.create_torch_pipeline(
                        encoding=encoding_name,
                        device=device,
                    )
                )

            if print_intermediates or intermediates_dir:
                export_path = (
                    str(intermediates_dir)
                    if intermediates_dir is not None
                    else None
                )
                hook = torch_print_hook.TorchPrintHook(export_path=export_path)
                hook.name_layers(torch_pipeline_and_tokenizer.model)

            print(f"Running {pipeline_name} model on Torch")
            results = run_torch_model(
                pipeline_oracle=pipeline_oracle,
                torch_pipeline_and_tokenizer=torch_pipeline_and_tokenizer,
                device=torch_device,
                inputs=inputs,
                num_steps=num_steps,
            )
        else:
            raise NotImplementedError(
                f"Framework {framework_name!r} not implemented"
            )

    if print_output:
        print(f"Framework:    {framework_name}")
        print(f"Pipeline:     {pipeline_name}")
        print(f"Encoding:     {encoding_name}")
        print(f"Device specs: {device_specs}")
        print("Results:")
        print(results)
    with open(output_path, "w") as f:
        f.write(numpy_encoder.NumpyEncoder().encode(results))


if __name__ == "__main__":
    main()
