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

import os
import sys
import tempfile

# Standard library
from datetime import datetime
from pathlib import Path
from typing import Any

# 3rd-party
import click
import torch
from create_pipelines import PIPELINE_ORACLES, GenericOracle
from max import driver
from max.entrypoints.cli import DevicesOptionType
from max.entrypoints.cli.entrypoint import configure_cli_logging
from max.pipelines.lib.device_specs import (
    device_specs_from_normalized_device_handle,
    normalize_device_specs_input,
)
from run_models import (
    Flake,
    _detect_hf_flakes,
    get_max_default_encoding,
    get_torch_device,
    maybe_log_hf_downloads,
    run_max_model,
    run_torch_model,
    run_vllm_model,
)

# Tests
from test_common import (
    numpy_encoder,
)
from test_common.evaluate import NUM_STEPS, ModelOutput
from test_common.github_utils import github_log_group

# This is far from a universal standard, but this is the closest to a standard
# that I could find: BSD-derived programs sometimes use exit codes from
# "sysexits.h", which defines this exit code as "temp failure; user is invited
# to retry".  generate_llm_logits will emit this if it detects a failure is
# likely caused by a network flake and could be resolved by a retry.
EX_TEMPFAIL = 75


@click.command()
@click.option(
    "--framework",
    "framework_name",
    type=click.Choice(["max", "torch", "vllm"]),
    default="max",
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
    type=str,
    default=None,
    help="Path to output resulting goldens JSON. If omitted, will output to tmp/<timestamp>_<pipeline_name>_<framework_name>.json",
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
@click.option(
    "--generate-logprobs",
    "generate_logprobs",
    is_flag=True,
    default=False,
    help="Generate logprobs in addition to logits.",
)
def main(
    device_type: str | list[int],
    framework_name: str,
    pipeline_name: str,
    encoding_name: str | None,
    output_path: str | None,
    reference_path: Path | None,
    print_output: bool,
    max_batch_size: int | None,
    log_hf_downloads: bool,
    mini: bool,
    generate_logprobs: bool,
) -> None:
    if "gemma3" in pipeline_name:
        # Running into dynamo error:
        # https://huggingface.co/google/gemma-3-4b-it/discussions/51
        torch._dynamo.config.disable = True

    if reference_path is not None:
        reference_logits = numpy_encoder.NumpyDecoder().decode(
            reference_path.read_text()
        )
    else:
        reference_logits = None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pipeline_name_no_slash = pipeline_name.replace("/", "-")
    default_output_path = Path(
        f"{timestamp}_{pipeline_name_no_slash}_{framework_name}.json"
    )
    if output_path is None:
        final_output_path = Path(tempfile.gettempdir()) / default_output_path
    elif output_path.endswith(".json"):
        final_output_path = Path(output_path)
    elif Path(output_path).is_dir():
        final_output_path = Path(output_path) / default_output_path
    else:
        raise ValueError(
            f"Invalid output path: {output_path}. Please provide a valid file path ending with .json or a directory."
        )
    try:
        generate_llm_logits(
            device_specs=device_specs_from_normalized_device_handle(
                normalize_device_specs_input(device_type)
            ),
            framework_name=framework_name,
            pipeline_name=pipeline_name,
            encoding_name=encoding_name,
            output_path=final_output_path,
            reference=reference_logits,
            print_output=print_output,
            max_batch_size=max_batch_size,
            log_hf_downloads=log_hf_downloads,
            mini=mini,
            generate_logprobs=generate_logprobs,
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
    mini: bool = False,
    generate_logprobs: bool = False,
) -> None:
    """Output logits to a file for a model based on a fixed set of prompts.

    The resulting logit golden files for two different frameworks can be used
    with //max/tests/integration/architectures/llama3/verify to check their
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
        num_steps = getattr(pipeline_oracle, "num_steps", NUM_STEPS)

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

            with maybe_log_hf_downloads(log_hf_downloads):
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
                generate_logprobs=generate_logprobs,
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

            print(f"Running {pipeline_name} model on Torch")
            results = run_torch_model(
                pipeline_oracle=pipeline_oracle,
                torch_pipeline_and_tokenizer=torch_pipeline_and_tokenizer,
                device=torch_device,
                inputs=inputs,
                num_steps=num_steps,
                generate_logprobs=generate_logprobs,
            )
        elif framework_name == "vllm":
            # We don't call `get_torch_device()` here to avoid premature CUDA
            # initialization, which can cause vLLM to crash in certain
            # multiprocessing contexts.

            print(f"Running {pipeline_name} model on vLLM")

            vllm_pipeline = pipeline_oracle.create_vllm_pipeline(
                encoding=encoding_name,
                device_specs=device_specs,
            )

            results = run_vllm_model(
                pipeline_oracle=pipeline_oracle,
                vllm_pipeline=vllm_pipeline,
                inputs=inputs,
                num_steps=num_steps,
                max_batch_size=max_batch_size,
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

    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(numpy_encoder.NumpyEncoder().encode(results))
        print(f"Results written to {output_path}")


if __name__ == "__main__":
    main()
