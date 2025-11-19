# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import functools
import os
import sys
import traceback

# Standard library
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import TypeVar

# 3rd-party
import click
import huggingface_hub
import requests
import torch
from create_pipelines import PIPELINE_ORACLES
from debugging_utils import add_max_hooks
from max import driver, pipelines
from max.entrypoints.cli import DevicesOptionType
from max.entrypoints.cli.entrypoint import configure_cli_logging
from max.interfaces import PipelineTask

# Tests
from test_common import (
    evaluate,
    evaluate_embeddings,
    numpy_encoder,
    torch_print_hook,
    torch_utils,
)
from test_common.evaluate import NUM_STEPS, ModelOutput
from test_common.github_utils import github_log_group
from typing_extensions import ParamSpec

# This is far from a universal standard, but this is the closest to a standard
# that I could find: BSD-derived programs sometimes use exit codes from
# "sysexits.h", which defines this exit code as "temp failure; user is invited
# to retry".  generate_llm_logits will emit this if it detects a failure is
# likely caused by a network flake and could be resolved by a retry.
EX_TEMPFAIL = 75


@contextmanager
def maybe_log_hf_downloads(enable_logging: bool):  # noqa: ANN201
    """Context manager that conditionally logs HuggingFace file downloads."""
    if not enable_logging:
        yield
        return

    original_hf_hub_download = huggingface_hub.hf_hub_download

    def logged_hf_hub_download(*args, **kwargs):  # noqa: ANN202
        repo_id = kwargs.get("repo_id") or (
            args[0] if len(args) > 0 else "unknown"
        )
        filename = kwargs.get("filename") or (
            args[1] if len(args) > 1 else "unknown"
        )
        print(f"Accessing {filename} from {repo_id}")
        result = original_hf_hub_download(*args, **kwargs)
        print(f"-> Located at: {result}\n")
        return result

    huggingface_hub.hf_hub_download = logged_hf_hub_download
    try:
        yield
    finally:
        huggingface_hub.hf_hub_download = original_hf_hub_download


class Flake(Exception):
    """A failure has occurred that appears to be of a temporary nature.

    It is likely that retrying the operation would succeed.
    """


_ParamsT = ParamSpec("_ParamsT")
_ReturnT = TypeVar("_ReturnT")


def _detect_hf_flakes(
    inner: Callable[_ParamsT, _ReturnT],
) -> Callable[_ParamsT, _ReturnT]:
    """Decorator to exit with a distinct status on Hugging Face flake."""

    def is_client_error(exc: requests.RequestException) -> bool:
        if not isinstance(exc, requests.HTTPError):
            return False
        if exc.response is None:
            return False
        # 4xx status codes indicate client error.
        return 400 <= exc.response.status_code < 500

    def get_all_exceptions_in_chain(
        exc: Exception,
    ) -> list[Exception]:
        """Gets all exceptions in the exception chain."""
        to_visit = [exc]
        visited = set()
        all_exceptions = []

        while to_visit:
            current_exc = to_visit.pop(0)

            if id(current_exc) in visited:
                continue
            visited.add(id(current_exc))

            all_exceptions.append(current_exc)

            cause = current_exc.__cause__
            if cause is not None and isinstance(cause, Exception):
                to_visit.append(cause)

            context = current_exc.__context__
            if context is not None and isinstance(context, Exception):
                to_visit.append(context)

        return all_exceptions

    @functools.wraps(inner)
    def wrapper(*args, **kwargs):  # noqa: ANN202
        try:
            return inner(*args, **kwargs)
        except Exception as exc:
            request_exceptions = [
                e
                for e in get_all_exceptions_in_chain(exc)
                if isinstance(e, requests.RequestException)
            ]
            for req_exc in request_exceptions:
                if (
                    req_exc.request is not None
                    and req_exc.request.url is not None
                    and "huggingface.co" in req_exc.request.url
                    and not is_client_error(req_exc)
                ):
                    # This is probably a Hugging Face flake.
                    print(
                        "Seems like a Hugging Face flake has occurred:",
                        file=sys.stderr,
                    )
                    traceback.print_exc()
                    print(
                        "-- End of Hugging Face flake traceback --",
                        file=sys.stderr,
                    )
                    raise Flake("Hugging Face API flake detected") from exc
            raise

    return wrapper


@click.command()
@click.option(
    "--device",
    "--devices",
    "device_type",
    type=DevicesOptionType(),
    required=True,
    help="Type of device to run pipeline with",
)
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
    type=click.Choice(sorted(list(PIPELINE_ORACLES.keys()))),
    required=True,
    help="Pipeline to run",
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

    device_specs = DevicesOptionType.device_specs(device_type)

    if encoding_name is None:
        device_name = device_specs[0].device_type
        device_encoding_map = PIPELINE_ORACLES[
            pipeline_name
        ].device_encoding_map
        if device_name not in device_encoding_map:
            raise ValueError(
                f"Device type {device_name} not supported for pipeline {pipeline_name}. "
                f"Supported device types are: {device_encoding_map.keys()}"
            )
        if len(device_encoding_map[device_name]) > 1:
            raise ValueError(
                f"Multiple encodings supported for device type {device_name}: "
                f"{device_encoding_map[device_name]}. Please specify an encoding."
            )
        encoding_name = device_encoding_map[device_name][0]

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
    encoding_name: str,
    output_path: Path,
    print_output: bool,
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

    pipeline_oracle = PIPELINE_ORACLES[pipeline_name]

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

    for device_spec in device_specs:
        if not pipeline_oracle.is_supported(
            encoding=encoding_name,
            device_spec=device_spec,
        ):
            raise ValueError(
                f"Unsupported combination of encoding '{encoding_name}' and "
                f"device '{device_spec.device_type}'. For pipeline "
                f"'{pipeline_name}', supported combinations are: "
                f"{pipeline_oracle.device_encoding_map}"
            )

    title = f"{pipeline_name} - {framework_name.upper()} - {encoding_name}"
    with github_log_group(title):
        if framework_name == "max":
            hooks_ctx = (
                add_max_hooks(output_directory=intermediates_dir)
                if (print_intermediates or intermediates_dir is not None)
                else nullcontext()
            )
            with hooks_ctx, maybe_log_hf_downloads(log_hf_downloads):
                max_pipeline_and_tokenizer = (
                    pipeline_oracle.create_max_pipeline(
                        encoding=encoding_name,
                        device_specs=device_specs,
                    )
                )

            print(f"Running {pipeline_name} model on MAX")
            if pipeline_oracle.task == PipelineTask.TEXT_GENERATION:
                assert isinstance(
                    max_pipeline_and_tokenizer.pipeline,
                    pipelines.TextGenerationPipeline,
                )
                results = evaluate.run_model(
                    max_pipeline_and_tokenizer.pipeline,
                    max_pipeline_and_tokenizer.tokenizer,
                    requests=inputs,
                    num_steps=num_steps,
                    print_outputs=True,
                    batch_size=evaluation_batch_size,
                    reference=reference,
                )
            elif pipeline_oracle.task == PipelineTask.EMBEDDINGS_GENERATION:
                assert isinstance(
                    max_pipeline_and_tokenizer.pipeline,
                    pipelines.EmbeddingsPipeline,
                )
                if not isinstance(evaluation_batch_size, int):
                    raise ValueError(
                        "Data parallel mode not supported for embeddings generation."
                    )
                results = evaluate_embeddings.encode(
                    max_pipeline_and_tokenizer.pipeline,
                    max_pipeline_and_tokenizer.tokenizer,
                    prompts=(inp.prompt for inp in inputs),
                    batch_size=evaluation_batch_size,
                )
            else:
                raise ValueError(
                    f"Evaluating task {pipeline_oracle.task} is not supported."
                )
        elif framework_name == "torch":
            print(f"Running {pipeline_name} model on Torch")
            torch_device: torch.device
            if device_specs[0].device_type == "cpu":
                torch_device = torch.device("cpu")
            elif device_specs[0].device_type == "gpu":
                torch_device = torch.device("cuda:0")

            # For multi-gpu, use auto to handle mapping automatically.
            if len(device_specs) > 1:
                device = "auto"
            else:
                device = torch_device

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

            if pipeline_oracle.task == PipelineTask.TEXT_GENERATION:
                results = pipeline_oracle.run_torch_text_generation(
                    torch_pipeline_and_tokenizer=torch_pipeline_and_tokenizer,
                    device=torch_device,
                    num_steps=num_steps,
                )
            elif pipeline_oracle.task == PipelineTask.EMBEDDINGS_GENERATION:
                results = torch_utils.run_embeddings_generation(
                    model=torch_pipeline_and_tokenizer.model,
                    data_processor=torch_pipeline_and_tokenizer.data_processor,
                    device=torch_device,
                    prompts=(inp.prompt for inp in inputs),
                )
            else:
                raise ValueError(
                    f"Evaluating task {pipeline_oracle.task} is not supported."
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
