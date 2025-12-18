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

import json
import os
import sys

# Standard library
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, cast

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
from max.nn.module_v3.module import Module as ModuleV3
from max.pipelines.lib.model_config import MAXModelConfig
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
    torch_print_hook,
)
from test_common.github_utils import github_log_group
from test_common.test_data import MockTextGenerationRequest

# This is far from a universal standard, but this is the closest to a standard
# that I could find: BSD-derived programs sometimes use exit codes from
# "sysexits.h", which defines this exit code as "temp failure; user is invited
# to retry".  debug_model will emit this if it detects a failure is
# likely caused by a network flake and could be resolved by a retry.
EX_TEMPFAIL = 75


def _apply_config_overrides(
    config: Any, overrides: dict[str, Any], config_type: str = "config"
) -> None:
    """Apply overrides to a HuggingFace config object with validation.

    Args:
        config: The HuggingFace config object to modify.
        overrides: Dictionary of key-value pairs to override in the config.
        config_type: Description of config type for error messages (e.g., "AutoConfig").

    Raises:
        ValueError: If any override keys are not valid config attributes.
    """
    try:
        config_dict = config.to_dict()
    except Exception:
        config_dict = dict(getattr(config, "__dict__", {}))

    valid_keys = set(config_dict.keys())
    invalid_keys = [k for k in overrides if k not in valid_keys]

    if invalid_keys:
        valid_lines = "\n  - ".join(sorted(valid_keys))
        invalid_lines = "\n  - ".join(sorted(invalid_keys))
        raise ValueError(
            f"Invalid override key(s):"
            f"{invalid_lines}\n\nAllowed {config_type} keys that can be overridden:"
            f"{valid_lines}"
        )

    for key, value in overrides.items():
        setattr(config, key, value)


@contextmanager
def apply_hf_config_override(
    hf_config_overrides: dict[str, Any],
) -> Iterator[None]:
    """Apply overrides to HuggingFace config property.

    TODO (MODELS-792): This patch is a temporary workaround to allow overriding
    the HuggingFace config. In a future version of the MAXModelConfig class,
    we should be able to edit the object directly.
    """
    orig_hf_prop = MAXModelConfig.huggingface_config
    if not isinstance(orig_hf_prop, property) or orig_hf_prop.fget is None:
        raise RuntimeError(
            "Expected MAXModelConfig.huggingface_config to be a @property."
        )
    original_getter = orig_hf_prop.fget

    def _patched_getter(self: Any) -> Any:
        cfg = original_getter(self)
        _apply_config_overrides(cfg, hf_config_overrides, "AutoConfig")
        return cfg

    MAXModelConfig.huggingface_config = property(_patched_getter)
    try:
        yield
    finally:
        MAXModelConfig.huggingface_config = orig_hf_prop


@contextmanager
def apply_non_strict_load() -> Iterator[None]:
    """Wrap load_state_dict methods to use strict=False."""

    def _wrap_non_strict(original_fn: Any) -> Any:
        def _wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
            kwargs["strict"] = False
            return original_fn(self, *args, **kwargs)

        return _wrapped

    orig_max_load = Module.load_state_dict
    cast(Any, Module).load_state_dict = _wrap_non_strict(orig_max_load)

    orig_max_v3_load = getattr(ModuleV3, "load_state_dict", None)
    if orig_max_v3_load is not None:
        cast(Any, ModuleV3).load_state_dict = _wrap_non_strict(orig_max_v3_load)

    try:
        yield
    finally:
        cast(Any, Module).load_state_dict = orig_max_load
        if orig_max_v3_load is not None:
            cast(Any, ModuleV3).load_state_dict = orig_max_v3_load


@contextmanager
def apply_max_hooks(output_directory: Path | None) -> Iterator[PrintHook]:
    """Create and manage MAX print hooks."""
    hook = PrintHook()
    orig_infer_init: Any = None

    if output_directory is not None:
        orig_infer_init = InferenceSession.__init__

        def _patched_inference_init(
            session_self: InferenceSession, *args: Any, **kwargs: Any
        ) -> None:
            orig_infer_init(session_self, *args, **kwargs)
            session_self.set_debug_print_options(
                style=PrintStyle.BINARY_MAX_CHECKPOINT,
                output_directory=output_directory,
            )

        InferenceSession.__init__ = _patched_inference_init  # type: ignore[method-assign,assignment]

    try:
        yield hook
    finally:
        hook.remove()
        if orig_infer_init is not None:
            InferenceSession.__init__ = orig_infer_init  # type: ignore[method-assign]


@contextmanager
def apply_name_layers_after_state_load(hook: PrintHook) -> Iterator[None]:
    """Wrap Module.load_state_dict to name layers after loading."""
    orig_load = Module.load_state_dict

    def _name_layers_after_load(
        module_self: Any, *args: Any, **kwargs: Any
    ) -> Any:
        result = orig_load(module_self, *args, **kwargs)
        hook.name_layers(module_self)
        return result

    cast(Any, Module).load_state_dict = _name_layers_after_load
    try:
        yield
    finally:
        cast(Any, Module).load_state_dict = orig_load


@contextmanager
def debug_context(
    *,
    output_directory: Path | None,
    hf_config_overrides: dict[str, Any] | None,
) -> Iterator[None]:
    """Context manager to manage model execution when debugging.

    This context manages:
    1. HuggingFace config overrides to modify the model configuration
    2. Places print hooks for both MAX and Torch models to inspect intermediate tensors
    3. Names layers after state dict loading
    """
    with ExitStack() as stack:
        if hf_config_overrides is not None:
            stack.enter_context(apply_hf_config_override(hf_config_overrides))
            stack.enter_context(apply_non_strict_load())
        hook = stack.enter_context(apply_max_hooks(output_directory))
        stack.enter_context(apply_name_layers_after_state_load(hook))
        yield


@click.command()
@click.option(
    "--framework",
    "framework_name",
    type=click.Choice(["max", "torch"]),
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
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory to write intermediate tensors. If omitted, tensors are printed to the console.",
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
    "--num-steps",
    "num_steps",
    type=int,
    default=1,
    help="The number of steps to run the model for (default: 1).",
)
@click.option(
    "--prompt",
    "prompt",
    type=str,
    required=False,
    help="Override the default TEXT prompt (plain text only). For multimodal inputs pass images via --image. If omitted, uses the pipeline's first default prompt.",
)
@click.option(
    "--image",
    "images",
    type=str,
    multiple=True,
    required=False,
    help="Image URL or path for multimodal pipelines. Can be passed multiple times.",
)
@click.option(
    "--hf-config-overrides",
    "hf_config_overrides",
    type=str,
    default=None,
    help="JSON dict of overrides applied to HuggingFace AutoConfig fields.",
)
def main(
    device_type: str | list[int],
    framework_name: str,
    pipeline_name: str,
    encoding_name: str | None,
    output_path: Path,
    max_batch_size: int | None,
    log_hf_downloads: bool,
    num_steps: int,
    prompt: str | None,
    images: tuple[str, ...] | None,
    hf_config_overrides: str | None,
) -> None:
    if "gemma3" in pipeline_name:
        # Running into dynamo error:
        # https://huggingface.co/google/gemma-3-4b-it/discussions/51
        torch._dynamo.config.disable = True

    parsed_overrides: dict[str, Any] | None
    if hf_config_overrides:
        try:
            parsed = json.loads(hf_config_overrides)
            if not isinstance(parsed, dict):
                raise ValueError("hf_config_overrides must be a JSON object.")
            parsed_overrides = cast(dict[str, Any], parsed)
        except Exception as e:
            raise click.UsageError(
                f"Invalid --hf-config-overrides JSON: {e}"
            ) from e
    else:
        parsed_overrides = None

    try:
        debug_model(
            device_specs=DevicesOptionType.device_specs(device_type),
            framework_name=framework_name,
            pipeline_name=pipeline_name,
            encoding_name=encoding_name,
            output_path=output_path,
            max_batch_size=max_batch_size,
            log_hf_downloads=log_hf_downloads,
            num_steps=num_steps,
            prompt=prompt,
            images=images,
            hf_config_overrides=parsed_overrides,
        )
    except Flake:
        sys.exit(EX_TEMPFAIL)


@_detect_hf_flakes
def debug_model(
    device_specs: list[driver.DeviceSpec],
    framework_name: str,
    pipeline_name: str,
    output_path: Path,
    encoding_name: str | None = None,
    max_batch_size: int | None = None,
    log_hf_downloads: bool = False,
    num_steps: int = 1,
    prompt: str | None = None,
    images: tuple[str, ...] | None = None,
    hf_config_overrides: dict[str, Any] | None = None,
) -> None:
    """Run a model with print hooks enabled and write intermediate tensors.

    Intermediate tensors are written to the output directory if specified.
    Config overrides can be applied to both MAX and Torch models via hf_config_overrides.
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

    # Build input based on user-provided prompt and/or images
    if prompt is None and not images:
        inputs = pipeline_oracle.inputs[:1]
    elif images and len(images) > 0:
        inputs = [
            MockTextGenerationRequest.with_images(
                prompt=prompt
                if prompt is not None
                else pipeline_oracle.inputs[0].prompt,
                images=list(images),
            )
        ]
    else:
        inputs = [
            MockTextGenerationRequest.text_only(
                prompt
                if prompt is not None
                else pipeline_oracle.inputs[0].prompt
            )
        ]

    evaluation_batch_size: int | list[int]
    if max_batch_size is None:
        if pipeline_oracle.default_batch_size is None:
            evaluation_batch_size = 1
        else:
            evaluation_batch_size = pipeline_oracle.default_batch_size
    else:
        evaluation_batch_size = max_batch_size

    title = f"{pipeline_name} - {framework_name.upper()} - {encoding_name or 'Default Encoding'}"
    with (
        debug_context(
            output_directory=output_path,
            hf_config_overrides=hf_config_overrides,
        ),
        github_log_group(title),
    ):
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
            run_max_model(
                task=pipeline_oracle.task,
                max_pipeline_and_tokenizer=max_pipeline_and_tokenizer,
                inputs=inputs,
                num_steps=num_steps,
                evaluation_batch_size=evaluation_batch_size,
                reference=None,
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

            # Apply HuggingFace config overrides directly to the model config
            if hf_config_overrides:
                _apply_config_overrides(
                    torch_pipeline_and_tokenizer.model.config,
                    hf_config_overrides,
                    "config",
                )

            export_path = str(output_path) if output_path is not None else None
            hook = torch_print_hook.TorchPrintHook(export_path=export_path)
            hook.name_layers(torch_pipeline_and_tokenizer.model)

            print(f"Running {pipeline_name} model on Torch")
            run_torch_model(
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


if __name__ == "__main__":
    main()
