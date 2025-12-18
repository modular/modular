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

from pathlib import Path

import torch
from max.driver.tensor import load_max_tensor
from max.entrypoints.cli import DevicesOptionType
from max.tests.integration.tools.debug_model import debug_model


def load_intermediate_tensors(
    model: str,
    framework: str,
    output_dir: Path = Path("/tmp/intermediate_tensors/torch"),
    device_type: str = "default",
    encoding_name: str | None = None,
) -> dict[str, torch.Tensor]:
    """Run a Transformers model using Torch with print hooks enabled and return intermediate tensors as a dictionary mapping tensor name to torch.Tensor.

    Args:
        model: Hugging Face model id (e.g., "google/gemma-3-1b-it").
        framework: Framework to run the model on (e.g., "torch", "max").
        dir: Output directory where Torch print hooks will write .pt files; may contain subdirectories.
        device_type: Device selector passed through DevicesOptionType (e.g., "default", "gpu", "cpu", "gpu:0,1,2").
        encoding_name: Optional explicit encoding/dtype (e.g., "bfloat16"). If None, the pipeline default is used.

    Returns:
        Dict keyed by emitted file name (str) with loaded torch.Tensor values.
    """

    debug_model(
        pipeline_name=model,
        framework_name=framework,
        output_path=output_dir,
        device_specs=DevicesOptionType.device_specs(device_type),
        encoding_name=encoding_name if encoding_name else None,
    )
    tensors_map: dict[str, torch.Tensor] = {}
    if framework == "torch":
        files = sorted(
            output_dir.rglob("*.pt"), key=lambda p: p.stat().st_mtime
        )
        for file in files:
            torch_tensor = torch.load(file)
            tensors_map[file.name] = torch_tensor
    elif framework == "max":
        files = sorted(
            output_dir.rglob("*.max"), key=lambda p: p.stat().st_mtime
        )
        for file in files:
            tensor = load_max_tensor(file)
            torch_tensor = torch.from_dlpack(tensor).cpu()
            tensors_map[file.name] = torch_tensor
    else:
        raise ValueError(f"Framework not supported: {framework}")
    return tensors_map


def get_torch_testdata(
    model: str,
    module_name: str,
    output_dir: Path = Path("/tmp/intermediate_tensors/torch"),
    device_type: str = "default",
    encoding_name: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get input and output tensors for a specific module from a torch model run.

    This function runs the model with print hooks enabled and extracts the input
    and output tensors for the specified module. Module naming follows the pattern
    used by TorchPrintHook.name_layers(), where modules are named "model.{module_name}"
    (see torch_print_hook.py).

    Args:
        model: Hugging Face model id (e.g., "google/gemma-3-1b-it").
        module_name: Name of the module to get tensors for (e.g., "encoder.layer.0").
                    The "model." prefix is added automatically. Pass empty string for
                    the top-level model.
        output_dir: Output directory where Torch print hooks write .pt files.
        device_type: Device selector passed through DevicesOptionType (e.g., "default", "gpu").
        encoding_name: Optional explicit encoding/dtype (e.g., "bfloat16").

    Returns:
        Tuple of (input_tensor, output_tensor) for the specified module.
        Input tensor corresponds to the first argument passed to the module's forward method.
        Output tensor corresponds to the result of the module's forward method.

    Raises:
        KeyError: If the specified module tensors are not found in the output.
                 The error message will list all available tensor keys.

    Example:
        >>> input_tensor, output_tensor = get_torch_testdata(
        ...     model="gpt2",
        ...     module_name="transformer.h.0",
        ... )
    """
    # Load all intermediate tensors from the torch run
    tensors_map = load_intermediate_tensors(
        model=model,
        framework="torch",
        output_dir=output_dir,
        device_type=device_type,
        encoding_name=encoding_name,
    )

    # Construct the expected module name following TorchPrintHook.name_layers() pattern
    # See torch_print_hook.py line 45: name = f"model.{module_name}" if module_name else "model"
    full_module_name = f"model.{module_name}" if module_name else "model"

    # Look for input and output tensors
    # Forward hooks save tensors with different suffixes for inputs/outputs
    # Typical patterns: {module_name}.input.pt, {module_name}.output.pt
    # or {module_name}.args.0.pt for first input argument
    output_key = f"{full_module_name}.output.pt"
    input_key = f"{full_module_name}.input.pt"

    # Try alternative naming patterns if primary keys not found
    if output_key not in tensors_map:
        # Output might be saved without .output suffix
        output_key = f"{full_module_name}.pt"

    if input_key not in tensors_map:
        # Input might be saved as args.0 (first positional argument)
        input_key = f"{full_module_name}.args.0.pt"

    # Check if we found the tensors
    available_keys = sorted(tensors_map.keys())

    if output_key not in tensors_map:
        raise KeyError(
            f"Output tensor for module '{full_module_name}' not found.\n"
            f"Tried keys: '{full_module_name}.output.pt', '{full_module_name}.pt'\n"
            f"Available tensors: {available_keys}"
        )

    if input_key not in tensors_map:
        raise KeyError(
            f"Input tensor for module '{full_module_name}' not found.\n"
            f"Tried keys: '{full_module_name}.input.pt', '{full_module_name}.args.0.pt'\n"
            f"Available tensors: {available_keys}"
        )

    input_tensor = tensors_map[input_key]
    output_tensor = tensors_map[output_key]

    return input_tensor, output_tensor


__all__ = ["get_torch_testdata", "load_intermediate_tensors"]
