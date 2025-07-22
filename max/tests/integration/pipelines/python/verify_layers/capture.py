# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Layer capture utilities for MAX and PyTorch models."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import traceback
from functools import wraps
from pathlib import Path
from typing import Any, Callable

import torch
from generate_llm_logits import PIPELINE_ORACLES
from max._core.engine import PrintStyle
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, TensorValue, ops
from max.interfaces import TextGenerationRequest
from max.nn.kv_cache import KVCacheInputsSequence
from max.nn.layer import Module
from test_common.evaluate import ModelOutput

from verify_layers.collect_subclasses import all_subclasses
from verify_layers.layer_utils import (
    normalize_max_layer_name,
    normalize_pytorch_layer_name,
)


class LayerIOCapture:
    """Manages layer I/O capture state for MAX models."""

    def __init__(
        self,
        input_injection: bool = False,
        torch_layers: dict[str, Any] | None = None,
        export_path: Path | None = None,
    ):
        self.export_path = export_path or Path.cwd()
        self.max_layers_export_path = Path(self.export_path / "max_layers")
        self.max_layers_export_path.mkdir(parents=True, exist_ok=True)

        self.layer_count = 0
        self.captured_layers: dict[str, dict[str, Any]] = {}
        self._original_call_methods: dict[type, Callable] = {}
        # Track active calls to avoid double-counting layers
        self._active_calls: set[str] = set()

        # For improved naming: track block type counts
        self._block_type_counter: dict[str, int] = {}

        # Input injection parameters
        self.input_injection = input_injection
        self.torch_layers = torch_layers or {}
        self.torch_export_path = self.export_path / "torch_layers"

        # Clear/truncate module_names file at the start of each run
        try:
            module_names_file = self.export_path / "module_names_max.txt"
            with open(module_names_file, "w") as f:
                pass  # Just create/clear the file
        except Exception as e:
            logging.error(f"Failed to clear {module_names_file}: {e}")

    def _find_matching_torch_layer(
        self, max_layer_name: str
    ) -> dict[str, Any] | None:
        """Find a matching PyTorch layer for the given MAX layer name.

        Args:
            max_layer_name: The name of the MAX layer

        Returns:
            PyTorch layer data if a match is found, None otherwise
        """

        # Try to find a matching layer using the same normalization logic as in layer_utils

        normalized_max_name = normalize_max_layer_name(max_layer_name)

        for torch_layer_name, torch_layer_data in self.torch_layers.items():
            normalized_torch_name = normalize_pytorch_layer_name(
                torch_layer_name
            )
            if normalized_max_name == normalized_torch_name:
                return torch_layer_data

        return None

    def _load_torch_input_tensors(
        self, torch_layer_data: dict[str, Any]
    ) -> list[Any]:
        """Load PyTorch input tensors for injection.

        Args:
            torch_layer_data: PyTorch layer data containing input information

        Returns:
            List of loaded tensors that can be used as MAX inputs
        """

        loaded_tensors = []
        torch_inputs = torch_layer_data.get("inputs", [])

        for input_info in torch_inputs:
            try:
                # Get the tensor file path
                if "file" in input_info:
                    tensor_file = Path(input_info["file"])
                elif "name" in input_info:
                    tensor_file = (
                        self.torch_export_path / f"{input_info['name']}.pt"
                    )
                else:
                    continue

                if tensor_file.exists():
                    # Load PyTorch tensor and convert to MAX TensorValue
                    from verify_layers.tensor_io import load_pytorch_tensor

                    torch_tensor = load_pytorch_tensor(tensor_file)

                    if torch_tensor is not None:
                        # Convert PyTorch tensor to MAX TensorValue using ops.constant
                        try:
                            # Special case for bfloat16: convert to float32 for numpy compatibility
                            if torch_tensor.dtype == torch.bfloat16:
                                print(
                                    "Converting BFloat16 tensor while preserving BFloat16 dtype for MAX"
                                )
                                torch_tensor_float32 = torch_tensor.float()
                                numpy_tensor = (
                                    torch_tensor_float32.detach().cpu().numpy()
                                )
                            else:
                                numpy_tensor = (
                                    torch_tensor.detach().cpu().numpy()
                                )

                            # Use MAX's built-in dtype conversion
                            max_dtype = DType.from_torch(torch_tensor.dtype)

                            # Create TensorValue using ops.constant with proper dtype
                            max_tensor = ops.constant(
                                numpy_tensor,
                                dtype=max_dtype,
                                device=DeviceRef.GPU(),
                            )

                            loaded_tensors.append(max_tensor)
                            print(
                                f"Successfully loaded PyTorch input tensor from {tensor_file} for injection (shape: {numpy_tensor.shape}, dtype: {max_dtype})"
                            )
                        except Exception as conversion_error:
                            print(
                                f"Warning: Failed to convert PyTorch tensor to MAX TensorValue: {conversion_error}"
                            )
                            continue

            except Exception as e:
                print(f"Warning: Failed to load PyTorch input tensor: {e}")
                continue

        return loaded_tensors

    def monkey_patch_module(self, module_class: type[Module]) -> None:
        """Monkey patch a Module subclass to capture I/O and log module names.

        In addition to capturing input/output tensors, this will append the module's class name
        to /home/ubuntu/modular/module_names.txt every time the module is called.
        Uses improved fallback naming: layers.{block_count}.{block_name}
        """
        if module_class in self._original_call_methods:
            return  # Already patched

        original_call = module_class.__call__
        self._original_call_methods[module_class] = original_call

        # Capture self reference for use in closure
        layer_capture = self

        @wraps(original_call)
        def wrapped_call(module_self, *args, **kwargs):  # noqa: ANN001
            # Generate layer name - try to get meaningful name from layer weights
            layer_name = f"layer_{layer_capture.layer_count:03d}_{type(module_self).__name__}"

            # Try to get a more meaningful name from layer weights if available
            try:
                if (
                    hasattr(module_self, "layer_weights")
                    and "weight" in module_self.layer_weights
                ):
                    weight_name = module_self.layer_weights["weight"].name
                    if weight_name:
                        layer_name = weight_name.removesuffix(".weight")

            except (AttributeError, KeyError):
                pass  # Fall back to generic name

            # If no meaningful name, use improved fallback naming
            if layer_name.startswith("layer_"):
                class_name = type(module_self).__name__
                count = layer_capture._block_type_counter.get(class_name, 0)
                layer_name = f"layers.{count}.{class_name}"
                layer_capture._block_type_counter[class_name] = count + 1

            # See if the layer is in the active calls set (this can happen if a
            # layer calls super().__call__())
            if layer_name in layer_capture._active_calls:
                return original_call(module_self, *args, **kwargs)
            layer_capture._active_calls.add(layer_name)

            # Write layer name to module_names_max.txt in working directory
            try:
                module_names_file = (
                    layer_capture.export_path / "module_names_max.txt"
                )
                with open(module_names_file, "a") as f:
                    f.write(f"{layer_name}\n")
            except Exception as e:
                # Log error but do not interrupt execution
                logging.error(
                    f"Failed to write module name to {module_names_file}: {e}"
                )

            # Input injection logic
            original_args = args
            injected_inputs = False

            if layer_capture.input_injection:
                # Check if we have a matching PyTorch layer
                matching_torch_layer = layer_capture._find_matching_torch_layer(
                    layer_name
                )

                if matching_torch_layer is not None:
                    try:
                        # Load PyTorch input tensors
                        torch_input_tensors = (
                            layer_capture._load_torch_input_tensors(
                                matching_torch_layer
                            )
                        )

                        if torch_input_tensors:
                            # Replace the positional arguments with PyTorch inputs
                            # Note: This assumes the first N arguments are the tensor inputs
                            # You may need to adjust this logic based on your specific layer signatures
                            injected_args = list(args)

                            # Replace tensor arguments with injected tensors
                            tensor_count = 0
                            for i, arg in enumerate(args):
                                if isinstance(
                                    arg, TensorValue
                                ) and tensor_count < len(torch_input_tensors):
                                    injected_args[i] = torch_input_tensors[
                                        tensor_count
                                    ]
                                    tensor_count += 1

                            # If we have more injected tensors than original tensor args, append them
                            while tensor_count < len(torch_input_tensors):
                                injected_args.append(
                                    torch_input_tensors[tensor_count]
                                )
                                tensor_count += 1

                            args = tuple(injected_args)
                            injected_inputs = True
                            print(
                                f"Injected {len(torch_input_tensors)} PyTorch input tensors into layer {layer_name}"
                            )

                    except Exception as e:
                        print(
                            f"Warning: Input injection failed for layer {layer_name}: {e}"
                        )
                        # Continue with original inputs if injection fails
                        args = original_args

            layer_capture.layer_count += 1

            layer_data: dict[str, Any] = {
                "layer_name": layer_name,
                "layer_type": type(module_self).__name__,
                "inputs": [],
                "outputs": [],
                "input_injection_used": injected_inputs,
            }

            # Export input tensors (either original or injected)
            for i, arg in enumerate(args):
                if isinstance(arg, TensorValue):
                    input_name = f"{layer_name}_input_{i}"
                    if injected_inputs:
                        input_name += "_injected"
                    arg.print(input_name)
                    # Convert shape to list, handling symbolic dimensions
                    shape: list[int | str] = []
                    for dim in arg.shape:
                        try:
                            shape.append(int(dim))
                        except (TypeError, ValueError):
                            # If it's a symbolic dimension, convert to string
                            shape.append(str(dim))

                    layer_data["inputs"].append(
                        {
                            "index": i,
                            "shape": shape,
                            "dtype": str(arg.dtype),
                            "name": input_name,
                            "injected": injected_inputs,
                        }
                    )

            # Export keyword argument tensors
            for arg_name, arg in kwargs.items():
                if isinstance(arg, TensorValue):
                    input_name = f"{layer_name}_input_{arg_name}"
                    if injected_inputs:
                        input_name += "_injected"
                    arg.print(input_name)
                    shape = []
                    for dim in arg.shape:
                        try:
                            shape.append(int(dim))
                        except (TypeError, ValueError):
                            # If it's a symbolic dimension, convert to string
                            shape.append(str(dim))

                    layer_data["inputs"].append(
                        {
                            "name": arg_name,
                            "shape": shape,
                            "dtype": str(arg.dtype),
                            "file_name": input_name,
                            "injected": injected_inputs,
                        }
                    )

            # Call original function
            result = original_call(module_self, *args, **kwargs)
            layer_capture._active_calls.remove(layer_name)

            # Export output tensors
            if isinstance(result, (list, tuple)):
                outputs = []
                for i, res in enumerate(result):
                    if isinstance(res, TensorValue):
                        output_name = f"{layer_name}_output_{i}"
                        res.print(output_name)
                        # Convert shape to list, handling symbolic dimensions
                        shape = []
                        for dim in res.shape:
                            try:
                                shape.append(int(dim))
                            except (TypeError, ValueError):
                                # If it's a symbolic dimension, convert to string
                                shape.append(str(dim))

                        layer_data["outputs"].append(
                            {
                                "index": i,
                                "shape": shape,
                                "dtype": str(res.dtype),
                                "name": output_name,
                            }
                        )
                    outputs.append(res)
                result = type(result)(outputs)
            elif isinstance(result, TensorValue):
                output_name = f"{layer_name}_output"
                result.print(output_name)
                # Convert shape to list, handling symbolic dimensions
                shape = []
                for dim in result.shape:
                    try:
                        shape.append(int(dim))
                    except (TypeError, ValueError):
                        # If it's a symbolic dimension, convert to string
                        shape.append(str(dim))

                layer_data["outputs"].append(
                    {
                        "shape": shape,
                        "dtype": str(result.dtype),
                        "name": output_name,
                    }
                )

            layer_capture.captured_layers[layer_name] = layer_data
            return result

        module_class.__call__ = wrapped_call  # type: ignore[assignment]

    def restore_original_methods(self) -> None:
        """Restore original __call__ methods."""
        for module_class, original_call in self._original_call_methods.items():
            module_class.__call__ = original_call  # type: ignore[method-assign]
        self._original_call_methods.clear()

    def save_metadata(self) -> None:
        """Save layer metadata to JSON file."""
        metadata_file = self.max_layers_export_path / "layer_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(self.captured_layers, f, indent=2)


class TorchLayerIOCapture:
    """Hook to capture PyTorch layer inputs and outputs."""

    def __init__(self, export_path: Path | None = None):
        # Handle None export_path case
        if export_path is None:
            export_path = Path.cwd()

        self.torch_layers_export_path = Path(export_path / "torch_layers")
        self.torch_layers_export_path.mkdir(parents=True, exist_ok=True)

        self.layer_count = 0
        self.captured_layers: dict[str, dict[str, Any]] = {}
        self.hooks: list[Any] = []

        # Working directory for module names and other files
        self.export_path = export_path

        # Clear/truncate module_names file at the start of each run
        try:
            module_names_file = self.export_path / "module_names_torch.txt"
            with open(module_names_file, "w") as f:
                pass  # Just create/clear the file
        except Exception as e:
            logging.error(f"Failed to clear {module_names_file}: {e}")

    def register_hooks(self, model: torch.nn.Module) -> None:
        """Register forward hooks for all modules in the model."""
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Only leaf modules
                hook = module.register_forward_hook(
                    self._make_hook(name, module)
                )
                self.hooks.append(hook)

    def _process_and_save_tensor(
        self,
        tensor: torch.Tensor,
        output_file: Path,
        additional_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Process, save a tensor, and return its metadata.

        Args:
            tensor: The tensor to process and save
            output_file: Path where the tensor should be saved
            additional_metadata: Optional additional metadata fields to include

        Returns:
            Dictionary containing tensor metadata
        """
        # Remove batch dimension if batch size is 1 to match MAX tensor shapes
        tensor_to_save = tensor.detach().cpu()
        if tensor_to_save.shape[0] == 1 and len(tensor_to_save.shape) > 1:
            tensor_to_save = tensor_to_save.squeeze(0)

        # Save the tensor
        torch.save(tensor_to_save, output_file)

        # Create base metadata
        metadata = {
            "shape": list(tensor_to_save.shape),
            "dtype": str(tensor_to_save.dtype),
            "file": str(output_file),
        }

        # Add any additional metadata fields
        if additional_metadata:
            metadata.update(additional_metadata)

        return metadata

    def _make_hook(self, module_name: str, module: torch.nn.Module):
        """Create a forward hook for a specific module."""

        def hook_fn(module, inputs, outputs) -> None:  # noqa: ANN001
            # Use the original module name, replacing dots with underscores for valid filenames
            layer_name = module_name
            self.layer_count += 1

            layer_data: dict[str, Any] = {
                "layer_name": layer_name,
                "layer_type": type(module).__name__,
                "module_name": module_name,
                "inputs": [],
                "outputs": [],
            }
            # Write layer name to module_names_torch.txt in working directory
            try:
                module_names_file = self.export_path / "module_names_torch.txt"
                with open(module_names_file, "a") as f:
                    f.write(f"{layer_name}\n")
            except Exception as e:
                # Log error but do not interrupt execution
                logging.error(
                    f"Failed to write module name to {module_names_file}: {e}"
                )

            # Capture inputs
            for i, inp in enumerate(inputs):
                if isinstance(inp, torch.Tensor):
                    input_file = (
                        self.torch_layers_export_path
                        / f"{layer_name}_input_{i}.pt"
                    )
                    input_metadata = self._process_and_save_tensor(
                        inp, input_file, additional_metadata={"index": i}
                    )
                    layer_data["inputs"].append(input_metadata)

            # Capture outputs
            if isinstance(outputs, torch.Tensor):
                output_file = (
                    self.torch_layers_export_path / f"{layer_name}_output.pt"
                )
                output_metadata = self._process_and_save_tensor(
                    outputs, output_file
                )
                layer_data["outputs"].append(output_metadata)
            elif isinstance(outputs, (list, tuple)):
                for i, output in enumerate(outputs):
                    if isinstance(output, torch.Tensor):
                        output_file = (
                            self.torch_layers_export_path
                            / f"{layer_name}_output_{i}.pt"
                        )
                        output_metadata = self._process_and_save_tensor(
                            output,
                            output_file,
                            additional_metadata={"index": i},
                        )
                        layer_data["outputs"].append(output_metadata)

            self.captured_layers[layer_name] = layer_data

        return hook_fn

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def save_metadata(self) -> None:
        """Save layer metadata to JSON file."""
        metadata_file = self.torch_layers_export_path / "layer_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(self.captured_layers, f, indent=2)


def capture_max_layer_outputs(
    pipeline: str,
    encoding: str,
    device_specs: list,
    prompt: str,
    reference: list[ModelOutput] | None = None,
    input_injection: bool = False,
    torch_layers: dict[str, Any] | None = None,
    export_path: Path | None = None,
) -> dict[str, Any]:
    """Capture MAX model layer outputs using monkey patching.

    Args:
        prompt: The prompt string to use for input context.
        input_injection: Whether to inject PyTorch inputs into MAX layers.
        torch_layers: PyTorch layer data for input injection (if enabled).
        export_path: Path to the layer data directory.

    Returns:
        tuple of (captured_layers_metadata, input_tensor_for_pytorch)
    """

    # Set up layer capture
    if export_path is None:
        export_path = Path.cwd()

    max_layers_export_path = export_path / "max_layers"
    torch_export_path = (
        export_path / "torch_layers" if input_injection else None
    )

    layer_capture = LayerIOCapture(
        input_injection=input_injection,
        torch_layers=torch_layers,
        export_path=export_path,
    )

    # Set environment variable for tensor export
    old_logdir = os.environ.get("MODULAR_LOGDIR")
    os.environ["MODULAR_LOGDIR"] = str(max_layers_export_path)

    # Monkey patch InferenceSession to set debug print options on all new sessions
    original_init = InferenceSession.__init__

    def patched_init(self, *args, **kwargs) -> None:  # noqa: ANN001
        # Call original init
        original_init(self, *args, **kwargs)
        # Set debug print options on this session
        self.set_debug_print_options(
            style=PrintStyle.BINARY_MAX_CHECKPOINT,
            output_directory=str(max_layers_export_path),
        )

    try:
        # Apply the monkey patch using setattr to avoid mypy errors
        setattr(InferenceSession, "__init__", patched_init)  # noqa: B010

        # Find all Module subclasses and patch them
        # This is a simplified approach - in practice you might need to be more selective
        for cls in all_subclasses():
            layer_capture.monkey_patch_module(cls)

        pipeline_oracle = PIPELINE_ORACLES[pipeline]

        max_pipeline_data = pipeline_oracle.create_max_pipeline(
            encoding=encoding, device_specs=device_specs
        )

        # Create input context for MAX
        async def create_max_context():
            return await max_pipeline_data.tokenizer.new_context(
                TextGenerationRequest(
                    id="test",
                    index=0,
                    prompt=prompt,
                    model_name=pipeline,
                )
            )

        context = asyncio.run(create_max_context())

        # For models with KV cache, we need to claim cache rows
        if hasattr(max_pipeline_data.model, "kv_manager"):
            if not max_pipeline_data.model.kv_manager.contains(
                context.cache_seq_id
            ):
                max_pipeline_data.model.kv_manager.external_claim(
                    [context.cache_seq_id]
                )

            # Fetch kv inputs

            kv_cache_inputs = max_pipeline_data.model.kv_manager.fetch(
                [context]
            )
            kv_cache_inputs_sequence = KVCacheInputsSequence(
                kv_cache_inputs=kv_cache_inputs
            )
        else:
            kv_cache_inputs_sequence = None

        # Prepare model inputs
        print("Preparing MAX model inputs...")
        model_inputs = max_pipeline_data.model.prepare_initial_token_inputs(
            context_batch=[context],
            kv_cache_inputs=kv_cache_inputs_sequence,
        )

        # Run a single inference step to capture layers
        print("Running MAX model with single inference step...")
        model_outputs = max_pipeline_data.model.execute(model_inputs)

        # We no longer need to extract and return the first layer's input tensor
        # from MAX because the PyTorch path now tokenizes the prompt itself.

        # Save captured layer metadata
        layer_capture.save_metadata()

        # Load and return the captured layer data
        metadata_file = max_layers_export_path / "layer_metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                captured_data = json.load(f)
            print(f"Captured {len(captured_data)} MAX layers")
            return captured_data
        else:
            print("No MAX layer metadata file found")
            return {}

    except Exception as e:
        print(f"Error in MAX layer capture: {e}")
        traceback.print_exc()
        return {}
    finally:
        # Restore original InferenceSession.__init__ using setattr
        setattr(InferenceSession, "__init__", original_init)  # noqa: B010

        # Restore original methods
        layer_capture.restore_original_methods()

        # Restore environment variable
        if old_logdir is not None:
            os.environ["MODULAR_LOGDIR"] = old_logdir
        elif "MODULAR_LOGDIR" in os.environ:
            del os.environ["MODULAR_LOGDIR"]


def capture_torch_layer_outputs(
    pipeline: str,
    encoding: str,
    device: str,
    prompt: str,
    export_path: Path | None = None,
) -> dict[str, Any]:
    """Capture PyTorch model layer outputs using hooks, deriving input tensor from prompt.

    Args:
        prompt: The prompt string to use for input context.
        export_path: Path to the layer data directory.
    """
    # Set up layer capture hook
    layer_hook = TorchLayerIOCapture(export_path=export_path)

    try:
        # Get the pipeline oracle and create torch model
        pipeline_oracle = PIPELINE_ORACLES[pipeline]
        torch_device = torch.device("cuda:0" if "gpu" in device else "cpu")

        # Create the PyTorch pipeline
        torch_pipeline_data = pipeline_oracle.create_torch_pipeline(
            encoding=encoding, device=torch_device
        )

        # Tokenize the prompt to get input tensor
        tokenizer = torch_pipeline_data.data_processor
        # Try common tokenization methods
        if hasattr(tokenizer, "encode"):
            input_ids = tokenizer.encode(prompt, return_tensors="pt")
        else:
            raise RuntimeError(
                "Could not tokenize prompt for PyTorch pipeline."
            )

        input_tensor = input_ids.to(torch_device)
        print(
            f"Tokenized prompt to input tensor with shape: {input_tensor.shape}"
        )

        # Register hooks on the model
        layer_hook.register_hooks(torch_pipeline_data.model)
        print(f"Registered hooks on {len(layer_hook.hooks)} PyTorch modules")

        model = torch_pipeline_data.model

        # Convert to correct device and ensure proper format
        if len(input_tensor.shape) == 1:
            # Add batch dimension if needed
            input_tensor = input_tensor.unsqueeze(0)

        input_tensor = input_tensor.to(torch_device)

        # For generative models, the input should be token IDs
        # Run single forward pass (NOT generation) to capture layers
        print(
            f"Running PyTorch model forward pass with input shape: {input_tensor.shape}"
        )

        with torch.no_grad():
            # Run a single forward pass - this will trigger all the hooks
            outputs = model(input_tensor)

        print(
            f"PyTorch forward pass completed, captured {len(layer_hook.captured_layers)} layers"
        )

        # Save metadata
        layer_hook.save_metadata()

        # Load and return captured data
        if export_path is None:
            export_path = Path.cwd()

        metadata_file = export_path / "torch_layers" / "layer_metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                captured_data = json.load(f)
            print(f"Captured {len(captured_data)} PyTorch layers")
            return captured_data
        else:
            return {}

    except Exception as e:
        print(f"Error in PyTorch layer capture: {e}")
        traceback.print_exc()
        return {}
    finally:
        layer_hook.remove_hooks()

        # Clean up GPU memory to make room for MAX model - be very aggressive
        try:
            print("Starting PyTorch memory cleanup...")

            # Delete ALL major variables that could hold GPU memory
            if "model" in locals():
                del model
            if "torch_pipeline_data" in locals():
                del torch_pipeline_data
            if "input_tensor" in locals():
                del input_tensor
            if "input_ids" in locals():
                del input_ids
            if "tokenizer" in locals():
                del tokenizer
            if "torch_device" in locals():
                del torch_device
            if "pipeline_oracle" in locals():
                del pipeline_oracle
            if "outputs" in locals():
                del outputs

            # Clear CUDA cache multiple times for thorough cleanup
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            # Add memory synchronization to ensure all GPU operations are complete
            torch.cuda.synchronize()

            # Additional aggressive cleanup
            torch.cuda.empty_cache()

            print("PyTorch memory cleanup completed")

        except Exception as cleanup_error:
            print(f"Warning: Error during GPU memory cleanup: {cleanup_error}")
            # Don't fail the entire function due to cleanup errors
