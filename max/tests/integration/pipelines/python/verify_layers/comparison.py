# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Tensor comparison utilities for layer verification."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from verify import (
    DEFAULT_ABSOLUTE_TOLERANCE,
    DEFAULT_COS_DIST_THRESHOLD,
    DEFAULT_KL_DIV_THRESHOLD,
    DEFAULT_RELATIVE_TOLERANCE,
    DiscrepancyReport,
)

from verify_layers.layer_utils import find_matching_layers
from verify_layers.tensor_io import load_max_tensor_data, load_pytorch_tensor
from verify_layers.tensor_rules import apply_tensor_processing_rules


@dataclass
class LayerVerificationResult:
    """Result of layer-by-layer verification."""

    layer_name: str
    passed: bool
    discrepancy_report: Optional[DiscrepancyReport] = None
    error_message: Optional[str] = None
    input_shapes: list[list[Union[int, str]]] = field(default_factory=list)
    output_shapes: list[list[Union[int, str]]] = field(default_factory=list)
    mse: Optional[float] = None
    rms: Optional[float] = None


@dataclass
class LayerVerificationReport:
    """Complete report of layer-by-layer verification."""

    pipeline_name: str
    encoding: str
    device_type: str
    layer_results: list[LayerVerificationResult]

    @property
    def total_layers(self) -> int:
        return len(self.layer_results)

    @property
    def passed_layers(self) -> int:
        return sum(1 for result in self.layer_results if result.passed)

    @property
    def failed_layers(self) -> int:
        return self.total_layers - self.passed_layers

    @property
    def overall_passed(self) -> bool:
        return self.failed_layers == 0


def get_component_name(layer_name: str) -> str:
    """Extract component name from layer name for grouping."""
    parts = layer_name.split(".")
    if "layers" in parts:
        try:
            idx = (
                parts.index("layers") + 2
            )  # layer number is right after 'layers'
            return ".".join(parts[idx:])
        except IndexError:
            return layer_name
    else:
        return layer_name


def generate_comparison_text_report(
    layer_indices: list[int],
    mse_values: list[float],
    layer_names: list[str],
    results: list[LayerVerificationResult],
    save_report_path: str,
    execution_order_passed: bool = True,
    execution_order_message: str = "",
) -> None:
    """Generate a detailed text report with layer-by-layer comparison results.

    Args:
        layer_indices: Execution order indices for each layer
        mse_values: MSE values for each layer
        layer_names: Names of the layers
        results: List of LayerVerificationResult objects
        save_report_path: Path to save the text report
        execution_order_passed: Whether execution order check passed
        execution_order_message: Message about execution order check
    """
    try:
        with open(save_report_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("LAYER-BY-LAYER NUMERICAL COMPARISON REPORT\n")
            f.write("=" * 80 + "\n\n")

            # Execution order verification section
            f.write("EXECUTION ORDER VERIFICATION:\n")
            if execution_order_passed:
                f.write(
                    f"✅ Execution order check PASSED: {len(layer_indices)} layers in identical order\n"
                )
                f.write(
                    "✅ Execution order consistency verified - MAX and PyTorch models execute layers in identical sequence\n\n"
                )
            else:
                f.write("❌ Execution order check FAILED\n")
                f.write(f"❌ {execution_order_message}\n\n")

            # Summary statistics - only numerical data
            total_layers = len(results)

            f.write("SUMMARY:\n")
            f.write(f"  Total Layers: {total_layers}\n\n")

            # Filter out infinite values for statistics
            finite_mse_values = [v for v in mse_values if not np.isinf(v)]

            if finite_mse_values:
                f.write("MSE STATISTICS:\n")
                f.write(f"  Min: {min(finite_mse_values):.6e}\n")
                f.write(f"  Max: {max(finite_mse_values):.6e}\n")
                f.write(f"  Mean: {np.mean(finite_mse_values):.6e}\n")
                f.write(f"  Median: {np.median(finite_mse_values):.6e}\n")
                f.write(
                    f"  Finite Values: {len(finite_mse_values)}/{len(mse_values)}\n\n"
                )

            # Group by component for statistics
            grouped: dict[str, dict[str, list]] = {}
            for i, (name, mse, result) in enumerate(
                zip(layer_names, mse_values, results)
            ):
                comp = get_component_name(name)
                if comp not in grouped:
                    grouped[comp] = {
                        "mse_values": [],
                        "results": [],
                        "indices": [],
                    }
                if not np.isinf(mse):
                    grouped[comp]["mse_values"].append(mse)
                grouped[comp]["results"].append(result)
                grouped[comp]["indices"].append(layer_indices[i])

            # Per-component statistics - only numerical data
            f.write("PER-COMPONENT STATISTICS:\n")
            for comp_name, data in sorted(grouped.items()):
                comp_results = data["results"]
                comp_mse = data["mse_values"]
                comp_total = len(comp_results)

                f.write(f"  {comp_name}:\n")
                f.write(f"    Layers: {comp_total}\n")
                if comp_mse:
                    f.write(f"    MSE Min: {min(comp_mse):.6e}\n")
                    f.write(f"    MSE Max: {max(comp_mse):.6e}\n")
                    f.write(f"    MSE Mean: {np.mean(comp_mse):.6e}\n")
                f.write("\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("DETAILED LAYER RESULTS (Execution Order)\n")
            f.write("=" * 80 + "\n\n")

            # Layer-by-layer results - only numerical data
            for i, (idx, name, mse, result) in enumerate(  # noqa: B007
                zip(layer_indices, layer_names, mse_values, results)
            ):
                mse_str = f"{mse:.6e}" if not np.isinf(mse) else "inf"

                f.write(f"[{idx:3d}] {name}\n")
                f.write(f"      MSE: {mse_str}\n")

                # Input/output shapes
                if result.input_shapes:
                    f.write(f"      Input shapes: {result.input_shapes}\n")
                if result.output_shapes:
                    f.write(f"      Output shapes: {result.output_shapes}\n")

                # Only show technical errors (not tolerance-related failures)
                if result.error_message and not result.error_message.startswith(
                    "Tolerance exceeded"
                ):
                    f.write(f"      Error: {result.error_message}\n")

                f.write("\n")

            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")

        print(f"Numerical comparison report saved to: {save_report_path}")

    except Exception as e:
        print(f"Error generating comparison text report: {e}")


def plot_mse_by_component(
    layer_indices: list[int],
    mse_values: list[float],
    layer_names: list[str],
    save_plot_path: str,
) -> None:
    """Plot MSE values by component type in execution order.

    Args:
        layer_indices: Execution order indices for each layer
        mse_values: MSE values for each layer
        layer_names: Names of the layers
        save_plot_path: Path to save the plot
    """
    try:
        # Filter out infinite values for better visualization
        finite_values = []
        finite_indices = []
        finite_names = []

        for i, value in enumerate(mse_values):
            if not np.isinf(value):
                finite_values.append(value)
                finite_indices.append(layer_indices[i])
                finite_names.append(layer_names[i])

        if finite_values:
            # Group indices and values by component name
            grouped: dict[str, dict[str, list]] = {}
            for i, name in enumerate(finite_names):
                comp = get_component_name(name)
                grouped.setdefault(comp, {"indices": [], "values": []})
                grouped[comp]["indices"].append(finite_indices[i])
                grouped[comp]["values"].append(finite_values[i])

            plt.figure(figsize=(15, 8))

            # Color scheme for different components

            try:
                cmap = plt.get_cmap("tab20")
                colors = cmap(np.linspace(0, 1, len(grouped)))
            except ValueError:
                # Fallback if 'tab20' is not available
                cmap = plt.get_cmap("Set3")
                colors = cmap(np.linspace(0, 1, len(grouped)))

            for (comp_name, data), color in zip(grouped.items(), colors):
                plt.plot(
                    data["indices"],
                    data["values"],
                    marker="o",
                    linestyle="-",
                    label=comp_name,
                    color=color,
                    alpha=0.8,
                    markersize=4,
                )

            plt.xlabel("Layer Index (Execution Order)")
            plt.ylabel("MSE (MAX vs PyTorch)")
            plt.title(
                "MSE Between MAX and PyTorch Layer Outputs by Component (Execution Order)"
            )
            plt.grid(True, alpha=0.3)
            plt.legend(loc="best", fontsize="small")

            plt.tight_layout()
            plt.savefig(save_plot_path, dpi=300, bbox_inches="tight")
            print(f"MSE plot saved to: {save_plot_path}")
            print(f"Plotted {len(finite_values)} layers with finite MSE values")

            # Print some statistics
            if finite_values:
                print("MSE Statistics:")
                print(f"  Min: {min(finite_values):.6e}")
                print(f"  Max: {max(finite_values):.6e}")
                print(f"  Mean: {np.mean(finite_values):.6e}")
                print(f"  Median: {np.median(finite_values):.6e}")

                # Print per-component statistics
                print("Per-component MSE statistics:")
                for comp_name, data in grouped.items():
                    values = data["values"]
                    print(f"  {comp_name}:")
                    print(f"    Min: {min(values):.6e}")
                    print(f"    Max: {max(values):.6e}")
                    print(f"    Mean: {np.mean(values):.6e}")
                    print(f"    Count: {len(values)}")
        else:
            print("No finite MSE values to plot")

    except Exception as e:
        print(f"Error creating MSE plot: {e}")


def calculate_mse(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    """Calculate Mean Squared Error (MSE) between two tensors.

    Args:
        tensor1: First tensor
        tensor2: Second tensor

    Returns:
        MSE as a float
    """
    try:
        # Ensure tensors have the same shape
        if tensor1.shape != tensor2.shape:
            return float("inf")  # Return infinity for shape mismatches

        # Convert to same dtype for comparison
        if tensor1.dtype != tensor2.dtype:
            tensor1 = tensor1.float()
            tensor2 = tensor2.float()

        # Calculate MSE: mean of squared differences
        diff = tensor1 - tensor2
        mse = torch.mean(diff * diff).item()
        return mse
    except Exception as e:
        print(f"Error calculating MSE: {e}")
        return float("inf")


def compare_tensors(
    expected: torch.Tensor,
    actual: torch.Tensor,
    absolute_tolerance: float = DEFAULT_ABSOLUTE_TOLERANCE,
    relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE,
) -> tuple[bool, str]:
    """Compare two torch tensors numerically.

    Returns:
        Tuple of (passed, error_message)
    """
    try:
        # Check shapes match
        if expected.shape != actual.shape:
            return (
                False,
                f"Shape mismatch: expected {expected.shape}, got {actual.shape}",
            )

        # Handle dtype compatibility
        original_expected_dtype = expected.dtype
        original_actual_dtype = actual.dtype

        # Convert to float32 for comparison if needed for better numerical stability
        if expected.dtype != actual.dtype or expected.dtype in [
            torch.bfloat16,
            torch.float16,
        ]:
            print(
                f"Converting dtypes for comparison: expected {expected.dtype} -> float32, actual {actual.dtype} -> float32"
            )
            try:
                expected = expected.float()  # Convert to float32
                actual = actual.float()  # Convert to float32
            except Exception as e:
                return (
                    False,
                    f"Failed to convert dtypes for comparison (expected: {original_expected_dtype}, actual: {original_actual_dtype}): {e}",
                )

        # Use torch.allclose for the comparison
        try:
            is_close = torch.allclose(
                actual,
                expected,
                atol=absolute_tolerance,
                rtol=relative_tolerance,
            )
            if is_close:
                return True, ""
        except Exception as e:
            return False, f"torch.allclose comparison failed: {e}"

        # If allclose failed, compute detailed statistics
        abs_diff = torch.abs(expected - actual)

        # Handle relative difference calculation with better numerical stability
        expected_abs = torch.abs(expected)
        denominator = torch.maximum(
            expected_abs, torch.tensor(1e-8)
        )  # Use max to avoid division by very small numbers
        rel_diff = abs_diff / denominator

        # Get statistics
        max_abs_diff = torch.max(abs_diff).item()
        max_rel_diff = torch.max(rel_diff).item()
        mean_abs_diff = torch.mean(abs_diff).item()
        mean_rel_diff = torch.mean(rel_diff).item()

        # Find the percentage of elements that fail tolerance
        abs_failures = torch.sum(abs_diff > absolute_tolerance).item()
        rel_failures = torch.sum(rel_diff > relative_tolerance).item()
        total_elements = torch.numel(expected)

        error_msg = (
            f"Tolerance exceeded - "
            f"max_abs_diff: {max_abs_diff:.2e} (tol: {absolute_tolerance:.2e}), "
            f"max_rel_diff: {max_rel_diff:.2e} (tol: {relative_tolerance:.2e}), "
            f"mean_abs_diff: {mean_abs_diff:.2e}, "
            f"mean_rel_diff: {mean_rel_diff:.2e}, "
            f"abs_failures: {abs_failures}/{total_elements} ({abs_failures / total_elements * 100:.1f}%), "
            f"rel_failures: {rel_failures}/{total_elements} ({rel_failures / total_elements * 100:.1f}%)"
        )

        return False, error_msg

    except Exception as e:
        return False, f"Comparison error: {str(e)}"


def check_execution_order_consistency(
    max_layers: dict[str, Any],
    torch_layers: dict[str, Any],
    layer_data_path: Path | None = None,
) -> tuple[bool, str]:
    """Check if the execution order of modules is identical between MAX and PyTorch.

    Args:
        max_layers: Dictionary of MAX layer information
        torch_layers: Dictionary of PyTorch layer information
        layer_data_path: Path to the layer data directory

    Returns:
        Tuple of (is_consistent, error_message)
    """
    try:
        # Get matches with execution order preserved
        matches = find_matching_layers(
            max_layers, torch_layers, layer_data_path=layer_data_path
        )

        if not matches:
            return False, "No matching layers found between MAX and PyTorch"

        # Extract execution order information
        max_layer_names = list(max_layers.keys())
        torch_layer_names = list(torch_layers.keys())

        print(f"MAX execution order ({len(max_layer_names)} layers):")
        for i, name in enumerate(
            max_layer_names[:10]
        ):  # Show first 10 for brevity
            print(f"  [{i:2d}] {name}")
        if len(max_layer_names) > 10:
            print(f"  ... and {len(max_layer_names) - 10} more layers")

        print(f"\nPyTorch execution order ({len(torch_layer_names)} layers):")
        for i, name in enumerate(
            torch_layer_names[:10]
        ):  # Show first 10 for brevity
            print(f"  [{i:2d}] {name}")
        if len(torch_layer_names) > 10:
            print(f"  ... and {len(torch_layer_names) - 10} more layers")

        # Create normalized sequences in execution order
        max_normalized_sequence = []
        torch_normalized_sequence = []

        # Build mapping from original names to normalized names
        max_name_to_normalized = {}
        torch_name_to_normalized = {}

        for max_name, torch_name, normalized_name in matches:
            max_name_to_normalized[max_name] = normalized_name
            torch_name_to_normalized[torch_name] = normalized_name

        # Build normalized sequences preserving execution order
        for max_name in max_layer_names:
            if max_name in max_name_to_normalized:
                max_normalized_sequence.append(max_name_to_normalized[max_name])

        for torch_name in torch_layer_names:
            if torch_name in torch_name_to_normalized:
                torch_normalized_sequence.append(
                    torch_name_to_normalized[torch_name]
                )

        # Compare sequences
        if len(max_normalized_sequence) != len(torch_normalized_sequence):
            return (
                False,
                f"Different number of matched layers: MAX={len(max_normalized_sequence)}, PyTorch={len(torch_normalized_sequence)}",
            )

        # Find first difference
        differences = []
        for i, (max_norm, torch_norm) in enumerate(
            zip(max_normalized_sequence, torch_normalized_sequence)
        ):
            if max_norm != torch_norm:
                differences.append((i, max_norm, torch_norm))

        if differences:
            error_msg = f"Execution order mismatch found at {len(differences)} positions:\n"
            # Show first few differences
            for i, (pos, max_norm, torch_norm) in enumerate(differences[:5]):  # noqa: B007
                error_msg += f"  Position {pos}: MAX='{max_norm}' vs PyTorch='{torch_norm}'\n"
            if len(differences) > 5:
                error_msg += (
                    f"  ... and {len(differences) - 5} more differences"
                )
            return False, error_msg.strip()

        print(
            f"\n✅ Execution order check PASSED: {len(max_normalized_sequence)} layers in identical order"
        )
        return True, ""

    except Exception as e:
        return False, f"Error during execution order check: {str(e)}"


def compare_layer_outputs(
    max_layers: dict[str, Any],
    torch_layers: dict[str, Any],
    max_export_path: Path,
    torch_export_path: Path,
    absolute_tolerance: float = DEFAULT_ABSOLUTE_TOLERANCE,
    relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE,
    cos_dist_threshold: float = DEFAULT_COS_DIST_THRESHOLD,
    kl_div_threshold: float = DEFAULT_KL_DIV_THRESHOLD,
    plot_mse: bool = True,
    save_plot_path: str = "mse_plot.png",
    save_report_path: str = "comparison_report.txt",
    layer_data_path: Path | None = None,
) -> list[LayerVerificationResult]:
    """Compare layer outputs between MAX and PyTorch models with actual tensor data."""

    # First, check execution order consistency
    print("Checking execution order consistency between MAX and PyTorch...")
    order_consistent, order_error = check_execution_order_consistency(
        max_layers, torch_layers, layer_data_path
    )

    if not order_consistent:
        print("❌ EXECUTION ORDER CHECK FAILED:")
        print(f"   {order_error}")
        print(
            "   This may indicate a fundamental issue with the MAX implementation."
        )
        print(
            "   Proceeding with tensor comparison, but results should be interpreted carefully.\n"
        )

        # Create a special result to indicate the execution order issue
        order_error_result = LayerVerificationResult(
            layer_name="EXECUTION_ORDER_CHECK",
            passed=False,
            error_message=f"Execution order mismatch: {order_error}",
        )
    else:
        print(
            "✅ Execution order check passed - proceeding with tensor comparison\n"
        )
        order_error_result = None

    results = []
    matches = find_matching_layers(
        max_layers, torch_layers, layer_data_path=layer_data_path
    )

    print(f"Matches: {matches}")

    print(f"Found {len(matches)} matching layers for comparison")

    # Lists to store MSE values and layer names for plotting
    mse_values = []
    layer_names = []
    layer_indices = []

    for idx, (max_layer_name, torch_layer_name, normalized_name) in enumerate(
        matches
    ):
        try:
            max_layer = max_layers[max_layer_name]
            torch_layer = torch_layers[torch_layer_name]

            # Get output information
            max_outputs = max_layer.get("outputs", [])
            torch_outputs = torch_layer.get("outputs", [])

            layer_comparison_passed = True
            layer_error_messages = []
            layer_mse = float(
                "inf"
            )  # Default to infinity for failed comparisons

            if len(max_outputs) != len(torch_outputs):
                layer_comparison_passed = False
                layer_error_messages.append(
                    f"Output count mismatch: MAX={len(max_outputs)}, PyTorch={len(torch_outputs)}"
                )
            elif max_outputs and torch_outputs:
                # Compare each output tensor by loading actual data first
                # For MSE, we'll use the first output tensor

                max_out = max_outputs[0]  # Use first output for MSE calculation
                torch_out = torch_outputs[0]

                # Load actual tensor data first
                max_tensor_file = max_export_path / f"{max_out['name']}.max"
                torch_tensor_file = Path(torch_out["file"])

                max_data = load_max_tensor_data(max_tensor_file)
                torch_data = load_pytorch_tensor(torch_tensor_file)

                if max_data is not None and torch_data is not None:
                    # Apply special processing rules before comparison
                    processed_max_data, processed_torch_data = (
                        apply_tensor_processing_rules(
                            normalized_name, max_data, torch_data
                        )
                    )

                    # Calculate MSE for plotting
                    layer_mse = calculate_mse(
                        processed_max_data, processed_torch_data
                    )

                    # Continue with all output comparisons
                    for i, (max_out, torch_out) in enumerate(
                        zip(max_outputs, torch_outputs)
                    ):
                        # Load actual tensor data
                        max_tensor_file = (
                            max_export_path / f"{max_out['name']}.max"
                        )
                        torch_tensor_file = Path(torch_out["file"])

                        max_data = load_max_tensor_data(max_tensor_file)
                        torch_data = load_pytorch_tensor(torch_tensor_file)

                        # Apply special processing rules before comparison
                        processed_max_data, processed_torch_data = (
                            apply_tensor_processing_rules(
                                normalized_name, max_data, torch_data
                            )
                        )

                        # Use processed tensor shapes for comparison
                        max_shape = list(processed_max_data.shape)
                        torch_shape = list(processed_torch_data.shape)

                        # Check shape compatibility using processed shapes
                        if max_shape != torch_shape:
                            layer_comparison_passed = False
                            layer_error_messages.append(
                                f"Output {i} processed shape mismatch: MAX={max_shape}, PyTorch={torch_shape}"
                            )
                            continue

                        # Perform numerical comparison using processed tensor data
                        tensor_passed, error_msg = compare_tensors(
                            expected=processed_torch_data,
                            actual=processed_max_data,
                            absolute_tolerance=absolute_tolerance,
                            relative_tolerance=relative_tolerance,
                        )

                        if not tensor_passed:
                            layer_comparison_passed = False
                            layer_error_messages.append(
                                f"Tensor {i}: {error_msg}"
                            )

            # Store data for plotting
            if normalized_name == "lm_head":
                print(f"layer_mse: {layer_mse}")
            mse_values.append(layer_mse)
            layer_names.append(normalized_name)
            layer_indices.append(idx)

            # Create result using actual loaded shapes if available, otherwise fall back to metadata
            try:
                # Try to get input shapes from actual tensor data if possible
                input_shapes: list[list[Union[int, str]]] = []
                for inp in max_layer.get("inputs", []):
                    input_file = (
                        max_export_path / f"{inp['name']}.max"
                        if "name" in inp
                        else None
                    )
                    if input_file and input_file.exists():
                        input_data = load_max_tensor_data(input_file)
                        if input_data is not None:
                            # Convert int shapes to Union[int, str] for type compatibility
                            input_shapes.append(
                                [int(dim) for dim in input_data.shape]
                            )
                        else:
                            input_shapes.append(inp.get("shape", []))
                    else:
                        input_shapes.append(inp.get("shape", []))

                # Try to get output shapes from actual tensor data if possible
                output_shapes: list[list[Union[int, str]]] = []
                for out in max_layer.get("outputs", []):
                    output_file = max_export_path / f"{out['name']}.max"
                    if output_file.exists():
                        output_data = load_max_tensor_data(output_file)
                        if output_data is not None:
                            # Convert int shapes to Union[int, str] for type compatibility
                            output_shapes.append(
                                [int(dim) for dim in output_data.shape]
                            )
                        else:
                            output_shapes.append(out.get("shape", []))
                    else:
                        output_shapes.append(out.get("shape", []))
            except Exception:
                # Fall back to metadata shapes if there's any error loading actual data
                input_shapes = [
                    inp.get("shape", []) for inp in max_layer.get("inputs", [])
                ]
                output_shapes = [
                    out.get("shape", []) for out in max_layer.get("outputs", [])
                ]

            error_message = (
                "; ".join(layer_error_messages)
                if layer_error_messages
                else None
            )

            result = LayerVerificationResult(
                layer_name=normalized_name,
                passed=layer_comparison_passed,
                error_message=error_message,
                input_shapes=input_shapes,
                output_shapes=output_shapes,
            )

        except Exception as e:
            # Still add to plotting data even if there's an error
            mse_values.append(float("inf"))
            layer_names.append(
                normalized_name
                if "normalized_name" in locals()
                else f"layer_{idx}"
            )
            layer_indices.append(idx)

            result = LayerVerificationResult(
                layer_name=normalized_name
                if "normalized_name" in locals()
                else f"layer_{idx}",
                passed=False,
                error_message=f"Layer comparison error: {str(e)}",
            )

        results.append(result)

    # Add execution order check result if there was an issue
    if order_error_result is not None:
        results.insert(
            0, order_error_result
        )  # Add at the beginning for visibility

    # Create MSE plot
    if plot_mse and mse_values:
        try:
            print(f"layer names: {layer_names}")
            print(f"mse_values: {mse_values}")
            print(f"layer_error_messages: {layer_error_messages}")
            plot_mse_by_component(
                layer_indices, mse_values, layer_names, save_plot_path
            )
        except Exception as e:
            print(f"Error creating MSE plot: {e}")

    # Generate text-based comparison report
    if mse_values and results:
        try:
            generate_comparison_text_report(
                layer_indices,
                mse_values,
                layer_names,
                results,
                save_report_path,
                order_consistent,
                order_error,
            )
        except Exception as e:
            print(f"Error generating comparison text report: {e}")

    return results
