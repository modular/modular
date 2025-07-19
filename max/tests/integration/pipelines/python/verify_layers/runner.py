# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Main runner for layer verification."""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Optional

from max.entrypoints.cli import DevicesOptionType

from verify_layers.capture import (
    capture_max_layer_outputs,
    capture_torch_layer_outputs,
)
from verify_layers.comparison import compare_layer_outputs


def run_layer_verification(
    device_type: str,
    devices: str,
    pipeline: str,
    encoding: str,
    export_path: Optional[Path] = None,
    layer_data_path: Optional[Path] = None,
    input_injection: bool = False,
):
    """Run layer-by-layer verification for a given pipeline.

    Args:
        device_type: Type of device to use
        devices: Device specifications
        pipeline: Pipeline name
        encoding: Encoding to use
        export_path: Base directory for reports, plots, and other output files. Defaults to current working directory.
        layer_data_path: Directory for layer tensor data. If None, uses export_path.
        input_injection: Whether to inject PyTorch inputs into MAX layers during capture.
    """

    # Use working directory if provided, otherwise default to current directory with model_outputs
    if export_path is None:
        export_path = Path.cwd() / "layer_verification_output"

    export_path.mkdir(parents=True, exist_ok=True)

    # Use layer_data_path if provided, otherwise fall back to export_path
    if layer_data_path is None:
        layer_data_path = export_path

    layer_data_path.mkdir(parents=True, exist_ok=True)

    def run_verification(export_path: Path, layer_data_path: Path) -> None:
        # Parse device specs
        device_specs = DevicesOptionType.device_specs(devices)
        print(f"device_specs: {device_specs}")

        try:
            # Capture layer outputs to layer data directory
            print(f"Capturing layers to: {layer_data_path}")
            print(f"Saving reports to: {export_path}")

            # Define the prompt to use for both MAX and PyTorch
            prompt = "The meaning of life is not a single enumerator"

            # Capture PyTorch layer outputs
            print(f"Capturing PyTorch layer outputs for {pipeline}...")
            torch_layers = capture_torch_layer_outputs(
                pipeline=pipeline,
                encoding=encoding,
                device=devices,
                prompt=prompt,
                export_path=layer_data_path,  # Use layer data path
            )

            # Capture MAX layer outputs
            print(f"Capturing MAX layer outputs for {pipeline}...")
            max_layers = capture_max_layer_outputs(
                pipeline=pipeline,
                encoding=encoding,
                device_specs=device_specs,
                prompt=prompt,
                input_injection=input_injection,
                torch_layers=torch_layers if input_injection else None,
                export_path=layer_data_path,  # Use layer data path
            )

            # Use layer data paths for tensor data
            max_export_path = layer_data_path / "max_layers"
            torch_export_path = layer_data_path / "torch_layers"

            # Compare layer outputs (save reports to export_path)
            print("Comparing layer outputs...")
            layer_results = compare_layer_outputs(
                max_layers=max_layers,
                torch_layers=torch_layers,
                max_export_path=max_export_path,
                torch_export_path=torch_export_path,
                plot_mse=True,
                save_plot_path=str(
                    export_path / "mse_plot.png"
                ),  # Reports go to export_path
                save_report_path=str(
                    export_path  # Reports go to export_path
                    / f"layer_verification_report_{pipeline}_{encoding}.txt"
                ),
                layer_data_path=layer_data_path,  # For other report files
            )

            return

        except Exception as e:
            print(f"Error during layer verification: {e}")
            traceback.print_exc()
            return

    return run_verification(export_path, layer_data_path)
