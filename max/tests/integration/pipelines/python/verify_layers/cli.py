# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Command-line interface for layer verification tool."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import click

from verify_layers.constants import DeviceKind
from verify_layers.pipelines import PipelineRunner


@click.command()
@click.option(
    "--devices", "devices_str", help="Devices to run pipeline on", default="cpu"
)
@click.option("--pipeline", help="Pipeline to run", required=True)
@click.option(
    "--encoding", help="Encoding to use for the pipeline", required=True
)
@click.option(
    "--export-path",
    "export_path",
    type=click.Path(path_type=Path),
    help="Export directory for all output files and results",
    default="layer_verification_output",
)
@click.option(
    "--save-layers",
    "save_layers",
    is_flag=True,
    default=False,
    help="Save layer data permanently in export-path. If False, uses temporary directories for layer data.",
)
def main(
    devices_str: str,
    pipeline: str,
    encoding: str,
    export_path: Path,
    save_layers: bool,
) -> None:
    """Run layer-by-layer verification between MAX and PyTorch models."""

    if workspace_dir := os.getenv("BUILD_WORKSPACE_DIRECTORY"):
        os.chdir(workspace_dir)

    # Ensure working directory exists
    export_path.mkdir(parents=True, exist_ok=True)

    # Create layer data path based on save_layers option
    if save_layers:
        layer_data_path = export_path
        print(f"Saving layer data to: {layer_data_path}")
    else:
        # Create a single temporary directory for all layer data
        temp_dir = tempfile.mkdtemp(prefix="layer_verification_data_")
        layer_data_path = Path(temp_dir)

    device_type = DeviceKind.CPU if "cpu" in devices_str else DeviceKind.GPU

    pipeline_runner = PipelineRunner(
        pipeline=pipeline,
        encoding=encoding,
        export_path=export_path,
        layer_data_path=layer_data_path,  # Pass the layer data path
        device_str=devices_str,
    )

    print(f"Running layer verification for pipeline: {pipeline}")
    print(f"Encoding: {encoding}")
    print(f"Device: {devices_str}")
    print(f"Export path: {export_path}")
    print(f"Layer data path: {layer_data_path}")

    print("\n" + "=" * 60)
    print("DUAL-MODE VERIFICATION: Running both scenarios automatically")
    print("  1. WITHOUT input injection (standard mode)")
    print("  2. WITH input injection (PyTorch inputs injected into MAX)")
    print("=" * 60)

    # Run layer verification for both scenarios
    pipeline_runner.run()

    print("\n" + "=" * 70)
    print("LAYER VERIFICATION COMPLETE - BOTH SCENARIOS")
    print("=" * 70)


if __name__ == "__main__":
    main()
