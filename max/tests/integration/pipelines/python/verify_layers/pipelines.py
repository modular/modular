# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Data models for layer verification."""

from __future__ import annotations

from multiprocessing import Process
from pathlib import Path
from typing import Optional

from verify_layers.runner import run_layer_verification


def run_verification_subprocess(
    device_str: str,
    pipeline: str,
    encoding: str,
    export_path: Path,
    layer_data_path: Path,
    input_injection: bool,
) -> None:
    run_layer_verification(
        device_type=device_str,
        devices=device_str,
        pipeline=pipeline,
        encoding=encoding,
        export_path=export_path,
        layer_data_path=layer_data_path,
        input_injection=input_injection,
    )


class PipelineRunner:
    """Runner for pipeline verification with GPU memory cleanup between runs."""

    def __init__(
        self,
        pipeline: str,
        encoding: str,
        device_str: str,
        export_path: Optional[Path] = None,
        layer_data_path: Optional[Path] = None,
    ) -> None:
        self.pipeline = pipeline
        self.encoding = encoding
        self.export_path = export_path
        self.layer_data_path = layer_data_path
        self.device_str = device_str

    def run(self) -> None:
        """Run each scenario in a separate subprocess to free GPU memory between runs."""

        # Run without input injection
        print("\n" + "🔄 SCENARIO 1: Running WITHOUT input injection...")
        scenario1_export_path = (
            self.export_path / "without_injection" if self.export_path else None
        )
        scenario1_layer_data_path = (
            self.layer_data_path / "without_injection"
            if self.layer_data_path
            else None
        )

        process1 = Process(
            target=run_verification_subprocess,
            args=(
                self.device_str,
                self.pipeline,
                self.encoding,
                scenario1_export_path,
                scenario1_layer_data_path,
                False,  # input_injection
            ),
        )
        process1.start()
        process1.join()

        # Run with input injection
        print("\n" + "🔄 SCENARIO 2: Running WITH input injection...")
        scenario2_export_path = (
            self.export_path / "with_injection" if self.export_path else None
        )
        scenario2_layer_data_path = (
            self.layer_data_path / "with_injection"
            if self.layer_data_path
            else None
        )

        process2 = Process(
            target=run_verification_subprocess,
            args=(
                self.device_str,
                self.pipeline,
                self.encoding,
                scenario2_export_path,
                scenario2_layer_data_path,
                True,  # input_injection
            ),
        )
        process2.start()
        process2.join()

        print(
            "\n✅ Both scenarios completed with GPU memory cleanup between runs."
        )
