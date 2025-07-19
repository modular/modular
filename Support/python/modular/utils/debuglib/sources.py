# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Utilities for handling Mojo sources and compiled binaries."""

import subprocess
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any


class MojoCompilationError(Exception):
    pass


class MojoSource:
    """Class that represents a source file."""

    def __init__(self, path: Path) -> None:
        self.path = path

    @staticmethod
    def test_input(file_name: str):
        """Create a MojoSource that lives in the /Inputs directory."""
        return MojoSource(Path(__file__).parent.parent / "Inputs" / file_name)

    def find_lines_with_text(self, text: str) -> Generator[int, None, None]:
        """Generate the 1-indexed line numbers at which the given text is found
        in the source file."""
        with open(self.path) as source:
            i = 1
            for line in source:
                if text in line:
                    yield i
                i += 1

    def build(self, output_path: Path, suppress_output: bool = False) -> None:
        """Build the SourceFile as a binary in the given `output_path`."""

        try:
            subprocess.run(
                [
                    "mojo",
                    "build",
                    "--debug-level",
                    "full",
                    "-O0",
                    self.path,
                    "-o",
                    output_path,
                ],
                stdout=subprocess.DEVNULL if suppress_output else None,
                stderr=subprocess.DEVNULL if suppress_output else None,
                check=True,
            )
        except Exception as e:
            raise MojoCompilationError(
                f"Couldn't build the mojo file {self.path}."
            ) from e


class MojoBinary:
    def __init__(
        self,
        source: MojoSource,
        suppress_build_output: bool = False,
    ) -> None:
        self.source = source
        self.suppress_build_output = suppress_build_output
        self.out_dir = tempfile.TemporaryDirectory()
        self.bin_file = Path(self.out_dir.name) / (
            self.source.path.name + ".exe"
        )
        self.source.build(self.bin_file, self.suppress_build_output)

    def __enter__(self) -> Path:
        return self.bin_file

    def __exit__(self, exc: Any, value: Any, tb: Any):
        self.out_dir.cleanup()
