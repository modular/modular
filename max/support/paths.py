# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import hashlib
import logging
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


def _eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


class Error(Exception):
    """Base error for max paths."""


@dataclass
class MojoCompilationError(Error):
    """Error encountered compiling a Mojo source package."""

    path: Path
    command: list[str]
    stdout: str
    stderr: str

    def __str__(self):
        command = shlex.join(self.command)
        return (
            f"error compiling {self.path}. Command: {command}\n\n{self.stderr}"
        )


def is_mojo_source_package_path(path: Path) -> bool:
    """Returns True if the given path is a Mojo package source directory.

    A Mojo package source directory is a directory that contains an `__init__.mojo`
    or `__init__.🔥` file.

    Args:
        path: The path to check

    Returns:
        bool: True if the path is a Mojo source package directory
    """
    if not path.is_dir():
        return False

    init_mojo = path / "__init__.mojo"
    init_fire = path / "__init__.🔥"

    return init_mojo.is_file() or init_fire.is_file()


def is_mojo_binary_package_path(path: Path) -> bool:
    """Returns True if the given path is a Mojo binary package file, i.e.
    a file ending in ".mojopkg" or ".📦".
    """

    if not path.is_file():
        return False

    return path.suffix in [".mojopkg", ".📦"]


def _build_mojo_source_package(path: Path) -> Path:
    assert is_mojo_source_package_path(path)

    # Create a deterministic path in the temp directory based on the source path
    path_hash = hashlib.md5(str(path.absolute()).encode()).hexdigest()
    tmp_path = (
        Path(tempfile.gettempdir())
        / ".modular"
        / "mojo_pkg"
        / f"mojo_pkg_{path_hash}.mojopkg"
    )

    # Ensure parent directories exist
    tmp_path.parent.mkdir(parents=True, exist_ok=True)

    args = ["mojo", "package", str(path), "-o", str(tmp_path)]

    try:
        # TODO(GEX-2033): Either locate `mojo` more robustly, so this still
        #   works when `mojo` is not on the users runtime `PATH`, or call
        #   directly into the lower-level Mojo compiler packaging code.
        package_result = subprocess.run(args, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        error = MojoCompilationError(
            path, args, e.stdout.decode(), e.stderr.decode()
        )
        logging.error(str(error))
        raise error from e

    return tmp_path
