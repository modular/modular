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
from typing import Optional


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


@dataclass
class MojoModulePath:
    """Represents a path to the root file of a Mojo module on the file system."""

    path: Path
    """Mojo source file that is the root of the module. Either an `__init__` file
    or a signle-file module. """

    is_package: bool


def is_mojo_source_package_path(path: Path) -> bool:
    """Returns True if the given path is a Mojo package source directory.

    A Mojo package source directory is a directory that contains an `__init__.mojo`
    or `__init__.🔥` file.

    Args:
        path: The path to check

    Returns:
        bool: True if the path is a Mojo source package directory
    """
    return _mojo_source_package_root_file(path) != None


def find_mojo_module_in_dir(
    dir_path: Path, module_name: str
) -> Optional[MojoModulePath]:
    """Searches a directory for Mojo package or single file module.

    Returns:
        A `MojoModulePath` if found, otherwise None.
    """
    # Check for package first: <dir_path>/<module_name>/__init__.mojo or .🔥
    if init_file_path := _mojo_source_package_root_file(dir_path / module_name):
        return MojoModulePath(init_file_path, True)

    # If not a package, check for single file: <dir_path>/<module_name>.mojo or .🔥
    for ext in ["mojo", "🔥"]:
        potential_file = dir_path / f"{module_name}.{ext}"
        if potential_file.is_file():
            # Found single file.
            return MojoModulePath(potential_file, False)

    # Not found in this directory
    return None


def _mojo_source_package_root_file(path: Path) -> Optional[Path]:
    """Returns the path to the `__init__.mojo` or `__init__.🔥` package root
    file if this is a Mojo source package directory, otherwise None.

    A Mojo package source directory is a directory that contains an `__init__.mojo`
    or `__init__.🔥` file.

    Args:
        path: The path to check

    Returns:
        Path: Path to the root module file if the path is a Mojo source package
          directory
    """
    if not path.is_dir():
        return None

    init_mojo = path / "__init__.mojo"
    init_fire = path / "__init__.🔥"

    if init_mojo.is_file():
        return init_mojo
    elif init_fire.is_file():
        return init_fire
    else:
        return None


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
