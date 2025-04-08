# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import subprocess
import sys
import tempfile
from logging import (
    debug as log_debug,
)
from logging import (
    error as log_error,
)
from pathlib import Path


def _eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


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

    # FIXME(GEX-2032): Delete this source package to avoid cluttering
    #   the users temporary directory.
    tmp = tempfile.NamedTemporaryFile(suffix=".mojopkg", delete=False)

    try:
        # TODO(GEX-2033): Either locate `mojo` more robustly, so this still
        #   works when `mojo` is not on the users runtime `PATH`, or call
        #   directly into the lower-level Mojo compiler packaging code.
        package_result = subprocess.run(
            ["mojo", "package", str(path), "-o", tmp.name],
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        log_error(
            "ERROR: `mojo package` invocation failed with exit code",
            e.returncode,
            "while building source package at:\n  ",
            path,
        )
        log_debug("\nRaw `mojo package` output follows.")
        log_debug("\n===== STDERR =====\n")
        log_debug(e.stderr.decode("utf-8"))
        log_debug("\n===== STDOUT =====\n")
        log_debug(e.stdout.decode("utf-8"))
        log_debug(
            "\n\nERROR: `mojo package` invocation failed. See above"
            " output for more information."
        )

        raise RuntimeError(
            f"An error occurred compiling the specified Mojo source package at {path}."
        ) from e

    return Path(tmp.name)
