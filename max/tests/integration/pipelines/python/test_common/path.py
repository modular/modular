# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Common path-related functions."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Literal

from max.pipelines.lib import SupportedEncoding


def find_runtime_path(
    fname: str,
    local_dir: Path | None = None,
    bazel_dir: Path = Path("test_llama_golden"),
) -> Path:
    """Returns the runtime path for the a file.

    Args:
        fname: Filename.
        local_dir: Path to a local testdirectory that this file is located
            in (when not running in bazel).
        bazel_dir: Name of the bazel directory that this file is in. If
            the data is in an `http_archive`, this value should be set to the
            name of the http archive.

    Returns:
        Path to file if it exists.

    Raises:
        Error if file could not be found.
    """
    try:
        from python.runfiles import runfiles
    except ModuleNotFoundError:
        # Default to expecting data in the local testdata directory when running
        # outside Bazel.
        return _find_local_path(fname, local_dir)

    r = runfiles.Create()
    assert r
    path = r.Rlocation(str(bazel_dir / fname))

    if path is None:
        raise FileNotFoundError(f"Runtime path for {fname} was not found.")
    else:
        print(f"Runtime path for {fname} was located at {path}")
    return Path(path)


def _find_local_path(fname: str, dir: Path | None):
    path = (dir / fname) if dir else Path(fname)
    if path.exists():
        return path
    else:
        abspath = os.path.abspath(path)
        raise FileNotFoundError(f"File {abspath} does not exist.")


def golden_data_fname(
    model: str,
    encoding: SupportedEncoding,
    *,
    framework: Literal["max", "torch"] = "max",
):
    """Returns the golden json filename."""
    # TODO(MSDK-948): Actually support a distinction between device and encoding
    # instead of letting bfloat16 _imply_ GPU as is done multiple times in this file
    if encoding == "bfloat16":
        result = subprocess.run(
            [os.environ["MODULAR_CUDA_QUERY_PATH"]], stdout=subprocess.PIPE
        )
        hardware = (
            result.stdout.decode()
            .split("name:")[1]
            .split("\n")[0]
            .strip()
            .replace(" ", "")
        )
    else:
        # TODO: MSDK-968 address the hardware variance that makes
        # this untenable
        # info = _system_info()
        # hardware = info["arch"]
        hardware = "all"

    # This becomes a file path
    hardware = hardware.replace(" ", "")

    if framework == "max":
        return f"{model}_{encoding}_{hardware}_golden.json"
    else:
        return f"{framework}_{model}_{encoding}_{hardware}_golden.json"


def _system_info():
    result = subprocess.run(
        [os.environ["MODULAR_SYSTEM_INFO_PATH"]], stdout=subprocess.PIPE
    )
    system_info = {}
    for line in result.stdout.decode().split("\n"):
        try:
            k, v = line.split(": ")
            system_info[k.strip()] = v.strip()
        except:
            pass

    return system_info
