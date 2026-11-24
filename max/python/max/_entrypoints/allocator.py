# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Puts the ``max`` CLI process on jemalloc instead of glibc malloc."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from mojo._package_root import get_package_root

logger = logging.getLogger("max._entrypoints")

_LIBRARY_NAME = "libjemalloc_preload.so"
_RUNFILES_PATH = f"_main/AsyncRT/{_LIBRARY_NAME}"
_ALLOCATOR_ENV_VAR = "MODULAR_MAX_ALLOCATOR"
_REEXEC_ENV_VAR = "MODULAR_MAX_ALLOCATOR_REEXEC"


def _library_mapped(preload: str) -> bool:
    """Reports whether the preload library is loaded into this process.

    The memory map is the ground truth: after a re-exec ``LD_PRELOAD`` names
    the library whether or not the dynamic loader managed to load it. The
    variable is consulted only when the map cannot be read.
    """
    try:
        with open("/proc/self/maps") as maps:
            return any(_LIBRARY_NAME in line for line in maps)
    except OSError:
        return _LIBRARY_NAME in preload


def _installed_library() -> Path | None:
    try:
        root = get_package_root()
    except RuntimeError:
        return None
    return None if root is None else root / "lib" / _LIBRARY_NAME


def _runfiles_library() -> Path | None:
    """Locates the library under Bazel without depending on rules_python.

    The wheel cannot carry ``python.runfiles``, so this reads the two
    variables Bazel sets for a running binary: a manifest of
    ``rlocation real_path`` lines, or a materialized runfiles directory.
    """
    if manifest := os.environ.get("RUNFILES_MANIFEST_FILE"):
        try:
            with open(manifest) as entries:
                for entry in entries:
                    rlocation, _, real_path = entry.rstrip("\n").partition(" ")
                    if rlocation == _RUNFILES_PATH:
                        return Path(real_path)
        except OSError:
            return None
    if runfiles_dir := os.environ.get("RUNFILES_DIR"):
        return Path(runfiles_dir) / _RUNFILES_PATH
    return None


def _find_library() -> Path | None:
    for candidate in (_installed_library(), _runfiles_library()):
        if candidate is not None and candidate.is_file():
            return candidate
    return None


def reexec_with_jemalloc() -> None:
    """Replaces this process with one that allocates through jemalloc.

    The graph compiler and the Mojo compiler run inside this process as shared
    libraries, so the allocator they use is the one the interpreter was started
    with. ``LD_PRELOAD`` is the only way to change it, and it has to be set
    before the process starts, hence the re-exec. Model-worker subprocesses
    inherit the variable and so inherit the allocator.

    Does nothing off Linux, when a sanitizer runtime already owns ``malloc``,
    when the library is not installed, or when ``MODULAR_MAX_ALLOCATOR`` is set
    to ``system`` rather than its default of ``jemalloc``.

    Only the ``max`` CLI calls this. A script that drives MAX through
    :class:`max.engine.InferenceSession` keeps the allocator its interpreter
    started with; to give it jemalloc too, put the installed library,
    ``modular/lib/libjemalloc_preload.so`` under site-packages, in
    ``LD_PRELOAD`` before starting Python.
    """
    if sys.platform != "linux":
        return

    requested = os.environ.get(_ALLOCATOR_ENV_VAR, "jemalloc")
    if requested != "jemalloc":
        if requested != "system":
            logger.warning(
                "Ignoring unrecognized %s=%s; expected 'jemalloc' or 'system'",
                _ALLOCATOR_ENV_VAR,
                requested,
            )
        return

    preload = os.environ.get("LD_PRELOAD", "")
    if "libclang_rt" in preload:
        return

    mapped = _library_mapped(preload)
    if _REEXEC_ENV_VAR in os.environ:
        if not mapped:
            logger.warning(
                "Re-executed with LD_PRELOAD=%s but jemalloc is not mapped;"
                " using the system allocator (the dynamic loader's own error"
                " above says why)",
                preload,
            )
        return
    if mapped:
        return

    library = _find_library()
    if library is None:
        logger.debug("No %s found; using the system allocator", _LIBRARY_NAME)
        return

    env = dict(os.environ)
    env["LD_PRELOAD"] = f"{library}:{preload}" if preload else str(library)
    env[_REEXEC_ENV_VAR] = "1"
    logger.debug("Re-executing with LD_PRELOAD=%s", env["LD_PRELOAD"])
    os.execve(sys.executable, sys.orig_argv, env)
