# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

import hashlib
import logging
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from importlib.util import spec_from_file_location
from pathlib import Path
from uuid import uuid4

from filelock import FileLock, Timeout

from .paths import MojoCompilationError, MojoModulePath, find_mojo_module_in_dir
from .run import subprocess_run_mojo

# ---------------------------------------
# Helper Functions
# ---------------------------------------


def _calculate_mojo_source_hash(mojo_dir: Path) -> str:
    """Calculates a truncated SHA256 hash of all .mojo/.🔥 files in a directory."""
    # Find all .mojo and .🔥 files recursively
    source_files = sorted((*mojo_dir.rglob("*.mojo"), *mojo_dir.rglob("*.🔥")))

    if not source_files:
        # This should be unreachable if the caller validates that mojo_dir
        # contains Mojo source files before calling this function.
        raise ImportError(
            f"Internal Error: No .mojo or .🔥 files found in directory '{mojo_dir}' for hashing."
        )

    hasher = hashlib.sha256()
    for file_path in source_files:
        try:
            # Add file path to hash to distinguish identical content in different files
            hasher.update(str(file_path.relative_to(mojo_dir)).encode("utf-8"))
            # Add file content to hash
            with open(file_path, "rb") as f:
                hasher.update(f.read())
        except (ValueError, UnicodeError, OSError) as e:
            raise ImportError(
                f"Could not process Mojo source file '{file_path}' for hashing"
            ) from e

    # Return only the first 16 characters of the hex digest, since the full
    # hash is quite long and this is just a best-effort heuristic to check for
    # changes.
    return hasher.hexdigest()[:16]


def _compile_mojo_to_so(root_mojo_path: Path, output_so_path: Path) -> None:
    """Compiles a Mojo file to a shared object library."""
    # Assertions from _build_mojo_file_to_python_extension_module
    assert root_mojo_path.is_file()
    assert output_so_path.suffix == ".so"

    mojo_cli_args = [
        # First arg is implicitly the `mojo` executable (handled by subprocess_run_mojo)
        "build",
        str(root_mojo_path),
        "--emit",
        "shared-lib",
        "-o",
        str(output_so_path),
    ]

    try:
        # Run the `mojo` that's embedded in the `max` package layout via subprocess_run_mojo.
        subprocess_run_mojo(mojo_cli_args, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        error = MojoCompilationError.from_subprocess_error(
            root_mojo_path, mojo_cli_args, e
        )
        logging.error(str(error))
        # Propagate compilation errors as ImportError
        raise ImportError(
            "Import of Mojo module failed due to compilation error."
        ) from error
    except FileNotFoundError:
        # Handle case where mojo executable is not found
        raise ImportError(
            "Mojo executable not found via subprocess_run_mojo."
        ) from None


def _cleanup_stale_cache_files(
    cache_dir: Path, *, stem: str, current_hash: str
) -> None:
    """Removes outdated cache files for a given Mojo module.

    This function is tolerant to failures. The stale file may
    be cleaned up on a subsequent run.
    """
    if not cache_dir.is_dir():
        return

    current_cache_filename = f"{stem}.hash-{current_hash}.so"
    for old_cache_file in cache_dir.glob(f"{stem}.hash-*.so"):
        if old_cache_file.name == current_cache_filename:
            continue

        try:
            os.remove(old_cache_file)
        except (OSError, PermissionError) as e:
            logging.debug(
                f"Could not remove stale cache file '{old_cache_file}': {e}"
            )


# ---------------------------------------
# Define custom importer
# ---------------------------------------


# Resources:
#    https://docs.python.org/3/reference/import.html#the-meta-path
#    https://docs.python.org/3/library/importlib.html#importlib.abc.MetaPathFinder.find_spec
#    https://docs.python.org/3/library/importlib.html#importlib.machinery.ExtensionFileLoader
#    https://peps.python.org/pep-0489/#module-creation-phase
class MojoImporter:
    def find_spec(  # noqa: ANN201
        self,
        name: str,
        import_path: Sequence[str] | None,
        target: object | None,
    ):
        # This importer only handles top-level imports. `import foo.bar` is not
        # supported.
        if "." in name or import_path is not None:
            return None

        mojo_module: MojoModulePath | None = None
        # Search sys.path for the Mojo source file or package
        for path_entry in sys.path:
            # Use the helper function to check this directory
            mojo_module = find_mojo_module_in_dir(Path(path_entry), name)

            if mojo_module:
                break  # Found the source, stop searching sys.path

        # If no Mojo source found, let other importers handle it.
        if not mojo_module:
            return None

        # `root_mojo_path` is the path to the specific Mojo source file.
        root_mojo_path = mojo_module.path

        # `mojo_dir` is the directory containing the Mojo source file(s); it is
        # the directory that will be hashed to check for changes.
        mojo_dir = root_mojo_path.parent

        # Determine cache location and directory to hash
        cache_dir = mojo_dir / "__mojocache__"

        # Ensure cache directory exists.
        os.makedirs(cache_dir, exist_ok=True)

        # Calculate hash.
        current_hash = _calculate_mojo_source_hash(mojo_dir)

        expected_cache_file = (
            cache_dir / f"{root_mojo_path.stem}.hash-{current_hash}.so"
        )

        # Compile if cache doesn't exist or is invalid
        if not expected_cache_file.is_file():
            lock_path = cache_dir / f".{root_mojo_path.stem}.lock"
            # No matching cached file exists, so compile the Mojo source file.
            try:
                # Acquire a file lock to prevent concurrent compilations.
                with FileLock(lock_path, timeout=60):
                    # Double-check if file exists after acquiring lock.
                    if not expected_cache_file.is_file():
                        # Atomic write with temp file to prevent partial reads
                        tmp_so_path = (
                            cache_dir
                            / f"{root_mojo_path.stem}.hash-{current_hash}.{os.getpid()}.{uuid4().hex}.tmp"
                        )

                        try:
                            _compile_mojo_to_so(root_mojo_path, tmp_so_path)

                            # Move temp file to expected cache file atomically.
                            tmp_so_path.replace(expected_cache_file)
                        finally:
                            # Ensure the temporary file is cleaned up, even on failure.
                            if tmp_so_path.is_file():
                                os.remove(tmp_so_path)

                        # Safely clean up stale cache files.
                        # Only after the new cache file is successfully in place, remove
                        # ant outdated cache files for this module to prevent cache bloat.
                        _cleanup_stale_cache_files(
                            cache_dir,
                            stem=root_mojo_path.stem,
                            current_hash=current_hash,
                        )
            except Timeout:
                # briefly wait to see if the expected cache file appears
                time.sleep(0.2)
                if not expected_cache_file.is_file():
                    raise ImportError(
                        f"Could not acquire lock for compiling Mojo module '{name}' "
                        f"at '{lock_path}'. Another process may be holding it."
                    ) from None

        # A file may not be immediately visible or accessible to all processes after creation.
        # This short retry loop handles that transition inconsistency.
        for _ in range(3):
            if expected_cache_file.is_file() and os.access(
                expected_cache_file, os.R_OK
            ):
                break
            time.sleep(0.1)
        else:
            raise ImportError(
                f"Compiled Mojo module '{expected_cache_file}' not found or "
                "inaccessible after compilation."
            )

        # Constructs an ExtensionFileLoader automatically based on the .so
        # file extension.
        return spec_from_file_location(
            name, str(expected_cache_file), submodule_search_locations=None
        )


# -------------------------------------------------------
# Side Effect: Add custom importer to the Python metapath
# -------------------------------------------------------

sys.meta_path.append(MojoImporter())  # type: ignore
