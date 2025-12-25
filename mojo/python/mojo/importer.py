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
from collections import defaultdict
from collections.abc import Sequence
from importlib.util import spec_from_file_location
from pathlib import Path
from itertools import chain
import threading

from .paths import MojoCompilationError, MojoModulePath, find_mojo_module_in_dir
from .run import subprocess_run_mojo

# ---------------------------------------
# Helper Functions
# ---------------------------------------
_thread_local_storage = threading.local()

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


def _calculate_comptime_variables_hash(comptime_variables: dict[str, str] | None) -> str:
    """Calculates a truncated SHA256 hash of compile-time variables."""
    if not comptime_variables:
        return ""

    hasher = hashlib.sha256()
    # Sort items to ensure consistent hashing regardless of dict ordering
    for key, value in sorted(comptime_variables.items()):
        hasher.update(key.encode("utf-8"))
        hasher.update(b"=")
        hasher.update(value.encode("utf-8"))
        hasher.update(b"\0")  # null separator between entries

    return hasher.hexdigest()[:16]


def _compile_mojo_to_so(
    root_mojo_path: Path,
    output_so_path: Path,
    comptime_variables: dict[str, str] | None = None
) -> None:
    """Compiles a Mojo file to a shared object library."""
    # Assertions from _build_mojo_file_to_python_extension_module
    assert root_mojo_path.is_file()
    assert output_so_path.suffix == ".so"

    mojo_cli_args = [
        "build",
        str(root_mojo_path),
        "--emit",
        "shared-lib",
        "-o",
        str(output_so_path),
    ]

    # add comptime variables if set
    if comptime_variables is not None:
        extra_cli_args = [
            ('-D', f"{key}={value}")
            for key, value in comptime_variables.items()
        ]
        extra_cli_args_flat = chain.from_iterable(extra_cli_args)
        mojo_cli_args.extend(extra_cli_args_flat)

    try:
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


# TODO: Instead of being careful about only deleting old files, we could just
#   delete all files in the cache directory?
def _delete_matching_cached_files(
    cache_dir: Path, *, stem: str, ext: str
) -> None:
    """Removes outdated cache files for a given Mojo module."""
    if not cache_dir.is_dir():
        return

    for old_cache_file in cache_dir.glob(f"{stem}.*.{ext}"):
        os.remove(old_cache_file)


def set_comptime_variables(module_name: str, values: dict[str, str], ) -> None:
    """Set compile-time variables for a Mojo module imports."""
    # Initialize the comptime_variables_by_module dict if it doesn't exist
    if not hasattr(_thread_local_storage, 'comptime_variables_by_module'):
        _thread_local_storage.comptime_variables_by_module = defaultdict(dict)

    # Store variables for a specific module
    _thread_local_storage.comptime_variables_by_module[module_name].update(values)

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

        # Calculate hash of source.
        source_hash = _calculate_mojo_source_hash(mojo_dir)

        # Retrieve comptime variables for this specific module
        comptime_variables = _thread_local_storage.comptime_variables_by_module.get(name, None)

        # Calculate hash of comptime variables
        comptime_hash = _calculate_comptime_variables_hash(comptime_variables)

        # Build cache filename: <module>.<source_hash>.<comptime_hash>.so
        # If no comptime variables, the format is: <module>.<source_hash>..so
        cache_filename = f"{root_mojo_path.stem}.{source_hash}.{comptime_hash}.so"
        expected_cache_file = cache_dir / cache_filename

        # Compile if cache doesn't exist or is invalid
        if not expected_cache_file.is_file():
            # No matching cached file exists, so compile the Mojo source file.
            os.makedirs(cache_dir, exist_ok=True)
            # Delete any non-matching cached .so's, to prevent the number of
            # stale cached files from growing without bound.
            _delete_matching_cached_files(
                cache_dir, stem=root_mojo_path.stem, ext="so"
            )

            # compile mojo source
            _compile_mojo_to_so(
                root_mojo_path,
                expected_cache_file,
                comptime_variables=comptime_variables,
            )

        # If we reach here, expected_cache_file should exist (either pre-existing or just compiled)
        assert expected_cache_file.is_file()

        # Constructs an ExtensionFileLoader automatically based on the .so
        # file extension.
        return spec_from_file_location(
            name, str(expected_cache_file), submodule_search_locations=None
        )


# -------------------------------------------------------
# Side Effect: Add custom importer to the Python metapath
# -------------------------------------------------------

sys.meta_path.append(MojoImporter())  # type: ignore
