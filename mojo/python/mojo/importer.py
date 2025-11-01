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
import subprocess
import sys
from collections.abc import Sequence
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec
from importlib.util import spec_from_file_location
from pathlib import Path
from typing import ClassVar

from .paths import MojoCompilationError, MojoModulePath, find_mojo_module_in_dir
from .run import subprocess_run_mojo

# ---------------------------------------
# Helper Functions
# ---------------------------------------


def _calculate_mojo_source_hash(mojo_dir: Path) -> str:
    """Calculates a truncated SHA256 hash of all .mojo/.🔥 files in a directory."""
    if not mojo_dir.is_dir():
        raise ImportError(
            f"Expected mojo_dir to be a directory, got: {mojo_dir}"
        )

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
            hasher.update(str(file_path.relative_to(mojo_dir)).encode())
            hasher.update(file_path.read_bytes())
        except (ValueError, UnicodeError, OSError) as exc:
            raise ImportError(
                f"Could not process Mojo source file '{file_path}' for hashing"
            ) from exc

    # Return only the first 16 characters of the hex digest, since the full
    # hash is quite long and this is just a best-effort heuristic to check for
    # changes.
    return hasher.hexdigest()[:16]


def _compile_mojo_to_so(root_mojo_path: Path, output_so_path: Path) -> None:
    """Compiles a Mojo file to a shared object library."""
    if not root_mojo_path.is_file():
        raise ImportError(
            f"Expected root_mojo_path to be a file, got: {root_mojo_path}"
        )
    if not output_so_path.suffix == ".so":
        raise ImportError(
            f"Expected output_so_path to have .so suffix, got: {output_so_path}"
        )

    mojo_cli_args = [
        "build",
        str(root_mojo_path),
        "--emit",
        "shared-lib",
        "-o",
        str(output_so_path),
    ]

    try:
        subprocess_run_mojo(mojo_cli_args, capture_output=True, check=True)
    except subprocess.CalledProcessError as exc:
        error = MojoCompilationError.from_subprocess_error(
            root_mojo_path, mojo_cli_args, exc
        )
        raise ImportError(
            "Import of Mojo module failed due to compilation error."
        ) from error
    except FileNotFoundError as exc:
        raise ImportError(
            "Mojo executable not found via subprocess_run_mojo."
        ) from exc


# TODO: Instead of being careful about only deleting old files, we could just
#   delete all files in the cache directory?
def _delete_old_cached_files(
    cache_dir: Path,
    stem: str,
    extension: str = "so",
) -> None:
    """Removes outdated cache files for a given Mojo module."""
    if not cache_dir.is_dir():
        return

    for old_cache_file in cache_dir.glob(f"{stem}.*.{extension}"):
        old_cache_file.unlink(missing_ok=True)


# ---------------------------------------
# Define custom Mojo importer
# ---------------------------------------
# Resources:
#    https://docs.python.org/3/reference/import.html#the-meta-path
#    https://docs.python.org/3/library/importlib.html#importlib.abc.MetaPathFinder.find_spec
#    https://docs.python.org/3/library/importlib.html#importlib.machinery.ExtensionFileLoader
#    https://peps.python.org/pep-0489/#module-creation-phase
class MojoImporter(MetaPathFinder):
    caches: ClassVar[list[Path]] = []

    def find_spec(
        self,
        name: str,
        import_path: Sequence[str] | None,
        target: object | None,
    ) -> ModuleSpec | None:
        search_path = import_path if import_path else sys.path

        mojo_module: MojoModulePath | None = None
        for path_entry in search_path:
            mojo_module = find_mojo_module_in_dir(
                dir_path=Path(path_entry),
                module_name=name.split(".")[-1],
            )
            if mojo_module:
                break

        # If no Mojo source found, let other importers handle it.
        if not mojo_module:
            return None

        mojo_source_path = mojo_module.path
        mojo_source_dir = mojo_source_path.parent

        cache_dir = mojo_source_dir / "__mojocache__"
        source_file_hash = _calculate_mojo_source_hash(mojo_source_dir)

        mojo_extension = (
            cache_dir / f"{mojo_source_path.stem}.{source_file_hash}.so"
        )

        if not mojo_extension.is_file():
            cache_dir.mkdir(exist_ok=True)
            _delete_old_cached_files(
                cache_dir,
                stem=mojo_source_path.stem,
                extension="so",
            )
            _compile_mojo_to_so(mojo_source_path, mojo_extension)

        if not mojo_extension.is_file():
            raise ImportError(
                f"Failed to compile Mojo module '{name}' to shared object."
            )

        self.add_cached_file(mojo_extension)

        # Constructs an ExtensionFileLoader based on the compiled .so file extension.
        return spec_from_file_location(
            name,
            str(mojo_extension),
            submodule_search_locations=None,
        )

    @classmethod
    def add_cached_file(cls, cache_file: Path) -> None:
        """Registers a compiled Mojo cached file for future invalidation."""
        cls.caches.append(cache_file)

    @classmethod
    def invalidate_caches(cls) -> None:
        """Invalidates all cached Mojo compiled files."""
        for cached_file in cls.caches:
            cached_file.unlink(missing_ok=True)


# -------------------------------------------------------
# Side Effect: Add custom importer to the Python metapath
# -------------------------------------------------------
sys.meta_path.append(MojoImporter())  # type: ignore
