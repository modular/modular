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

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from mojo.paths import MojoCompilationError, _build_mojo_source_package


def _fake_precompile(args: list[str], **kwargs: object) -> MagicMock:
    """Stands in for `subprocess_run_mojo`, honoring the `-o` output contract
    the production code relies on: a successful run leaves the requested
    artifact on disk."""
    Path(args[args.index("-o") + 1]).write_bytes(b"mojoc")
    return MagicMock()


def test_build_mojo_source_package_path_is_user_specific() -> None:
    """The temp path must include the user ID to isolate per-user caches.

    Without user isolation, two OS users sharing /tmp will collide on
    /tmp/.modular/mojo_pkg/ — the second user cannot write into a
    directory owned by the first.
    """
    fake_src = Path("/fake/mojo/package")
    uid = os.getuid()

    with (
        patch("mojo.paths.is_mojo_source_package_path", return_value=True),
        patch("mojo.paths.subprocess_run_mojo", side_effect=_fake_precompile),
        tempfile.TemporaryDirectory() as tmp_dir,
        patch("mojo.paths.tempfile.gettempdir", return_value=tmp_dir),
    ):
        result = _build_mojo_source_package(fake_src)

    assert f".modular_{uid}" in str(result), (
        f"Expected path to contain '.modular_{uid}', got: {result}"
    )
    assert "/mojo_pkg/" in str(result)
    assert result.name.endswith(".mojoc")


def test_build_mojo_source_package_no_shared_directory_collision() -> None:
    """Verify the path does NOT use a shared .modular/ directory.

    The old path /tmp/.modular/mojo_pkg/ is shared across all users.
    The fixed path must use a user-specific prefix instead.
    """
    fake_src = Path("/fake/mojo/package")

    with (
        patch("mojo.paths.is_mojo_source_package_path", return_value=True),
        patch("mojo.paths.subprocess_run_mojo", side_effect=_fake_precompile),
        tempfile.TemporaryDirectory() as tmp_dir,
        patch("mojo.paths.tempfile.gettempdir", return_value=tmp_dir),
    ):
        result = _build_mojo_source_package(fake_src)

    # The path must NOT contain the old shared ".modular/mojo_pkg" pattern
    relative = str(result.relative_to(tmp_dir))
    assert not relative.startswith(".modular/mojo_pkg"), (
        f"Path still uses shared .modular/ directory: {result}"
    )


def test_build_mojo_source_package_publishes_atomically() -> None:
    """A successful build leaves exactly the final artifact in the cache dir.

    The cache path is keyed only by source path and shared between
    processes, so the compile must land in per-process staging and be
    published with an atomic rename — a concurrent reader must never see a
    half-written .mojoc, and no staging debris may accumulate.
    """
    fake_src = Path("/fake/mojo/package")

    with (
        patch("mojo.paths.is_mojo_source_package_path", return_value=True),
        patch("mojo.paths.subprocess_run_mojo", side_effect=_fake_precompile),
        tempfile.TemporaryDirectory() as tmp_dir,
        patch("mojo.paths.tempfile.gettempdir", return_value=tmp_dir),
    ):
        result = _build_mojo_source_package(fake_src)

        assert result.read_bytes() == b"mojoc"
        assert {p.name for p in result.parent.iterdir()} == {result.name}, (
            f"Staging debris left next to the artifact: "
            f"{sorted(p.name for p in result.parent.iterdir())}"
        )


def test_build_mojo_source_package_failure_leaves_no_artifact() -> None:
    """A failed compile must not publish anything nor leave staging debris.

    Before the atomic-rename publish, the compiler wrote the shared cache
    path in place, so a failed or interrupted compile could leave a partial
    file that a concurrent reader would load.
    """
    fake_src = Path("/fake/mojo/package")
    compile_error = subprocess.CalledProcessError(
        1, ["mojo"], output=b"", stderr=b"boom"
    )

    with (
        patch("mojo.paths.is_mojo_source_package_path", return_value=True),
        patch("mojo.paths.subprocess_run_mojo", side_effect=compile_error),
        tempfile.TemporaryDirectory() as tmp_dir,
        patch("mojo.paths.tempfile.gettempdir", return_value=tmp_dir),
    ):
        with pytest.raises(MojoCompilationError):
            _build_mojo_source_package(fake_src)

        cache_dir = Path(tmp_dir) / f".modular_{os.getuid()}" / "mojo_pkg"
        assert list(cache_dir.iterdir()) == [], (
            f"Failed build left files behind: "
            f"{sorted(p.name for p in cache_dir.iterdir())}"
        )
