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
"""Regression: Mojo runtime init when interpreter ops load on a background thread.

``InferenceSession`` on the main thread creates the global MAX context and
AsyncRT runtime. ``max._interpreter_ops`` pulls in Mojo-built extension modules;
their ``PyInit`` / ``PythonModuleBuilder.finalize`` path calls
``_ensure_runtime_init``, which must register the CompilerRT global runtime via
``GetOrCreateRuntime`` without relying on another thread's TLS (see
``std.builtin._startup``).

Uses a subprocess so import order is not affected by other tests in the same
pytest worker process.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import python.runfiles


def _subprocess_entry_script(relative_repo_path: str) -> Path:
    """Resolve a workspace test helper under the Bazel runfiles root."""
    runfiles = python.runfiles.Create()
    assert runfiles is not None, "runfiles unavailable under bazel test"
    rloc = f"_main/{relative_repo_path}"
    loc = runfiles.Rlocation(rloc)
    assert loc is not None, f"Rlocation {rloc!r} could not be resolved"
    return Path(loc)


def _run_subprocess_entry(
    relative_repo_path: str,
) -> subprocess.CompletedProcess[str]:
    script = _subprocess_entry_script(relative_repo_path)
    return subprocess.run(
        [sys.executable, str(script)],
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )


def test_import_interpreter_ops_on_thread_after_inference_session_subprocess() -> (
    None
):
    """Background thread: first import of Mojo ops after session on main thread."""
    result = _run_subprocess_entry(
        "max/tests/tests/mojo_runtime_background_thread_subprocess_thread.py",
    )
    assert result.returncode == 0, (
        f"subprocess failed rc={result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def test_import_interpreter_ops_via_asyncio_to_thread_after_session_subprocess() -> (
    None
):
    """Asyncio default thread pool: import Mojo ops after session on main thread."""
    result = _run_subprocess_entry(
        "max/tests/tests/mojo_runtime_background_thread_subprocess_asyncio.py",
    )
    assert result.returncode == 0, (
        f"subprocess failed rc={result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
