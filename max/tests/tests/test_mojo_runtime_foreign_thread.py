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
"""Regression: Mojo CompilerRT calls on a foreign thread must not segfault.

GEX-3592: ``Runtime::getCurrentRuntimeOrNull()`` returned null for threads not
managed by AsyncRT, causing any Mojo CompilerRT function that dereferenced the
result to segfault.
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


def test_compile_and_execute_on_asyncio_thread_subprocess() -> None:
    """Asyncio thread pool: compile and run a model on a foreign thread.

    Regression test for GEX-3592.  The ``InferenceSession`` is created on the
    event-loop thread; ``asyncio.to_thread`` spawns a ``ThreadPoolExecutor``
    thread that has no AsyncRT TLS.  The test verifies that compiling and
    executing a MAX graph from that thread does not segfault and produces the
    correct output.
    """
    result = _run_subprocess_entry(
        "max/tests/tests/mojo_runtime_foreign_thread_subprocess.py",
    )
    assert result.returncode == 0, (
        f"subprocess failed rc={result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
