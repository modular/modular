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

"""Session initialization regressions for modular/modular#7531.

The CPU runtime and Python CPU cache are process-wide, so each lifecycle runs
in a fresh interpreter. These tests initialize sessions and graphs only; they
do not compile or execute models.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


def _run(script: str, config_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(sys.path),
            # Isolate local modular.cfg settings and any inherited thread count.
            "MODULAR_HOME": str(config_dir),
            "MODULAR_RUNTIME_NUM_THREADS": "0",
        },
    )


def _assert_success(result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode == 0, (
        f"subprocess failed rc={result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


@pytest.mark.parametrize("num_threads", [1, 2])
def test_explicit_thread_count_initializes_host_device_and_graph(
    num_threads: int, tmp_path: Path
) -> None:
    result = _run(
        f"""
        import resource
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        from max.engine import InferenceSession
        session = InferenceSession(num_threads={num_threads})

        # Inspect the host device only after reproducing the issue: constructing
        # CPU() first would establish a different initialization order.
        from max.driver import CPU
        assert session.devices == [CPU()]
        from max.graph import Graph
        with Graph("after_explicit_session"):
            pass
        """,
        tmp_path,
    )
    _assert_success(result)


@pytest.mark.parametrize("num_threads", [1, 2])
def test_compatible_sessions_reuse_configuration(
    num_threads: int, tmp_path: Path
) -> None:
    result = _run(
        f"""
        from max.engine import InferenceSession
        first = InferenceSession(num_threads={num_threads})
        matching = InferenceSession(num_threads={num_threads})
        assert first.devices == matching.devices
        """,
        tmp_path,
    )
    _assert_success(result)


def test_default_sessions_and_graph(tmp_path: Path) -> None:
    result = _run(
        """
        from max.engine import InferenceSession
        first = InferenceSession()
        second = InferenceSession()
        zero = InferenceSession(num_threads=0)
        assert first.devices == second.devices == zero.devices
        from max.graph import Graph
        with Graph("after_default_session"):
            pass
        """,
        tmp_path,
    )
    _assert_success(result)


@pytest.mark.parametrize("num_threads", [None, 2])
def test_thread_configuration_conflict_is_rejected(
    num_threads: int | None, tmp_path: Path
) -> None:
    result = _run(
        f"""
        import resource
        # The expected native fatal error must not leave a core dump in CI.
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        from max.engine import InferenceSession
        first = InferenceSession(num_threads=1)
        print("FIRST_SESSION_READY", flush=True)
        conflicting = InferenceSession(num_threads={num_threads!r})
        """,
        tmp_path,
    )
    diagnostics = (
        f"rc={result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    # An import failure or an abort in the first session must not count as
    # successfully rejecting the second session's incompatible request.
    assert "FIRST_SESSION_READY" in result.stdout, diagnostics
    assert result.returncode != 0, diagnostics
    assert "differentInit::Options" in result.stderr, diagnostics
