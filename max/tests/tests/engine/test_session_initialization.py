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


def _run(script: str, config_dir: Path, num_threads: int | None = None) -> None:
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(sys.path),
        "MODULAR_HOME": str(config_dir),
    }
    if num_threads is None:
        env.pop("MODULAR_RUNTIME_NUM_THREADS", None)
    else:
        env["MODULAR_RUNTIME_NUM_THREADS"] = str(num_threads)
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("setting", "configured"),
    [(None, 0), ("2", 2), ("0x2", 2), ("010", 8)],
    ids=["unset", "decimal", "hexadecimal", "octal"],
)
def test_session_thread_argument(
    setting: str | None, configured: int, tmp_path: Path
) -> None:
    if setting is not None:
        (tmp_path / "modular.cfg").write_text(
            f"[runtime]\nnum_threads = {setting}\n"
        )
    _run(
        rf"""
        import ctypes
        import warnings
        import pytest
        from max.engine import InferenceSession

        warnings.simplefilter("error", DeprecationWarning)
        size_t_bits = 8 * ctypes.sizeof(ctypes.c_size_t)
        with pytest.raises(ValueError, match="does not match runtime[.]num_threads="):
            InferenceSession(num_threads=2**size_t_bits - 1)
        for count in (-1, 2**size_t_bits):
            with pytest.raises(ValueError, match="num_threads must be between"):
                InferenceSession(num_threads=count)
        with pytest.raises(ValueError, match=r"runtime[.]num_threads={configured}\b"):
            InferenceSession(num_threads=1)

        session = InferenceSession()
        assert InferenceSession(num_threads=None).devices == session.devices
        with pytest.warns(DeprecationWarning, match="Omit num_threads") as caught:
            assert InferenceSession(num_threads=0).devices == session.devices
        assert caught[0].filename == "<string>"
        """,
        tmp_path,
    )


@pytest.mark.parametrize("num_threads", [1, 2])
@pytest.mark.parametrize(
    "first",
    ["InferenceSession()", "CPU()", 'Graph("first")'],
    ids=["session-first", "cpu-first", "graph-first"],
)
def test_runtime_initialization_order(
    first: str, num_threads: int, tmp_path: Path
) -> None:
    _run(
        f"""
        import os
        import pytest
        from max.driver import CPU
        from max.engine import InferenceSession
        from max.graph import Graph

        first = {first}
        os.environ["MODULAR_RUNTIME_NUM_THREADS"] = "3"
        with pytest.warns(DeprecationWarning):
            session = InferenceSession(devices=[CPU()], num_threads={num_threads})
        with Graph("after_session"):
            assert session.devices == [CPU()]
        """,
        tmp_path,
        num_threads,
    )
