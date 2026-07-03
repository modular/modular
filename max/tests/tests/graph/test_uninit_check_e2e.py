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
"""End-to-end test for uninitialized memory read detection via the Python API.

Verifies that MODULAR_MAX_DEBUG_UNINITIALIZED_READ_CHECK=true enables the
full pipeline: debug allocator poison + Mojo MOJO_STDLIB_SIMD_UNINIT_CHECK
define.

Each test spawns a subprocess so the env var is read fresh by
InferenceSession.__init__.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from max.driver import accelerator_count

_SCRIPTS_DIR = Path(__file__).parent / "uninit_check"


def test_uninit_check_no_false_positives_e2e() -> None:
    """A well-behaved model should run without aborting when the check is on."""
    result = subprocess.run(
        [sys.executable, str(_SCRIPTS_DIR / "well_behaved_model.py")],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=120,
        env={
            **os.environ,
            "MODULAR_MAX_DEBUG_UNINITIALIZED_READ_CHECK": "true",
        },
    )
    assert result.returncode == 0, (
        f"Model execution failed (exit code {result.returncode}).\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "ALL CHECKS PASSED" in result.stdout


def test_uninit_check_disabled_by_default_e2e() -> None:
    """Without the env var, the check should be disabled."""
    env = {
        k: v
        for k, v in os.environ.items()
        if k != "MODULAR_MAX_DEBUG_UNINITIALIZED_READ_CHECK"
    }
    result = subprocess.run(
        [sys.executable, str(_SCRIPTS_DIR / "well_behaved_model.py")],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=120,
        env=env,
    )
    assert result.returncode == 0, (
        f"Model execution failed (exit code {result.returncode}).\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "ALL CHECKS PASSED" in result.stdout


def test_uninit_check_env_var_sets_allocator_and_define() -> None:
    """InferenceSession should set the allocator env var and the define."""
    result = subprocess.run(
        [sys.executable, str(_SCRIPTS_DIR / "env_var_plumbing.py")],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=120,
        env={
            **os.environ,
            "MODULAR_MAX_DEBUG_UNINITIALIZED_READ_CHECK": "true",
        },
    )
    assert result.returncode == 0, (
        f"Plumbing check failed (exit code {result.returncode}).\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "ALLOCATOR_SET" in result.stdout
    assert "PLUMBING_OK" in result.stdout


@pytest.mark.skipif(
    accelerator_count() == 0,
    reason="GPU required — debug allocator poisons device memory",
)
def test_uninit_check_detects_uninitialized_read_e2e() -> None:
    """The debug allocator poisons device memory; reading it on the CPU
    side of execute() should abort with a clear message."""
    result = subprocess.run(
        [sys.executable, str(_SCRIPTS_DIR / "trigger_uninit_read.py")],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=120,
        env={
            **os.environ,
            "MODULAR_MAX_DEBUG_UNINITIALIZED_READ_CHECK": "true",
            "UNINIT_OPS_PATH": os.environ["UNINIT_OPS_PATH"],
        },
    )
    combined_output = result.stdout + result.stderr
    assert result.returncode != 0, (
        "Expected the process to abort, but it exited successfully.\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "NO ABORT" not in result.stdout, (
        "The process ran to completion without detecting the "
        "uninitialized read."
    )
    # The abort message appears on stdout; the debug allocator logs to
    # stderr.  Either confirms the poison was detected.
    assert (
        "use of uninitialized memory" in combined_output
        or "DEBUG_ALLOC_POISON" in combined_output
    ), (
        "Expected poison detection output.\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
