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
"""Tests for the open-file soft limit raise."""

from __future__ import annotations

import resource
import subprocess
import sys
from collections.abc import Iterator

import pytest
from max.experimental.cascade.core.fd_limits import raise_nofile_soft_limit


@pytest.fixture
def nofile_limits() -> Iterator[tuple[int, int]]:
    """Yields the current limits and restores them after the test."""
    limits = resource.getrlimit(resource.RLIMIT_NOFILE)
    yield limits
    resource.setrlimit(resource.RLIMIT_NOFILE, limits)


def test_raises_launchd_default_to_capped_hard_limit(
    nofile_limits: tuple[int, int],
) -> None:
    # The test environment usually leaves the hard limit unlimited, so on
    # macOS this also covers the launchd case where the target is the
    # `kern.maxfilesperproc` cap rather than the hard limit.
    _, hard = nofile_limits
    resource.setrlimit(resource.RLIMIT_NOFILE, (256, hard))
    raise_nofile_soft_limit()
    expected = hard
    if sys.platform == "darwin":
        cap = subprocess.check_output(["sysctl", "-n", "kern.maxfilesperproc"])
        expected = min(hard, int(cap))
    assert resource.getrlimit(resource.RLIMIT_NOFILE) == (expected, hard)


def test_never_lowers_soft_limit(nofile_limits: tuple[int, int]) -> None:
    _, hard = nofile_limits
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    raise_nofile_soft_limit()
    assert resource.getrlimit(resource.RLIMIT_NOFILE) == (hard, hard)
