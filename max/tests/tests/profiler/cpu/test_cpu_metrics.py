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
from collections import namedtuple

import psutil
import pytest
from max.profiler.cpu import (
    CPUMetrics,
    CPUMetricsCollector,
    collect_pids_for_port,
)
from pytest_mock import MockerFixture

# ---------------------------------------------------------------------------
# CPUMetrics dataclass
# ---------------------------------------------------------------------------


def test_cpu_metrics_fields() -> None:
    m = CPUMetrics(
        user=1.5,
        user_percent=75.0,
        system=0.5,
        system_percent=25.0,
        elapsed=2.0,
    )
    assert m.user == 1.5
    assert m.user_percent == 75.0
    assert m.system == 0.5
    assert m.system_percent == 25.0
    assert m.elapsed == 2.0


# ---------------------------------------------------------------------------
# CPUMetricsCollector
# ---------------------------------------------------------------------------

CpuTimes = namedtuple("CpuTimes", ["user", "system"])


def test_collector_current_process() -> None:
    """Collect stats for the current process — a basic integration smoke test."""
    pid = os.getpid()
    collector = CPUMetricsCollector([pid])
    with collector:
        # Burn a little CPU so user_percent > 0.
        total = sum(range(500_000))
        assert total >= 0
    stats = collector.get_stats()

    assert isinstance(stats, CPUMetrics)
    assert stats.elapsed > 0
    assert stats.user >= 0
    assert stats.system >= 0
    assert stats.user_percent >= 0
    assert stats.system_percent >= 0


def test_collector_get_stats_without_context_raises() -> None:
    collector = CPUMetricsCollector([1])
    with pytest.raises(RuntimeError, match="Must use CPUMetricsCollector"):
        collector.get_stats()


def test_collector_get_stats_mid_context_raises() -> None:
    collector = CPUMetricsCollector([os.getpid()])
    with collector:
        with pytest.raises(RuntimeError, match="Must use CPUMetricsCollector"):
            collector.get_stats()


def test_collector_handles_vanished_pid() -> None:
    """PIDs that disappear between enter/exit should not crash get_stats."""
    fake_pid = 2_000_000_000  # almost certainly not running
    collector = CPUMetricsCollector([fake_pid])
    with collector:
        pass
    stats = collector.get_stats()

    assert stats.user == 0.0
    assert stats.system == 0.0
    assert stats.elapsed > 0


def test_collector_aggregates_multiple_pids() -> None:
    """Verify that CPU times from multiple PIDs are summed."""
    collector = CPUMetricsCollector([1, 2])
    collector.clock_start = 100.0
    collector.clock_end = 102.0
    collector.cpu_times_start = {
        1: CpuTimes(user=10.0, system=2.0),
        2: CpuTimes(user=5.0, system=1.0),
    }
    collector.cpu_times_end = {
        1: CpuTimes(user=12.0, system=3.0),
        2: CpuTimes(user=6.0, system=1.5),
    }
    stats = collector.get_stats()

    assert stats.user == pytest.approx(3.0)  # (12-10) + (6-5)
    assert stats.system == pytest.approx(1.5)  # (3-2) + (1.5-1)
    assert stats.elapsed == pytest.approx(2.0)
    assert stats.user_percent == pytest.approx(150.0)  # 3/2 * 100
    assert stats.system_percent == pytest.approx(75.0)  # 1.5/2 * 100


def test_collector_skips_none_times() -> None:
    """If a PID had NoSuchProcess at enter/exit, its entry is None."""
    collector = CPUMetricsCollector([1, 2])
    collector.clock_start = 100.0
    collector.clock_end = 101.0
    collector.cpu_times_start = {
        1: CpuTimes(user=10.0, system=2.0),
        2: None,
    }
    collector.cpu_times_end = {
        1: CpuTimes(user=11.0, system=2.5),
        2: None,
    }
    stats = collector.get_stats()

    assert stats.user == pytest.approx(1.0)
    assert stats.system == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# collect_pids_for_port
# ---------------------------------------------------------------------------


def test_collect_pids_for_port_no_listeners(mocker: MockerFixture) -> None:
    """When nothing listens on the port, returns empty list."""
    mocker.patch("psutil.net_connections", return_value=[])
    assert collect_pids_for_port(12345) == []


def test_collect_pids_for_port_finds_listener(mocker: MockerFixture) -> None:
    """Finds the PID listening on the requested port."""
    mock_conn = mocker.MagicMock()
    mock_conn.laddr.port = 8000
    mock_conn.status = psutil.CONN_LISTEN
    mock_conn.pid = 42

    mocker.patch("psutil.net_connections", return_value=[mock_conn])
    mocker.patch("psutil.process_iter", return_value=[])
    assert 42 in collect_pids_for_port(8000)
