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
"""Tests for host-memory introspection."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import mock_open, patch

import pytest
from max.support import host_memory
from max.support.host_memory import (
    _cgroups,
    _headroom,
    _v1,
    _v2,
    available_host_memory,
    host_memory_limit,
)

GIB = 1024**3


def write_cgroup(
    directory: Path,
    *,
    limit: str,
    usage: str,
    stat: str | None = None,
    v1: bool = False,
) -> str:
    """Writes one cgroup's memory files and returns its directory."""
    limit_name = "memory.limit_in_bytes" if v1 else "memory.max"
    usage_name = "memory.usage_in_bytes" if v1 else "memory.current"
    (directory / limit_name).write_text(limit)
    (directory / usage_name).write_text(usage)
    if stat is not None:
        (directory / "memory.stat").write_text(stat)
    return str(directory)


def test_host_memory_limit__reports_a_plausible_size() -> None:
    limit = host_memory_limit()
    assert limit is not None
    assert limit > 0


def test_available_host_memory__reports_a_plausible_size() -> None:
    available = available_host_memory()
    assert available is not None
    assert available > 0


def test_available_host_memory__never_exceeds_the_limit() -> None:
    limit = host_memory_limit()
    available = available_host_memory()
    assert limit is not None and available is not None
    assert available <= limit


@pytest.mark.parametrize(
    ("proc_self_cgroup", "expected"),
    [
        pytest.param(
            "0::/system.slice/max-serve.service\n",
            "/sys/fs/cgroup/system.slice/max-serve.service/memory.max",
            id="v2-systemd-unit",
        ),
        pytest.param(
            "4:memory:/docker/abc123\n",
            "/sys/fs/cgroup/memory/docker/abc123/memory.limit_in_bytes",
            id="v1-memory-controller",
        ),
    ],
)
def test_cgroups__include_this_process_own_cgroup(
    proc_self_cgroup: str, expected: str
) -> None:
    with patch("builtins.open", mock_open(read_data=proc_self_cgroup)):
        cgroups = _cgroups()

    assert expected in [cgroup.limit for cgroup in cgroups]
    assert cgroups[0].limit == "/sys/fs/cgroup/memory.max"


def test_cgroups__root_cgroup_adds_nothing() -> None:
    with patch("builtins.open", mock_open(read_data="0::/\n")):
        assert [cgroup.limit for cgroup in _cgroups()] == [
            "/sys/fs/cgroup/memory.max",
            "/sys/fs/cgroup/memory/memory.limit_in_bytes",
        ]


def test_cgroups__unreadable_proc_falls_back() -> None:
    with patch("builtins.open", side_effect=OSError):
        assert [cgroup.limit for cgroup in _cgroups()] == [
            "/sys/fs/cgroup/memory.max",
            "/sys/fs/cgroup/memory/memory.limit_in_bytes",
        ]


def test_headroom__is_the_limit_less_what_is_charged(tmp_path: Path) -> None:
    assert (
        _headroom(_v2(write_cgroup(tmp_path, limit="1000", usage="600"))) == 400
    )


def test_headroom__discounts_reclaimable_page_cache(tmp_path: Path) -> None:
    directory = write_cgroup(
        tmp_path,
        limit="1000",
        usage="900",
        stat="anon 100\nfile 700\nshmem 100\n",
    )
    # 900 charged, of which 700 - 100 = 600 is reclaimable cache.
    assert _headroom(_v2(directory)) == 700


def test_headroom__keeps_the_raw_usage_without_both_stat_keys(
    tmp_path: Path,
) -> None:
    directory = write_cgroup(
        tmp_path, limit="1000", usage="900", stat="anon 100\nfile 700\n"
    )
    assert _headroom(_v2(directory)) == 100


def test_headroom__an_unlimited_cgroup_reports_none(tmp_path: Path) -> None:
    assert (
        _headroom(_v2(write_cgroup(tmp_path, limit="max", usage="600"))) is None
    )


def test_headroom__a_v1_sentinel_limit_reports_none(tmp_path: Path) -> None:
    directory = write_cgroup(
        tmp_path, limit=str((1 << 63) - 4096), usage="600", v1=True
    )
    assert _headroom(_v1(directory)) is None


def test_headroom__an_over_limit_cgroup_reports_zero(tmp_path: Path) -> None:
    assert (
        _headroom(_v2(write_cgroup(tmp_path, limit="1000", usage="1200"))) == 0
    )


def test_headroom__missing_or_unparsable_files_report_none(
    tmp_path: Path,
) -> None:
    assert _headroom(_v2(str(tmp_path))) is None

    garbage = tmp_path / "garbage"
    garbage.mkdir()
    assert (
        _headroom(_v2(write_cgroup(garbage, limit="1000", usage="nonsense")))
        is None
    )


def test_headroom__reads_the_v1_spelling(tmp_path: Path) -> None:
    directory = write_cgroup(
        tmp_path,
        limit="1000",
        usage="900",
        stat="total_cache 700\ntotal_shmem 100\n",
        v1=True,
    )
    assert _headroom(_v1(directory)) == 700


@pytest.mark.parametrize(
    ("machine_free", "expected"),
    [
        pytest.param(2048 * GIB, 60 * GIB, id="cgroup-binds"),
        pytest.param(8 * GIB, 8 * GIB, id="machine-binds"),
    ],
)
def test_available_host_memory__takes_the_smaller_bound(
    machine_free: int,
    expected: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = write_cgroup(tmp_path, limit=str(64 * GIB), usage=str(4 * GIB))
    monkeypatch.setattr(host_memory, "_cgroups", lambda: [_v2(directory)])
    monkeypatch.setattr(
        host_memory.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=machine_free),
    )

    assert available_host_memory() == expected


def test_available_host_memory__unknown_when_nothing_can_be_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise() -> None:
        raise OSError("host memory unavailable")

    monkeypatch.setattr(host_memory, "_cgroups", list)
    monkeypatch.setattr(host_memory.psutil, "virtual_memory", _raise)

    assert available_host_memory() is None
