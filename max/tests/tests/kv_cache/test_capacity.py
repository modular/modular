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

"""Unit tests for the KV cache host and disk capacity preflights."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from max.pipelines.kv_cache.connectors import tier_connector


def _set_available_host_memory(
    monkeypatch: pytest.MonkeyPatch, available: int | None
) -> None:
    monkeypatch.setattr(
        tier_connector, "available_host_memory", lambda: available
    )


def test_host_capacity_rejects_oversized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_available_host_memory(monkeypatch, 1024)

    with pytest.raises(RuntimeError, match="host_offload_max_gb"):
        tier_connector._check_host_memory_capacity(2048)


def test_host_capacity_accepts_fitting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_available_host_memory(monkeypatch, 4096)

    tier_connector._check_host_memory_capacity(4096)


def test_host_capacity_skips_when_unknown(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _set_available_host_memory(monkeypatch, None)

    tier_connector._check_host_memory_capacity(1 << 60)
    assert "skipping KV cache host capacity preflight" in caplog.text


def test_default_host_offload_is_one_and_a_half_times_the_device_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_available_host_memory(monkeypatch, 1 << 40)

    assert tier_connector._default_host_offload_bytes(8 << 30) == 12 << 30


def test_default_host_offload_is_capped_to_what_the_process_may_use(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _set_available_host_memory(monkeypatch, 20 << 30)

    assert tier_connector._default_host_offload_bytes(64 << 30) == 18 << 30
    assert "Reduced the default KV cache host offload budget" in caplog.text


def test_default_host_offload_uncapped_when_availability_is_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_available_host_memory(monkeypatch, None)

    assert tier_connector._default_host_offload_bytes(64 << 30) == 96 << 30


def test_disk_capacity_rejects_oversized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tier_connector.psutil,
        "disk_usage",
        lambda path: SimpleNamespace(free=1024),
    )

    with pytest.raises(RuntimeError, match="disk_offload_max_gb"):
        tier_connector._check_disk_capacity("/tmp", 2048)


def test_disk_capacity_accepts_fitting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tier_connector.psutil,
        "disk_usage",
        lambda path: SimpleNamespace(free=4096),
    )

    tier_connector._check_disk_capacity("/tmp", 4096)
