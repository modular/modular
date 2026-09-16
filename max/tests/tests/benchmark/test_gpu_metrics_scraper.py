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

"""Unit tests for the DCGM-exporter GPU metrics scraper."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from max.benchmark.benchmark_shared.gpu_metrics_scraper import (
    DCGMBackgroundRecorder,
    _dcgm_metrics_urls,
    dcgm_metrics_url,
    parse_dcgm_metrics,
)

_MIB = 1024 * 1024

# Two GPUs on one node, shaped like the default dcgm-exporter payload.
_DCGM_TEXT = """\
# HELP DCGM_FI_DEV_GPU_UTIL GPU utilization (in %).
# TYPE DCGM_FI_DEV_GPU_UTIL gauge
DCGM_FI_DEV_GPU_UTIL{gpu="0",UUID="GPU-aaa",device="nvidia0",modelName="NVIDIA H100",Hostname="node-1"} 85
DCGM_FI_DEV_GPU_UTIL{gpu="1",UUID="GPU-bbb",device="nvidia1",modelName="NVIDIA H100",Hostname="node-1"} 91
# HELP DCGM_FI_DEV_MEM_COPY_UTIL Memory utilization (in %).
# TYPE DCGM_FI_DEV_MEM_COPY_UTIL gauge
DCGM_FI_DEV_MEM_COPY_UTIL{gpu="0",UUID="GPU-aaa",device="nvidia0",Hostname="node-1"} 30
DCGM_FI_DEV_MEM_COPY_UTIL{gpu="1",UUID="GPU-bbb",device="nvidia1",Hostname="node-1"} 44
# HELP DCGM_FI_DEV_FB_USED Framebuffer memory used (in MiB).
# TYPE DCGM_FI_DEV_FB_USED gauge
DCGM_FI_DEV_FB_USED{gpu="0",UUID="GPU-aaa",device="nvidia0",Hostname="node-1"} 40000
DCGM_FI_DEV_FB_USED{gpu="1",UUID="GPU-bbb",device="nvidia1",Hostname="node-1"} 41000
# HELP DCGM_FI_DEV_FB_FREE Framebuffer memory free (in MiB).
# TYPE DCGM_FI_DEV_FB_FREE gauge
DCGM_FI_DEV_FB_FREE{gpu="0",UUID="GPU-aaa",device="nvidia0",Hostname="node-1"} 41000
DCGM_FI_DEV_FB_FREE{gpu="1",UUID="GPU-bbb",device="nvidia1",Hostname="node-1"} 40000
"""

# One GPU on a second node, so its device keys are disjoint from _DCGM_TEXT.
_DCGM_TEXT_NODE2 = """\
# TYPE DCGM_FI_DEV_GPU_UTIL gauge
DCGM_FI_DEV_GPU_UTIL{gpu="0",UUID="GPU-ccc",Hostname="node-2"} 60
# TYPE DCGM_FI_DEV_FB_USED gauge
DCGM_FI_DEV_FB_USED{gpu="0",UUID="GPU-ccc",Hostname="node-2"} 20000
# TYPE DCGM_FI_DEV_FB_FREE gauge
DCGM_FI_DEV_FB_FREE{gpu="0",UUID="GPU-ccc",Hostname="node-2"} 61000
"""


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("10.0.0.5", "http://10.0.0.5:9400/metrics"),
        ("10.0.0.5:9401", "http://10.0.0.5:9401/metrics"),
        ("dcgm.svc", "http://dcgm.svc:9400/metrics"),
        ("http://host:9400/metrics", "http://host:9400/metrics"),
        ("https://host/metrics", "https://host/metrics"),
        ("fe80::1", "http://[fe80::1]:9400/metrics"),
        ("[fe80::1]:9402", "http://[fe80::1]:9402/metrics"),
    ],
)
def test_dcgm_metrics_url(host: str, expected: str) -> None:
    assert dcgm_metrics_url(host) == expected


def test_parse_dcgm_metrics_builds_per_gpu_stats() -> None:
    stats = parse_dcgm_metrics(_DCGM_TEXT)

    assert set(stats) == {"node-1:gpu0", "node-1:gpu1"}

    gpu0 = stats["node-1:gpu0"]
    assert gpu0.utilization.gpu_usage_percent == 85
    assert gpu0.utilization.memory_activity_percent == 30
    assert gpu0.memory.used_bytes == 40000 * _MIB
    assert gpu0.memory.free_bytes == 41000 * _MIB
    assert gpu0.memory.total_bytes == 81000 * _MIB
    assert gpu0.clocks is None

    assert stats["node-1:gpu1"].utilization.gpu_usage_percent == 91


def test_parse_dcgm_metrics_ignores_unrelated_families() -> None:
    text = (
        "# TYPE DCGM_FI_DEV_POWER_USAGE gauge\n"
        'DCGM_FI_DEV_POWER_USAGE{gpu="0",Hostname="n"} 250.0\n'
        "# TYPE DCGM_FI_DEV_GPU_UTIL gauge\n"
        'DCGM_FI_DEV_GPU_UTIL{gpu="0",Hostname="n"} 77\n'
    )
    stats = parse_dcgm_metrics(text)
    assert set(stats) == {"n:gpu0"}
    assert stats["n:gpu0"].utilization.gpu_usage_percent == 77


def test_parse_dcgm_metrics_empty_payload() -> None:
    assert parse_dcgm_metrics("") == {}


def test_parse_dcgm_metrics_keys_by_uuid_without_hostname() -> None:
    text = (
        "# TYPE DCGM_FI_DEV_GPU_UTIL gauge\n"
        'DCGM_FI_DEV_GPU_UTIL{UUID="GPU-xyz"} 50\n'
    )
    stats = parse_dcgm_metrics(text)
    assert set(stats) == {"GPU-xyz"}


@pytest.mark.parametrize(
    ("hosts", "expected"),
    [
        # Single host stays a single URL.
        ("10.0.0.5", ["http://10.0.0.5:9400/metrics"]),
        # Comma-separated string expands to one URL per endpoint.
        (
            "10.0.0.5,10.0.0.6",
            [
                "http://10.0.0.5:9400/metrics",
                "http://10.0.0.6:9400/metrics",
            ],
        ),
        # A sequence is accepted too; blanks are dropped, order preserved.
        (
            ["10.0.0.5", " ", "10.0.0.6"],
            [
                "http://10.0.0.5:9400/metrics",
                "http://10.0.0.6:9400/metrics",
            ],
        ),
        # Duplicates collapse so an endpoint isn't scraped twice.
        ("10.0.0.5,10.0.0.5", ["http://10.0.0.5:9400/metrics"]),
        ("", []),
    ],
)
def test_dcgm_metrics_urls(hosts: str, expected: list[str]) -> None:
    assert _dcgm_metrics_urls(hosts) == expected


@patch("max.benchmark.benchmark_shared.gpu_metrics_scraper.fetch_metrics")
def test_background_recorder_collects_snapshot(mock_fetch: MagicMock) -> None:
    # A large interval samples exactly once on entry, then blocks until the
    # context exit sets the stop event -- deterministic, no sleep race.
    mock_fetch.return_value = _DCGM_TEXT
    with DCGMBackgroundRecorder(hosts="10.0.0.5", interval=60.0) as recorder:
        pass

    assert len(recorder.stats) == 1
    snapshot = recorder.stats[0]
    assert snapshot["node-1:gpu0"].utilization.gpu_usage_percent == 85
    mock_fetch.assert_called_with("http://10.0.0.5:9400/metrics", timeout_s=2.0)


@patch("max.benchmark.benchmark_shared.gpu_metrics_scraper.fetch_metrics")
def test_background_recorder_merges_multiple_endpoints(
    mock_fetch: MagicMock,
) -> None:
    """Each interval scrapes every endpoint and unions their device keys."""
    mock_fetch.side_effect = [_DCGM_TEXT, _DCGM_TEXT_NODE2]
    with DCGMBackgroundRecorder(
        hosts="10.0.0.5,10.0.0.6", interval=60.0
    ) as recorder:
        pass

    assert len(recorder.stats) == 1
    snapshot = recorder.stats[0]
    # Both nodes' devices present in the single merged snapshot.
    assert set(snapshot) == {"node-1:gpu0", "node-1:gpu1", "node-2:gpu0"}
    assert snapshot["node-2:gpu0"].utilization.gpu_usage_percent == 60
    assert mock_fetch.call_count == 2


@patch("max.benchmark.benchmark_shared.gpu_metrics_scraper.fetch_metrics")
def test_background_recorder_tolerates_one_failed_endpoint(
    mock_fetch: MagicMock,
) -> None:
    """One endpoint failing skips only its slice; the rest still land."""
    mock_fetch.side_effect = [RuntimeError("connection refused"), _DCGM_TEXT]
    with DCGMBackgroundRecorder(
        hosts="10.0.0.5,10.0.0.6", interval=60.0
    ) as recorder:
        pass

    assert len(recorder.stats) == 1
    # Only the reachable endpoint's devices survive; no KeyError, no raise.
    assert set(recorder.stats[0]) == {"node-1:gpu0", "node-1:gpu1"}


@patch("max.benchmark.benchmark_shared.gpu_metrics_scraper.fetch_metrics")
def test_background_recorder_skips_failed_scrape(mock_fetch: MagicMock) -> None:
    mock_fetch.side_effect = RuntimeError("connection refused")
    with DCGMBackgroundRecorder(hosts="10.0.0.5", interval=60.0) as recorder:
        pass

    # A failed scrape is logged and dropped, not raised or recorded.
    assert recorder.stats == []


def test_background_recorder_rejects_negative_interval() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        DCGMBackgroundRecorder(hosts="10.0.0.5", interval=-1.0)
