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

import threading
from collections.abc import Callable
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


@pytest.mark.parametrize("host", ["10.0.0.5:abc", "10.0.0.5:99999"])
def test_dcgm_metrics_url_falls_back_on_malformed_port(host: str) -> None:
    """A typo in the host must not abort the run before it starts."""
    assert dcgm_metrics_url(host) == "http://10.0.0.5:9400/metrics"


def test_parse_dcgm_metrics_builds_per_gpu_stats() -> None:
    stats = parse_dcgm_metrics(_DCGM_TEXT)

    assert set(stats) == {"GPU-aaa", "GPU-bbb"}

    gpu0 = stats["GPU-aaa"]
    assert gpu0.utilization.gpu_usage_percent == 85
    assert gpu0.utilization.memory_activity_percent == 30
    assert gpu0.memory.used_bytes == 40000 * _MIB
    assert gpu0.memory.free_bytes == 41000 * _MIB
    assert gpu0.memory.total_bytes == 81000 * _MIB
    assert gpu0.clocks is None

    assert stats["GPU-bbb"].utilization.gpu_usage_percent == 91


def test_parse_dcgm_metrics_ignores_unrelated_families() -> None:
    text = (
        "# TYPE DCGM_FI_DEV_POWER_USAGE gauge\n"
        'DCGM_FI_DEV_POWER_USAGE{gpu="0",UUID="GPU-aaa"} 250.0\n'
        "# TYPE DCGM_FI_DEV_GPU_UTIL gauge\n"
        'DCGM_FI_DEV_GPU_UTIL{gpu="0",UUID="GPU-aaa"} 77\n'
    )
    stats = parse_dcgm_metrics(text)
    assert set(stats) == {"GPU-aaa"}
    assert stats["GPU-aaa"].utilization.gpu_usage_percent == 77


def test_parse_dcgm_metrics_empty_payload() -> None:
    assert parse_dcgm_metrics("") == {}


def test_parse_dcgm_metrics_keys_by_uuid_over_hostname() -> None:
    """The UUID names the device, so a pod rename can't split its series.

    An exporter restart gives the DaemonSet pod a new name; keying on that
    would count one physical GPU as two devices in the aggregate.
    """
    before = parse_dcgm_metrics(
        "# TYPE DCGM_FI_DEV_GPU_UTIL gauge\n"
        'DCGM_FI_DEV_GPU_UTIL{gpu="0",UUID="GPU-xyz",Hostname="dcgm-abc"} 50\n'
    )
    after = parse_dcgm_metrics(
        "# TYPE DCGM_FI_DEV_GPU_UTIL gauge\n"
        'DCGM_FI_DEV_GPU_UTIL{gpu="0",UUID="GPU-xyz",Hostname="dcgm-def"} 70\n'
    )
    assert set(before) == set(after) == {"GPU-xyz"}


def test_parse_dcgm_metrics_falls_back_to_hostname_without_uuid() -> None:
    """An exporter that omits UUIDs still yields a node-qualified key."""
    text = (
        "# TYPE DCGM_FI_DEV_GPU_UTIL gauge\n"
        'DCGM_FI_DEV_GPU_UTIL{gpu="0",Hostname="node-1"} 50\n'
    )
    assert set(parse_dcgm_metrics(text)) == {"node-1:gpu0"}


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
    assert snapshot["GPU-aaa"].utilization.gpu_usage_percent == 85
    mock_fetch.assert_called_with("http://10.0.0.5:9400/metrics", timeout_s=2.0)


def _one_sweep(
    *responses: str | Exception,
) -> tuple[Callable[..., str], threading.Event]:
    """Fake ``fetch_metrics`` answering one sweep, and an event for its end.

    The recorder checks the stop event between endpoints, so a test that
    exits the context straight away races the sweep and can skip every
    endpoint after the first. Waiting on the event before exiting pins the
    whole sweep.
    """
    remaining = list(responses)
    reached_last = threading.Event()

    def fetch(url: str, **kwargs: object) -> str:
        response = remaining.pop(0)
        if not remaining:
            reached_last.set()
        if isinstance(response, Exception):
            raise response
        return response

    return fetch, reached_last


@patch("max.benchmark.benchmark_shared.gpu_metrics_scraper.fetch_metrics")
def test_background_recorder_merges_multiple_endpoints(
    mock_fetch: MagicMock,
) -> None:
    """Each interval scrapes every endpoint and unions their device keys."""
    mock_fetch.side_effect, swept = _one_sweep(_DCGM_TEXT, _DCGM_TEXT_NODE2)
    with DCGMBackgroundRecorder(
        hosts="10.0.0.5,10.0.0.6", interval=60.0
    ) as recorder:
        assert swept.wait(timeout=5)

    assert len(recorder.stats) == 1
    snapshot = recorder.stats[0]
    # Both nodes' devices present in the single merged snapshot.
    assert set(snapshot) == {"GPU-aaa", "GPU-bbb", "GPU-ccc"}
    assert snapshot["GPU-ccc"].utilization.gpu_usage_percent == 60
    assert mock_fetch.call_count == 2


@patch("max.benchmark.benchmark_shared.gpu_metrics_scraper.fetch_metrics")
def test_background_recorder_tolerates_one_failed_endpoint(
    mock_fetch: MagicMock,
) -> None:
    """One endpoint failing skips only its slice; the rest still land."""
    mock_fetch.side_effect, swept = _one_sweep(
        RuntimeError("connection refused"), _DCGM_TEXT
    )
    with DCGMBackgroundRecorder(
        hosts="10.0.0.5,10.0.0.6", interval=60.0
    ) as recorder:
        assert swept.wait(timeout=5)

    assert len(recorder.stats) == 1
    # Only the reachable endpoint's devices survive; no KeyError, no raise.
    assert set(recorder.stats[0]) == {"GPU-aaa", "GPU-bbb"}


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


@patch("max.benchmark.benchmark_shared.gpu_metrics_scraper.fetch_metrics")
def test_background_recorder_survives_unparseable_payload(
    mock_fetch: MagicMock,
) -> None:
    """A malformed payload costs that endpoint's slice, not the thread.

    The parse runs inside the same try as the fetch, so an exporter serving
    garbage is skipped like an unreachable one instead of killing the
    sampling thread and silently ending collection.
    """
    mock_fetch.side_effect, swept = _one_sweep(
        "not prometheus text @@@", _DCGM_TEXT
    )
    with DCGMBackgroundRecorder(
        hosts="10.0.0.5,10.0.0.6", interval=60.0
    ) as recorder:
        assert swept.wait(timeout=5)

    # The second endpoint was still reached, so the first one's parse failure
    # did not take the sweep -- or the thread -- down with it.
    assert set(recorder.stats[0]) == {"GPU-aaa", "GPU-bbb"}


@patch("max.benchmark.benchmark_shared.gpu_metrics_scraper.fetch_metrics")
def test_background_recorder_stops_between_endpoints(
    mock_fetch: MagicMock,
) -> None:
    """Shutdown during a sweep skips the endpoints not yet scraped.

    Endpoints are scraped serially, so without this a slow shutdown owes one
    timeout per remaining endpoint.
    """
    recorder = DCGMBackgroundRecorder(
        hosts="10.0.0.5,10.0.0.6,10.0.0.7", interval=60.0
    )

    def stop_after_first(url: str, **kwargs: object) -> str:
        recorder._stop.set()
        return _DCGM_TEXT

    mock_fetch.side_effect = stop_after_first
    with recorder:
        pass

    assert mock_fetch.call_count == 1


@patch("max.benchmark.benchmark_shared.gpu_metrics_scraper.fetch_metrics")
def test_background_recorder_stats_frozen_when_scrape_outlives_join(
    mock_fetch: MagicMock,
) -> None:
    """A scrape that outlasts the join budget can't change what exit returned.

    A request timeout bounds socket inactivity, not a whole response, so a
    slow exporter can keep the worker alive past ``__exit__``.
    """
    release = threading.Event()

    def stalled_fetch(url: str, **kwargs: object) -> str:
        release.wait(timeout=10)
        return _DCGM_TEXT

    mock_fetch.side_effect = stalled_fetch
    recorder = DCGMBackgroundRecorder(
        hosts="10.0.0.5", interval=0.0, timeout_s=0.01
    )
    with recorder:
        worker = recorder._thread
        assert worker is not None
    assert worker.is_alive()

    release.set()
    worker.join(timeout=10)
    assert not worker.is_alive()
    assert recorder.stats == []


def test_background_recorder_join_budget_scales_with_endpoints() -> None:
    """The join must cover a serial sweep, not a single endpoint's wait."""
    one = DCGMBackgroundRecorder(hosts="10.0.0.5", interval=1.0, timeout_s=2.0)
    four = DCGMBackgroundRecorder(
        hosts="10.0.0.5,10.0.0.6,10.0.0.7,10.0.0.8",
        interval=1.0,
        timeout_s=2.0,
    )
    joins: list[float] = []

    for recorder in (one, four):
        thread = MagicMock()
        thread.join.side_effect = lambda timeout: joins.append(timeout)
        recorder._thread = thread
        recorder.__exit__(None, None, None)

    # 2s per endpoint + the interval + 1s of slack.
    assert joins == [4.0, 10.0]
