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

"""Scrape GPU utilization from remote DCGM-exporter Prometheus endpoints.

Off the accelerator node (e.g. a disaggregated Mammoth bench pod), local NVML
sees none of the engine's GPUs. The engine nodes' ``nvidia-dcgm-exporter`` does,
and the driver passes its endpoints to the bench pod as ``GPU_METRICS_HOST``.
This turns those endpoints into the same ``list[GPUStatsSnapshot]`` the local
recorder produces, so downstream aggregation, printing, JSON, and CSV are
unchanged.

Multi-node caveat: the exporter runs one pod per accelerator node, so a
multi-node engine has several. Scraping a single Service ClusterIP would
load-balance to one backing pod, so ``GPU_METRICS_HOST`` carries every ready
endpoint (comma-separated) and :class:`DCGMBackgroundRecorder` scrapes them all
each interval, merging their snapshots into one by the union of device keys.
:func:`_gpu_key` qualifies each key by ``Hostname``, so keys stay unique per
node and endpoints never clobber each other. A single endpoint failing an
interval is logged and skipped; the rest of that sample still lands.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from types import TracebackType
from typing import TypeAlias
from urllib.parse import urlparse

from max.profiler.gpu import (
    GPUStats,
    GpuStatsRecorder,
    MemoryStats,
    UtilizationStats,
)
from prometheus_client.parser import text_string_to_metric_families

from .prometheus_fetch import fetch_metrics

logger = logging.getLogger(__name__)

GPUDeviceKey: TypeAlias = str
"""Stable per-GPU key: ``"<Hostname>:gpu<N>"``, else ``UUID``, else ``"gpu<N>"``.

An alias of ``str`` (not a ``NewType``) so snapshots stay interchangeable with
the local NVML recorder's ``dict[str, GPUStats]`` the same aggregation consumes.
"""

GPUStatsSnapshot: TypeAlias = dict[GPUDeviceKey, GPUStats]
"""One point-in-time sample: every GPU's :class:`GPUStats`, keyed by device."""


# Default port the NVIDIA dcgm-exporter serves Prometheus metrics on.
DCGM_METRICS_PORT = 9400

_BYTES_PER_MIB = 1024 * 1024

# DCGM fields consumed here. Framebuffer is in MiB; utilization is percent.
_FB_USED = "DCGM_FI_DEV_FB_USED"
_FB_FREE = "DCGM_FI_DEV_FB_FREE"
_GPU_UTIL = "DCGM_FI_DEV_GPU_UTIL"
_MEM_COPY_UTIL = "DCGM_FI_DEV_MEM_COPY_UTIL"

# Host labels tried in order; preferred over ``UUID`` for keys as they read
# better in logs and are present even when the exporter omits device UUIDs.
_HOSTNAME_LABELS = ("Hostname", "instance")


def dcgm_metrics_url(host: str) -> str:
    """Return the Prometheus scrape URL for a DCGM-exporter ``host``.

    A bare host or ``host:port`` is expanded to ``http://<host>:9400/metrics``;
    anything already carrying a scheme is used verbatim.

    Args:
        host: DCGM-exporter host, ``host:port``, or full URL.

    Returns:
        A fully-qualified ``http(s)://.../metrics`` URL.
    """
    host = host.strip()
    if "://" in host:
        return host
    # Bracket a bare IPv6 literal so its trailing hextet isn't read as a port.
    if host.count(":") >= 2 and not host.startswith("["):
        return f"http://[{host}]:{DCGM_METRICS_PORT}/metrics"
    # ``urlparse`` needs a scheme to split host from port reliably.
    parsed = urlparse(f"//{host}", scheme="http")
    hostname = parsed.hostname or host
    if ":" in hostname:
        hostname = f"[{hostname}]"
    port = parsed.port or DCGM_METRICS_PORT
    return f"http://{hostname}:{port}/metrics"


def _dcgm_metrics_urls(hosts: str | Sequence[str]) -> list[str]:
    """Expand one or many DCGM-exporter hosts into distinct scrape URLs.

    ``hosts`` is either a comma-separated string (the ``GPU_METRICS_HOST``
    wire format the driver emits, one entry per exporter pod) or an already
    split sequence. Blank entries are dropped and duplicates are removed while
    preserving order, so a single host and a multi-node list share one path.

    Args:
        hosts: A comma-separated host string, or a sequence of hosts.

    Returns:
        The de-duplicated ``http(s)://.../metrics`` URLs, in input order.
    """
    if isinstance(hosts, str):
        hosts = hosts.split(",")
    urls: list[str] = []
    for host in hosts:
        if not host.strip():
            continue
        url = dcgm_metrics_url(host)
        if url not in urls:
            urls.append(url)
    return urls


def _gpu_key(labels: dict[str, str]) -> GPUDeviceKey:
    """Build a stable per-GPU key from a DCGM sample's labels."""
    gpu = labels.get("gpu")
    hostname = next(
        (labels[label] for label in _HOSTNAME_LABELS if labels.get(label)),
        None,
    )
    if hostname and gpu is not None:
        return f"{hostname}:gpu{gpu}"
    if uuid := labels.get("UUID"):
        return uuid
    if gpu is not None:
        return f"gpu{gpu}"
    # No device label: bucket under one key so the sample stays visible.
    return "gpu"


def parse_dcgm_metrics(raw_text: str) -> GPUStatsSnapshot:
    """Parse DCGM-exporter Prometheus text into per-GPU :class:`GPUStats`.

    Consumes only the utilization and framebuffer fields; clocks are left unset
    because DCGM omits the boost ceilings :class:`~max.profiler.gpu.ClockStats`
    needs. Missing numeric fields default to zero.

    Args:
        raw_text: Raw Prometheus text-format payload from the exporter.

    Returns:
        Mapping of each :data:`GPUDeviceKey` to its :class:`GPUStats` snapshot.
    """
    used_bytes: dict[GPUDeviceKey, int] = {}
    free_bytes: dict[GPUDeviceKey, int] = {}
    gpu_util: dict[GPUDeviceKey, int] = {}
    mem_util: dict[GPUDeviceKey, int] = {}

    for family in text_string_to_metric_families(raw_text):
        if family.name not in (_FB_USED, _FB_FREE, _GPU_UTIL, _MEM_COPY_UTIL):
            continue
        for sample in family.samples:
            # Skip derived series (e.g. ``_created``); only base-name samples
            # carry the gauge value.
            if sample.name != family.name:
                continue
            key = _gpu_key(sample.labels)
            value = sample.value
            if family.name == _FB_USED:
                used_bytes[key] = int(value) * _BYTES_PER_MIB
            elif family.name == _FB_FREE:
                free_bytes[key] = int(value) * _BYTES_PER_MIB
            elif family.name == _GPU_UTIL:
                gpu_util[key] = int(value)
            elif family.name == _MEM_COPY_UTIL:
                mem_util[key] = int(value)

    devices = (
        used_bytes.keys()
        | free_bytes.keys()
        | gpu_util.keys()
        | mem_util.keys()
    )
    stats: GPUStatsSnapshot = {}
    for key in sorted(devices):
        used = used_bytes.get(key, 0)
        free = free_bytes.get(key, 0)
        stats[key] = GPUStats(
            memory=MemoryStats(
                total_bytes=used + free,
                free_bytes=free,
                used_bytes=used,
                reserved_bytes=None,
            ),
            utilization=UtilizationStats(
                gpu_usage_percent=gpu_util.get(key, 0),
                memory_activity_percent=mem_util.get(key),
            ),
            clocks=None,
        )
    return stats


class DCGMBackgroundRecorder(GpuStatsRecorder):
    """Sample remote DCGM-exporter endpoints on an interval in a thread.

    A drop-in stand-in for :class:`max.profiler.gpu.BackgroundRecorder` when the
    benchmark runs off the accelerator node, exposing the same ``stats`` series
    so downstream aggregation is unchanged. Used as a context manager: sampling
    starts on enter and stops on exit.

    Multi-node: ``hosts`` may name several exporter pods (one per accelerator
    node). Each interval every endpoint is scraped and their per-GPU snapshots
    are merged into one by the union of device keys, so a multi-node engine
    reports every node's GPUs. A single endpoint failing an interval is logged
    and skipped -- it costs that endpoint's slice of one sample, not the run;
    an interval where every endpoint fails records nothing.

    Args:
        hosts: One DCGM-exporter host or a comma-separated list (also accepts a
            sequence). Each entry is a host, ``host:port``, or full URL (see
            :func:`dcgm_metrics_url`).
        interval: Seconds between scrapes. Must be non-negative.
        timeout_s: Per-scrape HTTP timeout in seconds.
    """

    def __init__(
        self,
        hosts: str | Sequence[str],
        *,
        interval: float = 1.0,
        timeout_s: float = 2.0,
    ) -> None:
        if interval < 0:
            raise ValueError("Interval must be non-negative")
        self._urls = _dcgm_metrics_urls(hosts)
        self._interval = interval
        self._timeout_s = timeout_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._stats: list[GPUStatsSnapshot] = []

    @property
    def stats(self) -> list[GPUStatsSnapshot]:
        """Time series of per-GPU snapshots collected so far."""
        return self._stats

    def _sample(self) -> GPUStatsSnapshot:
        merged: GPUStatsSnapshot = {}
        for url in self._urls:
            try:
                raw_text = fetch_metrics(url, timeout_s=self._timeout_s)
            except Exception as exc:
                # Tolerate one endpoint failing: costs its slice of a sample,
                # not the whole interval and not the run.
                logger.warning(
                    "Failed to collect DCGM metrics from %s: %s", url, exc
                )
                continue
            # _gpu_key qualifies by Hostname, so keys from different exporter
            # pods are unique and a later endpoint can't clobber an earlier's.
            merged.update(parse_dcgm_metrics(raw_text))
        return merged

    def _run(self) -> None:
        while True:
            snapshot = self._sample()
            if snapshot:
                self._stats.append(snapshot)
            # ``wait`` returns True once stopped, for prompt shutdown.
            if self._stop.wait(self._interval):
                return

    def __enter__(self) -> DCGMBackgroundRecorder:
        if self._thread is not None:
            raise RuntimeError("Recorder already running")
        self._thread = threading.Thread(
            target=self._run, name="dcgm-recorder", daemon=True
        )
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._timeout_s + self._interval + 1.0)
            self._thread = None
