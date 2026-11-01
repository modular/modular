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

"""Read the engine's GPU utilization from a Prometheus that scrapes DCGM.

Off the accelerator node, local NVML sees none of the engine's GPUs. A cluster
Prometheus that scrapes ``nvidia-dcgm-exporter`` with Kubernetes pod mapping
labels every sample with the pod that owns the GPU, so a label selector picks
out exactly the engine's devices, leaving idle GPUs on a shared node out of
the mean.

Rather than sample during the run, :class:`PrometheusGPURecorder` notes the
window it was entered for and, on exit, asks Prometheus for each device's
aggregate over that window. Those are the same aggregates
``_aggregate_gpu_stats`` computes from a local time series, so one snapshot
per run feeds the existing aggregation, printing, JSON, and CSV unchanged.

A disaggregated deployment's prefill and decode engines load their GPUs very
differently, so a mean across both hides which side is saturated. The pod
label names each device's engine, so the recorder also reports which role and
node every device belongs to.
"""

from __future__ import annotations

import logging
import math
import re
import time
from types import TracebackType

import requests
from max.profiler.gpu import (
    GPUStats,
    GpuStatsRecorder,
    MemoryStats,
    UtilizationStats,
)

from .gpu_metrics_scraper import GPUStatsSnapshot

logger = logging.getLogger(__name__)

_BYTES_PER_MIB = 1024 * 1024

# Each DCGM field with the reduction ``_aggregate_gpu_stats`` applies to it:
# peak memory is the max used, available memory the min free, and utilization
# the mean. Framebuffer is in MiB; utilization is percent.
_FB_USED = ("DCGM_FI_DEV_FB_USED", "max")
_FB_FREE = ("DCGM_FI_DEV_FB_FREE", "min")
_GPU_UTIL = ("DCGM_FI_DEV_GPU_UTIL", "mean")
_MEM_COPY_UTIL = ("DCGM_FI_DEV_MEM_COPY_UTIL", "mean")

# Disaggregated Mammoth engines run in pods named ``<app>-prefill-engine-<id>``
# and ``<app>-decode-engine-<id>``.
_ENGINE_ROLE = re.compile(r"-(prefill|decode)-engine-")
_ROLE_ORDER = ("prefill", "decode")

# Prometheus' Kubernetes discovery names the node; DCGM's own ``Hostname`` is
# the fallback for a scrape config that doesn't relabel it.
_NODE_LABELS = ("kubernetes_node", "Hostname")


def _window_query(field: tuple[str, str], selector: str, window_s: int) -> str:
    """Query one value per device ``UUID`` for ``field`` over the window.

    A restarted engine or exporter pod gives one GPU a second series under new
    ``pod`` or ``instance`` labels, so every reduction folds the series back
    into one device by ``UUID``. Max and min are exact across that split. The
    mean divides the summed samples by their count instead of taking the max
    of per-series means, which would report the busier part of the window.
    """
    metric, reduction = field
    series = f"{metric}{{{selector}}}[{window_s}s]"
    if reduction == "mean":
        return (
            f"sum by (UUID) (sum_over_time({series}))"
            f" / sum by (UUID) (count_over_time({series}))"
        )
    return f"{reduction} by (UUID) ({reduction}_over_time({series}))"


def _group_query(selector: str, window_s: int) -> str:
    """Query how many samples each device has under each pod and node."""
    labels = ", ".join(("UUID", "pod", *_NODE_LABELS))
    return (
        f"sum by ({labels}) (count_over_time("
        f"{_GPU_UTIL[0]}{{{selector}}}[{window_s}s]))"
    )


def _device_groups(
    series: list[tuple[dict[str, str], float]],
) -> dict[str, str]:
    """Label each device of a disaggregated engine with its role and node.

    A device seen under several pods (an engine pod restarted mid-window)
    takes the pod it has the most samples under. The label is the role alone
    when that role runs on one node, else ``"<role> node <n>"`` counted from 1
    over the role's nodes in name order, so each node of a multi-node role
    reports its own utilization. Returns nothing when no device belongs to a
    prefill or decode engine, as in an aggregated deployment.
    """
    samples: dict[str, dict[tuple[str, str], float]] = {}
    for labels, count in series:
        uuid = labels.get("UUID")
        match = _ENGINE_ROLE.search(labels.get("pod", ""))
        if not uuid or match is None:
            continue
        node = next((labels[k] for k in _NODE_LABELS if labels.get(k)), "")
        by_place = samples.setdefault(uuid, {})
        by_place[(match[1], node)] = by_place.get((match[1], node), 0) + count
    placement = {
        uuid: max(counts, key=counts.__getitem__)
        for uuid, counts in samples.items()
    }
    nodes_by_role: dict[str, list[str]] = {}
    for role, node in sorted(set(placement.values())):
        nodes_by_role.setdefault(role, []).append(node)
    groups: dict[str, str] = {}
    for uuid, (role, node) in sorted(
        placement.items(),
        key=lambda item: (_ROLE_ORDER.index(item[1][0]), item[1][1], item[0]),
    ):
        nodes = nodes_by_role[role]
        groups[uuid] = (
            role if len(nodes) == 1 else f"{role} node {nodes.index(node) + 1}"
        )
    return groups


class PrometheusGPURecorder(GpuStatsRecorder):
    """Report the engine's GPU stats for the recorded window from Prometheus.

    Used as a context manager around the measured run. Nothing is sampled
    while inside it; on exit it queries ``prometheus_url`` for each device's
    aggregate over the window, keyed by DCGM ``UUID``. A query that fails or
    finds no series is logged and leaves ``stats`` empty, which disables GPU
    stats for the run rather than failing it.

    Samples land at the scrape interval, so a window shorter than one interval
    may find none, and the last interval before exit may not be scraped yet.

    Args:
        prometheus_url: Base URL of the Prometheus HTTP API, such as
            ``http://kps-prometheus.kube-prometheus-stack.svc:9090``.
        selector: PromQL label matchers, without braces, that select the
            engine's GPUs, such as ``namespace="bench",pod=~"engine-.*"``.
            Required: an empty selector would average every GPU Prometheus
            scrapes.
        timeout_s: HTTP timeout for each query, in seconds.

    Raises:
        ValueError: If ``selector`` is empty.
    """

    def __init__(
        self,
        prometheus_url: str,
        selector: str,
        *,
        timeout_s: float = 10.0,
    ) -> None:
        if not selector.strip():
            raise ValueError(
                "A GPU metrics selector is required with a Prometheus URL"
            )
        self._query_url = f"{prometheus_url.rstrip('/')}/api/v1/query"
        self._selector = selector.strip()
        self._timeout_s = timeout_s
        self._start: float | None = None
        self._stats: list[GPUStatsSnapshot] = []
        self._device_groups: dict[str, str] = {}

    @property
    def stats(self) -> list[GPUStatsSnapshot]:
        """One snapshot of per-GPU aggregates for the window, or none."""
        return self._stats

    @property
    def device_groups(self) -> dict[str, str]:
        """Each disaggregated-engine device's role and node, keyed by ``UUID``.

        Values are ``"prefill"`` or ``"decode"``, or ``"<role> node <n>"``
        when that role spans several nodes. Empty for an aggregated
        deployment, or when the grouping query fails.
        """
        return self._device_groups

    def __enter__(self) -> PrometheusGPURecorder:
        self._start = time.time()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._start is None:
            return
        end = time.time()
        window_s = max(1, math.ceil(end - self._start))
        try:
            snapshot = self._query_window(end, window_s)
        except Exception as exc:
            logger.warning(
                "Failed to read GPU metrics from %s: %s", self._query_url, exc
            )
            return
        if not snapshot:
            logger.warning(
                "No GPU metrics in Prometheus match %s over the last %ds",
                self._selector,
                window_s,
            )
            return
        self._stats = [snapshot]
        # Grouping only splits the stats already read, so a failure here
        # keeps them and loses the per-role split.
        try:
            series = self._query_series(
                _group_query(self._selector, window_s), end
            )
        except Exception as exc:
            logger.warning(
                "Failed to read GPU roles from %s: %s", self._query_url, exc
            )
            return
        self._device_groups = {
            uuid: group
            for uuid, group in _device_groups(series).items()
            if uuid in snapshot
        }

    def _query_series(
        self, query: str, at: float
    ) -> list[tuple[dict[str, str], float]]:
        response = requests.get(
            self._query_url,
            params={"query": query, "time": f"{at:.3f}"},
            timeout=self._timeout_s,
        )
        response.raise_for_status()
        body = response.json()
        if body.get("status") != "success":
            raise RuntimeError(body.get("error") or "query failed")
        return [
            (series["metric"], float(series["value"][1]))
            for series in body["data"]["result"]
        ]

    def _query(self, query: str, at: float) -> dict[str, float]:
        return {
            uuid: value
            for labels, value in self._query_series(query, at)
            if (uuid := labels.get("UUID"))
        }

    def _query_window(self, end: float, window_s: int) -> GPUStatsSnapshot:
        used, free, gpu_util, mem_util = (
            self._query(_window_query(field, self._selector, window_s), end)
            for field in (_FB_USED, _FB_FREE, _GPU_UTIL, _MEM_COPY_UTIL)
        )
        snapshot: GPUStatsSnapshot = {}
        # Utilization is the field the triage report needs, so it decides
        # which devices count; memory fields only fill in what they have.
        for uuid in sorted(gpu_util):
            used_bytes = int(used.get(uuid, 0) * _BYTES_PER_MIB)
            free_bytes = int(free.get(uuid, 0) * _BYTES_PER_MIB)
            mem_activity = mem_util.get(uuid)
            snapshot[uuid] = GPUStats(
                memory=MemoryStats(
                    total_bytes=used_bytes + free_bytes,
                    free_bytes=free_bytes,
                    used_bytes=used_bytes,
                    reserved_bytes=None,
                ),
                utilization=UtilizationStats(
                    gpu_usage_percent=round(gpu_util[uuid]),
                    memory_activity_percent=(
                        None if mem_activity is None else round(mem_activity)
                    ),
                ),
                clocks=None,
            )
        return snapshot
