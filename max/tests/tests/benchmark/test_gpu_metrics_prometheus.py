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

from collections.abc import Mapping, Sequence
from unittest.mock import MagicMock, patch

import pytest
from max.benchmark.benchmark_shared.gpu_metrics_prometheus import (
    PrometheusGPURecorder,
)

_SELECTOR = 'namespace="bench",modelapplication_mammoth_modular_com_name="r"'
_MIB = 1024 * 1024


def _vector(values: Mapping[str, float]) -> dict[str, object]:
    """A Prometheus instant-query response with one series per UUID."""
    return {
        "status": "success",
        "data": {
            "resultType": "vector",
            "result": [
                {"metric": {"UUID": uuid}, "value": [0, str(value)]}
                for uuid, value in values.items()
            ],
        },
    }


def _group_vector(
    placements: Sequence[tuple[str, str, str, float]],
) -> dict[str, object]:
    """A grouping-query response: (UUID, pod, node, sample count) series."""
    return {
        "status": "success",
        "data": {
            "resultType": "vector",
            "result": [
                {
                    "metric": {
                        "UUID": uuid,
                        "pod": pod,
                        "kubernetes_node": node,
                    },
                    "value": [0, str(count)],
                }
                for uuid, pod, node, count in placements
            ],
        },
    }


def _is_group_query(query: str) -> bool:
    return query.startswith("sum by (UUID, pod,")


def _fake_get(
    by_metric: Mapping[str, Mapping[str, float]],
    placements: Sequence[tuple[str, str, str, float]] = (),
) -> tuple[MagicMock, list[str]]:
    """Fake ``requests.get`` that answers each query by the metric it names.

    The grouping query answers with ``placements``.
    """
    queries: list[str] = []

    def get(
        url: str, *, params: Mapping[str, str], timeout: float
    ) -> MagicMock:
        query = params["query"]
        queries.append(query)
        response = MagicMock()
        if _is_group_query(query):
            response.json.return_value = _group_vector(placements)
        else:
            metric = next(m for m in by_metric if m in query)
            response.json.return_value = _vector(by_metric[metric])
        return response

    return MagicMock(side_effect=get), queries


_TWO_GPUS: dict[str, dict[str, float]] = {
    "DCGM_FI_DEV_FB_USED": {"GPU-a": 1000, "GPU-b": 2000},
    "DCGM_FI_DEV_FB_FREE": {"GPU-a": 3000, "GPU-b": 4000},
    "DCGM_FI_DEV_GPU_UTIL": {"GPU-a": 84.6, "GPU-b": 40},
    "DCGM_FI_DEV_MEM_COPY_UTIL": {"GPU-a": 20.2},
}


def test_recorder_reports_window_aggregates_per_device() -> None:
    """One snapshot holds each GPU's window aggregates, keyed by UUID."""
    fake_get, _ = _fake_get(_TWO_GPUS)
    with patch(
        "max.benchmark.benchmark_shared.gpu_metrics_prometheus.requests.get",
        fake_get,
    ):
        with PrometheusGPURecorder("http://prom:9090/", _SELECTOR) as recorder:
            pass

    assert len(recorder.stats) == 1
    snapshot = recorder.stats[0]
    assert sorted(snapshot) == ["GPU-a", "GPU-b"]
    a = snapshot["GPU-a"]
    assert a.utilization.gpu_usage_percent == 85
    assert a.utilization.memory_activity_percent == 20
    assert a.memory.used_bytes == 1000 * _MIB
    assert a.memory.free_bytes == 3000 * _MIB
    assert a.memory.total_bytes == 4000 * _MIB
    # A field Prometheus has no series for stays unknown, not zero.
    assert snapshot["GPU-b"].utilization.memory_activity_percent is None
    assert fake_get.call_args.args[0] == "http://prom:9090/api/v1/query"


def test_recorder_queries_the_reduction_aggregation_applies() -> None:
    """Each field uses the reduction ``_aggregate_gpu_stats`` would apply.

    Peak memory is the max used, available memory the min free, and
    utilization the mean, over exactly the recorded window (rounded up to
    whole seconds) and evaluated at its end. Grouping by UUID folds a
    restarted pod's second series back into its GPU, and the mean divides
    summed samples by their count so a split series still averages the
    whole window.
    """
    fake_get, queries = _fake_get(_TWO_GPUS)
    with (
        patch(
            "max.benchmark.benchmark_shared.gpu_metrics_prometheus.requests.get",
            fake_get,
        ),
        patch(
            "max.benchmark.benchmark_shared.gpu_metrics_prometheus.time.time",
            side_effect=[1000.0, 1425.2],
        ),
    ):
        with PrometheusGPURecorder("http://prom:9090", _SELECTOR):
            pass

    sel = f"{{{_SELECTOR}}}[426s]"

    def mean(metric: str) -> str:
        return (
            f"sum by (UUID) (sum_over_time({metric}{sel}))"
            f" / sum by (UUID) (count_over_time({metric}{sel}))"
        )

    assert queries == [
        f"max by (UUID) (max_over_time(DCGM_FI_DEV_FB_USED{sel}))",
        f"min by (UUID) (min_over_time(DCGM_FI_DEV_FB_FREE{sel}))",
        mean("DCGM_FI_DEV_GPU_UTIL"),
        mean("DCGM_FI_DEV_MEM_COPY_UTIL"),
        "sum by (UUID, pod, kubernetes_node, Hostname)"
        f" (count_over_time(DCGM_FI_DEV_GPU_UTIL{sel}))",
    ]
    assert fake_get.call_args.kwargs["params"]["time"] == "1425.200"


def test_recorder_leaves_stats_empty_when_nothing_matches() -> None:
    """A selector that matches no series disables GPU stats, not the run."""
    fake_get, _ = _fake_get(
        {
            "DCGM_FI_DEV_FB_USED": {},
            "DCGM_FI_DEV_FB_FREE": {},
            "DCGM_FI_DEV_GPU_UTIL": {},
            "DCGM_FI_DEV_MEM_COPY_UTIL": {},
        }
    )
    with patch(
        "max.benchmark.benchmark_shared.gpu_metrics_prometheus.requests.get",
        fake_get,
    ):
        with PrometheusGPURecorder("http://prom:9090", _SELECTOR) as recorder:
            pass

    assert recorder.stats == []


def test_recorder_leaves_stats_empty_when_prometheus_is_unreachable() -> None:
    """An unreachable Prometheus is logged and skipped rather than raised."""
    with patch(
        "max.benchmark.benchmark_shared.gpu_metrics_prometheus.requests.get",
        side_effect=ConnectionError("refused"),
    ):
        with PrometheusGPURecorder("http://prom:9090", _SELECTOR) as recorder:
            pass

    assert recorder.stats == []


def test_recorder_leaves_stats_empty_on_a_query_error() -> None:
    """A PromQL error response is treated like an unreachable server."""
    response = MagicMock()
    response.json.return_value = {"status": "error", "error": "bad query"}
    with patch(
        "max.benchmark.benchmark_shared.gpu_metrics_prometheus.requests.get",
        return_value=response,
    ):
        with PrometheusGPURecorder("http://prom:9090", _SELECTOR) as recorder:
            pass

    assert recorder.stats == []


def test_recorder_rejects_an_empty_selector() -> None:
    """An empty selector would average every GPU the cluster scrapes."""
    with pytest.raises(ValueError, match="selector"):
        PrometheusGPURecorder("http://prom:9090", "  ")


_PREFILL = "mammoth-benchmark-1-prefill-engine-abc-x1"
_DECODE = "mammoth-benchmark-1-decode-engine-def-y1"


def _record_groups(
    placements: Sequence[tuple[str, str, str, float]],
) -> dict[str, str]:
    fake_get, _ = _fake_get(_TWO_GPUS, placements)
    with patch(
        "max.benchmark.benchmark_shared.gpu_metrics_prometheus.requests.get",
        fake_get,
    ):
        with PrometheusGPURecorder("http://prom:9090", _SELECTOR) as recorder:
            pass
    return recorder.device_groups


def test_recorder_groups_disaggregated_devices_by_role() -> None:
    """Prefill and decode GPUs are told apart by their engine pod's name."""
    groups = _record_groups(
        [("GPU-b", _DECODE, "node-2", 10), ("GPU-a", _PREFILL, "node-1", 10)]
    )
    assert groups == {"GPU-a": "prefill", "GPU-b": "decode"}
    # Prefill comes first, so console rows and triage keep a stable order.
    assert list(groups.values()) == ["prefill", "decode"]


def test_recorder_numbers_the_nodes_of_a_multi_node_role() -> None:
    """A role spanning nodes reports each node as its own group."""
    groups = _record_groups(
        [
            ("GPU-a", _DECODE, "node-2", 10),
            ("GPU-b", "mammoth-benchmark-1-decode-engine-def-y2", "node-1", 10),
        ]
    )
    assert groups == {"GPU-b": "decode node 1", "GPU-a": "decode node 2"}


def test_recorder_places_a_restarted_device_by_its_majority_pod() -> None:
    """A device seen under two pods takes the one with more samples."""
    groups = _record_groups(
        [
            ("GPU-a", _PREFILL, "node-1", 3),
            ("GPU-a", _DECODE, "node-1", 30),
            ("GPU-b", _DECODE, "node-1", 30),
        ]
    )
    assert groups == {"GPU-a": "decode", "GPU-b": "decode"}


def test_recorder_leaves_aggregated_devices_ungrouped() -> None:
    """Pods with no prefill or decode role keep the single mean."""
    groups = _record_groups(
        [("GPU-a", "mammoth-benchmark-1-engine-abc", "node-1", 10)]
    )
    assert groups == {}


def test_recorder_keeps_stats_when_the_grouping_query_fails() -> None:
    """A failed grouping query loses only the per-role split."""
    fake_get, _ = _fake_get(_TWO_GPUS)
    calls = fake_get.side_effect

    def get(
        url: str, *, params: Mapping[str, str], timeout: float
    ) -> MagicMock:
        if _is_group_query(params["query"]):
            raise ConnectionError("refused")
        return calls(url, params=params, timeout=timeout)

    with patch(
        "max.benchmark.benchmark_shared.gpu_metrics_prometheus.requests.get",
        side_effect=get,
    ):
        with PrometheusGPURecorder("http://prom:9090", _SELECTOR) as recorder:
            pass
    assert len(recorder.stats) == 1
    assert recorder.device_groups == {}
