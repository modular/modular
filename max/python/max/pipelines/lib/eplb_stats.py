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

"""Schema types for expert-parallel Load Balancing (EPLB) MoE profiling snapshots.

This module defines the schema for EPLB routing
histograms collected by the model worker and exposed over an internal
HTTP route. It is intentionally free of any serve, ZMQ, or GPU
imports so it can be reused by:

* the model worker (producer)
* the API server (consumer / serializer)
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
from max.driver import Buffer, Device
from max.dtype import DType
from numpy.typing import NDArray

from .eplb_rebalance import rebalance_experts

logger = logging.getLogger("max.serve")

__all__ = [
    "AggregatedEplbPlanner",
    "EplbPlacement",
    "EplbPlanner",
    "EplbStatsAccumulator",
    "EplbStatsMetadata",
    "EplbStatsSnapshot",
    "EplbTopology",
    "PerLayerEplbPlanner",
    "eplb_planner_for",
]


@dataclass(frozen=True)
class EplbStatsMetadata:
    """Static metadata describing the shape and semantics of a snapshot."""

    num_layers: int
    """Total transformer layers profiled. Equals the histogram row count."""

    num_moe_layers: int
    """Count of true MoE (routing) layers. Dense layers are all-zero rows."""

    moe_layer_indices: tuple[int, ...]
    """Histogram row indices that correspond to MoE layers (row i == layer i)."""

    num_logical_experts: int
    """Number of logical experts per layer."""

    num_experts_per_token: int
    """Top-k experts selected per token."""

    def __post_init__(self) -> None:
        assert self.num_layers > 0, (
            f"num_layers must be > 0, got {self.num_layers!r}"
        )
        assert 0 < self.num_moe_layers <= self.num_layers, (
            f"num_moe_layers must be in (0, {self.num_layers}], got "
            f"{self.num_moe_layers!r}"
        )
        assert len(self.moe_layer_indices) == self.num_moe_layers, (
            f"moe_layer_indices length {len(self.moe_layer_indices)} != "
            f"num_moe_layers {self.num_moe_layers}"
        )
        assert all(0 <= i < self.num_layers for i in self.moe_layer_indices), (
            f"moe_layer_indices out of range [0, {self.num_layers}): "
            f"{self.moe_layer_indices!r}"
        )
        assert self.num_logical_experts > 0, (
            f"num_logical_experts must be > 0, got {self.num_logical_experts!r}"
        )
        assert self.num_experts_per_token > 0, (
            f"num_experts_per_token must be > 0, got "
            f"{self.num_experts_per_token!r}"
        )
        assert self.num_experts_per_token <= self.num_logical_experts, (
            "num_experts_per_token must be <= num_logical_experts, got "
            f"{self.num_experts_per_token!r} and {self.num_logical_experts!r}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-safe dict representation."""
        return {
            "num_layers": self.num_layers,
            "num_moe_layers": self.num_moe_layers,
            "moe_layer_indices": list(self.moe_layer_indices),
            "num_logical_experts": self.num_logical_experts,
            "num_experts_per_token": self.num_experts_per_token,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EplbStatsMetadata:
        """Constructs an instance from a dict produced by ``to_dict``.

        Tolerates legacy snapshots that only had ``num_moe_layers`` (which then
        meant the row count): those map to ``num_layers`` with every row treated
        as MoE.
        """
        for key in ("num_logical_experts", "num_experts_per_token"):
            if key not in data:
                raise ValueError(f"Missing required field: {key}")
            if not isinstance(data[key], int) or isinstance(data[key], bool):
                raise ValueError(
                    f"Field {key} must be an int, got {data[key]!r}"
                )
        if "num_layers" in data:
            num_layers = int(data["num_layers"])
            num_moe_layers = int(data.get("num_moe_layers", num_layers))
        else:
            # Legacy: num_moe_layers was the row count.
            num_layers = int(data["num_moe_layers"])
            num_moe_layers = num_layers
        moe_layer_indices = tuple(
            int(i) for i in data.get("moe_layer_indices", range(num_layers))
        )
        return cls(
            num_layers=num_layers,
            num_moe_layers=num_moe_layers,
            moe_layer_indices=moe_layer_indices,
            num_logical_experts=int(data["num_logical_experts"]),
            num_experts_per_token=int(data["num_experts_per_token"]),
        )


@dataclass(frozen=True)
class EplbStatsSnapshot:
    """An immutable snapshot of per-layer logical-expert routing counts.

    The histogram has shape ``(num_moe_layers, num_logical_experts)``
    and dtype int64. Values are cumulative token-routings to each
    logical expert since the worker started accumulating.

    Snapshots are produced by the model worker and consumed by the
    API server. They are intentionally pure-data: no locks, no
    references back into the accumulator.

    The histogram is marked read-only after construction so a consumer
    holding a snapshot can never mutate it.
    """

    metadata: EplbStatsMetadata
    """Static shape/semantics descriptor."""

    histogram: NDArray[np.int64]
    """Int64 array of shape ``(num_layers, num_logical_experts)``. Read-only."""

    total_tokens: int = 0
    """Non-draft (confirmed)  tokens fed to the target. In spec decode this
    EXCLUDES draft candidates, so it is < total_routed_tokens."""

    total_routed_tokens: int = 0
    """Tokens actually routed through the target's experts (confirmed +
    verified drafts). Any MoE row sums to
    ``total_routed_tokens * num_experts_per_token``; equivalently
    ``total_routed_tokens == total_tokens + draft_candidate_tokens``."""

    hostname: str | None = None

    def __post_init__(self) -> None:
        assert isinstance(self.histogram, np.ndarray), (
            f"histogram must be a numpy ndarray, got "
            f"{type(self.histogram).__name__}"
        )
        expected_shape = (
            self.metadata.num_layers,
            self.metadata.num_logical_experts,
        )
        assert self.histogram.shape == expected_shape, (
            f"histogram shape mismatch: expected {expected_shape}, "
            f"got {self.histogram.shape}"
        )
        assert self.histogram.dtype == np.int64, (
            f"histogram dtype must be int64, got {self.histogram.dtype}"
        )
        assert self.total_tokens >= 0, (
            f"total_tokens must be >= 0, got {self.total_tokens}"
        )

        # Freeze the array so a consumer cannot mutate the snapshot.
        self.histogram.setflags(write=False)

    @property
    def draft_candidate_tokens(self) -> int:
        """Extra tokens routed by verified speculative draft candidates.

        0 on non-spec paths.
        """
        return max(0, self.total_routed_tokens - self.total_tokens)

    @classmethod
    def from_array(
        cls,
        metadata: EplbStatsMetadata,
        histogram: NDArray[np.int64],
        *,
        total_tokens: int = 0,
        total_routed_tokens: int = 0,
        hostname: str | None = None,
    ) -> EplbStatsSnapshot:
        """Builds a snapshot, defensively copying the histogram.

        Use this when the source array is owned by a live accumulator
        that may keep mutating after the snapshot is taken.

        Args:
            metadata: Static shape descriptor.
            histogram: 2D int64 array of cumulative counts. Will
                be copied.
            total_tokens: Total number of tokens contributing to the
                histogram so far.
            total_routed_tokens: Total number of tokens routed to experts,
                including speculative draft candidates.
            hostname: Optional worker hostname to tag the snapshot with.

        Returns:
            A new class instance of `EplbStatsSnapshot`.

        Raises:
            AssertionError: If shape/dtype/values are invalid.
        """
        return cls(
            metadata=metadata,
            histogram=np.array(histogram, dtype=np.int64, copy=True),
            total_tokens=int(total_tokens),
            total_routed_tokens=int(total_routed_tokens),
            hostname=hostname,
        )

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-safe dict representation.

        The histogram is serialized as a nested list[list[int]] and
        int64 values are converted to Python int via ndarray.tolist().

        Returns:
            A dict that is safe to pass directly to json.dumps.
        """
        d = {
            "metadata": self.metadata.to_dict(),
            "histogram": self.histogram.tolist(),
            "non_draft_tokens": self.total_tokens,
            "total_routed_tokens": self.total_routed_tokens,
            "hostname": self.hostname,
        }
        if self.draft_candidate_tokens > 0:
            d["draft_candidate_tokens"] = self.draft_candidate_tokens
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EplbStatsSnapshot:
        """Constructs a snapshot from a dict produced by method to_dict.

        Args:
            data: A dict produced by method to_dict.

        Returns:
            A new class instance of EplbStatsSnapshot.

        Raises:
            ValueError: If a required field is missing, has the wrong
                type,
            AssertionError: If the histogram fails shape/dtype
                validation against the metadata.
        """
        for key in (
            "metadata",
            "histogram",
            "non_draft_tokens",
            "hostname",
        ):
            if key not in data:
                raise ValueError(f"Missing required field: {key}")

        metadata_data = data["metadata"]
        if not isinstance(metadata_data, dict):
            raise ValueError(
                f"metadata must be a dict, got {type(metadata_data).__name__}"
            )
        metadata = EplbStatsMetadata.from_dict(metadata_data)
        histogram_data = data["histogram"]
        if not isinstance(histogram_data, list):
            raise ValueError(
                f"histogram must be a list, got {type(histogram_data).__name__}"
            )
        try:
            histogram = np.asarray(histogram_data, dtype=np.int64)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"histogram could not be parsed as int64 array: {e}"
            ) from e
        non_draft_tokens = data["non_draft_tokens"]
        if not isinstance(non_draft_tokens, int) or isinstance(
            non_draft_tokens, bool
        ):
            raise ValueError(
                f"non_draft_tokens must be an int, got {non_draft_tokens!r}"
            )
        total_routed_tokens = data.get("total_routed_tokens", non_draft_tokens)

        return cls.from_array(
            metadata=metadata,
            histogram=histogram,
            total_tokens=non_draft_tokens,
            total_routed_tokens=total_routed_tokens,
        )


class EplbStatsAccumulator:
    """Owns per-GPU persistent int64 histogram buffers.

    Lives in the model worker. Each forward atomically increments
    cells in these buffers via on-GPU scatter_nd_add ops emitted by
    _ep_forward. snapshot() pulls them to host and sums across GPUs.
    """

    def __init__(
        self,
        metadata: EplbStatsMetadata,
        devices: list[Device],
    ) -> None:
        """Initializes a zero-valued accumulator.

        Args:
            metadata: Static shape / semantics descriptor.
            devices: List of devices to allocate buffers on.
        """
        self._metadata = metadata
        self._devices = devices
        self._lock = threading.Lock()
        self._total_tokens: int = 0
        self._layer_device_buffers: list[list[Buffer]] = [
            [
                Buffer.zeros(
                    shape=(metadata.num_logical_experts,),
                    dtype=DType.int64,
                ).to(d)
                for d in devices
            ]
            for _ in range(metadata.num_layers)
        ]

    @property
    def device_buffers(self) -> list[Buffer]:
        """Layer-major X device-major flat list, matches input_types order."""
        return [
            buf
            for layer_bufs in self._layer_device_buffers
            for buf in layer_bufs
        ]

    @property
    def metadata(self) -> EplbStatsMetadata:
        """Static shape/semantics descriptor for the accumulator."""
        return self._metadata

    def record_batch_total_tokens(self, num_tokens: int) -> None:
        """Increments the cumulative token count by `num_tokens`."""
        with self._lock:
            self._total_tokens += int(num_tokens)

    def snapshot(self, hostname: str | None = None) -> EplbStatsSnapshot:
        """Returns a snapshot of the current per-layer routing histograms.

        Pulls each per-(layer, device) counter buffer to host, sums across
        devices, and packages the result with the cumulative token count
        and optional worker hostname.

        Args:
            hostname: Optional worker hostname to tag the snapshot with.

        Returns:
            An immutable `EplbStatsSnapshot`.
        """
        num_layers = self._metadata.num_layers
        num_experts = self._metadata.num_logical_experts
        histogram = np.zeros((num_layers, num_experts), dtype=np.int64)
        for layer_idx, layer_bufs in enumerate(self._layer_device_buffers):
            histogram[layer_idx] = sum(b.to_numpy() for b in layer_bufs)
        with self._lock:
            total = self._total_tokens
        # Tokens routed through the experts, derived from the histogram: each
        # routed token increments exactly ``top_k`` bins, so ``row_sum / top_k``
        # is the exact per-layer routed-token count. It is uniform across MoE
        # layers for target-only profiling; take the max non-zero row. This
        # already includes any speculative draft candidates the target verified
        # (wherever the harness injects them), so no separate host counter is
        # needed. Falls back to ``total`` when nothing has routed yet.
        row_sums = histogram.sum(axis=1)
        nonzero = row_sums[row_sums > 0]
        top_k = self._metadata.num_experts_per_token
        routed = int(nonzero.max() // top_k) if nonzero.size else total
        return EplbStatsSnapshot.from_array(
            metadata=self._metadata,
            histogram=histogram,
            total_tokens=total,
            total_routed_tokens=routed,
            hostname=hostname,
        )

    def reset(self) -> None:
        """Zeros every per-(layer, device) counter buffer and the token count.

        Zeros in place instead of reallocating: device-graph capture pins the
        pre-captured decode graph to the exact Buffer objects this accumulator
        owned at warmup. Reallocating orphans those buffers, so decode-step
        accumulations (which run via captured-graph replay) land in the orphaned
        buffers while snapshot() reads the new ones. Preserving buffer identity
        keeps the captured decode graph writing the live buffers. Runs on the
        model-worker thread between forward steps, so the zero cannot race a
        forward pass.
        """
        with self._lock:
            zero_host = Buffer.from_numpy(
                np.zeros((self._metadata.num_logical_experts,), dtype=np.int64)
            )
            for layer_bufs in self._layer_device_buffers:
                for buf in layer_bufs:
                    buf.inplace_copy_from(zero_host)
            self._total_tokens = 0


@dataclass(frozen=True)
class EplbPlacement:
    """Static log2phy expert placement derived from a snapshot.

    Built once at server startup; consumed by the EP dispatch graph
    via per-device log2phy / logcnt buffers.
    """

    phy2log: NDArray[np.int64]  # [num_layers, num_phy]
    log2phy: NDArray[np.int32]  # [num_layers, num_log, max_replicas]
    logcnt: NDArray[np.int32]  # [num_layers, num_log]

    @property
    def num_phy(self) -> int:
        """Total physical slot count (>= num_logical_experts)."""
        return int(self.phy2log.shape[1])

    @property
    def max_replicas(self) -> int:
        """Width of each ``log2phy`` row, at least the largest replica count.

        Rows are padded with ``-1`` to a width the fused router kernels can
        load as one vector, so this may exceed ``logcnt.max()``.
        """
        return int(self.log2phy.shape[-1])

    @classmethod
    def from_snapshot(
        cls,
        snap: EplbStatsSnapshot,
        *,
        ep_size: int,
        n_nodes: int,
        n_groups: int,
        eplb_redundant_experts_per_gpu: int = 0,
        planner: EplbPlanner | None = None,
    ) -> EplbPlacement:
        """Plans a placement from a routing-histogram snapshot.

        Args:
            snap: The routing histogram to balance.
            ep_size: Number of expert-parallel ranks.
            n_nodes: Number of nodes the ranks span.
            n_groups: Number of expert groups the router selects from.
            eplb_redundant_experts_per_gpu: Redundant expert slots to add per rank.
            planner: The balancing policy. Defaults to
                :class:`PerLayerEplbPlanner`.

        Returns:
            The placement, with one row per histogram row.
        """
        weight_per_layer = np.asarray(snap.histogram, dtype=np.int64)
        topology = EplbTopology(
            ep_size=ep_size,
            n_nodes=n_nodes,
            n_groups=n_groups,
            redundant_experts_per_gpu=eplb_redundant_experts_per_gpu,
        )
        topology.validate(num_logical=weight_per_layer.shape[1])
        placement = (planner or PerLayerEplbPlanner()).plan(
            weight_per_layer, topology
        )
        cls._log_summary(
            weight_per_layer, placement.phy2log, placement.logcnt, ep_size
        )
        return placement

    @classmethod
    def from_rebalance(
        cls, weight: NDArray[np.int64], topology: EplbTopology
    ) -> EplbPlacement:
        """Runs the hierarchical rebalance on every row of ``weight``."""
        phy2log, log2phy, logcnt = rebalance_experts(
            weight,
            topology.num_physical(weight.shape[1]),
            topology.n_groups,
            topology.n_nodes,
            topology.ep_size,
        )
        return cls(
            phy2log=phy2log,
            log2phy=_pad_replicas(log2phy).astype(np.int32),
            logcnt=logcnt.astype(np.int32),
        )

    @classmethod
    def identity(cls, num_layers: int, num_logical: int) -> EplbPlacement:
        """No-op placement: phy[i] = i, one replica per logical expert."""
        phy2log = np.broadcast_to(
            np.arange(num_logical, dtype=np.int64),
            (num_layers, num_logical),
        ).copy()
        log2phy = phy2log.astype(np.int32)[..., None]
        logcnt = np.ones((num_layers, num_logical), dtype=np.int32)
        return cls(phy2log=phy2log, log2phy=log2phy, logcnt=logcnt)

    @staticmethod
    def _log_summary(
        weight: NDArray[np.int64],
        phy2log: NDArray[np.int64],
        logcnt: NDArray[np.int64] | NDArray[np.int32],
        n_gpus: int,
    ) -> None:
        """Logs how far the busiest GPU is above the mean, per MoE layer.

        Each MoE layer ends in an all-to-all that waits for its slowest GPU,
        so the ratio that bounds step time is max/mean within a layer.
        Summing load over layers first would hide skew that differs between
        layers, which is exactly what a per-layer placement removes.
        Replicated experts split their load evenly across replicas.
        """
        num_layers = weight.shape[0]
        # Before: GPU g hosts the contiguous block of logical experts g.
        before = weight.reshape(num_layers, n_gpus, -1).sum(-1)
        counts_per_phy = np.take_along_axis(weight, phy2log, axis=-1)
        logcnt_per_phy = np.take_along_axis(
            logcnt.astype(np.int64), phy2log, axis=-1
        )
        normalized = counts_per_phy / np.maximum(logcnt_per_phy, 1)
        after = normalized.reshape(num_layers, n_gpus, -1).sum(-1)

        active = before.sum(-1) > 0
        if not active.any():
            logger.info("EPLB: snapshot has no routed tokens")
            return

        def skew(load: NDArray[np.float64]) -> NDArray[np.float64]:
            return load.max(-1) / np.maximum(load.mean(-1), 1e-9)

        b = skew(before[active].astype(np.float64))
        a = skew(after[active])
        logger.info(
            "EPLB: per-GPU load max/mean across %d MoE layers: median "
            "%.3f -> %.3f, worst %.3f -> %.3f",
            int(active.sum()),
            float(np.median(b)),
            float(np.median(a)),
            float(b.max()),
            float(a.max()),
        )


_REPLICA_WIDTHS = (1, 2, 4, 8, 16)
"""``log2phy`` row widths the fused router kernels load as one SIMD vector."""


def _pad_replicas(log2phy: NDArray[np.int64]) -> NDArray[np.int64]:
    """Pads ``log2phy``'s replica axis with ``-1`` to a supported width.

    Raises:
        ValueError: If some expert has more replicas than the widest width.
    """
    width = log2phy.shape[-1]
    padded = next((w for w in _REPLICA_WIDTHS if w >= width), None)
    if padded is None:
        raise ValueError(
            f"EPLB placement gives an expert {width} replicas; at most "
            f"{_REPLICA_WIDTHS[-1]} are supported. Lower "
            "eplb_redundant_experts_per_gpu."
        )
    return np.pad(
        log2phy,
        ((0, 0), (0, 0), (0, padded - width)),
        constant_values=-1,
    )


@dataclass(frozen=True)
class EplbTopology:
    """The expert-parallel layout a placement is planned for."""

    ep_size: int
    """Number of expert-parallel ranks."""

    n_nodes: int
    """Number of nodes the ranks span."""

    n_groups: int
    """Number of expert groups the router selects from."""

    redundant_experts_per_gpu: int = 0
    """Redundant expert slots added per rank."""

    def num_physical(self, num_logical: int) -> int:
        """Returns the physical slot count for ``num_logical`` experts."""
        return num_logical + self.redundant_experts_per_gpu * self.ep_size

    def validate(self, num_logical: int) -> None:
        """Raises ``ValueError`` if ``num_logical`` experts cannot fit."""
        if num_logical % self.ep_size or num_logical % self.n_groups:
            raise ValueError(
                f"num_logical={num_logical} must be divisible by "
                f"ep_size={self.ep_size} and n_groups={self.n_groups}"
            )
        if self.redundant_experts_per_gpu < 0:
            raise ValueError(
                "eplb_redundant_experts_per_gpu must be >= 0, got "
                f"{self.redundant_experts_per_gpu}"
            )


class EplbPlanner(ABC):
    """Turns a per-layer routing histogram into an expert placement.

    Implementations are free to choose how load is measured and combined, but
    must return one placement row per histogram row with the same physical
    slot count in every row, since every MoE layer shares one compiled
    subgraph.
    """

    @abstractmethod
    def plan(
        self, weight: NDArray[np.int64], topology: EplbTopology
    ) -> EplbPlacement:
        """Plans a placement.

        Args:
            weight: ``[num_layers, num_logical]`` routed-token counts.
            topology: The layout to plan for.

        Returns:
            The placement, with ``num_layers`` rows.
        """


class PerLayerEplbPlanner(EplbPlanner):
    """Balances every layer against its own routing histogram.

    Dense layers have all-zero rows; they get a valid but unused placement.
    """

    def plan(
        self, weight: NDArray[np.int64], topology: EplbTopology
    ) -> EplbPlacement:
        """Runs the hierarchical rebalance on each layer independently."""
        return EplbPlacement.from_rebalance(weight, topology)


class AggregatedEplbPlanner(EplbPlanner):
    """Balances the sum of all layers and applies that plan to every layer.

    Stable with fewer profiled tokens than the per-layer planner, but ignores
    load that differs between layers.
    """

    def plan(
        self, weight: NDArray[np.int64], topology: EplbTopology
    ) -> EplbPlacement:
        """Rebalances the layer sum once and broadcasts it to every layer."""
        num_layers = weight.shape[0]
        once = EplbPlacement.from_rebalance(
            weight.sum(axis=0, keepdims=True), topology
        )
        return EplbPlacement(
            phy2log=np.repeat(once.phy2log, num_layers, axis=0),
            log2phy=np.repeat(once.log2phy, num_layers, axis=0),
            logcnt=np.repeat(once.logcnt, num_layers, axis=0),
        )


_PLANNERS: dict[str, type[EplbPlanner]] = {
    "per_layer": PerLayerEplbPlanner,
    "aggregated": AggregatedEplbPlanner,
}


def eplb_planner_for(name: str) -> EplbPlanner:
    """Returns the planner registered under ``name``.

    Raises:
        ValueError: If no planner has that name.
    """
    if name not in _PLANNERS:
        raise ValueError(
            f"Unknown EPLB planner {name!r}; expected one of {sorted(_PLANNERS)}"
        )
    return _PLANNERS[name]()
