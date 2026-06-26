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

import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
from max.driver import Buffer, Device
from max.dtype import DType
from numpy.typing import NDArray


@dataclass(frozen=True)
class EplbStatsMetadata:
    """Static metadata describing the shape and semantics of a snapshot."""

    num_moe_layers: int
    """Number of MoE layers being profiled."""

    num_logical_experts: int
    """Number of logical experts per layer."""

    num_experts_per_token: int
    """Top-k experts selected per token."""

    def __post_init__(self) -> None:
        assert self.num_moe_layers > 0, (
            f"num_moe_layers must be > 0, got {self.num_moe_layers!r}"
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
            f"{self.num_experts_per_token!r} and "
            f"{self.num_logical_experts!r}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-safe dict representation."""
        return {
            "num_moe_layers": self.num_moe_layers,
            "num_logical_experts": self.num_logical_experts,
            "num_experts_per_token": self.num_experts_per_token,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EplbStatsMetadata:
        """Constructs an instance from a dict produced by method to_dict.
        Args:
            data: A dict produced by method to_dict.
        Returns:
            An `EplbStatsMetadata` class instance.
        Raises:
            ValueError: If a required field is missing or has the wrong
                type.
        """
        required = (
            "num_moe_layers",
            "num_logical_experts",
            "num_experts_per_token",
        )
        for key in required:
            if key not in data:
                raise ValueError(f"Missing required field: {key}")
            if not isinstance(data[key], int) or isinstance(data[key], bool):
                raise ValueError(
                    f"Field {key} must be an int, got {data[key]!r}"
                )

        return cls(
            num_moe_layers=data["num_moe_layers"],
            num_logical_experts=data["num_logical_experts"],
            num_experts_per_token=data["num_experts_per_token"],
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
    """Int64 array of shape ``(num_moe_layers, num_logical_experts)``. Read-only."""

    total_tokens: int = 0
    """Total number of tokens that contributed to the histogram.

    ``sum(histogram) == total_tokens * num_experts_per_token`` when
    accumulation is well-formed.
    """

    def __post_init__(self) -> None:
        assert isinstance(self.histogram, np.ndarray), (
            f"histogram must be a numpy ndarray, got "
            f"{type(self.histogram).__name__}"
        )
        expected_shape = (
            self.metadata.num_moe_layers,
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

    @classmethod
    def from_array(
        cls,
        metadata: EplbStatsMetadata,
        histogram: NDArray[np.int64],
        *,
        total_tokens: int = 0,
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
        Returns:
            A new class instance of `EplbStatsSnapshot`.
        Raises:
            AssertionError: If shape/dtype/values are invalid.
        """
        return cls(
            metadata=metadata,
            histogram=np.array(histogram, dtype=np.int64, copy=True),
            total_tokens=int(total_tokens),
        )

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-safe dict representation.
        The histogram is serialized as a nested list[list[int]] and
        int64 values are converted to Python int via ndarray.tolist().
        Returns:
            A dict that is safe to pass directly to json.dumps.
        """
        return {
            "metadata": self.metadata.to_dict(),
            "histogram": self.histogram.tolist(),
            "total_tokens": self.total_tokens,
        }

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
            "total_tokens",
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
        total_tokens = data["total_tokens"]
        if not isinstance(total_tokens, int) or isinstance(total_tokens, bool):
            raise ValueError(
                f"total_tokens must be an int, got {total_tokens!r}"
            )

        return cls.from_array(
            metadata=metadata,
            histogram=histogram,
            total_tokens=total_tokens,
        )


class EplbStatsAccumulator:
    """Per-device on-GPU expert-routing counter, drained on HTTP request."""

    def __init__(
        self,
        metadata: EplbStatsMetadata,
        devices: list[Device],
    ) -> None:
        self._metadata = metadata
        self._devices = devices

        num_layers = metadata.num_moe_layers
        num_experts = metadata.num_logical_experts
        self._layer_device_buffers: list[list[Buffer]] = [
            [
                Buffer.zeros(shape=(num_experts,), dtype=DType.int64).to(d)
                for d in devices
            ]
            for _ in range(num_layers)
        ]
        self._total_tokens: int = 0
        self._lock = threading.Lock()

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
        return self._metadata

    def record_batch_total_tokens(self, num_tokens: int) -> None:
        with self._lock:
            self._total_tokens += int(num_tokens)

    def snapshot(self) -> EplbStatsSnapshot:
        num_layers = self._metadata.num_moe_layers
        num_experts = self._metadata.num_logical_experts
        histogram = np.zeros((num_layers, num_experts), dtype=np.int64)
        for layer_idx, layer_bufs in enumerate(self._layer_device_buffers):
            histogram[layer_idx] = sum(b.to_numpy() for b in layer_bufs)
        with self._lock:
            total = self._total_tokens
        return EplbStatsSnapshot.from_array(
            metadata=self._metadata,
            histogram=histogram,
            total_tokens=total,
        )

    def reset(self) -> None:
        with self._lock:
            self._layer_device_buffers = [
                [
                    Buffer.zeros(
                        shape=(self.metadata.num_logical_experts,),
                        dtype=DType.int64,
                    ).to(d)
                    for d in self._devices
                ]
                for _ in range(self._metadata.num_moe_layers)
            ]
            self._total_tokens = 0
