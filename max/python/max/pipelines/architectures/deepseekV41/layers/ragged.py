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

"""Per-token table describing a ragged batch for DeepSeek-V4.1-Flash."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from max.dtype import DType
from max.graph import DeviceRef, Dim, DimLike, TensorValue, ops

from ...deepseekV4.layers.cache import scalar
from ...deepseekV4.layers.ragged import RaggedRows, arange_to, segment_ids

__all__ = ["ReplicaTokens", "V41RaggedRows"]


@dataclass
class V41RaggedRows(RaggedRows):
    """:class:`RaggedRows` plus each request's token count in this batch.

    ``starts`` and ``lengths`` share the offsets' batch dim ``b``, so they
    combine elementwise (``starts + lengths - 1``) and per-request tables can
    be sized by :attr:`batch`.
    """

    lengths: TensorValue | None = None
    """``[b]`` int32 tokens of each request in this batch."""

    @property
    def batch(self) -> Dim:
        """``b``, the offsets' batch dim."""
        assert self.lengths is not None
        return self.lengths.shape[0]

    @classmethod
    def from_offsets(
        cls,
        offsets: TensorValue,
        total: DimLike,
        starts: TensorValue | None = None,
        windows: Mapping[int, DimLike | None] | None = None,
    ) -> V41RaggedRows:
        """Builds from the graph's ``input_row_offsets`` and ``cache_lengths``.

        Args:
            offsets: ``[b + 1]`` ragged row offsets.
            total: Total token count across the batch.
            starts: ``[b]`` per-request cache lengths, rebound to the offsets'
                batch dim; ``None`` means every request starts at position 0.
            windows: Must be empty; window tables are not supported.

        Raises:
            ValueError: If ``windows`` is non-empty.
        """
        if windows:
            raise ValueError("V41RaggedRows builds no WindowRows")
        device = offsets.device
        off = ops.cast(offsets, DType.int32)
        b = off.shape[0] - 1
        i = arange_to(b, b, device)
        lengths = ops.gather(off, i + 1, axis=0) - ops.gather(off, i, axis=0)
        if starts is None:
            starts = ops.broadcast_to(scalar(0, device), [b])
        else:
            starts = ops.rebind(ops.cast(starts, DType.int32), [b])
        bid, index = segment_ids(off, total, device)
        local = index - ops.gather(off, bid, axis=0)
        positions = ops.gather(starts, bid, axis=0) + local
        return cls(
            offsets=ops.cast(offsets, DType.uint32),
            starts=starts,
            bid=bid,
            index=index,
            local=local,
            positions=positions,
            total=index.shape[0],
            device=device,
            lengths=lengths,
        )


@dataclass(frozen=True)
class ReplicaTokens:
    """Where each data-parallel device's tokens sit in the whole batch.

    The batch is replica-major: device ``i`` holds its replica's requests,
    whose tokens are rows ``ranges[i]`` of the batch's ``total`` tokens.
    """

    total: Dim
    """The whole batch's token count."""

    ranges: list[tuple[TensorValue, TensorValue]]
    """Per device, its first and past-the-end token row, int64 on CPU."""

    @classmethod
    def from_splits(
        cls,
        host_offsets: TensorValue,
        splits: TensorValue,
        total: DimLike,
        num_devices: int,
    ) -> ReplicaTokens:
        """Builds from the batch's host row offsets and replica splits.

        Args:
            host_offsets: ``[B + 1]`` row offsets of the whole batch, on CPU.
            splits: ``[num_devices + 1]`` int64 request boundaries of the
                replicas, on CPU.
            total: The whole batch's token count.
            num_devices: Devices, one replica each.
        """
        offsets = ops.cast(host_offsets, DType.int64)
        cpu = DeviceRef.CPU()
        return cls(
            total=Dim(total),
            ranges=[
                (
                    offsets[splits[i]].to(cpu),
                    offsets[splits[i + 1]].to(cpu),
                )
                for i in range(num_devices)
            ],
        )
