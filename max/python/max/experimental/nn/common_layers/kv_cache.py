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

"""Distributed PagedCacheValues."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import cast

from max.experimental.sharding import DeviceMapping
from max.experimental.tensor import Tensor
from max.graph import BufferValue, TensorValue
from max.nn.kv_cache.input_types import (
    KVCacheInputsPerDevice,
)
from max.nn.kv_cache.input_types import PagedCacheValues as _PagedCacheValues


@dataclass
class PagedCacheValues(KVCacheInputsPerDevice[Tensor, Tensor]):
    """Tensors holding the values for the allocated paged KV cache.

    May be located on multiple devices.
    """

    kv_blocks: Tensor
    cache_lengths: Tensor
    lookup_table: Tensor
    max_lengths: Tensor
    kv_scales: Tensor | None = None
    attention_dispatch_metadata: Tensor | None = None

    @classmethod
    def from_upstream(
        cls,
        per_device: Sequence[_PagedCacheValues],
        mapping: DeviceMapping,
    ) -> PagedCacheValues:
        """Constructs from per-device upstream PagedCacheValues."""

        def _wrap(
            values: Sequence[TensorValue | BufferValue],
        ) -> Tensor:
            return Tensor.from_shard_values(values, mapping)

        kv_scales: Tensor | None = None
        if per_device[0].kv_scales is not None:
            scale_values = cast(
                list[BufferValue], [d.kv_scales for d in per_device]
            )
            kv_scales = _wrap(scale_values)

        attention_dispatch_metadata: Tensor | None = None
        if per_device[0].attention_dispatch_metadata is not None:
            attention_dispatch_metadata_values = cast(
                list[TensorValue],
                [d.attention_dispatch_metadata for d in per_device],
            )
            attention_dispatch_metadata = _wrap(
                attention_dispatch_metadata_values
            )

        return cls(
            kv_blocks=_wrap([d.kv_blocks for d in per_device]),
            cache_lengths=_wrap([d.cache_lengths for d in per_device]),
            lookup_table=_wrap([d.lookup_table for d in per_device]),
            max_lengths=_wrap([d.max_lengths for d in per_device]),
            kv_scales=kv_scales,
            attention_dispatch_metadata=attention_dispatch_metadata,
        )

    @property
    def n_devices(self) -> int:
        """Returns the number of devices the paged KV cache is located on."""
        return len(self.kv_blocks.local_shards)

    def __iter__(self) -> Iterator[Tensor]:
        # Canonical paged KV ABI order (intentionally skip attention_dispatch_metadata).
        yield self.kv_blocks
        yield self.cache_lengths
        yield self.lookup_table
        yield self.max_lengths
        if self.kv_scales is not None:
            yield self.kv_scales

    def for_device(self, i: int) -> _PagedCacheValues:
        """Returns the local PagedCacheValues for the given device."""
        return _PagedCacheValues(
            kv_blocks=BufferValue(self.kv_blocks.local_shards[i]),
            cache_lengths=TensorValue(self.cache_lengths.local_shards[i]),
            lookup_table=TensorValue(self.lookup_table.local_shards[i]),
            max_lengths=TensorValue(self.max_lengths.local_shards[i]),
            kv_scales=BufferValue(self.kv_scales.local_shards[i])
            if self.kv_scales is not None
            else None,
            attention_dispatch_metadata=TensorValue(
                self.attention_dispatch_metadata.local_shards[i]
            )
            if self.attention_dispatch_metadata is not None
            else None,
        )
