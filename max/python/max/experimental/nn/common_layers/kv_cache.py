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
from dataclasses import dataclass, fields
from typing import cast

from max.experimental.sharding import DeviceMapping
from max.experimental.tensor import Tensor
from max.graph import BufferValue, TensorValue
from max.nn.kv_cache.input_types import KVCacheInputsPerDevice
from max.nn.kv_cache.input_types import PagedCacheValues as _PagedCacheValues


@dataclass
class PagedCacheValues(KVCacheInputsPerDevice[Tensor, Tensor]):
    """Distributed view of KV cache inputs."""

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

        mla_num_partitions: Tensor | None = None
        if per_device[0].mla_num_partitions is not None:
            mla_num_partitions = _wrap(
                cast(
                    list[TensorValue],
                    [d.mla_num_partitions for d in per_device],
                )
            )

        page_stride = _wrap(
            cast(list[TensorValue], [d.page_stride for d in per_device])
        )

        scales_page_stride: Tensor | None = None
        if per_device[0].scales_page_stride is not None:
            scales_page_stride = _wrap(
                cast(
                    list[TensorValue],
                    [d.scales_page_stride for d in per_device],
                )
            )

        scales_lookup_table: Tensor | None = None
        if per_device[0].scales_lookup_table is not None:
            scales_lookup_table = _wrap(
                cast(
                    list[TensorValue],
                    [d.scales_lookup_table for d in per_device],
                )
            )

        return cls(
            kv_blocks=_wrap([d.kv_blocks for d in per_device]),
            cache_lengths=_wrap([d.cache_lengths for d in per_device]),
            lookup_table=_wrap([d.lookup_table for d in per_device]),
            max_prompt_length=_wrap([d.max_prompt_length for d in per_device]),
            max_cache_length=_wrap([d.max_cache_length for d in per_device]),
            kv_scales=kv_scales,
            page_stride=page_stride,
            scales_page_stride=scales_page_stride,
            scales_lookup_table=scales_lookup_table,
            attention_dispatch_metadata=attention_dispatch_metadata,
            mla_num_partitions=mla_num_partitions,
        )

    def to_graph_values(self) -> _PagedCacheValues:
        """Returns this single-device cache as graph values.

        The blocks and scales become buffers, since attention kernels write
        them in place.
        """
        values = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, Tensor):
                is_buffer = f.name in ("kv_blocks", "kv_scales")
                value = BufferValue(value) if is_buffer else TensorValue(value)
            values[f.name] = value
        return _PagedCacheValues(**values)

    @property
    def n_devices(self) -> int:
        """Returns the number of devices the paged KV cache is located on."""
        return len(self.kv_blocks.local_shards)

    def __iter__(self) -> Iterator[Tensor]:
        # Canonical paged KV ABI order (intentionally skip attention_dispatch_metadata).
        yield self.kv_blocks
        yield self.cache_lengths
        yield self.lookup_table
        yield self.max_prompt_length
        yield self.max_cache_length
        if self.kv_scales is not None:
            yield self.kv_scales
