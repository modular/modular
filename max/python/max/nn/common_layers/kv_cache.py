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
from typing import Any, TypeAlias, cast

from max import tree
from max.graph import BufferValue, TensorValue
from max.nn.kv_cache.cache_params import KVCacheParamInterface
from max.nn.kv_cache.input_types import KVCacheInputsPerDevice
from max.nn.kv_cache.input_types import PagedCacheValues as _PagedCacheValues
from max.tensor import Tensor
from max.tensor.sharding import DeviceMapping, DeviceMesh

TensorLike: TypeAlias = TensorValue | BufferValue | Tensor
"""One device's shard of a cache field: a graph value, or a single-device
tensor in the graph being built."""


@dataclass
class PagedCacheValues(KVCacheInputsPerDevice[Tensor, Tensor]):
    """Distributed view of KV cache inputs."""

    @classmethod
    def from_upstream(
        cls,
        per_device: Sequence[_PagedCacheValues]
        | Sequence[KVCacheInputsPerDevice[Tensor, Tensor]],
        mapping: DeviceMapping,
    ) -> PagedCacheValues:
        """Constructs from per-device caches, as graph values or tensors.

        Each device's fields are its shards: graph values, single-device
        tensors in the graph being built, or, for a decoder called eagerly,
        real tensors.
        """

        def _wrap(values: Sequence[TensorLike]) -> Tensor:
            return cache_tensor(values, mapping)

        kv_scales: Tensor | None = None
        if per_device[0].kv_scales is not None:
            scale_values = cast(
                list[TensorLike], [d.kv_scales for d in per_device]
            )
            kv_scales = _wrap(scale_values)

        attention_dispatch_metadata: Tensor | None = None
        if per_device[0].attention_dispatch_metadata is not None:
            attention_dispatch_metadata_values = cast(
                list[TensorLike],
                [d.attention_dispatch_metadata for d in per_device],
            )
            attention_dispatch_metadata = _wrap(
                attention_dispatch_metadata_values
            )

        mla_num_partitions: Tensor | None = None
        if per_device[0].mla_num_partitions is not None:
            mla_num_partitions = _wrap(
                cast(
                    list[TensorLike],
                    [d.mla_num_partitions for d in per_device],
                )
            )

        page_stride = _wrap(
            cast(list[TensorLike], [d.page_stride for d in per_device])
        )

        scales_page_stride: Tensor | None = None
        if per_device[0].scales_page_stride is not None:
            scales_page_stride = _wrap(
                cast(
                    list[TensorLike],
                    [d.scales_page_stride for d in per_device],
                )
            )

        scales_lookup_table: Tensor | None = None
        if per_device[0].scales_lookup_table is not None:
            scales_lookup_table = _wrap(
                cast(
                    list[TensorLike],
                    [d.scales_lookup_table for d in per_device],
                )
            )

        mixed_step_split: Tensor | None = None
        if per_device[0].mixed_step_split is not None:
            records: list[TensorLike] = []
            for device in per_device:
                record = device.mixed_step_split
                assert isinstance(record, TensorLike), (
                    "every device of a replica declares the split record"
                )
                records.append(record)
            mixed_step_split = _wrap(records)

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
            split_head_dims=per_device[0].split_head_dims,
            mixed_step_split=mixed_step_split,
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


def cache_tensor(
    shards: Sequence[TensorLike], mapping: DeviceMapping | None = None
) -> Tensor:
    """Returns one cache input's shards as a tensor, keeping real ones.

    Like :meth:`~max.tensor.Tensor.from_shard_values`, except that real
    tensors, as a decoder called eagerly gets, keep their buffers, so ops
    that write the cache in place write the decoder's buffers.

    Args:
        shards: One graph value or tensor per device.
        mapping: How the shards make up the tensor. Required for more than
            one shard.

    Returns:
        A tensor backed by ``shards``.
    """
    real = [s for s in shards if isinstance(s, Tensor) and s.real]
    if len(real) != len(shards):
        return Tensor.from_shard_values(shards, mapping)
    if len(real) == 1:
        return real[0]
    assert mapping is not None
    devices = tuple(s.device for s in real)
    if devices != mapping.mesh.devices:
        # As in from_shard_values, a field that lives on the host, such as
        # the max cache length, keeps the mapping's grid on its own devices.
        mesh = mapping.mesh
        mapping = DeviceMapping(
            DeviceMesh(devices, mesh.mesh_shape, mesh.axis_names),
            mapping.placements,
        )
    return Tensor._from_shards(tuple(s.driver_tensor for s in real), mapping)


def kv_input_values(kv: tree.Tree[Tensor] | Sequence[Tensor]) -> list[Any]:
    """Returns a decoder's KV cache inputs for its params to unflatten.

    A traced decoder gets each input's graph value. A decoder called eagerly
    gets the real tensors, which have no graph values.

    Args:
        kv: The decoder's KV cache inputs.

    Returns:
        One value per input, in order.
    """
    leaves = tree.leaves(kv)
    if any(leaf.real for leaf in leaves):
        return leaves
    return [leaf._graph_value for leaf in leaves]


def per_device_caches(
    kv_params: KVCacheParamInterface, kv: tree.Tree[Tensor]
) -> list[KVCacheInputsPerDevice[Any, Any]]:
    """Returns a decoder's KV cache inputs as one record per device.

    A traced decoder gets records of graph values, as
    ``kv_params.unflatten_kv_inputs`` builds them. A decoder called eagerly
    gets real tensors, which have no graph values, so each record is a
    :class:`~max.nn.common_layers.kv_cache.PagedCacheValues` of them, whose
    blocks attention ops can write.

    Args:
        kv_params: The parameters of the cache ``kv`` holds.
        kv: The decoder's KV cache inputs, as its ``DecoderInputs`` holds
            them.

    Returns:
        One record per device, in device order.
    """
    values = kv_input_values(kv)
    records = tree.leaves(
        kv_params.unflatten_kv_inputs(iter(values)),
        leaf=KVCacheInputsPerDevice,
    )
    if not any(isinstance(v, Tensor) for v in values):
        return records
    return [
        PagedCacheValues(**{f.name: getattr(r, f.name) for f in fields(r)})
        for r in records
    ]


__all__ = [
    "PagedCacheValues",
    "TensorLike",
    "cache_tensor",
    "kv_input_values",
    "per_device_caches",
]
