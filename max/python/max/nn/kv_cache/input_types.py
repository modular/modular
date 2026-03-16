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

import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import (
    Any,
    Generic,
    Protocol,
    TypeAlias,
    cast,
    runtime_checkable,
)

from max.driver import Buffer
from max.dtype import DType
from max.graph import BufferType, BufferValue, TensorType, TensorValue, Value
from typing_extensions import TypeVar

logger = logging.getLogger("max.pipelines")

T = TypeVar("T", default=Any)


@dataclass
class NestedIterableDataclass(Generic[T]):
    """Base class for input symbols for KV cache managers.

    The derived class is responsible for defining the input symbols for the
    specific KV cache manager.
    For example, here's a derived class for a text KV cache manager:

    .. code-block:: python

        @dataclass
        class PagedCacheValues(NestedIterableDataclass[TensorType]):
            kv_blocks: TensorType
            cache_lengths: TensorType
            lookup_table: TensorType
            max_lengths: TensorType
    """

    def __iter__(self) -> Iterator[T]:
        """Iterates through each field in order."""
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            if value is None:
                continue
            if isinstance(value, NestedIterableDataclass):
                yield from value
            else:
                yield value

    def __getitem__(self, index: int | slice) -> Any:
        return list(self)[index]

    def flatten(self) -> list[T]:
        """Returns all leaf values as a flat list.

        Returns:
            A list containing every non-``None`` leaf value yielded by
            iterating this dataclass.
        """
        return list(self)


IterableInputSymbols: TypeAlias = NestedIterableDataclass[
    TensorType | BufferType
]


_DispatchMetadataT = TypeVar("_DispatchMetadataT", TensorType, TensorValue)


@dataclass
class AttentionDispatchMetadata(
    NestedIterableDataclass[_DispatchMetadataT],
    Generic[_DispatchMetadataT],
):
    """Wraps the scalar attention dispatch metadata tensor for a single device.

    The wrapped ``tensor`` must have dtype ``int64`` and rank 1. It encodes
    the four dispatch scalars consumed by ragged decode kernels: batch size,
    maximum query sequence length, number of partitions, and maximum cache
    valid length.
    """

    tensor: _DispatchMetadataT

    def __post_init__(self) -> None:
        if self.tensor.dtype != DType.int64:
            raise ValueError(
                "expected attention_dispatch_metadata dtype int64, got "
                f"{self.tensor.dtype}"
            )
        if self.tensor.rank != 1:
            raise ValueError(
                "expected attention_dispatch_metadata rank 1, got "
                f"{self.tensor.rank}"
            )


@dataclass
class PagedCacheInputSymbols(IterableInputSymbols):
    """Symbolic graph input types for a single device's paged KV cache."""

    kv_blocks: BufferType
    cache_lengths: TensorType
    lookup_table: TensorType
    max_lengths: TensorType
    kv_scales: BufferType | None = None  # KV scales for FP8 quantization
    dispatch_metadata: AttentionDispatchMetadata[TensorType] | None = None


@dataclass
class PagedCacheValues(NestedIterableDataclass[BufferValue | TensorValue]):
    """Concrete graph values for a single device's paged KV cache."""

    kv_blocks: BufferValue
    cache_lengths: TensorValue
    lookup_table: TensorValue
    max_lengths: TensorValue
    kv_scales: BufferValue | None = None  # KV scales for FP8 quantization
    dispatch_metadata: AttentionDispatchMetadata[TensorValue] | None = None

    def __iter__(self) -> Iterator[BufferValue | TensorValue]:
        # Canonical paged KV ABI order. (intentionally skip AttentionDispatchMetadata)
        yield self.kv_blocks
        yield self.cache_lengths
        yield self.lookup_table
        yield self.max_lengths
        if self.kv_scales is not None:
            yield self.kv_scales


def unflatten_ragged_attention_inputs(
    kv_inputs_flat: Sequence[Any], *, n_devices: int
) -> list[PagedCacheValues]:
    """Unmarshals flattened KV graph inputs into typed cache values.

    Args:
        kv_inputs_flat: Flattened graph values for all KV inputs.
            Elements may be ``Value`` instances or ``Tensor``-like objects
            with a ``_graph_value`` attribute.
        n_devices: Number of devices represented in ``kv_inputs_flat``.
    """
    # Extract graph values from Tensor-like objects.
    kv_inputs_flat = [
        v._graph_value if hasattr(v, "_graph_value") else v
        for v in kv_inputs_flat
    ]

    if n_devices <= 0:
        raise ValueError(f"n_devices must be positive, got {n_devices}")

    if len(kv_inputs_flat) % n_devices != 0:
        raise ValueError(
            "unexpected flattened KV input length: expected a multiple of "
            f"{n_devices}, got {len(kv_inputs_flat)}"
        )

    if any(not isinstance(value, Value) for value in kv_inputs_flat):
        raise TypeError("kv_inputs_flat must contain max.graph.Value instances")

    fields_per_device = len(kv_inputs_flat) // n_devices
    if fields_per_device not in (5, 6):
        raise ValueError(
            f"fields_per_device must be 5 or 6, got {fields_per_device}"
        )

    has_kv_scales = fields_per_device == 6
    kv_caches_per_dev: list[PagedCacheValues] = []
    for i in range(n_devices):
        start_idx = i * fields_per_device
        next_idx = start_idx + 4
        kv_scales = None
        if has_kv_scales:
            kv_scales = kv_inputs_flat[next_idx].buffer
            next_idx += 1

        metadata = AttentionDispatchMetadata(
            cast(TensorValue, kv_inputs_flat[next_idx].tensor)
        )

        kv_caches_per_dev.append(
            PagedCacheValues(
                kv_blocks=kv_inputs_flat[start_idx].buffer,
                cache_lengths=kv_inputs_flat[start_idx + 1].tensor,
                lookup_table=kv_inputs_flat[start_idx + 2].tensor,
                max_lengths=kv_inputs_flat[start_idx + 3].tensor,
                kv_scales=kv_scales,
                dispatch_metadata=metadata,
            )
        )

    return kv_caches_per_dev


def attention_dispatch_metadata(
    kv_collection: PagedCacheValues,
    *,
    device_idx: int | None = None,
) -> AttentionDispatchMetadata[TensorValue]:
    """Extracts the :class:`AttentionDispatchMetadata` from a KV collection.

    Args:
        kv_collection: The paged KV cache values to extract metadata from.
        device_idx: Optional device index included in the error message when
            ``dispatch_metadata`` is ``None``.

    Returns:
        The :class:`AttentionDispatchMetadata` stored on the collection.

    Raises:
        ValueError: If ``kv_collection.dispatch_metadata`` is ``None``.
    """
    dispatch_metadata = kv_collection.dispatch_metadata
    if dispatch_metadata is not None:
        return dispatch_metadata

    location = "" if device_idx is None else f" for device {device_idx}"
    raise ValueError(
        "Expected AttentionDispatchMetadata in kv_collection.dispatch_metadata"
        f"{location}."
    )


def attention_dispatch_metadata_list(
    kv_collections: Sequence[PagedCacheValues],
) -> list[AttentionDispatchMetadata[TensorValue]]:
    """Extracts :class:`AttentionDispatchMetadata` from each KV collection.

    Args:
        kv_collections: A sequence of per-device paged KV cache values.

    Returns:
        A list of :class:`AttentionDispatchMetadata` instances, one per
        device, in the same order as ``kv_collections``.

    Raises:
        ValueError: If any collection is missing its ``dispatch_metadata``.
    """
    return [
        attention_dispatch_metadata(kv_collection, device_idx=i)
        for i, kv_collection in enumerate(kv_collections)
    ]


@runtime_checkable
class FlattenableInputSymbols(Protocol):
    """A sequence-like collection of input symbols that can be flattened."""

    def __iter__(self) -> Iterator[Any]: ...
    def __getitem__(self, index: int | slice) -> Any: ...
    def __len__(self) -> int: ...
    def flatten(self) -> list[TensorType | BufferType]:
        """Returns all input symbols as a flat list.

        Returns:
            A flat list of every :class:`~max.graph.TensorType` and
            :class:`~max.graph.BufferType` contained in this collection.
        """
        ...


@dataclass
class PagedCacheInputSymbolsByReplica(
    Sequence[IterableInputSymbols], FlattenableInputSymbols
):
    """A class that holds the symbolic inputs for the paged ache for all replicas.

    This is separate from `MultiKVCacheInputSymbols` for more convenient typing.
    """

    values: Sequence[IterableInputSymbols]

    def __iter__(self) -> Iterator[IterableInputSymbols]:
        return iter(self.values)

    def __getitem__(self, index: int | slice) -> Any:
        return self.values[index]

    def __len__(self) -> int:
        return len(self.values)

    def flatten(self) -> list[TensorType | BufferType]:
        """Returns all per-replica input symbols as a flat list.

        Returns:
            A flat list of every :class:`~max.graph.TensorType` and
            :class:`~max.graph.BufferType` across all replicas.
        """
        items = []
        for item in self.values:
            items.extend(item.flatten())
        return items


@dataclass
class MultiKVCacheInputSymbols(
    Sequence[PagedCacheInputSymbolsByReplica], FlattenableInputSymbols
):
    """Aggregates symbolic KV cache inputs for all KV caches across all replicas."""

    values: list[PagedCacheInputSymbolsByReplica]

    def __iter__(self) -> Iterator[PagedCacheInputSymbolsByReplica]:
        return iter(self.values)

    def __getitem__(self, index: int | slice) -> Any:
        return self.values[index]

    def __len__(self) -> int:
        return len(self.values)

    def flatten(self) -> list[TensorType | BufferType]:
        """Returns all input symbols across all KV caches and replicas as a flat list.

        Returns:
            A flat list of every :class:`~max.graph.TensorType` and
            :class:`~max.graph.BufferType` across all KV caches and replicas.
        """
        items = []
        for item in self.values:
            items.extend(item.flatten())
        return items


@dataclass
class KVCacheInputsPerDevice:
    """Holds the concrete KV cache buffer inputs for a single device."""

    blocks: Buffer
    cache_lengths: Buffer
    lookup_table: Buffer
    max_lengths: Buffer
    kv_scales: Buffer | None = None  # Scale tensor for FP8 quantization
    attention_dispatch_metadata: Buffer | None = None

    def __iter__(self) -> Iterator[Buffer]:
        yield from self.as_list()

    def as_list(self) -> list[Buffer]:
        """Returns the non-``None`` KV cache buffers in ABI order.

        Returns:
            A list of :class:`~max.driver.Buffer` objects containing
            ``blocks``, ``cache_lengths``, ``lookup_table``, ``max_lengths``,
            and optionally ``kv_scales`` and ``attention_dispatch_metadata``.
        """
        return [
            self.blocks,
            self.cache_lengths,
            self.lookup_table,
            self.max_lengths,
            *((self.kv_scales,) if self.kv_scales else ()),
            *(
                (self.attention_dispatch_metadata,)
                if self.attention_dispatch_metadata
                else ()
            ),
        ]


@dataclass
class KVCacheInputs:
    """``KVCacheInputs`` is a sequence of :class:`KVCacheInputsPerDevice`.

    The number of `KVCacheInputsPerDevice` in the sequence is equal to the
    number of devices used to run the model. For example, if the model is run
    with DP=2 + TP=4 then there will be 8 items in the list.
    """

    inputs: Sequence[KVCacheInputsPerDevice]

    def __iter__(self) -> Iterator[Buffer]:
        for input in self.inputs:
            yield from input
