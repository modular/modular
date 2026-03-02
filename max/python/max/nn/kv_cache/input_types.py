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
    TypeGuard,
    overload,
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
        return list(self)


IterableInputSymbols: TypeAlias = NestedIterableDataclass[
    TensorType | BufferType
]


_DispatchMetadataT = TypeVar("_DispatchMetadataT", TensorType, TensorValue)


@dataclass
class MHADecodeDispatchMetadata(
    NestedIterableDataclass[_DispatchMetadataT],
    Generic[_DispatchMetadataT],
):
    tensor: _DispatchMetadataT

    def __post_init__(self) -> None:
        if self.tensor.dtype != DType.int64:
            raise ValueError(
                "expected mha_decode_dispatch_metadata dtype int64, got "
                f"{self.tensor.dtype}"
            )
        if self.tensor.rank != 1:
            raise ValueError(
                "expected mha_decode_dispatch_metadata rank 1, got "
                f"{self.tensor.rank}"
            )


@dataclass
class PagedCacheInputSymbols(IterableInputSymbols):
    kv_blocks: BufferType
    cache_lengths: TensorType
    lookup_table: TensorType
    max_lengths: TensorType
    kv_scales: BufferType | None = None  # KV scales for FP8 quantization
    dispatch_metadata: MHADecodeDispatchMetadata[TensorType] | None = None


@dataclass
class PagedCacheValues(NestedIterableDataclass[BufferValue | TensorValue]):
    kv_blocks: BufferValue
    cache_lengths: TensorValue
    lookup_table: TensorValue
    max_lengths: TensorValue
    kv_scales: BufferValue | None = None  # KV scales for FP8 quantization
    dispatch_metadata: MHADecodeDispatchMetadata[TensorValue] | None = None

    def __iter__(self) -> Iterator[BufferValue | TensorValue]:
        # Canonical paged KV ABI order.
        yield self.kv_blocks
        yield self.cache_lengths
        yield self.lookup_table
        yield self.max_lengths
        if self.kv_scales is not None:
            yield self.kv_scales


def unflatten_ragged_mha_decode_inputs(
    kv_inputs_flat: Sequence[Value[Any]], *, n_devices: int
) -> list[PagedCacheValues]:
    """Unmarshals flattened KV graph inputs into typed cache values.

    Args:
        kv_inputs_flat: Flattened graph values for all KV inputs.
        n_devices: Number of devices represented in ``kv_inputs_flat``.
    """
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

        metadata = MHADecodeDispatchMetadata(kv_inputs_flat[next_idx].tensor)

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


def mha_decode_dispatch_metadata(
    kv_collection: PagedCacheValues,
    *,
    device_idx: int | None = None,
) -> MHADecodeDispatchMetadata[TensorValue]:
    dispatch_metadata = kv_collection.dispatch_metadata
    if dispatch_metadata is not None:
        return dispatch_metadata

    location = "" if device_idx is None else f" for device {device_idx}"
    raise ValueError(
        "Expected MHADecodeDispatchMetadata in kv_collection.dispatch_metadata"
        f"{location}."
    )


def mha_decode_dispatch_metadata_list(
    kv_collections: Sequence[PagedCacheValues],
) -> list[MHADecodeDispatchMetadata[TensorValue]]:
    return [
        mha_decode_dispatch_metadata(kv_collection, device_idx=i)
        for i, kv_collection in enumerate(kv_collections)
    ]


@runtime_checkable
class FlattenableInputSymbols(Protocol):
    """A sequence-like collection of input symbols that can be flattened."""

    def __iter__(self) -> Iterator[Any]: ...
    def __getitem__(self, index: int | slice) -> Any: ...
    def __len__(self) -> int: ...
    def flatten(self) -> list[TensorType | BufferType]: ...


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
        items = []
        for item in self.values:
            items.extend(item.flatten())
        return items


@dataclass
class MultiKVCacheInputSymbols(
    Sequence[PagedCacheInputSymbolsByReplica], FlattenableInputSymbols
):
    values: list[PagedCacheInputSymbolsByReplica]

    def __iter__(self) -> Iterator[PagedCacheInputSymbolsByReplica]:
        return iter(self.values)

    def __getitem__(self, index: int | slice) -> Any:
        return self.values[index]

    def __len__(self) -> int:
        return len(self.values)

    def flatten(self) -> list[TensorType | BufferType]:
        items = []
        for item in self.values:
            items.extend(item.flatten())
        return items


_T = TypeVar("_T")


def _is_sequence_of(x: Any, ty: type[_T]) -> TypeGuard[Sequence[_T]]:
    return isinstance(x, Sequence) and all(isinstance(item, ty) for item in x)


@dataclass
class KVCacheInputs:
    """A base class that holds KV cache related (Tensor) inputs.

    It is meant to be subclassed by concrete KV cache input types.
    For example, here's a derived class for a text KV cache manager:

    .. code-block:: python

        @dataclass
        class RaggedKVCacheInputs(KVCacheInputs):
            blocks: Buffer
            cache_lengths: Buffer
            lookup_table: Buffer
            max_lengths: Buffer
    """

    def __iter__(self) -> Iterator[Buffer]:
        """Iterates through each Type in order."""
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            if value is None:
                continue
            if isinstance(value, KVCacheInputs):
                yield from value
            elif _is_sequence_of(value, KVCacheInputs):
                for item in value:
                    yield from item
            else:
                assert isinstance(value, Buffer)
                yield value

    @overload
    def __getitem__(self, index: int) -> Buffer: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[Buffer]: ...

    def __getitem__(self, index: Any) -> Any:
        return list(self)[index]

    def __len__(self) -> int:
        count = 0
        # Iterate over all fields in the dataclass. If we run into a sequence of
        # KVCacheInputs, we expand and recursively call `len` on the KVCacheInputs
        # elements.
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            if _is_sequence_of(value, KVCacheInputs):
                count += sum(len(x) for x in value)
            else:
                count += 1
        return count


@dataclass
class RaggedKVCacheInputs(KVCacheInputs):
    """``RaggedKVCacheInputs`` is a class that holds the inputs for
    KV cache when used together with ragged tensors.
    """

    blocks: Buffer
    cache_lengths: Buffer
    lookup_table: Buffer
    max_lengths: Buffer
    kv_scales: Buffer | None = None  # Scale tensor for FP8 quantization
    mha_decode_dispatch_metadata: Buffer | None = None


@dataclass
class KVCacheInputsSequence(KVCacheInputs):
    """``KVCacheInputsSequence`` is a sequence of :obj:`KVCacheInputs`.

    It is primarily used in our multistep execution to represent batched
    KVCacheInputs.
    """

    kv_cache_inputs: Sequence[KVCacheInputs]
