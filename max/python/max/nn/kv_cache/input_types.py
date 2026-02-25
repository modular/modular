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
from max.graph import BufferType, BufferValue, TensorType, TensorValue
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


@runtime_checkable
class InputSymbolInterface(Protocol):
    def flatten(self) -> list[TensorType | BufferType]: ...


IterableInputSymbols: TypeAlias = NestedIterableDataclass[
    TensorType | BufferType
]


@dataclass
class PagedCacheInputSymbols(IterableInputSymbols, InputSymbolInterface):
    kv_blocks: BufferType
    cache_lengths: TensorType
    lookup_table: TensorType
    max_lengths: TensorType
    kv_scales: BufferType | None = None  # KV scales for FP8 quantization


@dataclass
class PagedCacheValues(NestedIterableDataclass[BufferValue | TensorValue]):
    kv_blocks: BufferValue
    cache_lengths: TensorValue
    lookup_table: TensorValue
    max_lengths: TensorValue
    kv_scales: BufferValue | None = None  # KV scales for FP8 quantization


@dataclass
class PagedCacheInputSymbolsByReplica(
    Sequence[PagedCacheInputSymbols], InputSymbolInterface
):
    """A class that holds the symbolic inputs for the paged ache for all replicas.

    This is separate from `MultiKVCacheInputSymbols` for more convenient typing.
    """

    values: list[PagedCacheInputSymbols]

    def __iter__(self) -> Iterator[PagedCacheInputSymbols]:
        return iter(self.values)

    @overload
    def __getitem__(self, index: int) -> PagedCacheInputSymbols: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[PagedCacheInputSymbols]: ...

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
    Sequence[InputSymbolInterface], InputSymbolInterface
):
    values: list[InputSymbolInterface]

    def __iter__(self) -> Iterator[InputSymbolInterface]:
        return iter(self.values)

    @overload
    def __getitem__(self, index: int) -> InputSymbolInterface: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[InputSymbolInterface]: ...

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


@dataclass
class KVCacheInputsSequence(KVCacheInputs):
    """``KVCacheInputsSequence`` is a sequence of :obj:`KVCacheInputs`.

    It is primarily used in our multistep execution to represent batched
    KVCacheInputs.
    """

    kv_cache_inputs: Sequence[KVCacheInputs]
