# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

"""KVCacheInputs"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, TypeGuard, TypeVar, overload

from max.driver import Tensor

_T = TypeVar("_T")


def _is_sequence_of(x: Any, ty: type[_T]) -> TypeGuard[Sequence[_T]]:
    return isinstance(x, Sequence) and all(isinstance(item, ty) for item in x)


@dataclass
class KVCacheInputs:
    """
    A base class that holds KV cache related (Tensor) inputs.

    It is meant to be subclassed by concrete KV cache input types.
    For example, here's a derived class for a text KV cache manager:

    .. code-block:: python

        @dataclass
        class RaggedKVCacheInputs(KVCacheInputs):
            blocks: Tensor
            cache_lengths: Tensor
            lookup_table: Tensor
            max_lengths: Tensor
    """

    def __iter__(self) -> Iterator[Tensor]:
        """Iterates through each Type in order."""
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            if isinstance(value, KVCacheInputs):
                yield from value
            elif _is_sequence_of(value, KVCacheInputs):
                for item in value:
                    yield from item
            else:
                assert isinstance(value, Tensor)
                yield value

    @overload
    def __getitem__(self, index: int) -> Tensor: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[Tensor]: ...

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
    """
    ``RaggedKVCacheInputs`` is a class that holds the inputs for
    KV cache when used together with ragged tensors.
    """

    blocks: Tensor
    cache_lengths: Tensor
    lookup_table: Tensor
    max_lengths: Tensor


@dataclass
class KVCacheInputsSequence(KVCacheInputs):
    """
    ``KVCacheInputsSequence`` is a sequence of :obj:`KVCacheInputs`.

    It is primarily used in our multistep execution to represent batched
    KVCacheInputs.
    """

    kv_cache_inputs: Sequence[KVCacheInputs]
