# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any


@dataclass
class NestedIterableDataclass:
    """
    Base class for input symbols for KV cache managers.

    The derived class is responsible for defining the input symbols for the
    specific KV cache manager.
    For example, here's a derived class for a text KV cache manager:

    .. code-block:: python

        @dataclass
        class PagedCacheInputSymbols(NestedIterableDataclass):
            kv_blocks: TensorType
            cache_lengths: TensorType
            lookup_table: TensorType
            max_lengths: TensorType
    """

    def __iter__(self) -> Iterator[Any]:
        """Iterates through each field in order."""
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            if isinstance(value, NestedIterableDataclass):
                yield from value
            else:
                yield value

    def __getitem__(self, index: int | slice) -> Any:
        return list(self)[index]
