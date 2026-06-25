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

import itertools
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from max.driver import Buffer
from max.dtype import DType
from max.experimental.tensor import Tensor
from max.graph import BufferType, BufferValue, TensorType, TensorValue

_Tensor = TypeVar("_Tensor", TensorValue, TensorType, Buffer, Tensor)
_Buffer = TypeVar("_Buffer", BufferValue, BufferType, Buffer, Tensor)


def _verify_rank1_int64_tensor(name: str, t: _Tensor | None) -> None:
    if t is None:
        return
    if t.dtype != DType.int64:
        raise ValueError(
            f"Expected dtype int64, got {t.dtype} for tensor {name}"
        )
    if t.rank != 1:
        raise ValueError(f"Expected rank 1, got {t.rank} for tensor {t}")


@dataclass
class KVCacheInputsPerDevice(Generic[_Tensor, _Buffer]):
    """Symbolic graph input types for a single device's paged KV cache."""

    kv_blocks: _Buffer
    cache_lengths: _Tensor
    lookup_table: _Tensor
    max_prompt_length: _Tensor
    max_cache_length: _Tensor
    kv_scales: _Buffer | None = None  # KV scales for FP8 quantization
    attention_dispatch_metadata: _Tensor | None = None
    draft_attention_dispatch_metadata: _Tensor | None = None
    # Capturable-graph scalars: when present, the SM100 MLA dispatcher uses
    # these to align grid-time partition decisions with the kernel's divmod.
    # Populated only for MLA paths; ``None`` otherwise.
    mla_num_partitions: _Tensor | None = None
    draft_mla_num_partitions: _Tensor | None = None

    def __post_init__(self) -> None:
        _verify_rank1_int64_tensor(
            "attention_dispatch_metadata", self.attention_dispatch_metadata
        )
        _verify_rank1_int64_tensor(
            "draft_attention_dispatch_metadata",
            self.draft_attention_dispatch_metadata,
        )
        _verify_rank1_int64_tensor(
            "mla_num_partitions", self.mla_num_partitions
        )
        _verify_rank1_int64_tensor(
            "draft_mla_num_partitions", self.draft_mla_num_partitions
        )

    def flatten(self) -> list[_Tensor | _Buffer]:
        """Serialize fields into a flat list for graph input binding."""
        return [
            self.kv_blocks,
            self.cache_lengths,
            self.lookup_table,
            self.max_prompt_length,
            self.max_cache_length,
            *((self.kv_scales,) if self.kv_scales else ()),
            *(
                (self.attention_dispatch_metadata,)
                if self.attention_dispatch_metadata
                else ()
            ),
            *(
                (self.draft_attention_dispatch_metadata,)
                if self.draft_attention_dispatch_metadata
                else ()
            ),
            *((self.mla_num_partitions,) if self.mla_num_partitions else ()),
            *(
                (self.draft_mla_num_partitions,)
                if self.draft_mla_num_partitions
                else ()
            ),
        ]

    # TODO: FIX THIS HACK!!!
    def flatten_without_attention_dispatch_metadata(
        self,
    ) -> list[_Tensor | _Buffer]:
        return [
            self.kv_blocks,
            self.cache_lengths,
            self.lookup_table,
            self.max_prompt_length,
            self.max_cache_length,
            *((self.kv_scales,) if self.kv_scales else ()),
        ]

    def unflatten(
        self, it: Iterator[Any]
    ) -> KVCacheInputsPerDevice[TensorValue, BufferValue]:
        """Reconstruct from a flat iterator produced by ``flatten``.

        Consumes ``next(it)`` in the same order ``flatten`` emits elements;
        the two methods must stay in lock-step.
        """
        return KVCacheInputsPerDevice(
            kv_blocks=next(it),
            cache_lengths=next(it),
            lookup_table=next(it),
            max_prompt_length=next(it),
            max_cache_length=next(it),
            kv_scales=next(it) if self.kv_scales else None,
            attention_dispatch_metadata=next(it)
            if self.attention_dispatch_metadata
            else None,
            draft_attention_dispatch_metadata=next(it)
            if self.draft_attention_dispatch_metadata
            else None,
            mla_num_partitions=next(it) if self.mla_num_partitions else None,
            draft_mla_num_partitions=next(it)
            if self.draft_mla_num_partitions
            else None,
        )


PagedCacheValues = KVCacheInputsPerDevice[TensorValue, BufferValue]


class KVCacheInputsInterface(ABC, Generic[_Tensor, _Buffer]):
    """Common interface for KV cache graph inputs (leaf or tree)."""

    @abstractmethod
    def flatten(self) -> list[_Tensor | _Buffer]:
        """Flattens this (sub)tree into a flattened buffer/tensor list."""
        ...

    @abstractmethod
    def unflatten(
        self, it: Iterator[Any]
    ) -> KVCacheInputsInterface[TensorValue, BufferValue]:
        """Rebuilds this (sub)tree by consuming values from ``it``."""
        ...


@dataclass
class MultiKVCacheInputs(KVCacheInputsInterface[_Tensor, _Buffer]):
    """Symbolic graph input types for a tree of KV caches.

    This class is used to represent a tree of KV caches. For example, hybrid models
    like Gemma4 may have "sliding_window" and "full_attention" caches. Furthermore,
    we can also have "target" and "draft" caches for speculative decoding.
    """

    children: dict[str, KVCacheInputsInterface[_Tensor, _Buffer]]

    def flatten(self) -> list[_Tensor | _Buffer]:
        return list(
            itertools.chain.from_iterable(
                item.flatten() for item in self.children.values()
            )
        )

    def unflatten(
        self, it: Iterator[Any]
    ) -> MultiKVCacheInputs[TensorValue, BufferValue]:
        return MultiKVCacheInputs(
            children={
                key: item.unflatten(it) for key, item in self.children.items()
            }
        )


@dataclass
class KVCacheInputs(
    Generic[_Tensor, _Buffer], KVCacheInputsInterface[_Tensor, _Buffer]
):
    """Symbolic graph input types for a leaf KV cache.

    This contains the KV cache inputs for all TP shards."""

    inputs: Sequence[KVCacheInputsPerDevice[_Tensor, _Buffer]]

    def flatten(self) -> list[_Tensor | _Buffer]:
        return list(
            itertools.chain.from_iterable(
                item.flatten() for item in self.inputs
            )
        )

    def unflatten(
        self, it: Iterator[Any]
    ) -> KVCacheInputs[TensorValue, BufferValue]:
        return KVCacheInputs(
            inputs=[item.unflatten(it) for item in self.inputs]
        )
