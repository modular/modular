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
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from max.driver import Buffer
from max.dtype import DType
from max.experimental.tensor import Tensor
from max.graph import BufferType, BufferValue, TensorType, TensorValue

_Tensor = TypeVar("_Tensor", TensorValue, TensorType, Buffer, Tensor)
_Buffer = TypeVar("_Buffer", BufferValue, BufferType, Buffer, Tensor)


@dataclass
class KVCacheInputsPerDevice(Generic[_Tensor, _Buffer]):
    """Symbolic graph input types for a single device's paged KV cache."""

    kv_blocks: _Buffer
    cache_lengths: _Tensor
    lookup_table: _Tensor
    max_lengths: _Tensor
    kv_scales: _Buffer | None = None  # KV scales for FP8 quantization
    attention_dispatch_metadata: _Tensor | None = None
    draft_attention_dispatch_metadata: _Tensor | None = None
    # Capturable-graph scalars: when present, the SM100 MLA dispatcher uses
    # these to align grid-time partition decisions with the kernel's divmod
    # on scalar_args[2]. Only populated for MLA paths; None otherwise.
    mla_num_partitions: _Tensor | None = None
    mla_effective_split_len: _Tensor | None = None
    draft_mla_num_partitions: _Tensor | None = None
    draft_mla_effective_split_len: _Tensor | None = None

    def __post_init__(self) -> None:
        tensor = self.attention_dispatch_metadata
        if tensor is not None:
            if tensor.dtype != DType.int64:
                raise ValueError(
                    "expected attention_dispatch_metadata dtype int64, got "
                    f"{tensor.dtype}"
                )
            if tensor.rank != 1:
                raise ValueError(
                    "expected attention_dispatch_metadata rank 1, got "
                    f"{tensor.rank}"
                )
        for name in (
            "mla_num_partitions",
            "mla_effective_split_len",
            "draft_mla_num_partitions",
            "draft_mla_effective_split_len",
        ):
            t = getattr(self, name)
            if t is None:
                continue
            if t.dtype != DType.int64:
                raise ValueError(f"expected {name} dtype int64, got {t.dtype}")
            if t.rank != 1:
                raise ValueError(f"expected {name} rank 1, got {t.rank}")

    def flatten(self) -> list[_Tensor | _Buffer]:
        return [
            self.kv_blocks,
            self.cache_lengths,
            self.lookup_table,
            self.max_lengths,
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
                (self.mla_effective_split_len,)
                if self.mla_effective_split_len
                else ()
            ),
            *(
                (self.draft_mla_num_partitions,)
                if self.draft_mla_num_partitions
                else ()
            ),
            *(
                (self.draft_mla_effective_split_len,)
                if self.draft_mla_effective_split_len
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
            self.max_lengths,
            *((self.kv_scales,) if self.kv_scales else ()),
        ]

    def unflatten(
        self, it: Iterator[Any]
    ) -> KVCacheInputsPerDevice[TensorValue, BufferValue]:
        return KVCacheInputsPerDevice(
            kv_blocks=next(it),
            cache_lengths=next(it),
            lookup_table=next(it),
            max_lengths=next(it),
            kv_scales=next(it) if self.kv_scales else None,
            attention_dispatch_metadata=next(it)
            if self.attention_dispatch_metadata
            else None,
            draft_attention_dispatch_metadata=next(it)
            if self.draft_attention_dispatch_metadata
            else None,
            mla_num_partitions=next(it) if self.mla_num_partitions else None,
            mla_effective_split_len=next(it)
            if self.mla_effective_split_len
            else None,
            draft_mla_num_partitions=next(it)
            if self.draft_mla_num_partitions
            else None,
            draft_mla_effective_split_len=next(it)
            if self.draft_mla_effective_split_len
            else None,
        )


PagedCacheValues = KVCacheInputsPerDevice[TensorValue, BufferValue]


@dataclass
class KVCacheInputs(Generic[_Tensor, _Buffer]):
    """Symbolic graph input types for all devices' paged KV cache."""

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
