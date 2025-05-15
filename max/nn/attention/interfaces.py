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
"""General interface for Attention."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from max.graph import (
    BufferValue,
    TensorValue,
    TensorValueLike,
)

from ..kv_cache import (
    ContinuousBatchingKVCacheCollection,
    KVCacheParams,
    PagedKVCacheCollection,
)
from ..layer import Layer, Module
from ..linear import LinearV1


@dataclass
class AttentionImpl(Layer, ABC):
    """
    A generalized attention interface, that will be used upstream by a general Transformer.
    We would expect a seperate subclass, articulating each variation of Attention:

    - AttentionWithRope
    - AttentionWithAlibi
    - VanillaAttentionWithCausalMask
    - ...

    There are a series of shared attributes, however, more may be needed for each individual variant.
    For example, we may introduce an OptimizedRotaryEmbedding class for the AttentionWithRope class:

    .. code-block:: python

        @dataclass
        class AttentionWithRope(AttentionImpl):
            rope: OptimizedRotaryEmbedding
            ...

    We expect the ``__call__`` abstractmethod to remain relatively consistent, however the ``**kwargs``
    argument is exposed, allowing you to leverage additional arguments for each particular variant.
    For example, we may introduce an VanillaAttentionWithCausalMask class, which includes an attention
    mask:

    .. code-block:: python

        @dataclass
        class VanillaAttentionWithCausalMask(AttentionImpl):
            ...

            def __call__(
                self,
                x: TensorValueLike,
                kv_collection: ContinuousBatchingKVCacheCollection,
                valid_lengths: TensorValueLike,
                **kwargs,
            ) -> tuple[TensorValue, ContinuousBatchingKVCacheCollection]: ...

                if "attn_mask" not in kwargs:
                    raise ValueError("attn_mask not provided to VanillaAttentionWithCausalMask")

                # Which we can then use the attention mask downstream like so:
                op(
                    attn_mask = kwargs["attn_mask"]
                )
    """

    n_heads: int
    """The number of attention heads."""

    kv_params: KVCacheParams
    """KV Cache Params, including the number of kv heads, the head dim, and data type."""

    wqkv: TensorValue
    """The concatenation of q, k, and v weight vectors."""

    wo: LinearV1
    """A linear layer for the output projection."""

    scale: float
    """The scale factor for the attention."""

    def __post_init__(self) -> None:
        assert self.scale is not None, "scale must be provided to AttentionImpl"

        if not self.kv_params.cache_strategy.uses_opaque():
            raise ValueError(
                f"{self.kv_params.cache_strategy} cache strategy, not supported"
                " in Attention layer."
            )

    @abstractmethod
    def __call__(
        self,
        layer_idx: TensorValue,
        x: TensorValue,
        kv_collection: ContinuousBatchingKVCacheCollection
        | PagedKVCacheCollection,
        **kwargs,
    ) -> TensorValue: ...


class DistributedAttentionImpl(Module, ABC):
    """
    A generalized Distributed attention interface.
    """

    @abstractmethod
    def __call__(
        self,
        layer_idx: TensorValue,
        x: list[TensorValue],
        signal_buffers: list[BufferValue],
        kv_collections: list[
            ContinuousBatchingKVCacheCollection | PagedKVCacheCollection
        ],
        **kwargs,
    ) -> list[TensorValue]: ...


@dataclass
class AttentionImplQKV(Layer, ABC):
    """
    A generalized attention interface, that will be used upstream by a general Transformer.
    We would expect a seperate subclass, articulating each variation of Attention:

    - AttentionWithRope
    - AttentionWithAlibi
    - VanillaAttentionWithCausalMask
    - ...

    There are a series of shared attributes, however, more may be needed for each individual variant.
    For example, we may introduce an OptimizedRotaryEmbedding class for the AttentionWithRope class:

    .. code-block:: python

        @dataclass
        class AttentionWithRope(AttentionImpl):
            rope: OptimizedRotaryEmbedding
            ...

    We expect the ``__call__`` abstractmethod to remain relatively consistent, however the ``**kwargs``
    argument is exposed, allowing you to leverage additional arguments for each particular variant.
    For example, we may introduce an VanillaAttentionWithCausalMask class, which includes an attention
    mask:

    .. code-block:: python

        @dataclass
        class VanillaAttentionWithCausalMask(AttentionImpl):
            ...

            def __call__(
                self,
                x: TensorValueLike,
                kv_collection: ContinuousBatchingKVCacheCollection,
                valid_lengths: TensorValueLike,
                **kwargs,
            ) -> tuple[TensorValue, ContinuousBatchingKVCacheCollection]: ...

                if "attn_mask" not in kwargs:
                    raise ValueError("attn_mask not provided to VanillaAttentionWithCausalMask")

                # Which we can then use the attention mask downstream like so:
                op(
                    attn_mask = kwargs["attn_mask"]
                )
    """

    n_heads: int
    """The number of attention heads."""

    kv_params: KVCacheParams
    """KV Cache Params, including the number of kv heads, the head dim, and data type."""

    wq: TensorValueLike
    """The q weight vector."""

    wk: TensorValueLike
    """The k weight vector."""

    wv: TensorValueLike
    """The v weight vector."""

    wo: LinearV1
    """A linear layer for the output projection."""

    scale: float
    """The scale factor for the attention."""

    def __post_init__(self) -> None:
        assert self.scale is not None, (
            "scale must be provided to AttentionImplQKV"
        )

        if not self.kv_params.cache_strategy.uses_opaque():
            raise ValueError(
                f"{self.kv_params.cache_strategy} cache strategy, not supported"
                " in Attention layer."
            )

    @abstractmethod
    def __call__(
        self,
        layer_idx: TensorValue,
        x: TensorValue,
        kv_collection: ContinuousBatchingKVCacheCollection
        | PagedKVCacheCollection,
        **kwargs,
    ) -> TensorValue: ...
