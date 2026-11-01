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
"""The ``kv_cache`` component.

The KV cache a pipeline's graphs were exported for.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import ClassVar, TypeVar

from max.dtype import DType
from typing_extensions import Self

from .value_types import (
    ArtifactError,
    ArtifactType,
    decode_field,
    decode_object,
    decode_optional,
    expect,
    unknown_fields,
)

_T = TypeVar("_T")

POOL_GROUPS = ("full", "sliding_window", "recurrent", "scratch")
"""How the pool reuses and evicts a leaf's pages, as MAX's
``KVCacheGroupId.type`` names it."""


@dataclass(frozen=True)
class KVPoolLeaf:
    """One region of the cache pool a cache occupies.

    It mirrors MAX's ``KVLeafRegion``.

    Args:
        leaf_id: The id the pool keys the leaf's pages by.
        group: How the pool reuses and evicts the leaf's pages, one of
            :data:`POOL_GROUPS`.
        window_size: The sliding window in tokens of a ``sliding_window``
            leaf, else ``None``.
        bytes_per_page: The bytes one page of the leaf holds on one device.
        stride_alignment: What a padded page stride must be a multiple of, or
            ``None`` when the leaf's readers need packed pages.
    """

    leaf_id: str
    group: str
    window_size: int | None
    bytes_per_page: int
    stride_alignment: int | None


@dataclass(frozen=True)
class KVCacheQuantization:
    """How a cache's values are quantized.

    It mirrors MAX's ``KVCacheQuantizationConfig``.

    Args:
        scale_dtype: The dtype of the scales.
        quantization_granularity: The elements one scale covers along the
            head dimension.
        fp8_tail_dim: The trailing elements of each head row that a packed
            NVFP4 cache stores as raw FP8 bytes.
    """

    scale_dtype: DType
    quantization_granularity: int
    fp8_tail_dim: int


@dataclass(frozen=True)
class MHAKVCache:
    """A multi-head attention cache the graphs read.

    It mirrors MAX's ``MHAKVCacheParams``.

    Args:
        name: The cache's path in the cache tree, its child keys joined by
            ``.``, or empty for a cache that isn't in a tree.
        dtype: The dtype the cache stores its values in.
        num_layers: The layers the cache holds.
        n_kv_heads: The key-value heads, across every device.
        head_dim: The width of one key head.
        v_head_dim: The width of one value head when it differs from
            ``head_dim``, else ``None``.
        window_size: The sliding window in tokens, or ``None`` for full
            attention.
        allow_kv_head_replication: Whether a head is replicated across
            devices when there are more devices than heads.
        slots_per_page: The storage slots one page holds, fewer than the page
            size when the attention compresses tokens.
        stride_alignment: What a padded page stride must be a multiple of, or
            ``None`` when the cache's readers need packed pages.
        scratch: Whether the cache is one block per request that is never
            published.
        kvcache_quant_config: How the values are quantized, or ``None``.
        leaves: The pool leaves the cache occupies.
    """

    kind: ClassVar[str] = "mha"
    """The ``kind`` the cache is serialized with."""

    name: str
    dtype: DType
    num_layers: int
    n_kv_heads: int
    head_dim: int
    v_head_dim: int | None
    window_size: int | None
    allow_kv_head_replication: bool
    slots_per_page: int
    stride_alignment: int | None
    scratch: bool
    kvcache_quant_config: KVCacheQuantization | None
    leaves: tuple[KVPoolLeaf, ...]


@dataclass(frozen=True)
class MSAKVCache(MHAKVCache):
    """A multi-step attention cache.

    It mirrors MAX's ``MSAKVCacheParams``: a multi-head attention cache read
    through its own attention key.
    """

    kind: ClassVar[str] = "msa"


@dataclass(frozen=True)
class MLAKVCache:
    """A multi-head latent attention cache.

    It mirrors MAX's ``MLAKVCacheParams``.

    Args:
        name: The cache's path in the cache tree.
        dtype: The dtype the cache stores its values in.
        num_layers: The layers the cache holds.
        num_q_heads: The query heads, which the decode kernel's dispatch
            needs.
        head_dim: The width of one latent row.
        v_head_dim: The width of one value head when it differs from
            ``head_dim``, else ``None``.
        window_size: The sliding window in tokens, or ``None``.
        slots_per_page: The storage slots one page holds.
        sparse_indices_stride: The cache rows sparse MLA selects per token, or
            ``None`` for dense MLA.
        stride_alignment: What a padded page stride must be a multiple of, or
            ``None``.
        scratch: Whether the cache is one block per request that is never
            published.
        kvcache_quant_config: How the values are quantized, or ``None``.
        leaves: The pool leaves the cache occupies.
    """

    kind: ClassVar[str] = "mla"
    """The ``kind`` the cache is serialized with."""

    name: str
    dtype: DType
    num_layers: int
    num_q_heads: int
    head_dim: int
    v_head_dim: int | None
    window_size: int | None
    slots_per_page: int
    sparse_indices_stride: int | None
    stride_alignment: int | None
    scratch: bool
    kvcache_quant_config: KVCacheQuantization | None
    leaves: tuple[KVPoolLeaf, ...]


@dataclass(frozen=True)
class StateRegion:
    """One kind of recurrent state.

    It mirrors MAX's ``RecurrentStateRegion``.

    Args:
        leaf_id: The pool leaf the state occupies.
        num_layers: The layers the state holds.
        row_shape: The shape of one layer's state on one device.
        dtype: The dtype of the state.
        scratch: Whether the state is per-request scratch rather than a
            checkpoint.
    """

    leaf_id: str
    num_layers: int
    row_shape: tuple[int, ...]
    dtype: DType
    scratch: bool


@dataclass(frozen=True)
class RecurrentStateCache:
    """A cache of recurrent states.

    It mirrors MAX's ``RecurrentStateParams``.

    Args:
        name: The cache's path in the cache tree.
        regions: The states one request occupies, in flatten order.
        leaves: The pool leaves the cache occupies.
    """

    kind: ClassVar[str] = "recurrent"
    """The ``kind`` the cache is serialized with."""

    name: str
    regions: tuple[StateRegion, ...]
    leaves: tuple[KVPoolLeaf, ...]


KVCache = MHAKVCache | MSAKVCache | MLAKVCache | RecurrentStateCache
"""A cache of any kind the ``kv_cache`` component records."""


@dataclass(frozen=True)
class KVCacheLayout:
    """The KV cache a pipeline artifact's graphs were exported for.

    An engine serving the artifact allocates it.

    ``max export`` copies each value from the pipeline's KV cache parameters
    rather than serializing them, so the artifact's layout doesn't change when
    theirs does. Each cache records its ``kind``, every parameter its graphs
    were built with, and the pool leaves MAX computes for it.

    Args:
        page_size: The tokens one page holds, which every cache shares.
        caches: The caches, in the order the graphs take them as inputs.
    """

    page_size: int
    caches: tuple[KVCache, ...]

    def __serialize__(self) -> Mapping[str, ArtifactType]:
        """Returns the layout as a JSON object."""
        return {
            "page_size": self.page_size,
            "caches": [_serialize_cache(cache) for cache in self.caches],
        }

    @classmethod
    def __deserialize__(cls, value: Mapping[str, ArtifactType]) -> Self:
        """Rebuilds the layout written by :meth:`__serialize__`.

        Raises:
            ArtifactError: If a field is missing, unknown or malformed, or a
                cache is of a kind this layout doesn't record.
        """
        if unknown := unknown_fields(value, cls):
            raise ArtifactError(f"unknown fields {unknown}")
        return cls(
            page_size=decode_field(value, "page_size", _int),
            caches=decode_field(value, "caches", _tuple(_cache)),
        )


def _serialize_cache(cache: KVCache) -> dict[str, ArtifactType]:
    common: dict[str, ArtifactType] = {
        "name": cache.name,
        "kind": cache.kind,
        "leaves": [_serialize_leaf(leaf) for leaf in cache.leaves],
    }
    if isinstance(cache, RecurrentStateCache):
        return common | {
            "regions": [
                {
                    "leaf_id": region.leaf_id,
                    "num_layers": region.num_layers,
                    "row_shape": list(region.row_shape),
                    "dtype": region.dtype.name,
                    "scratch": region.scratch,
                }
                for region in cache.regions
            ]
        }
    attention: dict[str, ArtifactType] = {
        "dtype": cache.dtype.name,
        "num_layers": cache.num_layers,
        "head_dim": cache.head_dim,
        "v_head_dim": cache.v_head_dim,
        "window_size": cache.window_size,
        "slots_per_page": cache.slots_per_page,
        "stride_alignment": cache.stride_alignment,
        "scratch": cache.scratch,
        "kvcache_quant_config": _serialize_quantization(
            cache.kvcache_quant_config
        ),
    }
    if isinstance(cache, MLAKVCache):
        return (
            common
            | attention
            | {
                "num_q_heads": cache.num_q_heads,
                "sparse_indices_stride": cache.sparse_indices_stride,
            }
        )
    return (
        common
        | attention
        | {
            "n_kv_heads": cache.n_kv_heads,
            "allow_kv_head_replication": cache.allow_kv_head_replication,
        }
    )


def _serialize_leaf(leaf: KVPoolLeaf) -> dict[str, ArtifactType]:
    return {
        "leaf_id": leaf.leaf_id,
        "group": leaf.group,
        "window_size": leaf.window_size,
        "bytes_per_page": leaf.bytes_per_page,
        "stride_alignment": leaf.stride_alignment,
    }


def _serialize_quantization(
    quantization: KVCacheQuantization | None,
) -> ArtifactType:
    if quantization is None:
        return None
    return {
        "scale_dtype": quantization.scale_dtype.name,
        "quantization_granularity": quantization.quantization_granularity,
        "fp8_tail_dim": quantization.fp8_tail_dim,
    }


def _int(value: ArtifactType) -> int:
    return expect(value, int)


def _bool(value: ArtifactType) -> bool:
    return expect(value, bool)


def _str(value: ArtifactType) -> str:
    return expect(value, str)


def _dtype(value: ArtifactType) -> DType:
    name = expect(value, str)
    try:
        return DType[name]
    except KeyError:
        raise ValueError(f"unknown dtype {name!r}") from None


def _tuple(
    decoder: Callable[[ArtifactType], _T],
) -> Callable[[ArtifactType], tuple[_T, ...]]:
    return lambda value: tuple(decoder(item) for item in expect(value, list))


def _cache(value: ArtifactType) -> KVCache:
    obj = dict(expect(value, dict))
    kind = expect(obj.pop("kind"), str)
    if kind in (MHAKVCache.kind, MSAKVCache.kind):
        return _mha(obj, MSAKVCache if kind == MSAKVCache.kind else MHAKVCache)
    if kind == MLAKVCache.kind:
        return _mla(obj)
    if kind == RecurrentStateCache.kind:
        obj = decode_object(obj, RecurrentStateCache)
        return RecurrentStateCache(
            name=_str(obj["name"]),
            regions=_tuple(_region)(obj["regions"]),
            leaves=_tuple(_leaf)(obj["leaves"]),
        )
    raise ValueError(f"unknown cache kind {kind!r}")


def _mha(value: dict[str, ArtifactType], cls: type[MHAKVCache]) -> MHAKVCache:
    obj = decode_object(value, cls)
    return cls(
        name=_str(obj["name"]),
        dtype=_dtype(obj["dtype"]),
        num_layers=_int(obj["num_layers"]),
        n_kv_heads=_int(obj["n_kv_heads"]),
        head_dim=_int(obj["head_dim"]),
        v_head_dim=decode_optional(_int)(obj["v_head_dim"]),
        window_size=decode_optional(_int)(obj["window_size"]),
        allow_kv_head_replication=_bool(obj["allow_kv_head_replication"]),
        slots_per_page=_int(obj["slots_per_page"]),
        stride_alignment=decode_optional(_int)(obj["stride_alignment"]),
        scratch=_bool(obj["scratch"]),
        kvcache_quant_config=decode_optional(_quantization)(
            obj["kvcache_quant_config"]
        ),
        leaves=_tuple(_leaf)(obj["leaves"]),
    )


def _mla(value: dict[str, ArtifactType]) -> MLAKVCache:
    obj = decode_object(value, MLAKVCache)
    return MLAKVCache(
        name=_str(obj["name"]),
        dtype=_dtype(obj["dtype"]),
        num_layers=_int(obj["num_layers"]),
        num_q_heads=_int(obj["num_q_heads"]),
        head_dim=_int(obj["head_dim"]),
        v_head_dim=decode_optional(_int)(obj["v_head_dim"]),
        window_size=decode_optional(_int)(obj["window_size"]),
        slots_per_page=_int(obj["slots_per_page"]),
        sparse_indices_stride=decode_optional(_int)(
            obj["sparse_indices_stride"]
        ),
        stride_alignment=decode_optional(_int)(obj["stride_alignment"]),
        scratch=_bool(obj["scratch"]),
        kvcache_quant_config=decode_optional(_quantization)(
            obj["kvcache_quant_config"]
        ),
        leaves=_tuple(_leaf)(obj["leaves"]),
    )


def _region(value: ArtifactType) -> StateRegion:
    obj = decode_object(value, StateRegion)
    return StateRegion(
        leaf_id=_str(obj["leaf_id"]),
        num_layers=_int(obj["num_layers"]),
        row_shape=_tuple(_int)(obj["row_shape"]),
        dtype=_dtype(obj["dtype"]),
        scratch=_bool(obj["scratch"]),
    )


def _leaf(value: ArtifactType) -> KVPoolLeaf:
    obj = decode_object(value, KVPoolLeaf)
    group = _str(obj["group"])
    if group not in POOL_GROUPS:
        raise ValueError(f"unknown pool group {group!r}")
    return KVPoolLeaf(
        leaf_id=_str(obj["leaf_id"]),
        group=group,
        window_size=decode_optional(_int)(obj["window_size"]),
        bytes_per_page=_int(obj["bytes_per_page"]),
        stride_alignment=decode_optional(_int)(obj["stride_alignment"]),
    )


def _quantization(value: ArtifactType) -> KVCacheQuantization:
    obj = decode_object(value, KVCacheQuantization)
    return KVCacheQuantization(
        scale_dtype=_dtype(obj["scale_dtype"]),
        quantization_granularity=_int(obj["quantization_granularity"]),
        fp8_tail_dim=_int(obj["fp8_tail_dim"]),
    )
