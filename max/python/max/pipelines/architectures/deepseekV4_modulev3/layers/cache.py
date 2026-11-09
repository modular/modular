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
"""DeepSeek-V4's KV state as paged cache leaves, on ModuleV3.

The layout and why it is exact are in the ModuleV2 implementation
(``deepseekV4/layers/cache.py`` at dfc8109fc0c): a window leaf, per compress
ratio a zone leaf (``slots_per_page = page_size // ratio``) and an open-state
leaf, and the indexer's pair for ratio 4. Every
leaf is a ModuleV3 :class:`PagedCacheValues`; the slot geometry the kernels
infer from the buffer rides alongside in :class:`CacheLeaf`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from max import functional as F
from max import tree
from max.driver import CPU
from max.dtype import DType
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.kernels import store_k_cache_ragged, store_v_cache_ragged
from max.nn.kv_cache import KVCacheInputsPerDevice, KVCacheParamInterface
from max.tensor import Tensor
from max.tensor.sharding import DeviceMapping

from ..model_config import DeepseekV4Config

__all__ = ["KEY", "VALUE", "CacheLeaf", "DeepseekV4Cache", "LayerIndex"]

KEY = 0
VALUE = 1


@tree.dataclass
class LayerIndex:
    """Where one block's rows live in the leaves, as host uint32 scalars.

    Operands rather than constants baked into the ops, so a subgraph body
    shared by a group of layers (chunk 6) reads each caller's own leaves.
    """

    swa: Tensor
    """The window leaf's layer."""
    zone: Tensor | None = None
    """The compressed and state leaves' layer; ``None`` on a window-only
    layer."""

    @classmethod
    def constant(cls, swa: int, zone: int | None) -> LayerIndex:
        """Builds the index from fixed layer numbers, as host constants."""

        def host(layer: int) -> Tensor:
            return F.constant(layer, DType.uint32, device=CPU())

        if zone is None:
            return cls(swa=host(swa))
        return cls(swa=host(swa), zone=host(zone))


@dataclass
class CacheLeaf:
    """One leaf's cache values plus the slot geometry the kernels infer."""

    values: PagedCacheValues
    slots_per_page: int

    def __tree_flatten__(self) -> tuple[tuple[PagedCacheValues], int]:
        return (self.values,), self.slots_per_page

    @classmethod
    def __tree_unflatten__(
        cls, slots_per_page: int, children: Sequence[Any]
    ) -> CacheLeaf:
        (values,) = children
        return cls(values=values, slots_per_page=slots_per_page)

    @property
    def cache_lengths(self) -> Tensor:
        """``[batch]`` int32 tokens already in the cache."""
        return self.values.cache_lengths.cast(DType.int32)

    def store(
        self,
        layer: Tensor,
        kv_idx: int,
        rows: Tensor,
        offsets: Tensor,
        *,
        cache_lengths: Tensor | None = None,
        rows_per_seq: int | None = None,
        ratio: int = 1,
    ) -> None:
        """Write ``rows`` (``[total, head_dim]``) after each request's length.

        A compressed leaf passes its own ``cache_lengths`` (entries, not
        tokens) and the bounds are rescaled to match; see the V2
        ``CacheLeaf.store``.
        """
        values = self.values
        if cache_lengths is not None:
            assert rows_per_seq is not None
            mesh = values.max_prompt_length.mesh

            def bound(value: int) -> Tensor:
                return F.constant(
                    np.array([value], np.uint32), DType.uint32, device=mesh
                )

            values = replace(
                values,
                cache_lengths=cache_lengths.cast(DType.uint32),
                max_prompt_length=bound(rows_per_seq),
                max_cache_length=(
                    (values.max_cache_length + bound(ratio - 1)) // ratio
                ),
            )
        x = rows.unsqueeze(1)
        if kv_idx == KEY:
            store_k_cache_ragged(values, x, offsets, layer)
        else:
            store_v_cache_ragged(values, x, offsets, layer)


@tree.dataclass
class DeepseekV4Cache:
    """Every leaf the forward reads and writes.

    ``comp`` / ``state`` are keyed by compress ratio; a ratio the model does
    not use is absent, as are the indexer leaves without ratio-4 layers.
    """

    swa: CacheLeaf
    comp: dict[int, CacheLeaf]
    state: dict[int, CacheLeaf]
    idx_comp: CacheLeaf | None
    idx_state: CacheLeaf | None

    @property
    def cache_lengths(self) -> Tensor:
        """``[batch]`` uint32 tokens already processed, shared by every leaf."""
        return self.swa.values.cache_lengths

    @classmethod
    def from_values(
        cls, config: DeepseekV4Config, values: Mapping[str, PagedCacheValues]
    ) -> DeepseekV4Cache:
        """Bundles the leaves by their key in ``config.kv_leaf_specs()``."""
        page_size = config.kv_params.page_size
        specs = config.kv_leaf_specs()
        if set(values) != {spec.key for spec in specs}:
            raise ValueError(
                f"expected cache leaves {[s.key for s in specs]}, got "
                f"{sorted(values)}"
            )
        leaves = {
            spec.key: CacheLeaf(
                values=values[spec.key],
                slots_per_page=spec.slots_per_page(page_size),
            )
            for spec in specs
        }

        def of_kind(kind: str) -> dict[int, CacheLeaf]:
            return {s.ratio: leaves[s.key] for s in specs if s.kind == kind}

        return cls(
            swa=leaves["swa"],
            comp=of_kind("comp"),
            state=of_kind("state"),
            idx_comp=leaves.get("idx_c4a"),
            idx_state=leaves.get("idx_c4a_state"),
        )

    @classmethod
    def from_inputs(
        cls,
        config: DeepseekV4Config,
        kv_params: KVCacheParamInterface,
        kv_inputs: Sequence[Tensor],
        mapping: DeviceMapping,
    ) -> DeepseekV4Cache:
        """Unpacks the forward's flattened cache inputs, one leaf per key."""
        kv_tree = kv_params.unflatten_kv_inputs(
            iter(x._graph_value for x in kv_inputs)
        )
        assert isinstance(kv_tree, dict)
        return cls.from_values(
            config,
            {
                name: PagedCacheValues.from_upstream(
                    tree.leaves(child, leaf=KVCacheInputsPerDevice), mapping
                )
                for name, child in kv_tree.items()
            },
        )
