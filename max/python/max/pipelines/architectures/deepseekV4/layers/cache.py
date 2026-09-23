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

"""DeepSeek-V4's KV state as paged cache leaves.

The reference keeps, per compressed layer, a two-zone flat buffer (a 128-slot
sliding window of per-token latents plus one compressed entry per
``compress_ratio`` tokens), the compressor's open-window projections, and for
ratio-4 layers the same again for the lightning indexer. Here every one of
those is a leaf of one :class:`~max.nn.kv_cache.MultiKVCacheParams`, all paged
by token with one shared lookup table:

* the latent window is a ``sliding_window_group(128)`` leaf;
* a compressed zone is a full leaf with ``slots_per_page = page_size //
  ratio``, so page ``p`` holds the entries for tokens ``[128p, 128p + 128)``.
  The block table stays in tokens; only the buffer is narrower;
* the compressor's open state is a ``sliding_window_group(coff * ratio)`` leaf
  holding the raw ``wkv`` / ``wgate`` projections of the last ``coff * ratio``
  tokens (K and V of an MHA leaf). The pooled entry is recomputed from those
  rows, so no per-request state machine survives a forward.

Writes go through the stock ``mo.kv_cache.store.paged.ragged`` kernel. It
takes the page size from the buffer's static slot dimension, so a compressed
leaf is written by handing it ``cache_lengths // ratio``: entry ``j`` lands on
page ``j // slots_per_page``, slot ``j % slots_per_page``, which is the same
page the token page table already names.

Attention reads the window leaf and the zone leaf through the fused
``mo.latent_sparse_attention.ragged.paged`` kernel. The remaining graph-side
reads -- the compressor's open state and the indexer's candidate table --
gather rows out of the block buffer by ``(lookup_table[b, slot //
slots_per_page], slot % slots_per_page)``; each copies the leaf
(``buffer_load``), which is fine at bringup sizes and nowhere else.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.nn.kernels import store_k_cache_ragged, store_v_cache_ragged
from max.nn.kv_cache import PagedCacheValues
from max.support.math import ceildiv as ceildiv

from ..model_config import DeepseekV4Config

KEY = 0
VALUE = 1


def arange(n: int, device: DeviceRef) -> TensorValue:
    return ops.range(0, n, 1, out_dim=n, device=device, dtype=DType.int32)


def scalar(value: int, device: DeviceRef) -> TensorValue:
    return ops.constant(value, DType.int32, device)


def idiv(x: TensorValue, divisor: int) -> TensorValue:
    """Exact integer floor division of a non-negative integer tensor.

    ``TensorValue.__floordiv__`` is ``floor(div)`` and ``div`` promotes
    integers to float, so its result is not an index dtype. The values here
    (positions, slots) are far below 2**24, so the round trip is exact.
    """
    return ops.cast(ops.floor(ops.div(x, divisor)), x.dtype)


def row_offsets(
    batch: int, rows_per_seq: int, device: DeviceRef
) -> TensorValue:
    """``[batch + 1]`` uint32 ragged offsets for a padded batch."""
    return ops.cast(
        ops.range(
            0,
            (batch + 1) * rows_per_seq,
            rows_per_seq,
            out_dim=batch + 1,
            device=device,
            dtype=DType.int32,
        ),
        DType.uint32,
    )


@dataclass
class CacheLeaf:
    """One leaf's graph inputs plus the slot geometry the kernels infer."""

    values: PagedCacheValues
    slots_per_page: int

    @property
    def cache_lengths(self) -> TensorValue:
        """``[batch]`` int32 tokens already in the cache."""
        return ops.cast(self.values.cache_lengths, DType.int32)

    def gather(
        self,
        layer: int,
        kv_idx: int,
        slots: TensorValue,
        lut_rows: TensorValue | None = None,
    ) -> TensorValue:
        """Rows at ``slots`` (``[rows, n]`` int32) -> ``[rows, n, head_dim]``.

        ``lut_rows`` (``[rows]`` int32) names the request each row of
        ``slots`` addresses; without it the rows are the batch itself. Every
        slot must be non-negative and inside that request's allocated pages;
        callers clamp dead slots onto a live one and mask them out afterwards.
        Heads are always 1 on these leaves.
        """
        blocks = ops.buffer_load(self.values.kv_blocks)
        b, n = slots.shape[0], slots.shape[1]
        page_col = idiv(slots, self.slots_per_page)
        in_page = slots - page_col * self.slots_per_page
        lut = ops.cast(self.values.lookup_table, DType.int32)
        if lut_rows is not None:
            lut = ops.gather(lut, lut_rows, axis=0)
        else:
            # ``gather_nd`` wants the batch dims to match exactly, and the
            # table's is symbolic.
            lut = ops.rebind(lut, [b, lut.shape[1]])
        pages = ops.gather_nd(lut, ops.unsqueeze(page_col, -1), batch_dims=1)

        def const(v: int) -> TensorValue:
            return ops.broadcast_to(scalar(v, slots.device), [b, n])

        idx = ops.stack(
            [pages, const(kv_idx), const(layer), in_page, const(0)], axis=-1
        )
        return ops.gather_nd(blocks, idx)

    def store(
        self,
        layer: int,
        kv_idx: int,
        rows: TensorValue,
        offsets: TensorValue,
        *,
        cache_lengths: TensorValue | None = None,
        rows_per_seq: int | None = None,
        ratio: int = 1,
    ) -> None:
        """Write ``rows`` (``[total, head_dim]``) after each request's length.

        Row ``t`` of request ``b`` lands on slot ``cache_lengths[b] + t``.
        A compressed leaf passes its own ``cache_lengths`` (the token count
        divided by ``ratio``) so the slots count entries, not tokens; the
        bound inputs are rescaled the same way, ``rows_per_seq`` being an
        upper bound on the rows any one request stores.
        """
        values = self.values
        if cache_lengths is not None:
            assert rows_per_seq is not None
            device = values.max_prompt_length.device

            def bound(value: int) -> TensorValue:
                return ops.reshape(
                    ops.constant(value, DType.uint32, device), [1]
                )

            values = replace(
                values,
                cache_lengths=ops.cast(cache_lengths, DType.uint32),
                max_prompt_length=bound(rows_per_seq),
                max_cache_length=idiv(
                    values.max_cache_length + bound(ratio - 1), ratio
                ),
            )
        x = ops.unsqueeze(rows, 1)
        layer_idx = ops.constant(layer, DType.uint32, DeviceRef.CPU())
        if kv_idx == KEY:
            store_k_cache_ragged(values, x, offsets, layer_idx)
        else:
            store_v_cache_ragged(values, x, offsets, layer_idx)


@dataclass
class DeepseekV4Cache:
    """Every leaf one device's forward reads and writes.

    ``comp`` / ``state`` are keyed by compress ratio; a ratio the model does
    not use is absent. The indexer leaves exist only when there are ratio-4
    layers.
    """

    swa: CacheLeaf
    comp: dict[int, CacheLeaf]
    state: dict[int, CacheLeaf]
    idx_comp: CacheLeaf | None
    idx_state: CacheLeaf | None

    @property
    def cache_lengths(self) -> TensorValue:
        """``[batch]`` int32 tokens already processed, shared by every leaf."""
        return self.swa.cache_lengths

    @classmethod
    def from_groups(
        cls,
        config: DeepseekV4Config,
        groups: Sequence[Sequence[PagedCacheValues]],
        device_idx: int = 0,
    ) -> DeepseekV4Cache:
        """Bundle the leaves ``MultiKVCacheParams.unflatten_basic_kv_tree``
        returned, which are in ``config.kv_leaf_specs()`` order, one list per
        leaf holding one entry per device.
        """
        page_size = config.kv_params.page_size
        specs = config.kv_leaf_specs()
        if len(specs) != len(groups):
            raise ValueError(
                f"expected {len(specs)} cache leaves, got {len(groups)}"
            )
        leaves = {
            spec.key: CacheLeaf(
                values=group[device_idx],
                slots_per_page=spec.slots_per_page(page_size),
            )
            for spec, group in zip(specs, groups, strict=True)
        }
        by_kind = {spec.key: spec for spec in specs}
        return cls(
            swa=leaves["swa"],
            comp={
                by_kind[k].ratio: leaf
                for k, leaf in leaves.items()
                if by_kind[k].kind == "comp"
            },
            state={
                by_kind[k].ratio: leaf
                for k, leaf in leaves.items()
                if by_kind[k].kind == "state"
            },
            idx_comp=leaves.get("idx_c4a"),
            idx_state=leaves.get("idx_c4a_state"),
        )
