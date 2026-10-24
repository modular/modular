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
"""KV cache leaf geometry for DeepSeek-V4.1-Flash.

V4.1's attention state is split into several paged-cache leaves that share
one block table. This module describes those leaves and builds the KV cache
params from them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import (
    KVCacheParamInterface,
    KVCacheParams,
    MultiKVCacheParams,
)
from max.pipelines.lib import KVCacheConfig, PipelineConfig
from max.pipelines.speculative.config import SpeculativeMethod
from transformers import AutoConfig

from . import layer_resolver, model_config
from .layers.csa2 import COMPRESSED_RECORD_BYTES, WINDOW_RECORD_BYTES

CSA_KV_RECORDS = True
"""Whether the window and compressed leaves hold quantized CSA records by
default (see :func:`kv_leaf_specs`); the attention reads only records."""

MXFP4_INDEX_KEY_BYTES = 68
"""Bytes of one MXFP4 index key: 128 E2M1 values in 64 bytes, then a 32-bit
word of four UE8M0 scales."""

INDEX_KEY_STRIDE_ALIGNMENT = 16
"""The indexer ops take any block stride that is a multiple of 16 bytes."""

STATE_STRIDE_ALIGNMENT: int | None = 16
"""The compressor takes any block stride that is a multiple of 16 bytes.
``None`` keeps the state ring packed."""


def _storage_dtype_and_width(
    spec: V41KVLeafSpec, cache_dtype: DType
) -> tuple[DType, int]:
    """Returns a leaf's stored element type and row width in elements.

    A record leaf is opaque ``uint8`` records of ``record_bytes`` each, one
    per slot; its kernels lay each layer's page out as every record's value
    bytes, then every record's scale bytes. The other leaves store
    ``head_dim`` elements of ``float32`` or the cache dtype.
    """
    if spec.record_bytes is not None:
        return DType.uint8, spec.record_bytes
    return (DType.float32 if spec.float32 else cache_dtype), spec.head_dim


def _full_layers_by_ratio(
    huggingface_config: AutoConfig,
) -> dict[int, tuple[int, ...]]:
    """Layers that own a leaf's storage, grouped by ``compress_ratio``.

    Only ``attention_mode == "Full"`` layers (which compute their own KV and
    indexer results) own storage; DSpark stages never do.
    """
    by_ratio: dict[int, list[int]] = {}
    for row in layer_resolver.resolve(huggingface_config):
        if row.attention_mode == "Full":
            by_ratio.setdefault(row.compress_ratio, []).append(row.layer_id)
    return {ratio: tuple(layers) for ratio, layers in by_ratio.items()}


@dataclass(frozen=True)
class V41KVLeafSpec:
    """One paged-cache leaf of the V4.1 KV state.

    ``owning_layer_ids`` are the layers that read and write it.
    """

    key: str
    kind: str
    """``"swa"`` | ``"comp"`` | ``"state"`` | ``"idx_comp"``."""
    ratio: int
    """0 for ``swa`` (not ratio-keyed), else 1 or 2."""
    owning_layer_ids: tuple[int, ...]
    window_size: int | None
    head_dim: int
    kv_dim: int
    """1 (K-only / MLA-shaped) or 2 (K and V planes)."""
    float32: bool
    record_bytes: int | None = None
    """Bytes of the opaque ``uint8`` record each slot holds, or ``None`` for
    a row of ``head_dim`` elements."""
    stride_alignment: int | None = None
    """See :attr:`~max.nn.kv_cache.KVLeafRegion.stride_alignment`."""
    ring_rows: int | None = None
    """Rows of a ``state`` leaf's per-request ring, or ``None`` for a leaf
    paged by token."""

    @property
    def num_layers(self) -> int:
        return len(self.owning_layer_ids)

    def slots_per_page(self, page_size: int) -> int:
        """Storage slots per page: one per token or one per ``ratio`` tokens."""
        return (
            page_size // self.ratio
            if self.kind in ("comp", "idx_comp")
            else page_size
        )

    def device_slots_per_page(self, page_size: int) -> int:
        """Slots per page the device buffer allocates.

        Unlike :meth:`slots_per_page`, a ``comp`` leaf of element rows
        allocates one slot per token: attention stores one row per raw
        token. A ``comp`` record leaf stores one record per compression
        group, and ``idx_comp`` one slot per key.
        """
        if self.ring_rows is not None:
            return self.ring_rows
        if self.kind == "comp" and self.record_bytes is None:
            return page_size
        return self.slots_per_page(page_size)

    def page_bytes(self, page_size: int, cache_dtype: DType) -> int:
        """Device bytes of one page across every owning layer."""
        dtype, width = _storage_dtype_and_width(self, cache_dtype)
        return (
            self.kv_dim
            * self.num_layers
            * self.device_slots_per_page(page_size)
            * width
            * dtype.size_in_bytes
        )


def state_ring_rows(num_draft_tokens: int) -> int:
    """Returns the rows a compressor state ring holds.

    A verify step of ``k`` drafts writes ``k + 1`` rows and a rejection
    resumes on one of them, so the ring takes the power of two at or above
    ``k + 2``.

    Args:
        num_draft_tokens: Drafts a verify step carries, zero without
            speculative decoding.
    """
    rows = 2
    while rows < num_draft_tokens + 2:
        rows *= 2
    return rows


def kv_leaf_specs(
    huggingface_config: AutoConfig,
    *,
    csa_records: bool = CSA_KV_RECORDS,
    num_draft_tokens: int = 0,
) -> list[V41KVLeafSpec]:
    """Returns the V4.1 KV cache leaves, in params-tree order.

    Every layer owns the sliding-window leaf. Each compression ratio used by
    a layer that computes its own KV and indexer results adds a compressed-KV
    leaf, an index-K leaf and, for ratios above 1, a compressor state leaf.

    Args:
        huggingface_config: The model's ``text_config``.
        csa_records: Whether the window and compressed leaves hold quantized
            CSA records (a 528-byte MXFP8 record per token and a 288-byte
            NVFP4 record per compression group) instead of ``head_dim``
            cache-dtype rows per token.
        num_draft_tokens: Drafts a verify step carries, which sizes the
            compressor state ring (:func:`state_ring_rows`).
    """
    num_hidden_layers = huggingface_config.num_hidden_layers
    num_dspark_stages = model_config.num_dspark_stages(huggingface_config)
    head_dim = huggingface_config.head_dim
    index_head_dim = huggingface_config.index_head_dim
    sliding_window = huggingface_config.sliding_window
    full_layers = _full_layers_by_ratio(huggingface_config)

    specs = [
        V41KVLeafSpec(
            key="swa",
            kind="swa",
            ratio=0,
            owning_layer_ids=tuple(
                range(num_hidden_layers + num_dspark_stages)
            ),
            window_size=sliding_window,
            head_dim=head_dim,
            kv_dim=1,
            float32=False,
            record_bytes=WINDOW_RECORD_BYTES if csa_records else None,
        )
    ]
    for ratio in (2, 1):
        owning = full_layers.get(ratio, ())
        if not owning:
            continue
        key = f"c{ratio}a"
        specs.append(
            V41KVLeafSpec(
                key=key,
                kind="comp",
                ratio=ratio,
                owning_layer_ids=owning,
                window_size=None,
                head_dim=head_dim,
                kv_dim=1,
                float32=False,
                record_bytes=COMPRESSED_RECORD_BYTES if csa_records else None,
            )
        )
        if ratio > 1:
            # Ratio 1 compresses each token alone, so no partial group
            # carries over between steps and no compressor state is needed.
            # At ratio 2 only the open half of a pair does, and a prefix hit
            # lands on a closed pair, so the state is a per-request ring. Each
            # slot holds a token's float32 KV row then its score row.
            specs.append(
                V41KVLeafSpec(
                    key=f"{key}_state",
                    kind="state",
                    ratio=ratio,
                    owning_layer_ids=owning,
                    window_size=None,
                    head_dim=2 * head_dim,
                    kv_dim=1,
                    float32=True,
                    ring_rows=state_ring_rows(num_draft_tokens),
                    stride_alignment=STATE_STRIDE_ALIGNMENT,
                )
            )
        specs.append(
            V41KVLeafSpec(
                key=f"idx_{key}",
                kind="idx_comp",
                ratio=ratio,
                owning_layer_ids=owning,
                window_size=None,
                head_dim=index_head_dim,
                kv_dim=1,
                float32=False,
                record_bytes=MXFP4_INDEX_KEY_BYTES,
                stride_alignment=INDEX_KEY_STRIDE_ALIGNMENT,
            )
        )
    return specs


def build_kv_params(
    specs: Sequence[V41KVLeafSpec],
    *,
    kv_cache_config: KVCacheConfig,
    cache_dtype: DType,
    devices: Sequence[DeviceRef],
    num_q_heads: int,
    data_parallel_degree: int = 1,
    speculative_method: SpeculativeMethod | None = None,
    num_draft_tokens: int = 0,
) -> MultiKVCacheParams:
    """Builds one ``MultiKVCacheParams`` with a leaf per spec.

    Every leaf shares the token page size, so one block table serves all
    leaves; per-page slots come from
    :meth:`V41KVLeafSpec.device_slots_per_page`. A ring leaf is scratch.
    """
    page_size = kv_cache_config.kv_cache_page_size
    children: dict[str, KVCacheParams] = {}
    for spec in specs:
        dtype, width = _storage_dtype_and_width(spec, cache_dtype)
        children[spec.key] = kv_cache_config.to_params(
            dtype=dtype,
            n_kv_heads=1,
            head_dim=width,
            num_layers=spec.num_layers,
            devices=devices,
            data_parallel_degree=data_parallel_degree,
            is_mla=spec.kv_dim == 1,
            num_q_heads=num_q_heads,
            speculative_method=speculative_method,
            num_draft_tokens=num_draft_tokens,
            slots_per_page=spec.device_slots_per_page(page_size),
            window_size=spec.window_size,
            scratch=spec.ring_rows is not None,
            stride_alignment=spec.stride_alignment,
        )
    return MultiKVCacheParams.from_params(children)


def construct_kv_params(
    *,
    huggingface_config: AutoConfig,
    pipeline_config: PipelineConfig,
    devices: list[DeviceRef],
    kv_cache_config: KVCacheConfig,
    cache_dtype: DType,
    csa_records: bool = CSA_KV_RECORDS,
) -> KVCacheParamInterface:
    """Builds the V4.1 KV cache params from a raw ``text_config``.

    ``csa_records`` is passed to :func:`kv_leaf_specs`.
    """
    speculative_method: SpeculativeMethod | None = None
    num_draft_tokens = 0
    if pipeline_config.speculative:
        speculative_method = pipeline_config.speculative.speculative_method
        drafts = pipeline_config.speculative.num_speculative_tokens
        # The state ring's depth follows it, so it cannot default to zero.
        if speculative_method is not None and drafts is None:
            raise ValueError(
                f"{speculative_method} needs num_speculative_tokens resolved "
                "before the KV cache is sized."
            )
        num_draft_tokens = drafts or 0
    specs = kv_leaf_specs(
        huggingface_config,
        csa_records=csa_records,
        num_draft_tokens=num_draft_tokens,
    )
    return build_kv_params(
        specs,
        kv_cache_config=kv_cache_config,
        cache_dtype=cache_dtype,
        devices=devices,
        num_q_heads=huggingface_config.num_attention_heads,
        data_parallel_degree=pipeline_config.model.data_parallel_degree,
        speculative_method=speculative_method,
        num_draft_tokens=num_draft_tokens,
    )
