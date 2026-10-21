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

import dataclasses
import math
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
from .layers.csa2 import (
    COMPRESSED_DATA_BYTES,
    COMPRESSED_RECORD_BYTES,
    WINDOW_DATA_BYTES,
    WINDOW_RECORD_BYTES,
)

CSA_KV_RECORDS = True
"""Whether the window and compressed leaves hold quantized CSA records by
default (see :func:`kv_leaf_specs`); the attention reads only records."""

MXFP4_INDEX_KEY_BYTES = 68
"""Bytes of one MXFP4 index key: 128 E2M1 values in 64 bytes, then a 32-bit
word of four UE8M0 scales."""


def _storage_dtype_and_width(
    spec: V41KVLeafSpec, cache_dtype: DType
) -> tuple[DType, int]:
    """Returns a leaf's stored element type and row width in elements.

    A record leaf is opaque ``uint8`` records of ``record_bytes`` each, one
    per slot; its kernels lay each layer's page out as every record's value
    bytes, then every record's scale bytes, and the page spans
    ``record_pitch`` bytes per slot. The other leaves store ``head_dim``
    elements of ``float32`` or the cache dtype.
    """
    if spec.record_bytes is not None:
        return DType.uint8, spec.record_pitch or spec.record_bytes
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
    record_pitch: int | None = None
    """Bytes a record slot spans in the page, at least ``record_bytes``;
    ``None`` means ``record_bytes``. The excess pads each layer's page at
    its end, past the records' values and scales."""

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


def kv_leaf_specs(
    huggingface_config: AutoConfig,
    *,
    kv_cache_page_size: int = 128,
    csa_records: bool = CSA_KV_RECORDS,
) -> list[V41KVLeafSpec]:
    """Returns the V4.1 KV cache leaves, in params-tree order.

    Every layer owns the sliding-window leaf. Each compression ratio used by
    a layer that computes its own KV and indexer results adds a compressed-KV
    leaf, an index-K leaf and, for ratios above 1, a compressor state leaf.

    Args:
        huggingface_config: The model's ``text_config``.
        kv_cache_page_size: Tokens a page covers, which sizes the record
            leaves' padding.
        csa_records: Whether the window and compressed leaves hold quantized
            CSA records (a 528-byte MXFP8 record per token and a 288-byte
            NVFP4 record per compression group) instead of ``head_dim``
            cache-dtype rows per token.
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
            # Each slot holds a token's float32 projections, its KV row then
            # its score row, so a layer's pages read as one
            # ``[blocks, page_size, 2 * head_dim]`` array.
            specs.append(
                V41KVLeafSpec(
                    key=f"{key}_state",
                    kind="state",
                    ratio=ratio,
                    owning_layer_ids=owning,
                    window_size=ratio,
                    head_dim=2 * head_dim,
                    kv_dim=1,
                    float32=True,
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
            )
        )
    if csa_records:
        specs = _pad_record_pitches(specs, kv_cache_page_size)
    return specs


# The widest padding a record slot may take, as a multiple of the record.
_MAX_PITCH_FACTOR = 2
# Record pitches step in 8 bytes, which keeps a layer's page of two or more
# slots 16-byte aligned.
_PITCH_ALIGN = 8
# Bytes of a record's values, the rows the attention reads a page in.
_RECORD_DATA_BYTES = {"swa": WINDOW_DATA_BYTES, "comp": COMPRESSED_DATA_BYTES}


def _readable_page(slots: int, pitch: int, data_bytes: int) -> bool:
    """Returns whether the attention reads a page of ``slots`` records.

    A page is 16-byte aligned, and a page of two or more records is a whole
    number of ``data_bytes`` rows.

    Args:
        slots: Records a layer's page holds.
        pitch: Bytes each record spans.
        data_bytes: Bytes of a record's values.
    """
    return (slots * pitch) % (16 if slots == 1 else data_bytes) == 0


def _pad_record_pitches(
    specs: list[V41KVLeafSpec], page_size: int
) -> list[V41KVLeafSpec]:
    """Pads the window and compressed record slots to Jenga-friendly pitches.

    One Jenga huge page is the least common multiple of every leaf's page
    bytes, so a record's odd factors (528 = 16 * 3 * 11, 288 = 32 * 9) would
    multiply it. The target is the least common multiple of the other
    leaves' pages and of the padded leaves' layer counts. Each window and
    compressed record leaf takes the smallest aligned pitch whose page
    divides it, so the records add no factor to the huge page; a leaf with
    no such pitch under ``_MAX_PITCH_FACTOR`` records takes the smallest
    aligned pitch the attention accepts.

    The attention reads a page of two or more records in rows of a record's
    value bytes, so such a page must be a whole number of those rows.
    """
    padded_kinds = ("swa", "comp")
    target = math.lcm(
        *(
            spec.page_bytes(page_size, DType.uint8)
            for spec in specs
            if spec.kind not in padded_kinds
        ),
        *(spec.num_layers for spec in specs if spec.kind in padded_kinds),
    )
    out = []
    for spec in specs:
        if spec.kind in padded_kinds and spec.record_bytes is not None:
            slots = spec.device_slots_per_page(page_size)
            data = _RECORD_DATA_BYTES[spec.kind]
            first = -(-spec.record_bytes // _PITCH_ALIGN) * _PITCH_ALIGN
            pitch = next(
                (
                    p
                    for p in range(
                        first,
                        _MAX_PITCH_FACTOR * spec.record_bytes,
                        _PITCH_ALIGN,
                    )
                    if _readable_page(slots, p, data)
                    and target % (spec.num_layers * slots * p) == 0
                ),
                None,
            )
            if pitch is None:
                pitch = first
                while not _readable_page(slots, pitch, data):
                    pitch += _PITCH_ALIGN
            spec = dataclasses.replace(spec, record_pitch=pitch)
        out.append(spec)
    return out


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
    :meth:`V41KVLeafSpec.device_slots_per_page`.
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
    specs = kv_leaf_specs(
        huggingface_config,
        kv_cache_page_size=kv_cache_config.kv_cache_page_size,
        csa_records=csa_records,
    )
    speculative_method: SpeculativeMethod | None = None
    num_draft_tokens = 0
    if pipeline_config.speculative:
        speculative_method = pipeline_config.speculative.speculative_method
        num_draft_tokens = (
            pipeline_config.speculative.num_speculative_tokens or 0
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
