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
"""The DeepSeek-V4.1-Flash ModuleV3 interface: config, KV geometry, inputs.

The KV leaf geometry and the forward inputs after the KV leaves are derived
here from the model config, with no graph and no weights.

Every leaf matches the geometry the registered V4.1 (V2 pipeline) CSA2 ops
read, reused from :mod:`...deepseekV41.layers.csa2` and
:mod:`...deepseekV41.kv_leaves`: the window and compressed leaves hold
opaque ``uint8`` records (an MXFP8 record per token, an NVFP4 record per
compression group), the index-key leaves hold MXFP4 keys, and the ratio-2
compressor state is a per-request scratch ring of ``float32`` rows. The
record constants come from those modules, so this interface can never
drift from the kernels that consume its caches.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from max.dtype import DType
from max.graph import DeviceRef, TensorType
from max.nn.legacy.kv_cache import (
    KVCacheParams,
    MultiKVCacheParams,
)
from max.pipelines.lib import KVCacheConfig

from ..deepseekV41.kv_leaves import (
    INDEX_KEY_STRIDE_ALIGNMENT,
    MXFP4_INDEX_KEY_BYTES,
    STATE_STRIDE_ALIGNMENT,
    state_ring_rows,
)
from ..deepseekV41.layers.csa2 import (
    COMPRESSED_RECORD_BYTES,
    WINDOW_RECORD_BYTES,
)

PAGE_SIZE = 128
"""Tokens per KV cache page. This interface fixes the page size at 128; any
other value is rejected in :func:`build_kv_params`."""

_BATCH_SYMBOL = "replica_0_batch_size"
_NUM_LAST_TOKEN_LOOKBACK = 3

# DSpark draft stages when the checkpoint config names none, mirroring V2's
# ``deepseekV41.model_config._DEFAULT_DSPARK_STAGES`` so the two pipelines
# resolve the same window-leaf geometry.
_DEFAULT_DSPARK_STAGES = 3


@dataclass(frozen=True)
class V41Config:
    """The slice of the V4.1 text config the interface depends on."""

    vocab_size: int
    num_hidden_layers: int
    num_attention_heads: int
    head_dim: int
    sliding_window: int
    compress_ratios: tuple[int, ...]
    kv_source_layer_ids: tuple[int, ...]
    index_head_dim: int
    engram_num_embeddings: tuple[int, ...]
    engram_head_dim: int
    num_dspark_stages: int

    @classmethod
    def from_hf(cls, config: Mapping[str, Any]) -> V41Config:
        """Reads the interface fields off an HF ``config.json`` mapping."""
        text = config["text_config"]
        return cls(
            vocab_size=text["vocab_size"],
            num_hidden_layers=text["num_hidden_layers"],
            num_attention_heads=text["num_attention_heads"],
            head_dim=text["head_dim"],
            sliding_window=text["sliding_window"],
            compress_ratios=tuple(text["compress_ratios"]),
            kv_source_layer_ids=tuple(text["kv_source_layer_ids"]),
            index_head_dim=text["index_head_dim"],
            engram_num_embeddings=tuple(text["engram_num_embeddings"]),
            engram_head_dim=text["engram_head_dim"],
            # The draft-stage count that sizes the window leaf's extra
            # rows, resolved the way V2's
            # ``deepseekV41.model_config.num_dspark_stages`` resolves it.
            # The checkpoint's tap list (``dspark_target_layer_ids``) is a
            # different quantity and only coincides with it.
            num_dspark_stages=(
                text.get("n_mtp_layers")
                or text.get("num_nextn_predict_layers")
                or _DEFAULT_DSPARK_STAGES
            ),
        )

    def ratio(self, layer: int) -> int:
        """Returns the compression ratio of one trunk layer.

        Args:
            layer: The trunk layer id.

        Returns:
            The layer's compression ratio; 0 means sliding window only.
        """
        return self.compress_ratios[layer]

    @property
    def kv_layers_r2(self) -> tuple[int, ...]:
        """The KV source layers that run a ratio-2 compressor."""
        return tuple(l for l in self.kv_source_layer_ids if self.ratio(l) == 2)

    @property
    def kv_layers_r1(self) -> tuple[int, ...]:
        """The KV source layers that run a ratio-1 compressor."""
        return tuple(l for l in self.kv_source_layer_ids if self.ratio(l) == 1)


@dataclass(frozen=True)
class KVLeafSpec:
    """One paged-cache leaf of the V4.1 KV state.

    ``kind`` is one of ``"swa"``, ``"comp"``, ``"state"`` or
    ``"idx_comp"``. A record leaf (``record_bytes`` set) stores one opaque
    ``uint8`` record per slot -- the exact bytes the registered V2 CSA2
    store, attention and indexer kernels write and read -- whose kernels
    lay each layer's page out as every record's value bytes, then every
    record's scale bytes. The other leaves store ``head_dim`` elements:
    ``float32`` for the compressor state, one row per token of a
    per-request scratch ring of ``ring_rows`` rows.
    """

    key: str
    kind: str
    ratio: int
    window_size: int | None
    head_dim: int
    kv_dim: int
    """1 (K-only / MLA-shaped) or 2 (K and V planes); ``build_kv_params``
    passes ``is_mla=kv_dim == 1``, which selects the params class."""
    num_layers: int
    float32: bool = False
    record_bytes: int | None = None
    """Bytes of the opaque ``uint8`` record each slot holds, or ``None``
    for a row of ``head_dim`` elements."""
    stride_alignment: int | None = None
    """See :attr:`~max.nn.legacy.kv_cache.KVLeafRegion.stride_alignment`."""
    ring_rows: int | None = None
    """Rows of a ``state`` leaf's per-request ring, or ``None`` for a leaf
    paged by token."""

    @property
    def slots_per_page(self) -> int:
        """Storage slots one page holds at :data:`PAGE_SIZE`.

        One per ``ratio`` tokens on a compressed or index-key leaf, one per
        token on the window leaf, and the ring's rows on a state leaf.
        """
        if self.ring_rows is not None:
            return self.ring_rows
        if self.kind in ("comp", "idx_comp"):
            return PAGE_SIZE // self.ratio
        return PAGE_SIZE

    @property
    def row_bytes(self) -> int:
        """The leaf's row width: record bytes, or ``head_dim`` values."""
        if self.record_bytes is not None:
            return self.record_bytes
        return self.head_dim

    @property
    def dtype(self) -> DType:
        if self.record_bytes is not None:
            return DType.uint8
        return DType.float32 if self.float32 else DType.bfloat16


def kv_leaf_specs(config: V41Config) -> list[KVLeafSpec]:
    """Returns the V4.1 KV leaves in params-tree order, dropping unused
    ratios.

    Every layer owns the sliding-window leaf. Each compression ratio a KV
    source layer runs adds a compressed-record leaf, an index-key leaf and,
    above ratio 1, a compressor-state leaf. The record geometry is the
    canonical one the registered V2 ops address: MXFP8 window records
    (:data:`WINDOW_RECORD_BYTES`), NVFP4 compressed records
    (:data:`COMPRESSED_RECORD_BYTES`) and MXFP4 index keys
    (:data:`MXFP4_INDEX_KEY_BYTES`), one record per token or per closed
    compression group.

    Args:
        config: The interface config the leaf geometry derives from.

    Returns:
        The leaf specs, in the order :func:`build_kv_params` keys them
        into the params tree.
    """
    specs = [
        KVLeafSpec(
            key="swa",
            kind="swa",
            ratio=0,
            window_size=config.sliding_window,
            head_dim=config.head_dim,
            kv_dim=1,
            # DSpark stages share the window leaf, so adding them later
            # does not change the shape of ``kv_blocks``. The stage count
            # sizes the extra rows, the way V2's
            # ``model_config.num_dspark_stages`` does -- not the tap list,
            # which only coincides with it on the published checkpoint.
            num_layers=config.num_hidden_layers + config.num_dspark_stages,
            record_bytes=WINDOW_RECORD_BYTES,
        )
    ]
    for ratio, layers in ((2, config.kv_layers_r2), (1, config.kv_layers_r1)):
        n = len(layers)
        if not n:
            continue
        specs.append(
            KVLeafSpec(
                key=f"c{ratio}a",
                kind="comp",
                ratio=ratio,
                window_size=None,
                head_dim=config.head_dim,
                kv_dim=1,
                num_layers=n,
                record_bytes=COMPRESSED_RECORD_BYTES,
            )
        )
        if ratio > 1:
            # Ratio 1 compresses each token alone, so no partial group
            # carries over between forwards. At ratio 2 the open half of a
            # pair does, and a prefix hit lands on a closed pair, so the
            # state is a per-request scratch ring holding a token's raw
            # float32 KV row then its score row. Without speculative
            # decoding a verify step carries no drafts, so the ring is
            # two rows (:func:`state_ring_rows`);
            # ``DeepseekV41Config.construct_kv_params`` rejects a
            # speculative config until DSpark sizing lands.
            specs.append(
                KVLeafSpec(
                    key=f"c{ratio}a_state",
                    kind="state",
                    ratio=ratio,
                    window_size=None,
                    head_dim=2 * config.head_dim,
                    kv_dim=1,
                    num_layers=n,
                    float32=True,
                    stride_alignment=STATE_STRIDE_ALIGNMENT,
                    ring_rows=state_ring_rows(0),
                )
            )
        specs.append(
            KVLeafSpec(
                key=f"idx_c{ratio}a",
                kind="idx_comp",
                ratio=ratio,
                window_size=None,
                head_dim=config.index_head_dim,
                kv_dim=1,
                num_layers=n,
                record_bytes=MXFP4_INDEX_KEY_BYTES,
                stride_alignment=INDEX_KEY_STRIDE_ALIGNMENT,
            )
        )
    return specs


def build_kv_params(
    config: V41Config,
    kv_cache_config: KVCacheConfig,
    devices: Sequence[DeviceRef],
    data_parallel_degree: int = 1,
) -> MultiKVCacheParams:
    """Builds one ``MultiKVCacheParams`` with a leaf per :func:`kv_leaf_specs`
    entry.

    Every leaf shares the token page size, so one block table serves all
    of them; per-page slots come from each spec: one per token on the
    window leaf, one per closed compression group on the compressed and
    index-key leaves, and the ring's rows on the scratch state leaf.
    Record leaves are opaque ``uint8`` rows; the state ring is ``float32``
    and scratch -- drawn once per request, never published. The pool's own
    settings -- prefix caching, quantization, connectors -- flow through
    ``kv_cache_config.to_params`` untouched.

    ``data_parallel_degree`` names each replica's row dim after its replica
    (``replica_r_batch_size``, the deepseekV3 ``DP_EP`` pattern): replicas
    carry different request counts, so their KV metadata rows must be
    distinct symbolic dims. Degree 1 keeps every device on
    ``replica_0_batch_size`` — the tensor-parallel shape, where all devices
    do hold the same rows.

    Args:
        config: The interface config the leaf geometry derives from.
        kv_cache_config: The pool settings that flow through
            ``kv_cache_config.to_params`` untouched.
        devices: The devices the cache leaves live on.
        data_parallel_degree: The number of data-parallel replicas; 1
            keeps the tensor-parallel row dim naming.

    Returns:
        The params tree holding one child per leaf spec.

    Raises:
        ValueError: If ``kv_cache_config``'s page size is not
            :data:`PAGE_SIZE`. The leaves' slot counts are fixed to
            128-token pages and the same page size also drives allocation
            outside these params, so a mismatched config is rejected rather
            than silently overridden.
    """
    if kv_cache_config.kv_cache_page_size != PAGE_SIZE:
        raise ValueError(
            "DeepSeek-V4.1 requires a KV cache page size of "
            f"{PAGE_SIZE}, got {kv_cache_config.kv_cache_page_size}. Set "
            f"--kv-cache-page-size {PAGE_SIZE}."
        )
    children: dict[str, KVCacheParams] = {}
    for spec in kv_leaf_specs(config):
        children[spec.key] = kv_cache_config.to_params(
            dtype=spec.dtype,
            n_kv_heads=1,
            head_dim=spec.row_bytes,
            num_layers=spec.num_layers,
            devices=devices,
            data_parallel_degree=data_parallel_degree,
            is_mla=spec.kv_dim == 1,
            num_q_heads=config.num_attention_heads,
            slots_per_page=spec.slots_per_page,
            window_size=spec.window_size,
            stride_alignment=spec.stride_alignment,
            scratch=spec.ring_rows is not None,
        )
    return MultiKVCacheParams.from_params(children)


def extra_inputs(
    config: V41Config,
    device_refs: Sequence[DeviceRef],
) -> list[tuple[str, TensorType]]:
    """Returns the forward inputs after the KV leaves, in graph order.

    The four per-row fields name their rows _BATCH_SYMBOL: every device
    holds the same rows, so the name unifies them with each replica's KV
    metadata. Data parallelism is refused by the batch processor and the
    model, so there is no global-row dim to name.

    Args:
        config: The interface config the input extents derive from.
        device_refs: The devices the graph runs on; the per-row fields
            name their rows after the first.

    Returns:
        The ``(name, tensor type)`` pairs, in the order the forward reads
        them.
    """
    gpu0 = device_refs[0]
    cpu = DeviceRef.CPU()
    i32, u8 = DType.int32, DType.uint8
    extras = [
        ("swa_begin", TensorType(i32, [_BATCH_SYMBOL], gpu0)),
        ("global_write_floor", TensorType(i32, [_BATCH_SYMBOL], gpu0)),
        ("image_token_mask", TensorType(u8, ["total_seq_len"], gpu0)),
        (
            "engram_lookback",
            TensorType(i32, [_BATCH_SYMBOL, _NUM_LAST_TOKEN_LOOKBACK], gpu0),
        ),
        (
            "engram_lookback_dead",
            TensorType(u8, [_BATCH_SYMBOL, _NUM_LAST_TOKEN_LOOKBACK], gpu0),
        ),
    ]
    for ordinal, rows in enumerate(config.engram_num_embeddings):
        extras.append(
            (
                f"engram_table_{ordinal}_weight",
                TensorType(
                    DType.float8_e4m3fn,
                    [rows, config.engram_head_dim],
                    cpu,
                ),
            )
        )
        extras.append(
            (
                f"engram_table_{ordinal}_scale",
                TensorType(
                    DType.float8_e8m0fnu,
                    [rows, config.engram_head_dim // 32],
                    cpu,
                ),
            )
        )
    return extras


def extra_input_names(config: V41Config) -> list[str]:
    """Returns the names of :func:`extra_inputs`, in order.

    Args:
        config: The interface config the input extents derive from.

    Returns:
        The extra input names, in graph order.
    """
    return [name for name, _ in extra_inputs(config, [DeviceRef.GPU(0)])]
