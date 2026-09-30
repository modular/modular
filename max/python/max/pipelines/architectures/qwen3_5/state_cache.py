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
"""The shape of a Qwen3.5 Gated DeltaNet state, and how a layer reaches it.

Both the pool's leaves and the graph's input types are sized from the
geometry, so it is derived here once. The per-layer access built on top of it
belongs here too: it is the only reader of the order the leaves are declared
in, and keeping the two apart would put that order in two files.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from max import tree
from max.dtype import DType
from max.graph import BufferValue, TensorValue
from max.nn.kv_cache import (
    KVCacheParamInterface,
    KVCacheParams,
    MultiKVCacheParams,
    RecurrentStateInputsPerDevice,
    RecurrentStateRegion,
)

ATTN_CACHE_KEY = "attn"
STATE_CACHE_KEY = "state"
"""The two children of a Qwen3.5 cache tree."""

CONV_LEAF_ID = "linear_attn/conv"
RECURRENT_LEAF_ID = "linear_attn/recurrent"
"""The two leaves one Gated DeltaNet state spans.

Separate leaves because the kernels want each as its own uniformly-strided
tensor. They are allocated, published and evicted together.
"""

RING_LEAF_ID = "linear_attn/ring"
"""The scratch leaf of a speculative verify ring.

A scratch-group leaf of the state cache. Each request draws one block when it
is admitted and holds it until it is released, and it is never published.
"""

_LEAF_IDS = (CONV_LEAF_ID, RECURRENT_LEAF_ID)
"""Declaration order of the leaves every Gated DeltaNet state has.

:func:`linear_state_regions` checks its leading regions against this. The ring
leaf, when declared, comes after these.
"""

_CONV_LEAF = _LEAF_IDS.index(CONV_LEAF_ID)
_RECURRENT_LEAF = _LEAF_IDS.index(RECURRENT_LEAF_ID)
_VERIFY_LEAF = len(_LEAF_IDS)
"""Where a speculative verify's ring follows the live leaves."""

RING_DTYPE = DType.float32
"""Dtype of the ring pool. Float32 keeps the fold bit-exact against a
replay."""

COMPILED_RING_LENS = (2, 4, 8)
"""Ring lengths the ring kernels are compiled for."""

_RING_RECORD_ALIGN = 32
"""Elements a ring record's stride is a multiple of, one 128-byte line."""


def shadowed_leaf_ids(ring_len: int) -> tuple[str, ...]:
    """Returns the leaves a speculative verify writes in place.

    These must be copied to a shadow before the verify. The verify never
    writes the conv leaf, and with a ring it does not write the recurrent
    leaf either.

    Args:
        ring_len: The verify ring's record capacity, or zero for no ring.
    """
    return () if ring_len else (RECURRENT_LEAF_ID,)


def ring_len_for_window(window: int) -> int:
    """Returns the shortest compiled ring length that fits ``window``.

    Args:
        window: Tokens one verify advances a request by, ``K + 1``.

    Returns:
        The smallest entry of :data:`COMPILED_RING_LENS` that is at least
        ``window``.

    Raises:
        ValueError: If no compiled ring length is long enough.
    """
    for compiled in COMPILED_RING_LENS:
        if window <= compiled:
            return compiled
    raise ValueError(
        f"a {window}-token verify window needs a ring longer than any the "
        f"gated-delta ring kernels are compiled for ({COMPILED_RING_LENS}); "
        "add the length to both the kernel dispatch and COMPILED_RING_LENS"
    )


@tree.dataclass(frozen=True)
class GatedDeltaVerifyAccess:
    """One linear layer's verify ring, on one device."""

    pool: BufferValue
    row_id: TensorValue


@tree.dataclass(frozen=True)
class GatedDeltaStateAccess:
    """One linear layer's state access, on one device.

    Each pool travels with the row id it is indexed by, already reduced to
    this layer's column by :func:`layer_state_access`.
    """

    conv_pool: BufferValue
    conv_row_id: TensorValue
    recurrent_pool: BufferValue
    recurrent_row_id: TensorValue
    verify: GatedDeltaVerifyAccess | None = None
    """The verify's ring, or ``None`` outside a ring verify."""


def layer_state_access(
    state: Sequence[RecurrentStateInputsPerDevice[TensorValue, BufferValue]],
    layer: int,
) -> list[GatedDeltaStateAccess]:
    """Selects one linear layer's state row out of each device's leaves.

    Call this where a layer's inputs are assembled, never from inside a
    block body: the linear layers share one compiled subgraph.

    Args:
        state: Per-device recurrent-state inputs, leaves in declaration order.
        layer: The layer's index among the linear-attention layers, which is
            the row of ``live_row_ids`` it owns.

    Returns:
        One access per device, in the order ``state`` came in.
    """

    def verify(
        inputs: RecurrentStateInputsPerDevice[TensorValue, BufferValue],
    ) -> GatedDeltaVerifyAccess | None:
        assert len(inputs.leaves) in (len(_LEAF_IDS), _VERIFY_LEAF + 1), (
            "expected the live leaves alone or with a ring, got"
            f" {len(inputs.leaves)} leaves"
        )
        if len(inputs.leaves) == len(_LEAF_IDS):
            return None
        leaf = inputs.leaves[_VERIFY_LEAF]
        return GatedDeltaVerifyAccess(
            pool=leaf.pool, row_id=leaf.live_row_id(layer)
        )

    return [
        GatedDeltaStateAccess(
            conv_pool=inputs.leaves[_CONV_LEAF].pool,
            conv_row_id=inputs.leaves[_CONV_LEAF].live_row_id(layer),
            recurrent_pool=inputs.leaves[_RECURRENT_LEAF].pool,
            recurrent_row_id=inputs.leaves[_RECURRENT_LEAF].live_row_id(layer),
            verify=verify(inputs),
        )
        for inputs in state
    ]


def linear_conv_dim(
    *,
    key_head_dim: int,
    num_key_heads: int,
    value_head_dim: int,
    num_value_heads: int,
) -> int:
    """Returns the width of the causal convolution, unsharded.

    The conv kernel indexes a Q and a K over every key head, then a V over
    every value head.
    """
    return key_head_dim * num_key_heads * 2 + value_head_dim * num_value_heads


def linear_state_regions(
    *,
    num_linear_layers: int,
    key_head_dim: int,
    num_key_heads: int,
    value_head_dim: int,
    num_value_heads: int,
    conv_kernel_dim: int,
    dtype: DType,
    num_devices: int,
    ring_len: int = 0,
) -> tuple[RecurrentStateRegion, ...]:
    """Returns the pool leaves one request's state occupies, per device.

    Sharded here: a device holds only its own slice of the heads.

    Args:
        ring_len: Record capacity of the speculative verify ring, or zero to
            declare no ring. One of :data:`COMPILED_RING_LENS`.

    Raises:
        ValueError: If ``ring_len`` is neither zero nor a compiled length.
    """
    if ring_len and ring_len not in COMPILED_RING_LENS:
        raise ValueError(
            f"ring_len must be zero or one of {COMPILED_RING_LENS}, got "
            f"{ring_len}"
        )
    conv_dim = (
        linear_conv_dim(
            key_head_dim=key_head_dim,
            num_key_heads=num_key_heads,
            value_head_dim=value_head_dim,
            num_value_heads=num_value_heads,
        )
        // num_devices
    )
    regions = (
        RecurrentStateRegion(
            leaf_id=CONV_LEAF_ID,
            num_layers=num_linear_layers,
            row_shape=(conv_dim, conv_kernel_dim - 1),
            dtype=dtype,
        ),
        RecurrentStateRegion(
            leaf_id=RECURRENT_LEAF_ID,
            num_layers=num_linear_layers,
            row_shape=(
                num_value_heads // num_devices,
                key_head_dim,
                value_head_dim,
            ),
            dtype=dtype,
        ),
    )
    assert tuple(region.leaf_id for region in regions) == _LEAF_IDS
    if not ring_len:
        return regions

    num_key_heads_per_device = num_key_heads // num_devices
    ring_row_records = num_key_heads_per_device * ring_len
    record_stride = _ring_record_stride(
        record_elements=key_head_dim
        + (num_value_heads // num_key_heads) * (value_head_dim + 1),
        bytes_per_element=num_linear_layers
        * ring_row_records
        * RING_DTYPE.size_in_bytes,
        page_multiple_of=math.lcm(
            *(region.bytes_per_page for region in regions)
        ),
    )
    ring = RecurrentStateRegion(
        leaf_id=RING_LEAF_ID,
        num_layers=num_linear_layers,
        row_shape=(num_key_heads_per_device, ring_len, record_stride),
        dtype=RING_DTYPE,
        scratch=True,
    )
    return (*regions, ring)


def _ring_record_stride(
    *, record_elements: int, bytes_per_element: int, page_multiple_of: int
) -> int:
    """Returns the padded stride of one ring record.

    A huge block is the least common multiple of every leaf's page, so the
    ring's page is padded to divide ``page_multiple_of``, the live leaves'
    least common multiple, and never enlarges it. The kernels lay a record
    out as ``gated_delta_ring_record_elements`` does, the raw key and then
    each value head's delta row and decay, which ``record_elements`` counts.

    Args:
        record_elements: Elements one record holds before padding.
        bytes_per_element: Page bytes one element of stride costs.
        page_multiple_of: Bytes the ring's page must divide.

    Returns:
        The smallest aligned stride whose page divides ``page_multiple_of``,
        or the aligned record itself when no stride that small does.
    """
    aligned = -(-record_elements // _RING_RECORD_ALIGN) * _RING_RECORD_ALIGN
    for stride in range(
        aligned,
        page_multiple_of // bytes_per_element + 1,
        _RING_RECORD_ALIGN,
    ):
        if page_multiple_of % (stride * bytes_per_element) == 0:
            return stride
    return aligned


def attn_cache(params: KVCacheParamInterface) -> KVCacheParams:
    """Returns the attention half of a Qwen3.5 cache.

    A cache with no linear-attention layers is already that leaf.
    """
    if isinstance(params, KVCacheParams):
        return params
    assert isinstance(params, MultiKVCacheParams), (
        "A Qwen3.5 cache is either an attention leaf or a tree holding one,"
        f" got {type(params).__name__}"
    )
    attn = params.children[ATTN_CACHE_KEY]
    assert isinstance(attn, KVCacheParams)
    return attn
