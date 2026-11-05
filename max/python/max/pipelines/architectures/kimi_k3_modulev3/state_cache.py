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
"""The shape of a Kimi K3 KDA state, and how a layer reaches it.

Copied from ``kimi_k3.state_cache`` (V2) rather than imported, so this
ModuleV3 port has no build dependency on the V2 package.

K3 splits the KDA heads across devices, so the conv and recurrent state split
with them: device ``i`` holds ``conv_dim / n`` channels and ``num_heads / n``
heads of every request's state. The cache declares that state as the ``state``
child of its tree, one pool per leaf and device with a row per (request, KDA
layer), and a graph reaches a layer's rows through the ``[num_layers, batch]``
row-id table each leaf carries (the pooled recurrent ABI, the same one the
GLM-5.3-Flash KDA and the Qwen3.5 Gated DeltaNet use).

Both the leaves and the graph's input types are sized from the geometry, so it
is derived here once, together with the per-layer access built on top of it:
this is the only reader of the order the leaves are declared in.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Generic, TypeVar

from max import tree
from max.dtype import DType
from max.graph import BufferValue, TensorValue
from max.nn._graph_ops.state_space.kimi_delta import kda_ring_record_elements
from max.nn.kv_cache import (
    KVCacheParamInterface,
    KVCacheParams,
    MultiKVCacheParams,
    RecurrentLeafInputs,
    RecurrentStateInputsPerDevice,
    RecurrentStateRegion,
)
from max.tensor import Tensor

from ..qwen3_5.state_cache import (
    COMPILED_RING_LENS,
    RING_DTYPE,
    ring_record_stride,
)

__all__ = [
    "ATTN_CACHE_KEY",
    "COMPILED_RING_LENS",
    "CONV_LEAF_ID",
    "RECURRENT_LEAF_ID",
    "RING_DTYPE",
    "STATE_CACHE_KEY",
    "VERIFY_RING_LEAF_ID",
    "KdaStateAccess",
    "LeafBuffer",
    "LeafValue",
    "attn_cache",
    "has_verify_ring",
    "kda_state_regions",
    "layer_row",
    "layer_state_access",
    "leaf_inputs",
    "verify_ring_leaf",
    "with_verify_ring",
]

ATTN_CACHE_KEY = "attn"
STATE_CACHE_KEY = "state"
"""The two children of a K3 cache tree."""

CONV_LEAF_ID = "linear_attn/conv"
RECURRENT_LEAF_ID = "linear_attn/recurrent"
"""The two leaves one KDA state spans.

Separate leaves because the kernels want each as its own uniformly-strided
tensor, and because the conv window and the recurrent state have different
dtypes. They are allocated, published and evicted together.
"""

VERIFY_RING_LEAF_ID = "linear_attn/ring"
"""The scratch leaf a speculative verify records the recurrence into.

Declared only by a speculative cache, after the live leaves. Each request
draws one block when it is admitted, holds it until release, and never
publishes it. See :func:`kda_state_regions`.
"""

_LEAF_IDS = (CONV_LEAF_ID, RECURRENT_LEAF_ID)
"""Declaration order of the leaves.

``RecurrentStateInputsPerDevice.leaves`` is indexed positionally, in this
order. :func:`kda_state_regions` asserts the two agree, so reordering the
regions cannot silently point a leaf at the other's pool. The verify ring,
when one is declared, follows the two.
"""

_CONV_LEAF = _LEAF_IDS.index(CONV_LEAF_ID)
_RECURRENT_LEAF = _LEAF_IDS.index(RECURRENT_LEAF_ID)
_RING_LEAF = len(_LEAF_IDS)
"""Where a speculative verify's ring follows the live leaves."""

_ALL_LEAF_IDS = (*_LEAF_IDS, VERIFY_RING_LEAF_ID)
"""Every leaf id this cache can declare, in flatten order."""

LeafValue = TypeVar("LeafValue", TensorValue, Tensor)
"""A leaf's row-id table: a graph value, or the eager tensor holding it."""

LeafBuffer = TypeVar("LeafBuffer", BufferValue, Tensor)
"""A leaf's pool: a graph buffer, or an eager tensor's mutable backing."""


def leaf_inputs(
    device_state: RecurrentStateInputsPerDevice[LeafValue, LeafBuffer],
    leaf_id: str,
) -> RecurrentLeafInputs[LeafValue, LeafBuffer]:
    """Returns one device's inputs for ``leaf_id``.

    The leaves are positional, so the id is resolved against the declaration
    order :func:`kda_state_regions` asserts.

    Raises:
        ValueError: If the cache declares no leaf of that name -- asking for
            the verify ring of a cache that declares none.
    """
    index = _ALL_LEAF_IDS.index(leaf_id)
    if index >= len(device_state.leaves):
        raise ValueError(
            f"this cache declares no {leaf_id!r} leaf: a verify ring needs"
            " a speculative build to declare it"
        )
    return device_state.leaves[index]


def has_verify_ring(
    device_state: RecurrentStateInputsPerDevice[LeafValue, LeafBuffer],
) -> bool:
    """Returns whether one device's state inputs carry the verify ring."""
    return len(device_state.leaves) > len(_LEAF_IDS)


def layer_row(
    leaf: RecurrentLeafInputs[LeafValue, LeafBuffer], layer: int
) -> LeafValue:
    """Returns ``layer``'s row of a leaf's ``[num_layers, batch]`` table.

    The graph-value helper asserts a graph value, which an eager realized
    table is not; such a table slices the same way.
    """
    if isinstance(leaf.live_row_ids, Tensor):
        return leaf.live_row_ids[layer]
    return leaf.live_row_id(layer)


@tree.dataclass(frozen=True)
class KdaStateAccess(Generic[LeafValue, LeafBuffer]):
    """One KDA layer's state access, on one device.

    Each pool travels with the row id it is indexed by, already reduced to this
    layer's row by :func:`layer_state_access`. The two row ids are separate
    because the two leaves are: a page of one names nothing in the other.

    The live state alone: the verify ring is the rollback's, reached only
    through the binding :meth:`.spec_state.KimiK3RecurrentState.capturing`
    sets on the layer, never through this access.
    """

    conv_pool: LeafBuffer
    """``[conv_rows, conv_dim / n, conv_kernel - 1]``, every layer's rows."""

    conv_row_id: LeafValue
    """``[batch_size]`` conv row this layer runs each sequence in."""

    recurrent_pool: LeafBuffer
    """``[recurrent_rows, num_heads / n, head_dim, head_dim]``, every
    layer's rows."""

    recurrent_row_id: LeafValue
    """``[batch_size]`` recurrent row this layer runs each sequence in."""


def verify_ring_leaf(
    inputs: RecurrentStateInputsPerDevice[LeafValue, LeafBuffer],
) -> RecurrentLeafInputs[LeafValue, LeafBuffer]:
    """Returns the device's verify ring leaf, asserting it is one.

    The leaves carry no names at this level, so the ring is told by its
    position and shape: the leaf past the live two, a rank-4 float32 pool
    whose record axis is one of the compiled ring lengths -- a live leaf
    read as the ring has a head-dim-wide axis there and is refused here
    rather than corrupting the fold.

    Raises:
        AssertionError: If the third leaf is not shaped like the ring, or
            the cache declares neither two live leaves nor three.
    """
    assert len(inputs.leaves) == _RING_LEAF + 1, (
        "expected the live leaves alone or with a verify ring, got"
        f" {len(inputs.leaves)} state leaves"
    )
    leaf = inputs.leaves[_RING_LEAF]
    assert leaf.pool.rank == 4, (
        f"the verify ring must be a rank-4 pool, got {leaf.pool.rank}"
    )
    assert leaf.pool.dtype == RING_DTYPE, (
        f"the verify ring must be stored as {RING_DTYPE}, got {leaf.pool.dtype}"
    )
    assert leaf.pool.shape[2] in COMPILED_RING_LENS, (
        "the third state leaf's record axis is"
        f" {leaf.pool.shape[2]}, not one of the compiled ring lengths"
        f" {COMPILED_RING_LENS}: it is not the verify ring"
    )
    return leaf


def layer_state_access(
    state: Sequence[RecurrentStateInputsPerDevice[LeafValue, LeafBuffer]],
    layer: int,
) -> list[KdaStateAccess[LeafValue, LeafBuffer]]:
    """Selects one KDA layer's state row out of each device's leaves.

    Call this where a layer's inputs are assembled, never from inside a block
    body. The KDA layer index is the position among the KDA layers, not the
    decoder index: reading the decoder index instead binds layer 0's state to
    layer 3, which is neither a shape error nor a crash, just a worse model.

    Args:
        state: Per-device recurrent-state inputs, leaves in declaration order.
        layer: The layer's index among the KDA layers, which is the row of
            ``live_row_ids`` it owns.

    Returns:
        One access per device, in the order ``state`` came in.
    """
    return [
        KdaStateAccess(
            conv_pool=inputs.leaves[_CONV_LEAF].pool,
            conv_row_id=layer_row(inputs.leaves[_CONV_LEAF], layer),
            recurrent_pool=inputs.leaves[_RECURRENT_LEAF].pool,
            recurrent_row_id=layer_row(inputs.leaves[_RECURRENT_LEAF], layer),
        )
        for inputs in state
    ]


def kda_state_regions(
    *,
    num_kda_layers: int,
    conv_dim: int,
    conv_kernel_size: int,
    num_heads: int,
    head_dim: int,
    conv_dtype: DType,
    recurrent_dtype: DType,
    num_devices: int,
    ring_len: int = 0,
) -> tuple[RecurrentStateRegion, ...]:
    """Returns the pool leaves one request's KDA state occupies, per device.

    Sharded here: a device holds only its own slice of the channels and heads.

    Args:
        num_kda_layers: KDA layers, i.e. ``len(config.kda_layer_indices)``.
        conv_dim: The FULL fused q/k/v conv width; split across the devices.
        conv_kernel_size: Causal conv kernel size; the window is one less.
        num_heads: The FULL KDA head count; split across the devices.
        head_dim: Per-head width, shared by both recurrent state axes.
        conv_dtype: Storage dtype of the conv window. The conv kernel requires
            its history and ragged input to agree.
        recurrent_dtype: Storage dtype of the recurrent state.
        num_devices: Devices the channels and heads are split across.
        ring_len: Record capacity of the speculative verify's ring, one of
            :data:`COMPILED_RING_LENS`, or zero to declare no ring; see
            :func:`with_verify_ring` for the row layout.

    Raises:
        ValueError: If the conv width or head count does not divide across the
            devices, which would leave a rank owning a fraction of a head, or
            if ``ring_len`` is neither zero nor a compiled ring length.
    """
    if conv_dim % num_devices or num_heads % num_devices:
        raise ValueError(
            f"Kimi K3 state pools: conv width {conv_dim} and head count"
            f" {num_heads} must both divide by the tensor-parallel degree"
            f" {num_devices}"
        )
    regions = (
        RecurrentStateRegion(
            leaf_id=CONV_LEAF_ID,
            num_layers=num_kda_layers,
            row_shape=(conv_dim // num_devices, conv_kernel_size - 1),
            dtype=conv_dtype,
        ),
        RecurrentStateRegion(
            leaf_id=RECURRENT_LEAF_ID,
            num_layers=num_kda_layers,
            row_shape=(num_heads // num_devices, head_dim, head_dim),
            dtype=recurrent_dtype,
        ),
    )
    assert tuple(region.leaf_id for region in regions) == _LEAF_IDS
    if not ring_len:
        return regions
    return with_verify_ring(regions, ring_len)


def with_verify_ring(
    regions: Sequence[RecurrentStateRegion], ring_len: int
) -> tuple[RecurrentStateRegion, ...]:
    """Appends the speculative verify's ring to the live state leaves.

    A ring row holds one record per value head and window position, laid out
    as :func:`~max.nn._graph_ops.state_space.kda_ring_record_elements` counts it.
    The record stride is padded so the ring's page divides the live leaves'
    least common multiple -- the pool's block -- so declaring the ring never
    widens it. The name and argument order mirror GLM-5.3-Flash's
    ``glm5_next.state_cache.with_verify_ring``.

    Args:
        regions: The live leaves :func:`kda_state_regions` declares, per
            device.
        ring_len: Records one request's ring holds, one of
            :data:`COMPILED_RING_LENS`.

    Raises:
        ValueError: If ``ring_len`` is not a compiled ring length.
    """
    if ring_len not in COMPILED_RING_LENS:
        raise ValueError(
            f"Kimi K3 ring_len must be one of {COMPILED_RING_LENS},"
            f" got {ring_len}"
        )
    (recurrent,) = (
        region for region in regions if region.leaf_id == RECURRENT_LEAF_ID
    )
    heads, key_head_dim, value_head_dim = recurrent.row_shape
    record_stride = ring_record_stride(
        record_elements=kda_ring_record_elements(key_head_dim, value_head_dim),
        bytes_per_element=recurrent.num_layers
        * heads
        * ring_len
        * RING_DTYPE.size_in_bytes,
        page_multiple_of=math.lcm(
            *(region.bytes_per_page for region in regions)
        ),
    )
    return (
        *regions,
        RecurrentStateRegion(
            leaf_id=VERIFY_RING_LEAF_ID,
            num_layers=recurrent.num_layers,
            row_shape=(heads, ring_len, record_stride),
            dtype=RING_DTYPE,
            scratch=True,
        ),
    )


def attn_cache(params: KVCacheParamInterface) -> KVCacheParams:
    """Returns the MLA half of a K3 cache.

    A cache with no KDA layers is already that leaf.
    """
    if isinstance(params, KVCacheParams):
        return params
    assert isinstance(params, MultiKVCacheParams), (
        "A K3 cache is either an attention leaf or a tree holding one, got"
        f" {type(params).__name__}"
    )
    child = params.children[ATTN_CACHE_KEY]
    assert isinstance(child, KVCacheParams)
    return child
