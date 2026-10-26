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

__all__ = [
    "ATTN_CACHE_KEY",
    "CONV_LEAF_ID",
    "RECURRENT_LEAF_ID",
    "STATE_CACHE_KEY",
    "KdaStateAccess",
    "attn_cache",
    "kda_state_regions",
    "layer_state_access",
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

_LEAF_IDS = (CONV_LEAF_ID, RECURRENT_LEAF_ID)
"""Declaration order of the leaves.

``RecurrentStateInputsPerDevice.leaves`` is indexed positionally, in this
order. :func:`kda_state_regions` asserts the two agree, so reordering the
regions cannot silently point a leaf at the other's pool.
"""

_CONV_LEAF = _LEAF_IDS.index(CONV_LEAF_ID)
_RECURRENT_LEAF = _LEAF_IDS.index(RECURRENT_LEAF_ID)


@tree.dataclass(frozen=True)
class KdaStateAccess:
    """One KDA layer's state access, on one device.

    Each pool travels with the row id it is indexed by, already reduced to this
    layer's row by :func:`layer_state_access`. The two row ids are separate
    because the two leaves are: a page of one names nothing in the other.
    """

    conv_pool: BufferValue
    """``[conv_rows, conv_dim / n, conv_kernel - 1]``, every layer's rows."""

    conv_row_id: TensorValue
    """``[batch_size]`` conv row this layer runs each sequence in."""

    recurrent_pool: BufferValue
    """``[recurrent_rows, num_heads / n, head_dim, head_dim]``, every
    layer's rows."""

    recurrent_row_id: TensorValue
    """``[batch_size]`` recurrent row this layer runs each sequence in."""


def layer_state_access(
    state: Sequence[RecurrentStateInputsPerDevice[TensorValue, BufferValue]],
    layer: int,
) -> list[KdaStateAccess]:
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
            conv_row_id=inputs.leaves[_CONV_LEAF].live_row_id(layer),
            recurrent_pool=inputs.leaves[_RECURRENT_LEAF].pool,
            recurrent_row_id=inputs.leaves[_RECURRENT_LEAF].live_row_id(layer),
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

    Raises:
        ValueError: If the conv width or head count does not divide across the
            devices, which would leave a rank owning a fraction of a head.
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
    return regions


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
