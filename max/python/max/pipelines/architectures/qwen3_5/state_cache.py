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

_LEAF_IDS = (CONV_LEAF_ID, RECURRENT_LEAF_ID)
"""Region declaration order, which ``leaves`` is indexed by.

:func:`linear_state_regions` checks what it returns against this, so the two
indices below cannot drift into silently swapping conv and recurrent state.
"""

_CONV_LEAF = _LEAF_IDS.index(CONV_LEAF_ID)
_RECURRENT_LEAF = _LEAF_IDS.index(RECURRENT_LEAF_ID)


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
    return [
        GatedDeltaStateAccess(
            conv_pool=inputs.leaves[_CONV_LEAF].pool,
            conv_row_id=inputs.leaves[_CONV_LEAF].live_row_id(layer),
            recurrent_pool=inputs.leaves[_RECURRENT_LEAF].pool,
            recurrent_row_id=inputs.leaves[_RECURRENT_LEAF].live_row_id(layer),
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
) -> tuple[RecurrentStateRegion, ...]:
    """Returns the pool leaves one request's state occupies, per device.

    Sharded here: a device holds only its own slice of the heads.
    """
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
    return regions


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
