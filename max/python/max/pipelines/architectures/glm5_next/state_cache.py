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
"""The shape of a GLM-5.3-Flash recurrent state, as cache configuration.

Both the pool's leaves and the graph's input types are sized from this
geometry, so it is derived here once.

The 34 KDA layers keep two state kinds per layer, as Qwen3.5's Gated DeltaNet
does. The sparse-MLA layers keep a third: the DSA indexer's k-pool tail ring,
over a different layer set at a different dtype. The pool takes an arbitrary
number of leaves, each with its own layer count, row shape and dtype, so a
third kind needs nothing the contract does not already have.

The ring rides along for the row allocator and the copy-forward, not for the
reuse. A checkpoint copies the block a forward ran in into its successor, and a
prefix hit copies a published block into the block the request resumes in, so
the ring lands wherever the KDA state lands, off one ``RequestID -> row`` map.
Two maps over one request set can drift apart.

What the ring cannot earn is a hit of its own. A group checkpoints at page
boundaries, ``page_size`` is already required to be a multiple of
``index_kpool`` (see :meth:`Glm5NextConfig.construct_kv_params`), and
``mla_kpool_seed_tail`` stashes nothing when a prefix ends on a pool boundary.
So the ring holds no live token at any boundary a checkpoint can fall on, and
what the group publishes for it is dead bytes.

That it costs nothing to carry them is why the ring's layer set is the *indexer
cache's* rather than :attr:`Glm5NextConfig.sparse_attention_layers`. A pool's
block is the least common multiple of its leaves' pages, so a leaf with a
coprime layer count multiplies it -- and a ring sharing the indexer's layer
count has a page that divides the indexer's own, however that count factors.
"""

from __future__ import annotations

from typing import Any

from max.dtype import DType
from max.nn.kv_cache import RecurrentStateRegion

MLA_CACHE_KEY = "mla"
INDEXER_CACHE_KEY = "indexer"
STATE_CACHE_KEY = "state"
"""The three children of a GLM-5.3-Flash cache tree."""

CONV_LEAF_ID = "linear_attn/conv"
RECURRENT_LEAF_ID = "linear_attn/recurrent"
TAIL_RING_LEAF_ID = "sparse_attn/index_tail"
"""The three leaves one request's state spans.

Separate leaves because the kernels want each as its own uniformly-strided
tensor, and because the ring runs over the sparse layers rather than the KDA
ones. They are allocated, published and evicted together.
"""

STATE_LEAF_ORDER = (CONV_LEAF_ID, RECURRENT_LEAF_ID, TAIL_RING_LEAF_ID)
"""Positions within ``RecurrentStateInputsPerDevice.leaves``.

That tuple is indexed positionally, in the order the regions below are
declared. ``linear_state_regions`` asserts the two agree, so reordering the
regions cannot silently point a leaf at another leaf's pool.
"""


def leaf_inputs(device: Any, leaf_id: str) -> Any:
    """Returns one device's inputs for ``leaf_id``."""
    return device.leaves[STATE_LEAF_ORDER.index(leaf_id)]


TAIL_RING_DTYPE = DType.bfloat16
"""Dtype of the k-pool tail ring.

Not ``state_dtype``, which is the KDA pools' float32. The ring holds indexer
keys and their compression-gate scores, and the gate and its position table are
bfloat16 and unquantized in the checkpoint whatever the FP8 map says -- see
``deepseekV3_2.layers.indexer``. The tail kernel takes one dtype for the ring,
the keys and the gate together, so this follows the gate.
"""


def kda_conv_dim(*, num_heads: int, head_dim: int) -> int:
    """Returns the width of the causal convolution, unsharded.

    q, k and v share one depthwise convolution, so the width is three times
    one projection's.
    """
    return 3 * num_heads * head_dim


def state_regions(
    *,
    num_kda_layers: int,
    num_heads: int,
    head_dim: int,
    conv_kernel_dim: int,
    num_sparse_layers: int,
    index_kpool: int,
    index_head_dim: int,
    dtype: DType,
    num_devices: int,
) -> tuple[RecurrentStateRegion, ...]:
    """Returns the pool leaves one request's state occupies, per device.

    The two KDA leaves are sharded here: a device holds only its own slice of
    the heads, and because the split is by head *within* each of q, k and v,
    the conv width that falls out is exactly what the layer shard declares --
    pinned against it by test. The ring is not sharded; the indexer scores one
    key per token rather than one per head, so every device holds the same
    ring.

    Args:
        num_kda_layers: KDA layers, i.e. ``len(config.kda_layers)``.
        num_heads: KDA heads across all devices.
        head_dim: Per-head width, shared by both KDA state axes.
        conv_kernel_dim: Causal conv kernel size; the window is one less.
        num_sparse_layers: Sparse-MLA layers, i.e.
            ``len(config.sparse_attention_layers)`` -- the ring's layer set,
            not the KDA one.
        index_kpool: Tokens per indexer scoring pool.
        index_head_dim: Indexer key width.
        dtype: Storage dtype of the KDA leaves. The ring follows
            :data:`TAIL_RING_DTYPE` instead.
        num_devices: Devices the heads are split across.

    Raises:
        ValueError: If the head count does not divide across the devices, which
            would leave a rank owning a fraction of a head.
    """
    if num_heads % num_devices:
        raise ValueError(
            f"linear_num_heads ({num_heads}) must be divisible by the device "
            f"count ({num_devices})."
        )
    heads = num_heads // num_devices
    regions = (
        RecurrentStateRegion(
            leaf_id=CONV_LEAF_ID,
            num_layers=num_kda_layers,
            row_shape=(
                kda_conv_dim(num_heads=heads, head_dim=head_dim),
                conv_kernel_dim - 1,
            ),
            dtype=dtype,
        ),
        RecurrentStateRegion(
            leaf_id=RECURRENT_LEAF_ID,
            num_layers=num_kda_layers,
            row_shape=(heads, head_dim, head_dim),
            dtype=dtype,
        ),
        RecurrentStateRegion(
            leaf_id=TAIL_RING_LEAF_ID,
            num_layers=num_sparse_layers,
            # Index 0 holds keys, index 1 holds gate scores.
            row_shape=(2, index_kpool, index_head_dim),
            dtype=TAIL_RING_DTYPE,
        ),
    )
    assert tuple(r.leaf_id for r in regions) == STATE_LEAF_ORDER, (
        "STATE_LEAF_ORDER must mirror the declared region order"
    )
    return regions
