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
"""Tests the verify's shadow pools and their dense addressing.

The ring is not here: it is a scratch leaf of the state cache.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from max.driver import CPU, Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import BufferType, DeviceRef, Graph, TensorType, ops
from max.nn.kv_cache import RecurrentStateRegion
from max.pipelines.architectures.qwen3_5.model_config import Qwen3_5Config
from max.pipelines.architectures.qwen3_5.state_cache import (
    CONV_LEAF_ID,
    RECURRENT_LEAF_ID,
    RING_LEAF_ID,
    Qwen3_5SpecShadowPools,
    linear_state_regions,
    shadowed_leaf_ids,
    spec_shadow_bytes_per_request,
)
from max.pipelines.architectures.unified_mtp_qwen3_5.arch import (
    UnifiedMTPQwen3_5Config,
)
from max.pipelines.architectures.unified_mtp_qwen3_5.state_rollback import (
    snapshot_state_pools,
)
from max.pipelines.lib import PipelineConfig

NUM_LAYERS = 2
MAX_BATCH = 3
BATCH = 2


KEY_HEAD_DIM = 4
VALUE_HEAD_DIM = 4
NUM_HEADS = 2


CONV_KERNEL_DIM = 4
RING_LEN = 4

LIVE_ROWS = 13


POISON = np.float32(-7.75e18)


def _regions(
    ring_len: int = RING_LEN, num_devices: int = 1
) -> tuple[RecurrentStateRegion, ...]:
    return linear_state_regions(
        num_linear_layers=NUM_LAYERS,
        key_head_dim=KEY_HEAD_DIM,
        num_key_heads=NUM_HEADS,
        value_head_dim=VALUE_HEAD_DIM,
        num_value_heads=NUM_HEADS,
        conv_kernel_dim=CONV_KERNEL_DIM,
        dtype=DType.float32,
        num_devices=num_devices,
        ring_len=ring_len,
    )


def _pools(ring_len: int = RING_LEN) -> Qwen3_5SpecShadowPools:
    return Qwen3_5SpecShadowPools(
        _regions(ring_len), ring_len, MAX_BATCH, [Accelerator()]
    )


def _live_table() -> np.ndarray:
    """Returns a ``[NUM_LAYERS, BATCH]`` live table unlike the dense layout."""
    flat = (np.arange(BATCH * NUM_LAYERS) * 5 + 3) % LIVE_ROWS
    assert len(set(flat.tolist())) == flat.size
    assert not np.array_equal(flat, np.arange(flat.size))
    return flat.reshape(NUM_LAYERS, BATCH).astype(np.uint32)


def test_every_shadow_pool_is_max_batch_deep() -> None:
    """Checks every shadow pool is ``MAX_BATCH * NUM_LAYERS`` rows deep.

    Only the snapshot arm has a shadow.
    """
    pools = _pools(0)
    backed = [r for r in _regions(0) if r.leaf_id in shadowed_leaf_ids(0)]
    assert backed

    assert pools.rows == MAX_BATCH * NUM_LAYERS
    for region in backed:
        buffer = pools.pool(region.leaf_id, 0)
        assert tuple(buffer.shape) == (pools.rows, *region.row_shape)
        assert buffer.dtype == region.dtype


def test_the_ring_arm_backs_no_shadow() -> None:
    """Checks the ring arm has no shadow."""
    pools = _pools(RING_LEN)

    assert pools.shadow_pools(CONV_LEAF_ID) == []
    assert pools.shadow_pools(RECURRENT_LEAF_ID) == []


def test_the_snapshot_arm_backs_the_recurrent_shadow_alone() -> None:
    """Checks the snapshot arm has only the recurrent shadow."""
    pools = _pools(0)

    assert pools.shadow_pools(CONV_LEAF_ID) == []
    assert len(pools.shadow_pools(RECURRENT_LEAF_ID)) == 1


@pytest.mark.parametrize(
    ("rollback", "ring_len"), [("snapshot", 0), ("ring", RING_LEN)]
)
def test_the_state_cache_holds_the_ring_as_a_scratch_leaf(
    rollback: str, ring_len: int
) -> None:
    """Checks only the MTP ring arm adds a ring leaf to the state cache."""
    pipeline_config = cast(
        "PipelineConfig",
        SimpleNamespace(
            speculative=SimpleNamespace(
                recurrent_state_rollback=rollback, draft_width=RING_LEN - 1
            )
        ),
    )
    assert UnifiedMTPQwen3_5Config._verify_ring_len(pipeline_config) == ring_len
    assert Qwen3_5Config._verify_ring_len(pipeline_config) == 0
    scratch = [r.leaf_id for r in _regions(ring_len) if r.scratch]
    assert scratch == ([RING_LEAF_ID] if ring_len else [])


def _snapshot() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Snapshots ``BATCH`` requests' live recurrent rows into the shadow.

    Returns the shadow pool, the live pool and the table, all on the host.
    """
    device = Accelerator()
    gpu = DeviceRef.from_device(device)
    leaf = next(r for r in _regions(0) if r.leaf_id == RECURRENT_LEAF_ID)
    pools = Qwen3_5SpecShadowPools(_regions(0), 0, MAX_BATCH, [device])
    shadow = pools.pool(RECURRENT_LEAF_ID, 0)

    live_type = BufferType(
        DType.float32, [LIVE_ROWS, *leaf.row_shape], device=gpu
    )
    shadow_type = BufferType(
        DType.float32, [pools.rows, *leaf.row_shape], device=gpu
    )
    rows_type = TensorType(DType.uint32, [NUM_LAYERS, "batch_size"], device=gpu)

    with Graph(
        "snapshot", input_types=(live_type, shadow_type, rows_type)
    ) as graph:
        live_v, shadow_v, rows_v = graph.inputs
        span = ops.shape_to_tensor([rows_v.tensor.shape[1]])[0] * NUM_LAYERS
        snapshot_state_pools(
            [live_v.buffer], [shadow_v.buffer], [rows_v.tensor], span
        )
        graph.output()

    rng = np.random.default_rng(3)
    live = rng.standard_normal((LIVE_ROWS, *leaf.row_shape)).astype(np.float32)
    table = _live_table()

    shadow.inplace_copy_from(
        Buffer.from_numpy(
            np.full((pools.rows, *leaf.row_shape), POISON, dtype=np.float32)
        ).to(device)
    )
    session = InferenceSession(devices=[device])
    session.load(graph).execute(
        Buffer.from_numpy(live).to(device),
        shadow,
        Buffer.from_numpy(table).to(device),
    )
    return np.array(shadow.to(CPU()).to_numpy()), live, table


def test_the_snapshot_reads_the_rows_the_live_table_names() -> None:
    """Checks the snapshot copies live-table rows to dense shadow rows."""
    shadow, live, table = _snapshot()

    for request in range(BATCH):
        for layer in range(NUM_LAYERS):
            np.testing.assert_array_equal(
                shadow[layer * BATCH + request],
                live[table[layer, request]],
                err_msg=f"request {request} layer {layer} read the wrong row",
            )


def test_the_snapshot_leaves_the_rows_no_request_claimed_alone() -> None:
    """Checks the snapshot leaves unclaimed shadow rows unchanged."""
    shadow, _, _ = _snapshot()

    untouched = shadow[BATCH * NUM_LAYERS :]
    assert untouched.size
    np.testing.assert_array_equal(
        untouched, np.full(untouched.shape, POISON, dtype=np.float32)
    )


def test_the_priced_bytes_are_the_pools_that_get_allocated() -> None:
    """Checks the priced bytes match the allocated pools."""
    for ring_len in (0, RING_LEN):
        regions = _regions(ring_len)
        pools = _pools(ring_len)
        allocated = sum(
            int(np.prod(pools.pool(r.leaf_id, 0).shape)) * r.dtype.size_in_bytes
            for r in regions
            if r.leaf_id in shadowed_leaf_ids(ring_len)
        )
        assert allocated == MAX_BATCH * spec_shadow_bytes_per_request(
            regions, ring_len
        )
