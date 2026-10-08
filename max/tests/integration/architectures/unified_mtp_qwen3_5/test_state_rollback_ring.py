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

"""Tests the ring rollback against the recurrence over the accepted tokens.

The live pool must match the reference exactly. The state and the ring take
different row tables, and ring rows no request claims are poisoned.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from max.driver import CPU, Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import (
    BufferType,
    DeviceRef,
    Graph,
    TensorType,
    ops,
)
from max.nn.state_space import (
    gated_delta_recurrence_fwd,
    gated_delta_recurrence_verify_ring_fwd,
    verify_width_operand,
)
from max.pipelines.architectures.qwen3_5.state_cache import (
    RING_LEAF_ID,
    linear_state_regions,
)
from max.pipelines.architectures.unified_mtp_qwen3_5.state_rollback import (
    accepted_lengths,
    fold_state_pools,
)

RING_LEN = 4
NUM_DRAFTS = RING_LEN - 1
NUM_LAYERS = 2

# The recurrence kernel is only compiled for 128 x 128 heads.
KEY_HEAD_DIM = 128
VALUE_HEAD_DIM = 128
NUM_K_HEADS = 1
NUM_V_HEADS = 2
CONV_DIM = 2 * NUM_K_HEADS * KEY_HEAD_DIM + NUM_V_HEADS * VALUE_HEAD_DIM

MAX_BATCH = 3
POOL_ROWS = MAX_BATCH * NUM_LAYERS + 5

POISON = np.float32(-7.75e18)


def _ring_row_shape() -> tuple[int, ...]:
    """Returns the ring row the state cache declares for this geometry."""
    (ring,) = (
        region
        for region in linear_state_regions(
            num_linear_layers=NUM_LAYERS,
            key_head_dim=KEY_HEAD_DIM,
            num_key_heads=NUM_K_HEADS,
            value_head_dim=VALUE_HEAD_DIM,
            num_value_heads=NUM_V_HEADS,
            conv_kernel_dim=4,
            dtype=DType.bfloat16,
            num_devices=1,
            ring_len=RING_LEN,
        )
        if region.leaf_id == RING_LEAF_ID
    )
    return ring.row_shape


RING_ROW_SHAPE = _ring_row_shape()
RECORD_ELEMENTS = KEY_HEAD_DIM + (NUM_V_HEADS // NUM_K_HEADS) * (
    VALUE_HEAD_DIM + 1
)


def _rows(batch: int, kind: int) -> np.ndarray:
    """Returns a ``[NUM_LAYERS, batch]`` row table that differs per ``kind``."""
    flat = (np.arange(batch * NUM_LAYERS) * (2 * kind + 1) + kind) % POOL_ROWS
    assert len(set(flat.tolist())) == batch * NUM_LAYERS, (
        "a leaf's rows must be distinct or two requests share state"
    )
    return flat.reshape(NUM_LAYERS, batch).astype(np.uint32)


def _compile(
    seq_lengths: list[int],
    accepted: list[int],
    *,
    num_drafts: int = NUM_DRAFTS,
    rollback_width: int | None = None,
    rollback: bool = True,
    ring_row_shape: tuple[int, ...] = RING_ROW_SHAPE,
) -> Callable[[], dict[str, np.ndarray]]:
    """Compiles a verify and the ring rollback, plus a reference recurrence.

    The verify's own ``qkv`` stands in for the conv output. ``num_drafts`` is
    the verify width the ops read, and ``rollback_width`` overrides it for
    the fold alone. With ``rollback=False`` only the verify runs.
    ``ring_row_shape`` overrides the ring row the state cache declares.

    Returns a function that runs the graph and returns ``live``, the ring
    arm's pool, and ``reference``, the pool after
    `gated_delta_recurrence_fwd` over the host-sliced accepted tokens, with
    the first layer's ``readout`` and ``reference_readout``.
    """
    batch = len(seq_lengths)
    rng = np.random.default_rng(11)
    total = sum(seq_lengths)
    offsets = np.concatenate([[0], np.cumsum(seq_lengths)]).astype(np.uint32)

    qkv = rng.standard_normal((total, CONV_DIM)).astype(np.float32)
    decay = np.exp(-np.abs(rng.standard_normal((total, NUM_V_HEADS)))).astype(
        np.float32
    )
    beta = np.abs(rng.standard_normal((total, NUM_V_HEADS))).astype(np.float32)
    init_rec = rng.standard_normal(
        (POOL_ROWS, NUM_V_HEADS, KEY_HEAD_DIM, VALUE_HEAD_DIM)
    ).astype(np.float32)

    state_rows = _rows(batch, 0)
    ring_rows = _rows(batch, 1)

    keep = [
        n - num_drafts + a for a, n in zip(accepted, seq_lengths, strict=True)
    ]
    assert all(0 <= k <= n for k, n in zip(keep, seq_lengths, strict=True))
    ref_row_indices = np.concatenate(
        [
            np.arange(offsets[b], offsets[b] + keep[b])
            for b in range(batch)
            if keep[b]
        ]
        or [np.zeros(0)]
    ).astype(np.int64)
    ref_offsets = np.concatenate([[0], np.cumsum(keep)]).astype(np.uint32)

    gpu = DeviceRef.GPU()
    rec_pool_type = BufferType(
        DType.float32,
        [POOL_ROWS, NUM_V_HEADS, KEY_HEAD_DIM, VALUE_HEAD_DIM],
        device=gpu,
    )
    row_table_type = TensorType(
        DType.uint32, [NUM_LAYERS, "batch_size"], device=gpu
    )
    types: list[TensorType | BufferType] = [
        TensorType(DType.float32, [total, CONV_DIM], device=gpu),
        TensorType(DType.float32, [total, NUM_V_HEADS], device=gpu),
        TensorType(DType.float32, [total, NUM_V_HEADS], device=gpu),
        TensorType(DType.uint32, [batch + 1], device=gpu),
        TensorType(DType.int64, ["batch_size"], device=gpu),
        row_table_type,  # state rows
        row_table_type,  # ring rows
        TensorType(DType.int64, [len(ref_row_indices)], device=gpu),
        TensorType(DType.uint32, [batch + 1], device=gpu),
        rec_pool_type,  # live recurrent
        BufferType(DType.float32, [POOL_ROWS, *ring_row_shape], device=gpu),
        rec_pool_type,  # reference recurrent
    ]

    with Graph("mtp_ring_rollback", input_types=types) as graph:
        (
            qkv_v,
            decay_v,
            beta_v,
            offsets_v,
            accepted_v,
            state_rows_v,
            ring_rows_v,
            ref_rows_v,
            ref_offsets_v,
        ) = (v.tensor for v in graph.inputs[:9])
        live_rec, ring, ref_rec = (v.buffer for v in graph.inputs[9:])
        # A rank-0 input buffer is widened to rank 1, so K is a constant.
        k_v = ops.constant(num_drafts, DType.int64, device=gpu)
        verify_width = verify_width_operand(num_drafts)

        readouts = []
        for layer in range(NUM_LAYERS):
            readouts.append(
                gated_delta_recurrence_verify_ring_fwd(
                    qkv_conv_output=qkv_v,
                    decay_per_token=decay_v,
                    beta_per_token=beta_v,
                    recurrent_state=live_rec,
                    ring=ring,
                    slot_idx=state_rows_v[layer],
                    ring_slot_idx=ring_rows_v[layer],
                    input_row_offsets=offsets_v,
                    verify_width=verify_width,
                )
            )

        if rollback:
            fold_state_pools(
                [live_rec],
                [state_rows_v],
                [ring],
                [ring_rows_v],
                accepted_lengths(offsets_v, accepted_v, k_v),
                [],
                verify_width
                if rollback_width is None
                else verify_width_operand(rollback_width),
            )

        reference_readouts = []
        for layer in range(NUM_LAYERS):
            reference_readouts.append(
                gated_delta_recurrence_fwd(
                    qkv_conv_output=ops.gather(qkv_v, ref_rows_v, axis=0),
                    decay_per_token=ops.gather(decay_v, ref_rows_v, axis=0),
                    beta_per_token=ops.gather(beta_v, ref_rows_v, axis=0),
                    recurrent_state=ref_rec,
                    slot_idx=state_rows_v[layer],
                    input_row_offsets=ref_offsets_v,
                )
            )
        graph.output(readouts[0], reference_readouts[0])

    device = Accelerator()
    session = InferenceSession(devices=[device])
    model = session.load(graph)

    def execute() -> dict[str, np.ndarray]:
        def buf(values: np.ndarray) -> Buffer:
            return Buffer.from_numpy(np.ascontiguousarray(values)).to(device)

        def poisoned(shape: tuple[int, ...]) -> np.ndarray:
            return np.full(shape, POISON, dtype=np.float32)

        buffers = {
            "live": buf(init_rec),
            "ring": buf(poisoned((POOL_ROWS, *ring_row_shape))),
            "reference": buf(init_rec),
        }
        readout, reference_readout = model.execute(
            buf(qkv),
            buf(decay),
            buf(beta),
            buf(offsets),
            buf(np.array(accepted, dtype=np.int64)),
            buf(state_rows),
            buf(ring_rows),
            buf(ref_row_indices),
            buf(ref_offsets),
            *buffers.values(),
        )
        out = {
            name: np.array(b.to(CPU()).to_numpy())
            for name, b in buffers.items()
        }
        out["readout"] = np.array(readout.to(CPU()).to_numpy())
        out["reference_readout"] = np.array(
            reference_readout.to(CPU()).to_numpy()
        )
        out["init"] = init_rec
        out["ring_rows"] = ring_rows
        return out

    return execute


@pytest.mark.parametrize("accepted", [[0], [1], [NUM_DRAFTS]])
def test_the_fold_lands_on_the_accepted_prefix(accepted: list[int]) -> None:
    """Checks the fold matches the reference at each acceptance count."""
    pools = _compile([RING_LEN], accepted)()
    np.testing.assert_array_equal(pools["live"], pools["reference"])


def test_each_request_folds_its_own_length() -> None:
    """Checks requests accepting 3, 1 and 0 drafts in one batch."""
    pools = _compile([RING_LEN] * 3, [NUM_DRAFTS, 1, 0])()
    np.testing.assert_array_equal(pools["live"], pools["reference"])


def test_the_verify_leaves_the_live_pool_alone() -> None:
    """Checks the ring verify leaves the live pool unchanged."""
    pools = _compile([RING_LEN] * 2, [NUM_DRAFTS, 1], rollback=False)()
    np.testing.assert_array_equal(pools["live"], pools["init"])


def test_a_stale_fold_length_is_detectable() -> None:
    """Checks folding a different accepted count changes the result."""
    # Compile both before running either: recording stops at the first execute.
    full = _compile([RING_LEN], [NUM_DRAFTS])
    partial = _compile([RING_LEN], [1])
    assert not np.array_equal(full()["live"], partial()["reference"])


def test_the_ring_is_addressed_by_its_own_row_table() -> None:
    """Checks the ring fills only the records of the rows its table claims."""
    pools = _compile([RING_LEN] * 3, [NUM_DRAFTS, 2, 1])()
    ring = pools["ring"]
    claimed = pools["ring_rows"].reshape(-1)
    unclaimed = np.setdiff1d(np.arange(POOL_ROWS), claimed)
    assert not np.any(ring[claimed][..., :RECORD_ELEMENTS] == POISON), (
        "the ring left a claimed record unwritten"
    )
    assert np.all(ring[claimed][..., RECORD_ELEMENTS:] == POISON), (
        "the ring wrote a record's padding"
    )
    assert np.all(ring[unclaimed] == POISON), (
        "the ring wrote a row no request claimed"
    )


@pytest.mark.parametrize(
    "seq_lengths", [[1], [2], [3], [4], [5], [64], [1, 64]]
)
def test_a_committed_window_lands_on_the_forward(
    seq_lengths: list[int],
) -> None:
    """Checks a zero-width verify is the forward, with no ring or fold."""
    pools = _compile(seq_lengths, [0] * len(seq_lengths), num_drafts=0)()
    np.testing.assert_array_equal(pools["live"], pools["reference"])
    np.testing.assert_array_equal(pools["readout"], pools["reference_readout"])
    assert np.all(pools["ring"] == POISON), "the ring was written at width 0"


@pytest.mark.parametrize("length", [1, 2, 3, RING_LEN])
def test_rolling_back_a_committed_window_is_detectable(length: int) -> None:
    """Checks a fold that ignores the zero width advances the state twice."""
    pools = _compile([length], [0], num_drafts=0, rollback_width=NUM_DRAFTS)()
    assert not np.array_equal(pools["live"], pools["reference"])


def test_a_window_longer_than_the_ring_is_refused() -> None:
    """Checks a verify width the ring cannot hold raises."""
    with pytest.raises(Exception, match="needs a ring of at least"):
        _compile([RING_LEN + 1], [0], num_drafts=RING_LEN)()


def test_the_fold_skips_at_width_zero() -> None:
    """Checks a fold at width zero leaves the pool the verify left."""
    pools = _compile([RING_LEN], [NUM_DRAFTS], rollback_width=0)()
    np.testing.assert_array_equal(pools["live"], pools["init"])


def test_a_ring_record_too_narrow_is_refused() -> None:
    """Checks a ring whose records cannot hold a GQA group raises."""
    narrow = (*RING_ROW_SHAPE[:-1], RECORD_ELEMENTS - 1)
    with pytest.raises(Exception, match=r"needs \d+ elements"):
        _compile([RING_LEN], [NUM_DRAFTS], ring_row_shape=narrow)()
