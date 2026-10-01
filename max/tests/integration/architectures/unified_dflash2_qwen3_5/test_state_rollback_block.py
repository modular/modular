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

"""The Qwen3.5 state rollback, driven at a DFlash2 block rather than K = 3.

The Qwen3.5 MTP rollback is reused verbatim by the DFlash2 graph, so what
needs proving is that it is K-agnostic in fact and not only in argument. These
tests run the real state kernels over a verify window, on a ring as long as
the block, and check the live pools match a forward over the accepted prefix
alone, bit for bit.

The verify runs the conv with ``write_state=False`` and the replay writes the
window, so these tests also cover the conv rollback for both graphs. The
recurrence records into the ring and the fold applies the accepted records.
"""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import CPU, Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import BufferType, DeviceRef, Dim, Graph, TensorType, ops
from max.nn.state_space import (
    gated_delta_conv1d_fwd,
    gated_delta_recurrence_fwd,
    gated_delta_recurrence_verify_ring_fwd,
    verify_width_operand,
)
from max.pipelines.architectures.qwen3_5.layers.gated_deltanet import (
    GatedDeltaReplayInputs,
)
from max.pipelines.architectures.qwen3_5.state_cache import (
    RING_LEAF_ID,
    linear_state_regions,
    ring_len_for_window,
)
from max.pipelines.architectures.unified_mtp_qwen3_5.state_rollback import (
    accepted_lengths,
    accepted_row_plan,
    fold_state_pools,
    replay_conv_pools,
)

BLOCK = 8
"""DFlash2 verifies an anchor plus seven drafts in one window."""

NUM_DRAFTS = BLOCK - 1
# The recurrence kernel is only compiled for 128 x 128 heads, and conv_dim is
# the layout the projections produce: 2 x num_key_heads x key_head_dim for
# Q/K plus num_value_heads x value_head_dim for V.
KEY_HEAD_DIM = 128
VALUE_HEAD_DIM = 128
NUM_K_HEADS = 1
NUM_V_HEADS = 2
CONV_DIM = 2 * NUM_K_HEADS * KEY_HEAD_DIM + NUM_V_HEADS * VALUE_HEAD_DIM
CONV_KERNEL = 4
CARRY_FORWARD_BELOW = CONV_KERNEL - 1
"""Accepted lengths below this carry window slots from the pre-verify window."""

MAX_SLOTS = 4


def _ring_row_shape() -> tuple[int, ...]:
    """Returns the ring row the state cache declares for one block."""
    (ring,) = (
        region
        for region in linear_state_regions(
            num_linear_layers=1,
            key_head_dim=KEY_HEAD_DIM,
            num_key_heads=NUM_K_HEADS,
            value_head_dim=VALUE_HEAD_DIM,
            num_value_heads=NUM_V_HEADS,
            conv_kernel_dim=CONV_KERNEL,
            dtype=DType.float32,
            num_devices=1,
            ring_len=ring_len_for_window(BLOCK),
        )
        if region.leaf_id == RING_LEAF_ID
    )
    return ring.row_shape


RING_ROW_SHAPE = _ring_row_shape()


def _rand(rng: np.random.Generator, *shape: int) -> np.ndarray:
    return rng.standard_normal(shape).astype(np.float32)


def _run(
    accepted: list[int],
    *,
    replay_full_window: bool = False,
    verify_writes_conv_window: bool = False,
) -> dict[str, np.ndarray]:
    """Verifies a window, rolls back, and independently runs the prefix.

    Returns the live and reference pools. ``live`` is what the rollback
    produced; ``reference`` is what the forward kernels produce over a
    host-sliced accepted prefix, built without ``accepted_row_plan``, so the
    two agreeing is not a tautology.

    Args:
        accepted: Drafts each request accepted.
        replay_full_window: Roll back onto every verified row instead of the
            accepted prefix.
        verify_writes_conv_window: Let the verify write the conv window.
    """
    batch = len(accepted)
    rng = np.random.default_rng(7)
    total = batch * BLOCK
    qkv = _rand(rng, total, CONV_DIM)
    conv_weight = _rand(rng, CONV_DIM, CONV_KERNEL)
    decay = -np.abs(_rand(rng, total, NUM_V_HEADS))
    beta = np.abs(_rand(rng, total, NUM_V_HEADS))
    merged_offsets = np.arange(batch + 1, dtype=np.uint32) * BLOCK
    slots = np.arange(batch, dtype=np.uint32)
    init_conv = _rand(rng, MAX_SLOTS, CONV_DIM, CONV_KERNEL - 1)
    init_rec = _rand(rng, MAX_SLOTS, NUM_V_HEADS, KEY_HEAD_DIM, VALUE_HEAD_DIM)

    # The reference prefix, sliced on the host: rows [0, 1 + accepted) of each
    # request's window, which is what the step actually commits.
    keep = [1 + a for a in accepted]
    ref_rows = np.concatenate(
        [np.arange(b * BLOCK, b * BLOCK + keep[b]) for b in range(batch)]
    ).astype(np.int64)
    ref_offsets = np.concatenate([[0], np.cumsum(keep)]).astype(np.uint32)

    gpu = DeviceRef.GPU()
    conv_pool_type = BufferType(
        DType.float32, [MAX_SLOTS, CONV_DIM, CONV_KERNEL - 1], device=gpu
    )
    rec_pool_type = BufferType(
        DType.float32,
        [MAX_SLOTS, NUM_V_HEADS, KEY_HEAD_DIM, VALUE_HEAD_DIM],
        device=gpu,
    )
    types: list[TensorType | BufferType] = [
        TensorType(DType.float32, [total, CONV_DIM], device=gpu),
        TensorType(DType.float32, [CONV_DIM, CONV_KERNEL], device=gpu),
        TensorType(DType.float32, [total, NUM_V_HEADS], device=gpu),
        TensorType(DType.float32, [total, NUM_V_HEADS], device=gpu),
        # Symbolic, as the served graph declares it. A static length lets
        # fusion vectorize the plan's and the fold's shared offsets slice as
        # if it were aligned, which faults on a misaligned address.
        TensorType(DType.uint32, ["offsets_len"], device=gpu),
        TensorType(DType.int64, ["batch_size"], device=gpu),
        TensorType(DType.uint32, ["batch_size"], device=gpu),
        TensorType(DType.int64, [len(ref_rows)], device=gpu),
        TensorType(DType.uint32, [batch + 1], device=gpu),
        conv_pool_type,  # live conv
        rec_pool_type,  # live recurrent
        BufferType(DType.float32, [MAX_SLOTS, *RING_ROW_SHAPE], device=gpu),
        conv_pool_type,  # reference conv
        rec_pool_type,  # reference recurrent
    ]

    with Graph("dflash2_block_rollback", input_types=types) as graph:
        (
            qkv_v,
            conv_w,
            decay_v,
            beta_v,
            offsets_v,
            accepted_v,
            slots_v,
            ref_rows_v,
            ref_offsets_v,
        ) = (v.tensor for v in graph.inputs[:9])
        live_conv, live_rec, ring, ref_conv, ref_rec = (
            v.buffer for v in graph.inputs[9:]
        )

        # One linear layer here, so each row table is one layer deep, and
        # request ``r`` holds row ``r`` of every pool.
        live_rows = ops.unsqueeze(slots_v, 0)
        verify_width = verify_width_operand(NUM_DRAFTS)
        k_v = ops.constant(NUM_DRAFTS, DType.int64, device=gpu)

        # The recurrence reads the live pool and records into the ring.
        verify_conv = gated_delta_conv1d_fwd(
            qkv_input_ragged=qkv_v,
            conv_weight=conv_w,
            conv_state=live_conv,
            slot_idx=slots_v,
            input_row_offsets=offsets_v,
            write_state=verify_writes_conv_window,
        )
        gated_delta_recurrence_verify_ring_fwd(
            qkv_conv_output=ops.silu(verify_conv),
            decay_per_token=decay_v,
            beta_per_token=beta_v,
            recurrent_state=live_rec,
            ring=ring,
            slot_idx=slots_v,
            ring_slot_idx=slots_v,
            input_row_offsets=offsets_v,
            verify_width=verify_width,
        )

        rows, replay_offsets = accepted_row_plan(
            offsets_v, accepted_v, k_v, Dim(total), gpu
        )
        fold_accepted = accepted_lengths(offsets_v, accepted_v, k_v)
        if replay_full_window:
            rows = ops.range(
                start=0,
                stop=Dim(total),
                out_dim=Dim(total),
                device=gpu,
                dtype=DType.int64,
            )
            replay_offsets = offsets_v.cast(DType.int64)
            fold_accepted = accepted_v * 0 + BLOCK
        replay_conv_pools(
            [
                [
                    GatedDeltaReplayInputs(
                        qkv_v, conv_w, decay_v, beta_v, ops.silu(verify_conv)
                    )
                ]
            ],
            [live_conv],
            [live_rows],
            rows,
            replay_offsets,
            [],
            verify_width,
        )
        fold_state_pools(
            [live_rec],
            [live_rows],
            [ring],
            [live_rows],
            fold_accepted,
            [],
            verify_width,
        )

        # The independent reference: the same kernels over the host-sliced
        # accepted prefix, from a pool seeded with the same initial values.
        ref_conv_out = gated_delta_conv1d_fwd(
            qkv_input_ragged=ops.gather(qkv_v, ref_rows_v, axis=0),
            conv_weight=conv_w,
            conv_state=ref_conv,
            slot_idx=slots_v,
            input_row_offsets=ref_offsets_v,
        )
        gated_delta_recurrence_fwd(
            qkv_conv_output=ops.silu(ref_conv_out),
            decay_per_token=ops.gather(decay_v, ref_rows_v, axis=0),
            beta_per_token=ops.gather(beta_v, ref_rows_v, axis=0),
            recurrent_state=ref_rec,
            slot_idx=slots_v,
            input_row_offsets=ref_offsets_v,
        )
        graph.output()

    device = Accelerator()
    session = InferenceSession(devices=[device])
    model = session.load(graph)

    def pool(values: np.ndarray) -> Buffer:
        return Buffer.from_numpy(np.ascontiguousarray(values)).to(device)

    buffers = {
        "live_conv": pool(init_conv),
        "live_rec": pool(init_rec),
        "ring": pool(np.zeros((MAX_SLOTS, *RING_ROW_SHAPE), np.float32)),
        "ref_conv": pool(init_conv),
        "ref_rec": pool(init_rec),
    }
    model.execute(
        pool(qkv),
        pool(conv_weight),
        pool(decay),
        pool(beta),
        pool(merged_offsets),
        pool(np.array(accepted, dtype=np.int64)),
        pool(slots),
        pool(ref_rows),
        pool(ref_offsets),
        *buffers.values(),
    )
    return {
        name: np.array(buf.to(CPU()).to_numpy())
        for name, buf in buffers.items()
    }


def _assert_rolled_back(pools: dict[str, np.ndarray]) -> None:
    """Asserts both live pools match the reference replay exactly."""
    np.testing.assert_array_equal(pools["live_conv"], pools["ref_conv"])
    np.testing.assert_array_equal(pools["live_rec"], pools["ref_rec"])


@pytest.mark.parametrize("accepted", [0, 1, 2, 3, NUM_DRAFTS])
def test_the_block_rollback_lands_on_the_accepted_prefix(
    accepted: int,
) -> None:
    """Checks the rollback matches the reference at each accepted length.

    Accepted lengths 0 and 1 carry window slots forward from the pool, and 2
    is the boundary.
    """
    _assert_rolled_back(_run([accepted]))


def test_each_request_rolls_back_to_its_own_length() -> None:
    """Checks four requests on both sides of the carry-forward boundary."""
    _assert_rolled_back(_run([NUM_DRAFTS, 3, 1, 0]))


@pytest.mark.parametrize("accepted", [0, 1, 3, NUM_DRAFTS])
def test_a_verify_that_writes_the_conv_window_is_detectable(
    accepted: int,
) -> None:
    """Checks a verify that writes the conv window breaks the conv rollback.

    The conv window differs only below ``CARRY_FORWARD_BELOW``, since longer
    lengths rebuild it from input rows. The recurrent pool matches, since
    the ring records the verify's conv output rather than the conv replay's.
    """
    pools = _run([accepted], verify_writes_conv_window=True)
    np.testing.assert_array_equal(pools["live_rec"], pools["ref_rec"])
    keep = 1 + accepted
    if keep < CARRY_FORWARD_BELOW:
        assert not np.array_equal(pools["live_conv"], pools["ref_conv"])
    else:
        np.testing.assert_array_equal(
            pools["live_conv"],
            pools["ref_conv"],
        )


def test_rolling_back_onto_the_whole_window_is_detectable() -> None:
    """Checks keeping every verified row differs from the reference."""
    pools = _run([3], replay_full_window=True)
    assert not np.array_equal(pools["live_conv"], pools["ref_conv"])
    assert not np.array_equal(pools["live_rec"], pools["ref_rec"])
