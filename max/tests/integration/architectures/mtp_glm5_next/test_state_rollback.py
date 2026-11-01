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
"""Each GLM-5.3-Flash state leaf rolls back to its accepted prefix.

Every arm runs a verify over ``K + 1`` tokens per request, then the rollback
over the accepted count, and compares the live pool to a reference that ran the
plain forward over the accepted tokens alone. All three must match exactly.

- The KDA recurrent pool, through the verify ring and the fold. Needs the
  ``kda_verify_ring`` and ``kda_state_fold`` kernels.
- The KDA conv window, through Qwen3.5's conv replay over the KDA layer's
  captured inputs.
- The indexer tail ring, through the deferred seed and its replay.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from max.driver import CPU, Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import BufferType, DeviceRef, Dim, Graph, TensorType, ops
from max.nn.legacy.kernels import mla_kpool_seed_tail
from max.nn.legacy.state_space import (
    gated_delta_conv1d_fwd,
    gated_delta_conv1d_verify_fwd,
    kda_decode,
    kda_verify_ring,
    verify_width_operand,
)
from max.pipelines.architectures.deepseekV3_2.layers.indexer import (
    KpoolTailReplayInputs,
)
from max.pipelines.architectures.glm5_next.layers.kda import KdaReplayInputs
from max.pipelines.architectures.glm5_next.spec_state import (
    deferred_seed_offsets,
    fold_kda_state_pools,
    replay_tail_seeds,
)
from max.pipelines.architectures.glm5_next.state_cache import (
    VERIFY_RING_LEAF_ID,
    state_regions,
    with_verify_ring,
)
from max.pipelines.speculative.gated_delta_rollback import (
    accepted_lengths,
    accepted_row_plan,
    replay_conv_pools,
)

RING_LEN = 4
NUM_DRAFTS = RING_LEN - 1
NUM_LAYERS = 2

HEADS = 2
HEAD_DIM = 128
"""The head dim the KDA kernels are compiled for."""
CONV_KERNEL = 4
CONV_DIM = 3 * HEADS * HEAD_DIM

KPOOL = 4
INDEX_HEAD_DIM = 128

MAX_BATCH = 3
POOL_ROWS = MAX_BATCH * NUM_LAYERS + 5

POISON = np.float32(-7.75e18)


def _ring_row_shape() -> tuple[int, ...]:
    """Returns the ring row the speculative cache declares here."""
    regions = with_verify_ring(
        state_regions(
            num_kda_layers=NUM_LAYERS,
            num_heads=HEADS,
            head_dim=HEAD_DIM,
            conv_kernel_dim=CONV_KERNEL,
            num_sparse_layers=1,
            index_kpool=KPOOL,
            index_head_dim=INDEX_HEAD_DIM,
            dtype=DType.float32,
            num_devices=1,
        ),
        RING_LEN,
    )
    (ring,) = (r for r in regions if r.leaf_id == VERIFY_RING_LEAF_ID)
    return ring.row_shape


def _rows(batch: int, kind: int) -> np.ndarray:
    """Returns a ``[NUM_LAYERS, batch]`` row table that differs per ``kind``."""
    flat = (np.arange(batch * NUM_LAYERS) * (2 * kind + 1) + kind) % POOL_ROWS
    assert len(set(flat.tolist())) == batch * NUM_LAYERS
    return flat.reshape(NUM_LAYERS, batch).astype(np.uint32)


def _accepted_rows(
    seq_lengths: list[int], accepted: list[int], num_drafts: int
) -> tuple[np.ndarray, np.ndarray]:
    """Returns the rows a forward over the accepted tokens alone reads."""
    offsets = np.concatenate([[0], np.cumsum(seq_lengths)])
    keep = [
        n - num_drafts + a for a, n in zip(accepted, seq_lengths, strict=True)
    ]
    rows = np.concatenate(
        [np.arange(offsets[b], offsets[b] + keep[b]) for b in range(len(keep))]
    ).astype(np.int64)
    return rows, np.concatenate([[0], np.cumsum(keep)]).astype(np.uint32)


def _bfloat16_bits(values: np.ndarray) -> np.ndarray:
    """Returns float32 values truncated to bfloat16, as their uint16 bits."""
    return (values.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


def _host(value: np.ndarray) -> Buffer:
    """Stages a host array, reading uint16 as bfloat16 bits."""
    buffer = Buffer.from_numpy(np.ascontiguousarray(value))
    return buffer.view(DType.bfloat16) if value.dtype == np.uint16 else buffer


def _readback(buffer: Buffer) -> np.ndarray:
    host = buffer.to(CPU())
    if host.dtype == DType.bfloat16:
        host = host.view(DType.uint16)
    return np.array(host.to_numpy())


def _run(
    graph: Graph, inputs: list[np.ndarray], buffers: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Runs ``graph`` once and returns ``buffers`` as it left them.

    The graph's outputs are returned too, under ``output_<i>``.
    """
    device = Accelerator()
    model = InferenceSession(devices=[device]).load(graph)
    staged = {name: _host(value).to(device) for name, value in buffers.items()}
    outputs = model.execute(
        *(_host(value).to(device) for value in inputs), *staged.values()
    )
    result = {name: _readback(buf) for name, buf in staged.items()}
    for i, output in enumerate(outputs):
        assert isinstance(output, Buffer)
        result[f"output_{i}"] = _readback(output)
    return result


# ===--------------------------------------------------------------------=== #
# KDA recurrent: verify ring and fold
# ===--------------------------------------------------------------------=== #


def _compile_kda(
    seq_lengths: list[int],
    accepted: list[int],
    *,
    num_drafts: int = NUM_DRAFTS,
    rollback: bool = True,
) -> Callable[[], dict[str, np.ndarray]]:
    """Compiles the KDA verify, the fold, and the reference recurrence."""
    batch = len(seq_lengths)
    rng = np.random.default_rng(7)
    total = sum(seq_lengths)
    offsets = np.concatenate([[0], np.cumsum(seq_lengths)]).astype(np.uint32)
    ref_rows, ref_offsets = _accepted_rows(seq_lengths, accepted, num_drafts)

    head = (total, HEADS, HEAD_DIM)
    q, k, v = (rng.standard_normal(head).astype(np.float32) for _ in range(3))
    raw_gate = rng.standard_normal(head).astype(np.float32)
    beta = rng.standard_normal((total, HEADS)).astype(np.float32)
    a_log = rng.standard_normal(HEADS).astype(np.float32) * 0.1
    dt_bias = rng.standard_normal((HEADS, HEAD_DIM)).astype(np.float32) * 0.1
    pool_shape = (POOL_ROWS, HEADS, HEAD_DIM, HEAD_DIM)
    init = rng.standard_normal(pool_shape).astype(np.float32) * 0.1
    ring_shape = (POOL_ROWS, *_ring_row_shape())
    state_rows = _rows(batch, 0)
    ring_rows = _rows(batch, 1)

    gpu = DeviceRef.GPU()
    table = TensorType(DType.uint32, [NUM_LAYERS, "batch_size"], device=gpu)
    types: list[TensorType | BufferType] = [
        *(TensorType(DType.float32, head, device=gpu) for _ in range(4)),
        TensorType(DType.float32, [total, HEADS], device=gpu),
        TensorType(DType.float32, [HEADS], device=gpu),
        TensorType(DType.float32, [HEADS, HEAD_DIM], device=gpu),
        TensorType(DType.uint32, [batch + 1], device=gpu),
        TensorType(DType.int64, ["batch_size"], device=gpu),
        table,
        table,
        TensorType(DType.int64, [len(ref_rows)], device=gpu),
        TensorType(DType.uint32, [batch + 1], device=gpu),
        BufferType(DType.float32, pool_shape, device=gpu),
        BufferType(DType.float32, ring_shape, device=gpu),
        BufferType(DType.float32, pool_shape, device=gpu),
    ]
    with Graph("kda_ring_rollback", input_types=types) as graph:
        (
            q_v,
            k_v,
            v_v,
            gate_v,
            beta_v,
            a_log_v,
            dt_bias_v,
            offsets_v,
            accepted_v,
            state_rows_v,
            ring_rows_v,
            ref_rows_v,
            ref_offsets_v,
        ) = (value.tensor for value in graph.inputs[:13])
        live, ring, reference = (value.buffer for value in graph.inputs[13:])
        verify_width = verify_width_operand(num_drafts)
        cu_seqlens = offsets_v.cast(DType.int32)
        readouts = []
        for layer in range(NUM_LAYERS):
            readout = kda_verify_ring(
                q_v,
                k_v,
                v_v,
                gate_v,
                beta_v,
                a_log_v,
                dt_bias_v,
                cu_seqlens,
                live,
                state_rows_v[layer].cast(DType.int32),
                ring,
                ring_rows_v[layer],
                verify_width,
                output_dtype=DType.float32,
            )
            readouts.append(readout)
        if rollback:
            fold_kda_state_pools(
                [live],
                [state_rows_v],
                [ring],
                [ring_rows_v],
                accepted_lengths(
                    offsets_v,
                    accepted_v,
                    ops.constant(num_drafts, DType.int64, device=gpu),
                ),
                [offsets_v],
                [],
                verify_width,
            )

        reference_readouts = []
        for layer in range(NUM_LAYERS):
            reference_readout = kda_decode(
                ops.gather(q_v, ref_rows_v, axis=0),
                ops.gather(k_v, ref_rows_v, axis=0),
                ops.gather(v_v, ref_rows_v, axis=0),
                ops.gather(gate_v, ref_rows_v, axis=0),
                ops.gather(beta_v, ref_rows_v, axis=0),
                a_log_v,
                dt_bias_v,
                ref_offsets_v.cast(DType.int32),
                reference,
                state_rows_v[layer].cast(DType.int32),
                output_dtype=DType.float32,
                pin_fma=True,
            )
            reference_readouts.append(reference_readout)
        graph.output(readouts[0], reference_readouts[0])

    inputs: list[np.ndarray] = [
        q,
        k,
        v,
        raw_gate,
        beta,
        a_log,
        dt_bias,
        offsets,
        np.array(accepted, dtype=np.int64),
        state_rows,
        ring_rows,
        ref_rows,
        ref_offsets,
    ]

    def execute() -> dict[str, np.ndarray]:
        out = _run(
            graph,
            inputs,
            {
                "live": init,
                "ring": np.full(ring_shape, POISON, dtype=np.float32),
                "reference": init,
            },
        )
        out["init"] = init
        return out

    return execute


@pytest.mark.parametrize("accepted", [[0], [1], [NUM_DRAFTS]])
def test_the_fold_lands_the_kda_state_on_the_accepted_prefix(
    accepted: list[int],
) -> None:
    pools = _compile_kda([RING_LEN], accepted)()
    np.testing.assert_array_equal(pools["live"], pools["reference"])


def test_each_request_folds_its_own_kda_length() -> None:
    pools = _compile_kda([RING_LEN] * 3, [NUM_DRAFTS, 1, 0])()
    np.testing.assert_array_equal(pools["live"], pools["reference"])


def test_the_kda_verify_leaves_the_live_pool_alone() -> None:
    pools = _compile_kda([RING_LEN] * 2, [NUM_DRAFTS, 1], rollback=False)()
    np.testing.assert_array_equal(pools["live"], pools["init"])


@pytest.mark.parametrize("seq_lengths", [[1], [5], [1, 9]])
def test_a_committed_kda_window_lands_on_the_forward(
    seq_lengths: list[int],
) -> None:
    pools = _compile_kda(seq_lengths, [0] * len(seq_lengths), num_drafts=0)()
    np.testing.assert_array_equal(pools["live"], pools["reference"])
    np.testing.assert_array_equal(pools["output_0"], pools["output_1"])


# ===--------------------------------------------------------------------=== #
# KDA conv: replay over the captured verify inputs
# ===--------------------------------------------------------------------=== #


def _compile_conv(
    seq_lengths: list[int], accepted: list[int], *, num_drafts: int
) -> Callable[[], dict[str, np.ndarray]]:
    """Compiles the conv verify, the conv replay, and the reference conv."""
    batch = len(seq_lengths)
    rng = np.random.default_rng(3)
    total = sum(seq_lengths)
    offsets = np.concatenate([[0], np.cumsum(seq_lengths)]).astype(np.uint32)
    ref_rows, ref_offsets = _accepted_rows(seq_lengths, accepted, num_drafts)
    qkv = rng.standard_normal((total, CONV_DIM)).astype(np.float32)
    weight = rng.standard_normal((CONV_DIM, CONV_KERNEL)).astype(np.float32)
    pool_shape = (POOL_ROWS, CONV_DIM, CONV_KERNEL - 1)
    init = rng.standard_normal(pool_shape).astype(np.float32)
    rows = _rows(batch, 0)

    gpu = DeviceRef.GPU()
    types: list[TensorType | BufferType] = [
        TensorType(DType.float32, [total, CONV_DIM], device=gpu),
        TensorType(DType.float32, [CONV_DIM, CONV_KERNEL], device=gpu),
        TensorType(DType.uint32, [batch + 1], device=gpu),
        TensorType(DType.int64, ["batch_size"], device=gpu),
        TensorType(DType.uint32, [NUM_LAYERS, "batch_size"], device=gpu),
        TensorType(DType.int64, [len(ref_rows)], device=gpu),
        TensorType(DType.uint32, [batch + 1], device=gpu),
        BufferType(DType.float32, pool_shape, device=gpu),
        BufferType(DType.float32, pool_shape, device=gpu),
    ]
    with Graph("kda_conv_rollback", input_types=types) as graph:
        (
            qkv_v,
            weight_v,
            offsets_v,
            accepted_v,
            rows_v,
            ref_rows_v,
            ref_offsets_v,
        ) = (value.tensor for value in graph.inputs[:7])
        live, reference = (value.buffer for value in graph.inputs[7:])
        verify_width = verify_width_operand(num_drafts)
        captures: list[KdaReplayInputs] = []
        for layer in range(NUM_LAYERS):
            _, qkv_window = gated_delta_conv1d_verify_fwd(
                qkv_v,
                weight_v,
                live,
                rows_v[layer],
                offsets_v,
                verify_width,
                rollback=False,
            )
            captures.append(
                KdaReplayInputs(
                    qkv=qkv_v,
                    conv_weight=weight_v,
                    raw_gate=qkv_v,
                    beta_logits=qkv_v,
                    qkv_window=qkv_window,
                )
            )
        row_indices, replay_offsets = accepted_row_plan(
            offsets_v,
            accepted_v,
            ops.constant(num_drafts, DType.int64, device=gpu),
            qkv_window.shape[0],
            gpu,
        )
        replay_conv_pools(
            [captures],
            [live],
            [rows_v],
            row_indices,
            replay_offsets,
            [],
            verify_width,
        )
        for layer in range(NUM_LAYERS):
            gated_delta_conv1d_fwd(
                ops.gather(qkv_v, ref_rows_v, axis=0),
                weight_v,
                reference,
                rows_v[layer],
                ref_offsets_v,
            )
        graph.output()

    inputs: list[np.ndarray] = [
        qkv,
        weight,
        offsets,
        np.array(accepted, dtype=np.int64),
        rows,
        ref_rows,
        ref_offsets,
    ]
    return lambda: _run(graph, inputs, {"live": init, "reference": init})


@pytest.mark.parametrize("accepted", [[0], [2], [NUM_DRAFTS]])
def test_the_conv_replay_lands_on_the_accepted_prefix(
    accepted: list[int],
) -> None:
    pools = _compile_conv([RING_LEN], accepted, num_drafts=NUM_DRAFTS)()
    np.testing.assert_array_equal(pools["live"], pools["reference"])


def test_each_request_replays_its_own_conv_length() -> None:
    pools = _compile_conv(
        [RING_LEN] * 3, [NUM_DRAFTS, 1, 0], num_drafts=NUM_DRAFTS
    )()
    np.testing.assert_array_equal(pools["live"], pools["reference"])


def test_a_committed_conv_window_lands_on_the_forward() -> None:
    pools = _compile_conv([5, 2], [0, 0], num_drafts=0)()
    np.testing.assert_array_equal(pools["live"], pools["reference"])


# ===--------------------------------------------------------------------=== #
# Indexer tail ring: deferred seed and replay
# ===--------------------------------------------------------------------=== #


def _compile_tail(
    cache_lengths: list[int],
    accepted: list[int],
    *,
    num_drafts: int,
    defer: bool = True,
) -> Callable[[], dict[str, np.ndarray]]:
    """Compiles the verify's seed, its replay, and the reference seed.

    Every request verifies ``num_drafts + 1`` tokens from its cache length.
    The reference seeds only the accepted tokens into a ring that starts
    where the live one does. ``defer=False`` lets the verify seed every row,
    the bug the deferral exists to prevent.
    """
    batch = len(cache_lengths)
    rng = np.random.default_rng(5)
    seq_lengths = [num_drafts + 1] * batch
    total = sum(seq_lengths)
    offsets = np.concatenate([[0], np.cumsum(seq_lengths)]).astype(np.uint32)
    ref_rows, ref_offsets = _accepted_rows(seq_lengths, accepted, num_drafts)
    tail_shape = (POOL_ROWS, 2, KPOOL, INDEX_HEAD_DIM)
    key = _bfloat16_bits(rng.standard_normal((total, INDEX_HEAD_DIM)))
    gate = _bfloat16_bits(rng.standard_normal((total, INDEX_HEAD_DIM)))
    init = _bfloat16_bits(rng.standard_normal(tail_shape))
    slots = (np.arange(batch) * 3 + 1).astype(np.uint32)

    gpu = DeviceRef.GPU()
    types: list[TensorType | BufferType] = [
        TensorType(DType.bfloat16, [total, INDEX_HEAD_DIM], device=gpu),
        TensorType(DType.bfloat16, [total, INDEX_HEAD_DIM], device=gpu),
        TensorType(DType.uint32, [batch + 1], device=gpu),
        TensorType(DType.int64, ["batch_size"], device=gpu),
        TensorType(DType.uint32, ["batch_size"], device=gpu),
        TensorType(DType.uint32, ["batch_size"], device=gpu),
        TensorType(DType.int64, [len(ref_rows)], device=gpu),
        TensorType(DType.uint32, [batch + 1], device=gpu),
        BufferType(DType.bfloat16, tail_shape, device=gpu),
        BufferType(DType.bfloat16, tail_shape, device=gpu),
    ]
    with Graph("tail_ring_rollback", input_types=types) as graph:
        (
            key_v,
            gate_v,
            offsets_v,
            accepted_v,
            cache_lengths_v,
            slots_v,
            ref_rows_v,
            ref_offsets_v,
        ) = (value.tensor for value in graph.inputs[:8])
        live, reference = (value.buffer for value in graph.inputs[8:])

        (seed_offsets,) = (
            deferred_seed_offsets(Dim(num_drafts), [offsets_v])
            if defer
            else [offsets_v]
        )
        mla_kpool_seed_tail(
            live, key_v, gate_v, seed_offsets, cache_lengths_v, slots_v, KPOOL
        )
        num_drafts_v = ops.constant(num_drafts, DType.int64, device=gpu)
        row_indices, replay_offsets = accepted_row_plan(
            offsets_v, accepted_v, num_drafts_v, Dim(total), gpu
        )
        replay_tail_seeds(
            [
                [
                    KpoolTailReplayInputs(
                        k=key_v,
                        gate=gate_v,
                        tail=live,
                        slot_idx=slots_v,
                        cache_lengths=cache_lengths_v,
                        kpool=KPOOL,
                    )
                ]
            ],
            row_indices,
            replay_offsets,
            num_drafts_v,
            [],
        )
        mla_kpool_seed_tail(
            reference,
            ops.gather(key_v, ref_rows_v, axis=0),
            ops.gather(gate_v, ref_rows_v, axis=0),
            ref_offsets_v,
            cache_lengths_v,
            slots_v,
            KPOOL,
        )
        graph.output()

    inputs: list[np.ndarray] = [
        key,
        gate,
        offsets,
        np.array(accepted, dtype=np.int64),
        np.array(cache_lengths, dtype=np.uint32),
        slots,
        ref_rows,
        ref_offsets,
    ]
    return lambda: _run(graph, inputs, {"live": init, "reference": init})


@pytest.mark.parametrize(
    ("cache_length", "accepted"),
    [
        # Pending member at 8; the window 9..12 seeds 12 into its slot.
        (9, 0),
        (9, 2),
        (9, 3),
        # The window starts on a pool boundary.
        (8, 1),
        (10, 3),
    ],
)
def test_the_tail_ring_lands_on_the_accepted_prefix(
    cache_length: int, accepted: int
) -> None:
    rings = _compile_tail([cache_length], [accepted], num_drafts=NUM_DRAFTS)()
    np.testing.assert_array_equal(rings["live"], rings["reference"])


def test_each_request_seeds_its_own_tail() -> None:
    rings = _compile_tail([9, 10, 8], [0, 3, 1], num_drafts=NUM_DRAFTS)()
    np.testing.assert_array_equal(rings["live"], rings["reference"])


def test_a_verify_that_seeds_its_drafts_is_detectable() -> None:
    """Checks the reason for the deferral: a rejected draft at 12 overwrites
    the pending member at 8, which the next step closes its pool with."""
    rings = _compile_tail([9], [0], num_drafts=NUM_DRAFTS, defer=False)()
    assert not np.array_equal(rings["live"], rings["reference"])


def test_a_committed_tail_window_seeds_on_the_forward() -> None:
    """Checks a zero-width verify seeds the ring itself, and the replay
    leaves it there."""
    rings = _compile_tail([9, 6], [0, 0], num_drafts=0)()
    np.testing.assert_array_equal(rings["live"], rings["reference"])
