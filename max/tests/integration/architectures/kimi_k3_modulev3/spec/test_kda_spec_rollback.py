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
"""Each Kimi K3 (ModuleV3) KDA state leaf rolls back to its accepted prefix.

The GLM-5.3-Flash rollback design (``mtp_glm5_next/test_state_rollback.py``),
on K3's own leaves and gate plumbing: every arm runs the verify over ``K + 1``
rows per request, then the rollback over the accepted count, and the live
pools must match a reference that ran the plain ``kda_decode`` /
conv forward over the accepted rows alone, bit for bit -- the verify must
be bit-identical to the arm non-spec decoding runs. That reference decodes
one token per request per call -- the arm non-spec decode itself runs; on
gfx950 a multi-row ``kda_decode`` call takes the chunk pipeline the op
runs for prefill-shaped batches, whose rounding is not the decode arm's.

- The KDA recurrent pool, through the verify ring and the fold. The reference
  decodes with ``pin_fma=True``, the arm the pinned verify and fold kernels
  reproduce; K3's own decode arm has to pin the same way for the served path
  to match, as GLM's does.
- The KDA conv window, through the conv replay over the verify's captured
  input. The pool is K3's own bfloat16 ``ssm_dtype`` leaf, fed float32
  working values the way ``kimi_short_conv_qkv`` casts them.
- K3's own decode arm: ``_kda_decode`` must spell its multiply-adds exactly
  like the pinned legacy ``kda_decode``, or the state a verify folds
  lands a ULP off the state non-spec decoding leaves.
- The verify ring leaf the state cache declares: ``with_verify_ring``
  appended to :func:`kda_state_regions`, holding the ring the two ops above
  consume.

The sabotage parameters prove the comparisons can fail: a fold past the
accepted count, and the fold consuming the ring at the live rows and writing
the live pool at the ring rows -- the shape-valid form of a live/ring mixup,
since the ops reject literally swapped buffers at build time.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, NamedTuple

import numpy as np
import pytest
from max.driver import CPU, Accelerator, Buffer, accelerator_architecture_name
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import (
    BufferType,
    DeviceRef,
    Graph,
    TensorType,
    TensorValue,
    ops,
)
from max.nn._graph_ops.state_space import (
    gated_delta_conv1d_fwd,
    gated_delta_conv1d_verify_fwd,
    kda_decode,
    kda_ring_record_elements,
    kda_state_fold,
    kda_verify_ring,
    verify_width_operand,
)
from max.nn.kv_cache import RecurrentStateRegion
from max.pipelines.architectures.kimi_k3_modulev3.layers.kda_attention import (
    kda_decode as k3_kda_decode,
)
from max.pipelines.architectures.kimi_k3_modulev3.state_cache import (
    CONV_LEAF_ID,
    RECURRENT_LEAF_ID,
    VERIFY_RING_LEAF_ID,
    kda_state_regions,
    with_verify_ring,
)
from max.pipelines.speculative.gated_delta_rollback import (
    accepted_lengths,
    accepted_row_plan,
    replay_conv_pools,
)

KdaGateMode = Literal["original", "safe"]


NUM_LAYERS = 2
HEADS = 2
HEAD_DIM = 128
"""The head dim the KDA kernels are compiled for, and K3's own."""
CONV_KERNEL = 4
CONV_DIM = 3 * HEADS * HEAD_DIM
MAX_BATCH = 4
POOL_ROWS = MAX_BATCH * NUM_LAYERS + 5

RING_LENS = {1: 2, 7: 8}
"""Draft width K -> the ring length the verify records into, of the lengths
the ring kernels are compiled for (2, 4, 8). K3 drafts seven tokens."""
POISON = np.float32(-7.75e18)


def _regions() -> tuple[RecurrentStateRegion, ...]:
    """Returns K3's live state leaves, at the test's geometry."""
    return kda_state_regions(
        num_kda_layers=NUM_LAYERS,
        conv_dim=CONV_DIM,
        conv_kernel_size=CONV_KERNEL,
        num_heads=HEADS,
        head_dim=HEAD_DIM,
        conv_dtype=DType.bfloat16,
        recurrent_dtype=DType.float32,
        num_devices=1,
    )


def _ring_row_shape(ring_len: int) -> tuple[int, ...]:
    """Returns the ring row the K3 cache declares for ``ring_len``."""
    regions = with_verify_ring(_regions(), ring_len)
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


class _ConvCapture(NamedTuple):
    """The conv inputs a K3 verify arm records for the replay."""

    conv_weight: TensorValue
    qkv_window: TensorValue


# ===--------------------------------------------------------------------=== #
# The verify ring leaf the state cache must declare
# ===--------------------------------------------------------------------=== #


def test_the_cache_declares_the_verify_ring_leaf() -> None:
    """The ring rides the live leaves: scratch, float32, a row per layer
    per request, holding a record per head and window position."""
    assert VERIFY_RING_LEAF_ID == "linear_attn/ring"
    regions = with_verify_ring(_regions(), RING_LENS[7])
    assert [r.leaf_id for r in regions] == [
        CONV_LEAF_ID,
        RECURRENT_LEAF_ID,
        VERIFY_RING_LEAF_ID,
    ], "the ring must be appended, or the leaf order the row ids read breaks"
    (ring,) = (r for r in regions if r.leaf_id == VERIFY_RING_LEAF_ID)
    assert ring.scratch, "the ring is never published"
    assert ring.dtype == DType.float32, "the fold is bit-exact in float32"
    assert ring.num_layers == NUM_LAYERS
    heads, ring_len, record_stride = ring.row_shape
    assert (heads, ring_len) == (HEADS, RING_LENS[7])
    assert record_stride >= kda_ring_record_elements(HEAD_DIM, HEAD_DIM)


def test_the_cache_refuses_a_ring_len_it_has_not_compiled() -> None:
    """Ring length 16 is out of scope: no kernel is compiled for it."""
    with pytest.raises(ValueError, match="ring"):
        with_verify_ring(_regions(), 16)


# ===--------------------------------------------------------------------=== #
# KDA recurrent: verify ring and fold
# ===--------------------------------------------------------------------=== #


def _compile_kda(
    seq_lengths: list[int],
    accepted: list[int],
    *,
    num_drafts: int = 7,
    gate_mode: KdaGateMode = "safe",
    rollback: bool = True,
    sabotage: str | None = None,
) -> Callable[[], dict[str, np.ndarray]]:
    """Compiles the KDA verify, the fold, and the reference recurrence.

    K3's own gate plumbing feeds the kernels: ``safe`` pre-activates the
    gate and zeroes the per-head parameters, ``original`` passes K3's
    ``A_log[:, 0]`` slice and ``dt_bias`` raw, both as the layer builds them.
    """
    assert sabotage in (None, "accepted+1", "swap pools"), sabotage
    batch = len(seq_lengths)
    ring_len = RING_LENS[num_drafts] if num_drafts else RING_LENS[1]
    rng = np.random.default_rng(7)
    total = sum(seq_lengths)
    offsets = np.concatenate([[0], np.cumsum(seq_lengths)]).astype(np.uint32)
    ref_rows, ref_offsets = _accepted_rows(seq_lengths, accepted, num_drafts)
    # Rows each request keeps, host-side, so the reference below can step
    # the batch one token per call the way non-spec decode runs.
    keep = [
        n - num_drafts + a for a, n in zip(accepted, seq_lengths, strict=True)
    ]
    # One decode step per token position: which requests still advance,
    # and their window rows at that position.
    steps = [
        (
            [b for b in range(batch) if keep[b] > t],
            [int(offsets[b]) + t for b in range(batch) if keep[b] > t],
        )
        for t in range(max(keep))
    ]
    # The step-major position of every request-major reference row, so the
    # per-step readouts can be gathered back into ``ref_rows``' order.
    ref_step_major: list[int] = []
    step_position: dict[tuple[int, int], int] = {}
    position = 0
    for t, (members, _) in enumerate(steps):
        for i, b in enumerate(members):
            step_position[(b, t)] = position + i
        position += len(members)
    for b in range(batch):
        ref_step_major += [step_position[(b, t)] for t in range(keep[b])]

    head = (total, HEADS, HEAD_DIM)
    q, k, v = (_bfloat16_bits(rng.standard_normal(head)) for _ in range(3))
    raw_gate = rng.standard_normal(head).astype(np.float32)
    beta_logits = rng.standard_normal((total, HEADS)).astype(np.float32)
    a_log = rng.standard_normal((HEADS, HEAD_DIM)).astype(np.float32) * 0.1
    dt_bias = rng.standard_normal((HEADS, HEAD_DIM)).astype(np.float32) * 0.1
    next_head = (batch, HEADS, HEAD_DIM)
    q_next, k_next, v_next = (
        _bfloat16_bits(rng.standard_normal(next_head)) for _ in range(3)
    )
    gate_next = rng.standard_normal(next_head).astype(np.float32)
    beta_next = rng.standard_normal((batch, HEADS)).astype(np.float32)
    pool_shape = (POOL_ROWS, HEADS, HEAD_DIM, HEAD_DIM)
    init = rng.standard_normal(pool_shape).astype(np.float32) * 0.1
    ring_shape = (POOL_ROWS, *_ring_row_shape(ring_len))
    state_rows = _rows(batch, 0)
    ring_rows = _rows(batch, 1)

    gpu = DeviceRef.GPU()
    table = TensorType(DType.uint32, [NUM_LAYERS, "batch_size"], device=gpu)
    types: list[TensorType | BufferType] = [
        TensorType(DType.bfloat16, head, device=gpu),
        TensorType(DType.bfloat16, head, device=gpu),
        TensorType(DType.bfloat16, head, device=gpu),
        TensorType(DType.float32, head, device=gpu),
        TensorType(DType.float32, [total, HEADS], device=gpu),
        TensorType(DType.float32, [HEADS, HEAD_DIM], device=gpu),
        TensorType(DType.float32, [HEADS, HEAD_DIM], device=gpu),
        TensorType(DType.uint32, [batch + 1], device=gpu),
        TensorType(DType.int64, ["batch_size"], device=gpu),
        table,
        table,
        TensorType(DType.int64, [len(ref_rows)], device=gpu),
        TensorType(DType.uint32, [batch + 1], device=gpu),
        TensorType(DType.bfloat16, next_head, device=gpu),
        TensorType(DType.bfloat16, next_head, device=gpu),
        TensorType(DType.bfloat16, next_head, device=gpu),
        TensorType(DType.float32, next_head, device=gpu),
        TensorType(DType.float32, [batch, HEADS], device=gpu),
        BufferType(DType.float32, pool_shape, device=gpu),
        BufferType(DType.float32, ring_shape, device=gpu),
        BufferType(DType.float32, pool_shape, device=gpu),
    ]
    with Graph("k3_kda_ring_rollback", input_types=types) as graph:
        (
            q_v,
            k_v,
            v_v,
            raw_gate_v,
            beta_v,
            a_log_v,
            dt_bias_v,
            offsets_v,
            accepted_v,
            state_rows_v,
            ring_rows_v,
            ref_rows_v,
            ref_offsets_v,
            q_next_v,
            k_next_v,
            v_next_v,
            gate_next_v,
            beta_next_v,
        ) = (value.tensor for value in graph.inputs[:18])
        live, ring, reference = (value.buffer for value in graph.inputs[18:])
        if gate_mode == "safe":
            kernel_gate = (raw_gate_v + dt_bias_v) * ops.exp(a_log_v)
            kernel_a_log = ops.broadcast_to(
                ops.constant(0.0, DType.float32, device=gpu), [HEADS]
            )
            kernel_dt_bias = ops.broadcast_to(
                ops.constant(0.0, DType.float32, device=gpu),
                [HEADS, HEAD_DIM],
            )
            next_gate = (gate_next_v + dt_bias_v) * ops.exp(a_log_v)
        else:
            kernel_gate = raw_gate_v
            kernel_a_log = a_log_v[:, 0]
            kernel_dt_bias = dt_bias_v
            next_gate = gate_next_v
        verify_width = verify_width_operand(num_drafts)
        cu_seqlens = offsets_v.cast(DType.int32)
        num_drafts_v = ops.constant(num_drafts, DType.int64, device=gpu)
        readouts = []
        for layer in range(NUM_LAYERS):
            readouts.append(
                kda_verify_ring(
                    q_v,
                    k_v,
                    v_v,
                    kernel_gate,
                    beta_v,
                    kernel_a_log,
                    kernel_dt_bias,
                    cu_seqlens,
                    live,
                    state_rows_v[layer].cast(DType.int32),
                    ring,
                    ring_rows_v[layer],
                    verify_width,
                    output_dtype=DType.bfloat16,
                    gate_mode=gate_mode,
                    state_layout="K_FIRST",
                )
            )
        if rollback:
            fold_accepted = accepted_lengths(
                offsets_v, accepted_v, num_drafts_v
            )
            if sabotage == "accepted+1":
                fold_accepted = fold_accepted + 1
            # The sabotage swaps the two row tables, the shape-valid form of
            # a live/ring mixup: the ops reject literally swapped buffers.
            state_ids, ring_ids = (
                (ring_rows_v, state_rows_v)
                if sabotage == "swap pools"
                else (state_rows_v, ring_rows_v)
            )
            kda_state_fold(
                live,
                ring,
                state_ids,
                ring_ids,
                fold_accepted,
                offsets_v,
                verify_width,
                state_layout="K_FIRST",
            )

        # The reference decodes the accepted rows the way non-spec decode
        # runs them: one token per request per call. A single multi-row
        # call is not that arm -- on gfx950 a prefill-shaped `kda_decode`
        # batch runs the chunk pipeline inside the op, whose rounding is
        # the prefill's, not the decode arm the rollback owes its
        # bit-identity to. A window with no drafts is a prefill, so there
        # the reference keeps the one call both arms route alike.
        def _reference_decode(
            rows: TensorValue, offsets: TensorValue, state_rows: TensorValue
        ) -> TensorValue:
            """The pinned decode over exactly ``rows``, into the reference
            pool."""
            return kda_decode(
                ops.gather(q_v, rows, axis=0),
                ops.gather(k_v, rows, axis=0),
                ops.gather(v_v, rows, axis=0),
                ops.gather(kernel_gate, rows, axis=0),
                ops.gather(beta_v, rows, axis=0),
                kernel_a_log,
                kernel_dt_bias,
                offsets.cast(DType.int32),
                reference,
                state_rows.cast(DType.int32),
                output_dtype=DType.bfloat16,
                gate_mode=gate_mode,
                pin_fma=True,
            )

        reference_readouts = []
        for layer in range(NUM_LAYERS):
            if not num_drafts:
                reference_readouts.append(
                    _reference_decode(
                        ref_rows_v, ref_offsets_v, state_rows_v[layer]
                    )
                )
                continue
            step_readouts = []
            for members, member_rows in steps:
                members_v = ops.constant(members, DType.int64, device=gpu)
                rows = ops.constant(member_rows, DType.int64, device=gpu)
                step_offsets = ops.range(
                    0,
                    len(members) + 1,
                    out_dim=len(members) + 1,
                    device=gpu,
                    dtype=DType.uint32,
                )
                step_readouts.append(
                    _reference_decode(
                        rows,
                        step_offsets,
                        ops.gather(state_rows_v[layer], members_v, axis=0),
                    )
                )
            reference_readouts.append(
                ops.gather(
                    ops.concat(step_readouts, axis=0),
                    ops.constant(ref_step_major, DType.int64, device=gpu),
                    axis=0,
                )
            )

        # Traced after both pools' main passes: each pool's in-place writes
        # run in trace order, so this step must not run before the reference
        # decode that seeds it.
        next_readouts: list[TensorValue] = []
        if rollback:
            next_offsets = ops.range(
                0, batch + 1, out_dim=batch + 1, device=gpu, dtype=DType.uint32
            ).cast(DType.int32)
            for layer in range(NUM_LAYERS):
                for pool in (live, reference):
                    next_readouts.append(
                        kda_decode(
                            q_next_v,
                            k_next_v,
                            v_next_v,
                            next_gate,
                            beta_next_v,
                            kernel_a_log,
                            kernel_dt_bias,
                            next_offsets,
                            pool,
                            state_rows_v[layer].cast(DType.int32),
                            output_dtype=DType.bfloat16,
                            gate_mode=gate_mode,
                            pin_fma=True,
                        )
                    )
        outputs = [
            readouts[0],
            reference_readouts[0],
            # The verify's readout on the accepted rows is the decode's, bit
            # for bit, in every case -- those rows' logits are what greedy
            # acceptance reads, not only when the window commits itself.
            ops.gather(readouts[0], ref_rows_v, axis=0),
        ]
        outputs += next_readouts
        graph.output(*outputs)

    inputs: list[np.ndarray] = [
        q,
        k,
        v,
        raw_gate,
        beta_logits,
        a_log,
        dt_bias,
        offsets,
        np.array(accepted, dtype=np.int64),
        state_rows,
        ring_rows,
        ref_rows,
        ref_offsets,
        q_next,
        k_next,
        v_next,
        gate_next,
        beta_next,
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


def _assert_verify_matches_decode_on_the_accepted_rows(
    pools: dict[str, np.ndarray],
) -> None:
    """Checks the verify's readout on the accepted rows against the
    reference decode's: ``output_2`` is that readout, ``output_1`` the
    decode's. Those rows' logits are what greedy acceptance reads."""
    np.testing.assert_array_equal(pools["output_2"], pools["output_1"])


def test_the_fold_lands_on_the_original_gate_arm_too() -> None:
    """K3's other gate arm (``gate_lower_bound`` unset) verifies and folds
    the same way; only the tensors feeding the kernel differ."""
    pools = _compile_kda([8], [3], gate_mode="original")()
    np.testing.assert_array_equal(pools["live"], pools["reference"])
    _assert_verify_matches_decode_on_the_accepted_rows(pools)
    np.testing.assert_array_equal(pools["output_3"], pools["output_4"])


def test_a_ragged_batch_of_four_folds_each_request_its_own_length() -> None:
    """Four requests accept 0, 1, 3 and all 7 drafts in one verify, each
    landing its own prefix."""
    pools = _compile_kda([8] * 4, [0, 1, 3, 7])()
    np.testing.assert_array_equal(pools["live"], pools["reference"])
    _assert_verify_matches_decode_on_the_accepted_rows(pools)
    np.testing.assert_array_equal(pools["output_3"], pools["output_4"])
    np.testing.assert_array_equal(pools["output_5"], pools["output_6"])


def test_a_window_past_the_ring_commits_itself_beside_a_folding_one() -> None:
    """A mixed-width batch: the 12-row window exceeds the ring, so the
    verify commits it itself and the fold skips it, while the 8-row window
    records and folds four of its rows. Both must land where the reference
    decodes."""
    pools = _compile_kda([12, 8], [7, 3])()
    np.testing.assert_array_equal(pools["live"], pools["reference"])
    _assert_verify_matches_decode_on_the_accepted_rows(pools)
    np.testing.assert_array_equal(pools["output_3"], pools["output_4"])
    np.testing.assert_array_equal(pools["output_5"], pools["output_6"])


def test_the_kda_verify_leaves_the_live_pool_alone() -> None:
    """The verify records into the ring and writes nothing live."""
    pools = _compile_kda([8], [3], rollback=False)()
    np.testing.assert_array_equal(pools["live"], pools["init"])
    _assert_verify_matches_decode_on_the_accepted_rows(pools)


@pytest.mark.parametrize("seq_lengths", [[1, 9]])
def test_a_committed_kda_window_lands_on_the_forward(
    seq_lengths: list[int],
) -> None:
    """A window with no drafts has nothing to roll back: the verify lands
    the state itself and the fold launches nothing."""
    pools = _compile_kda(seq_lengths, [0] * len(seq_lengths), num_drafts=0)()
    np.testing.assert_array_equal(pools["live"], pools["reference"])
    np.testing.assert_array_equal(pools["output_0"], pools["output_1"])
    _assert_verify_matches_decode_on_the_accepted_rows(pools)


def test_a_fold_past_the_accepted_count_is_detectable() -> None:
    """One record folded too many advances the state past the prefix the
    request accepted, which the comparison must catch."""
    pools = _compile_kda([8], [3], sabotage="accepted+1")()
    assert not np.array_equal(pools["live"], pools["reference"])


def test_a_live_and_ring_crossed_fold_is_detectable() -> None:
    """A fold that reads the ring at the live rows and writes the live pool
    at the ring rows leaves the live pool where the verify left it."""
    pools = _compile_kda([8], [3], sabotage="swap pools")()
    assert not np.array_equal(pools["live"], pools["reference"])


# ===--------------------------------------------------------------------=== #
# K3's own decode arm: the spelling the ring and fold kernels match
# ===--------------------------------------------------------------------=== #


def _compile_k3_vs_pinned() -> Callable[[], dict[str, np.ndarray]]:
    """Compiles K3's own ``_kda_decode`` and the pinned legacy ``kda_decode``
    over the same inputs, each on its own copy of the same pool.

    Both take the kernel tensors K3's safe gate arm feeds: the pre-activated
    gate and the zeroed per-head parameters.
    """
    lengths = [9, 8]
    total = sum(lengths)
    rng = np.random.default_rng(11)
    head = (total, HEADS, HEAD_DIM)
    q, k, v = (_bfloat16_bits(rng.standard_normal(head)) for _ in range(3))
    a_log = rng.standard_normal((HEADS, HEAD_DIM)).astype(np.float32) * 0.1
    dt_bias = rng.standard_normal((HEADS, HEAD_DIM)).astype(np.float32) * 0.1
    raw_gate = rng.standard_normal(head).astype(np.float32)
    gate = (raw_gate + dt_bias) * np.exp(a_log)
    beta = rng.standard_normal((total, HEADS)).astype(np.float32)
    kernel_a_log = np.zeros(HEADS, dtype=np.float32)
    kernel_dt_bias = np.zeros((HEADS, HEAD_DIM), dtype=np.float32)
    offsets = np.cumsum([0, *lengths]).astype(np.uint32)
    rows = np.array([3, 7], dtype=np.uint32)
    pool_shape = (POOL_ROWS, HEADS, HEAD_DIM, HEAD_DIM)
    init = rng.standard_normal(pool_shape).astype(np.float32) * 0.1

    gpu = DeviceRef.GPU()
    types: list[TensorType | BufferType] = [
        TensorType(DType.bfloat16, head, device=gpu),
        TensorType(DType.bfloat16, head, device=gpu),
        TensorType(DType.bfloat16, head, device=gpu),
        TensorType(DType.float32, head, device=gpu),
        TensorType(DType.float32, [total, HEADS], device=gpu),
        TensorType(DType.float32, [HEADS], device=gpu),
        TensorType(DType.float32, [HEADS, HEAD_DIM], device=gpu),
        TensorType(DType.uint32, [len(lengths) + 1], device=gpu),
        TensorType(DType.uint32, [len(lengths)], device=gpu),
        BufferType(DType.float32, pool_shape, device=gpu),
        BufferType(DType.float32, pool_shape, device=gpu),
    ]
    with Graph("k3_pinned_decode", input_types=types) as graph:
        (
            q_v,
            k_v,
            v_v,
            gate_v,
            beta_v,
            a_log_v,
            dt_bias_v,
            offsets_v,
            rows_v,
        ) = (value.tensor for value in graph.inputs[:9])
        k3_pool, pinned_pool = (value.buffer for value in graph.inputs[9:])
        # K3's layer feeds rank-4 q/k/v/gate and a rank-3 beta, the shapes
        # ``_kda_decode`` hands the kernel.
        k3_readout = k3_kda_decode(
            ops.unsqueeze(q_v, 0),
            ops.unsqueeze(k_v, 0),
            ops.unsqueeze(v_v, 0),
            ops.unsqueeze(gate_v, 0),
            ops.reshape(beta_v, [1, total, HEADS]),
            a_log_v,
            dt_bias_v,
            offsets_v,
            k3_pool,
            rows_v,
            gate_mode="safe",
            out_type=TensorType(
                DType.bfloat16, [1, total, HEADS, HEAD_DIM], device=gpu
            ),
            # The spec build's arm: this decode is the one the ring and
            # fold kernels must reproduce bit for bit.
            pin_fma=True,
        )
        pinned_readout = kda_decode(
            q_v,
            k_v,
            v_v,
            gate_v,
            beta_v,
            a_log_v,
            dt_bias_v,
            offsets_v.cast(DType.int32),
            pinned_pool,
            rows_v.cast(DType.int32),
            output_dtype=DType.bfloat16,
            gate_mode="safe",
            pin_fma=True,
        )
        graph.output(ops.squeeze(k3_readout, 0), pinned_readout)

    inputs: list[np.ndarray] = [
        q,
        k,
        v,
        gate.astype(np.float32),
        beta,
        kernel_a_log,
        kernel_dt_bias,
        offsets,
        rows,
    ]
    return lambda: _run(
        graph,
        inputs,
        {"k3_pool": init, "pinned_pool": init},
    )


def test_k3s_decode_arm_is_the_pinned_one_the_fold_reproduces() -> None:
    """K3's ``_kda_decode`` must spell its multiply-adds exactly like
    the pinned ``kda_decode(pin_fma=True)`` the ring and fold kernels and the
    K==0 fallback reproduce, or the state a verify folds lands a ULP off
    the state non-spec decoding leaves."""
    pools = _compile_k3_vs_pinned()()
    np.testing.assert_array_equal(pools["k3_pool"], pools["pinned_pool"])
    np.testing.assert_array_equal(pools["output_0"], pools["output_1"])


# ===--------------------------------------------------------------------=== #
# KDA conv: replay over the captured verify inputs
# ===--------------------------------------------------------------------=== #


def _compile_conv(
    seq_lengths: list[int],
    accepted: list[int],
    *,
    num_drafts: int,
    sabotage: str | None = None,
) -> Callable[[], dict[str, np.ndarray]]:
    """Compiles the conv verify, the conv replay, and the reference conv."""
    assert sabotage in (None, "accepted+1"), sabotage
    batch = len(seq_lengths)
    rng = np.random.default_rng(3)
    total = sum(seq_lengths)
    offsets = np.concatenate([[0], np.cumsum(seq_lengths)]).astype(np.uint32)
    ref_rows, ref_offsets = _accepted_rows(seq_lengths, accepted, num_drafts)
    qkv = rng.standard_normal((total, CONV_DIM)).astype(np.float32)
    weight = rng.standard_normal((CONV_DIM, CONV_KERNEL)).astype(np.float32)
    pool_shape = (POOL_ROWS, CONV_DIM, CONV_KERNEL - 1)
    init = _bfloat16_bits(rng.standard_normal(pool_shape))
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
        BufferType(DType.bfloat16, pool_shape, device=gpu),
        BufferType(DType.bfloat16, pool_shape, device=gpu),
    ]
    with Graph("k3_kda_conv_rollback", input_types=types) as graph:
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
        captures: list[_ConvCapture] = []
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
            captures.append(_ConvCapture(weight_v, qkv_window))
        plan_accepted = (
            accepted_v + 1 if sabotage == "accepted+1" else accepted_v
        )
        row_indices, replay_offsets = accepted_row_plan(
            offsets_v,
            plan_accepted,
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


# The conv-replay model fuses ``accepted_lengths`` into a vectorized
# ``__cumsum`` elementwise whose 16-byte loads start at the int64
# ``offsets[1:]`` slice (byte offset 8); NVIDIA faults on the unaligned
# vector load (CUDA_ERROR_MISALIGNED_ADDRESS) while gfx9's loads take the
# 4-byte alignment. Lift when the elementwise lowering tracks slice
# alignment.
def _is_gfx9() -> bool:
    try:
        return accelerator_architecture_name().startswith("gfx9")
    except Exception:
        return False


_requires_gfx9 = pytest.mark.skipif(
    not _is_gfx9(),
    reason="the engine's elementwise takes 16-byte loads at an 8-byte-aligned"
    " int64 slice, which only gfx9 tolerates",
)


@_requires_gfx9
def test_a_ragged_batch_of_four_replays_each_request_its_own_length() -> None:
    pools = _compile_conv([8] * 4, [0, 1, 3, 7], num_drafts=7)()
    np.testing.assert_array_equal(pools["live"], pools["reference"])


@_requires_gfx9
def test_a_committed_conv_window_lands_on_the_forward() -> None:
    pools = _compile_conv([5, 2], [0, 0], num_drafts=0)()
    np.testing.assert_array_equal(pools["live"], pools["reference"])


@_requires_gfx9
def test_a_conv_replay_past_the_accepted_count_is_detectable() -> None:
    """Replaying one row too many slides the window past the accepted
    prefix, into the next request's tokens."""
    pools = _compile_conv([8], [3], num_drafts=7, sabotage="accepted+1")()
    assert not np.array_equal(pools["live"], pools["reference"])
