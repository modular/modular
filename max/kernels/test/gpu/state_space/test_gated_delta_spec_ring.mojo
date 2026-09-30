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
"""Tests the verify ring and fold against replaying the accepted tokens.

From the same pool and inputs, the ring arm runs
`gated_delta_recurrence_verify_ring_gpu` over the window and then
`gated_delta_state_fold_gpu`, and the replay arm runs
`gated_delta_recurrence_fwd_gpu` over the accepted tokens only. The pools and
the accepted readouts must match exactly.

Every layer gets its own inputs, so a fold that read another layer's records
would fold different numbers. The state pool and the ring take different row
tables, ring rows no request owns are poisoned, and every record is padded so a
kernel that ignored the record stride would read or write the padding.
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from layout import TileTensor, row_major
from state_space.gated_delta import (
    gated_delta_recurrence_fwd_gpu,
    gated_delta_recurrence_verify_ring_gpu,
    gated_delta_ring_record_elements,
    gated_delta_state_fold_gpu,
)
from std.random import rand, seed
from std.testing import TestSuite, assert_equal

comptime WORK_DTYPE = DType.float32
comptime STATE_DTYPE = DType.bfloat16
comptime RING_DTYPE = DType.float32

comptime POISON = Float32(-7.75e18)
"""Fill value for ring entries no request writes."""

comptime RECORD_PADDING = 3
"""Elements past each record, so a record stride differs from its size."""


struct RingCase(Copyable, Movable):
    """One verify window and the acceptance it settles on."""

    var batch_size: Int
    var num_layers: Int
    var num_value_heads: Int
    var num_key_heads: Int
    var seq_lengths: List[Int]
    var accepted: List[Int]

    def __init__(
        out self,
        num_layers: Int,
        num_value_heads: Int,
        num_key_heads: Int,
        var seq_lengths: List[Int],
        var accepted: List[Int],
    ):
        self.batch_size = len(seq_lengths)
        self.num_layers = num_layers
        self.num_value_heads = num_value_heads
        self.num_key_heads = num_key_heads
        self.seq_lengths = seq_lengths^
        self.accepted = accepted^


def _upload_table(
    ctx: DeviceContext, values: List[Int]
) raises -> DeviceBuffer[DType.uint32]:
    """Copies `values` into a new device buffer of uint32."""
    var host = ctx.enqueue_create_host_buffer[DType.uint32](len(values))
    for i in range(len(values)):
        host[i] = UInt32(values[i])
    var device = ctx.enqueue_create_buffer[DType.uint32](len(values))
    ctx.enqueue_copy(device, host.unsafe_ptr())
    ctx.synchronize()
    return device^


def run_ring_case[
    KEY_HEAD_DIM: Int,
    VALUE_HEAD_DIM: Int,
    RING_LEN: Int,
](
    spec: RingCase,
    ctx: DeviceContext,
    *,
    fold_accepted_override: List[Int] = List[Int](),
    expect_match: Bool = True,
    ring_silent_rows: List[Int] = List[Int](),
    expect_ring_silent: Bool = True,
) raises:
    """Runs both arms of one case and compares them.

    Parameters:
        KEY_HEAD_DIM: Key head dimension.
        VALUE_HEAD_DIM: Value head dimension.
        RING_LEN: Record capacity of one ring row.

    Args:
        spec: The window, the acceptance, and the pool geometry.
        ctx: Device to run on.
        fold_accepted_override: Acceptance passed to the fold instead of
            `spec.accepted`.
        expect_match: Whether the two pools must match. False asserts they
            differ.
        ring_silent_rows: Batch items whose ring rows must still hold
            `POISON`.
        expect_ring_silent: Whether those rows must be untouched. False
            asserts at least one of them was written.
    """
    var batch_size = spec.batch_size
    var num_layers = spec.num_layers
    var nv = spec.num_value_heads
    var nk = spec.num_key_heads
    var key_dim = nk * KEY_HEAD_DIM
    var value_dim = nv * VALUE_HEAD_DIM
    var conv_dim = key_dim * 2 + value_dim

    var window_rows = 0
    var accepted_rows = 0
    for b in range(batch_size):
        window_rows += spec.seq_lengths[b]
        accepted_rows += spec.accepted[b]
    # A zero-row replay still needs a nonempty buffer.
    var replay_rows = max(accepted_rows, 1)

    # ── Row tables, layer-major. Layer l of a request sits at pool row
    # ``block * num_layers + l``, with a different block per pool.
    var state_rows = (3 * batch_size + 2) * num_layers
    var ring_rows = (3 * batch_size + 4) * num_layers
    var table_size = batch_size * num_layers
    var state_table = List[Int]()
    var ring_table = List[Int]()
    for l in range(num_layers):
        for b in range(batch_size):
            state_table.append((2 * batch_size - 1 - b) * num_layers + l)
            ring_table.append((b + 3) * num_layers + l)

    # ── Per-token inputs, a window per layer ────────────────────────────────
    seed(0xC0FFEE)
    var qkv_layer = window_rows * conv_dim
    var gate_layer = window_rows * nv
    var qkv_h = ctx.enqueue_create_host_buffer[WORK_DTYPE](
        num_layers * qkv_layer
    )
    rand[WORK_DTYPE](
        qkv_h.unsafe_ptr(), num_layers * qkv_layer, min=-1.0, max=1.0
    )
    var decay_h = ctx.enqueue_create_host_buffer[WORK_DTYPE](
        num_layers * gate_layer
    )
    rand[WORK_DTYPE](
        decay_h.unsafe_ptr(), num_layers * gate_layer, min=0.05, max=0.99
    )
    var beta_h = ctx.enqueue_create_host_buffer[WORK_DTYPE](
        num_layers * gate_layer
    )
    rand[WORK_DTYPE](
        beta_h.unsafe_ptr(), num_layers * gate_layer, min=0.05, max=0.99
    )

    var window_offsets = List[Int]()
    var replay_offsets = List[Int]()
    window_offsets.append(0)
    replay_offsets.append(0)
    for b in range(batch_size):
        window_offsets.append(window_offsets[b] + spec.seq_lengths[b])
        replay_offsets.append(replay_offsets[b] + spec.accepted[b])

    var fold_accepted = List[Int]()
    for b in range(batch_size):
        fold_accepted.append(
            fold_accepted_override[b] if len(fold_accepted_override)
            > 0 else spec.accepted[b]
        )

    # ── Pre-verify pool. Nonzero, so a wrong decay changes the result.
    var row_elems = nv * KEY_HEAD_DIM * VALUE_HEAD_DIM
    var state_elems = state_rows * row_elems
    var pool_seed_h = ctx.enqueue_create_host_buffer[WORK_DTYPE](state_elems)
    rand[WORK_DTYPE](pool_seed_h.unsafe_ptr(), state_elems, min=-1.0, max=1.0)
    var pool_initial_h = ctx.enqueue_create_host_buffer[STATE_DTYPE](
        state_elems
    )
    for i in range(state_elems):
        pool_initial_h[i] = Scalar[STATE_DTYPE](pool_seed_h[i])

    # The accepted tokens of every layer, gathered into their own batch.
    var replay_qkv_layer = replay_rows * conv_dim
    var replay_gate_layer = replay_rows * nv
    var replay_qkv_h = ctx.enqueue_create_host_buffer[WORK_DTYPE](
        num_layers * replay_qkv_layer
    )
    var replay_decay_h = ctx.enqueue_create_host_buffer[WORK_DTYPE](
        num_layers * replay_gate_layer
    )
    var replay_beta_h = ctx.enqueue_create_host_buffer[WORK_DTYPE](
        num_layers * replay_gate_layer
    )
    for i in range(num_layers * replay_qkv_layer):
        replay_qkv_h[i] = 0
    for i in range(num_layers * replay_gate_layer):
        replay_decay_h[i] = 0
        replay_beta_h[i] = 0
    for l in range(num_layers):
        for b in range(batch_size):
            for t in range(spec.accepted[b]):
                var src = window_offsets[b] + t
                var dst = replay_offsets[b] + t
                for c in range(conv_dim):
                    replay_qkv_h[
                        l * replay_qkv_layer + dst * conv_dim + c
                    ] = qkv_h[l * qkv_layer + src * conv_dim + c]
                for h in range(nv):
                    replay_decay_h[
                        l * replay_gate_layer + dst * nv + h
                    ] = decay_h[l * gate_layer + src * nv + h]
                    replay_beta_h[
                        l * replay_gate_layer + dst * nv + h
                    ] = beta_h[l * gate_layer + src * nv + h]

    # ── Device buffers ──────────────────────────────────────────────────────
    var qkv_dev = ctx.enqueue_create_buffer[WORK_DTYPE](num_layers * qkv_layer)
    var decay_dev = ctx.enqueue_create_buffer[WORK_DTYPE](
        num_layers * gate_layer
    )
    var beta_dev = ctx.enqueue_create_buffer[WORK_DTYPE](
        num_layers * gate_layer
    )
    var replay_qkv_dev = ctx.enqueue_create_buffer[WORK_DTYPE](
        num_layers * replay_qkv_layer
    )
    var replay_decay_dev = ctx.enqueue_create_buffer[WORK_DTYPE](
        num_layers * replay_gate_layer
    )
    var replay_beta_dev = ctx.enqueue_create_buffer[WORK_DTYPE](
        num_layers * replay_gate_layer
    )
    var out_ring_dev = ctx.enqueue_create_buffer[WORK_DTYPE](
        num_layers * window_rows * value_dim
    )
    var out_replay_dev = ctx.enqueue_create_buffer[WORK_DTYPE](
        num_layers * replay_rows * value_dim
    )
    var pool_ring_dev = ctx.enqueue_create_buffer[STATE_DTYPE](state_elems)
    var pool_replay_dev = ctx.enqueue_create_buffer[STATE_DTYPE](state_elems)

    var record_elems = gated_delta_ring_record_elements[
        KEY_HEAD_DIM, VALUE_HEAD_DIM
    ](nv // nk)
    var record_stride = record_elems + RECORD_PADDING
    var ring_row_elems = nk * RING_LEN * record_stride
    var ring_elems = ring_rows * ring_row_elems
    var ring_dev = ctx.enqueue_create_buffer[RING_DTYPE](ring_elems)
    ctx.enqueue_memset(ring_dev, Scalar[RING_DTYPE](POISON))

    with ctx.push_context():
        ctx.enqueue_copy(qkv_dev, qkv_h.unsafe_ptr())
        ctx.enqueue_copy(decay_dev, decay_h.unsafe_ptr())
        ctx.enqueue_copy(beta_dev, beta_h.unsafe_ptr())
        ctx.enqueue_copy(replay_qkv_dev, replay_qkv_h.unsafe_ptr())
        ctx.enqueue_copy(replay_decay_dev, replay_decay_h.unsafe_ptr())
        ctx.enqueue_copy(replay_beta_dev, replay_beta_h.unsafe_ptr())
        ctx.enqueue_copy(pool_ring_dev, pool_initial_h.unsafe_ptr())
        ctx.enqueue_copy(pool_replay_dev, pool_initial_h.unsafe_ptr())
    ctx.synchronize()

    var window_offsets_dev = _upload_table(ctx, window_offsets)
    var replay_offsets_dev = _upload_table(ctx, replay_offsets)
    var state_table_dev = _upload_table(ctx, state_table)
    var ring_table_dev = _upload_table(ctx, ring_table)
    var accepted_dev = _upload_table(ctx, fold_accepted)

    var window_offsets_tt = TileTensor(
        window_offsets_dev, row_major(batch_size + 1)
    )
    var replay_offsets_tt = TileTensor(
        replay_offsets_dev, row_major(batch_size + 1)
    )
    var pool_ring_tt = TileTensor(
        pool_ring_dev, row_major(state_rows, nv, KEY_HEAD_DIM, VALUE_HEAD_DIM)
    )
    var pool_replay_tt = TileTensor(
        pool_replay_dev,
        row_major(state_rows, nv, KEY_HEAD_DIM, VALUE_HEAD_DIM),
    )
    var ring_tt = TileTensor(
        ring_dev, row_major(ring_rows, nk, RING_LEN, record_stride)
    )
    var state_table_tt = TileTensor(
        state_table_dev, row_major(num_layers, batch_size)
    )
    var ring_table_tt = TileTensor(
        ring_table_dev, row_major(num_layers, batch_size)
    )
    var accepted_tt = TileTensor(accepted_dev, row_major(batch_size))

    # Per-layer views are wrapped at each launch, because a `TileTensor`'s
    # type carries its buffer's origin. Every 2-D input shares one layout
    # type.
    comptime QkvLT = type_of(row_major(window_rows, conv_dim))
    var verify_kernel = ctx.compile_function[
        gated_delta_recurrence_verify_ring_gpu[
            WORK_DTYPE,
            STATE_DTYPE,
            RING_DTYPE,
            KEY_HEAD_DIM,
            VALUE_HEAD_DIM,
            RING_LEN,
            QkvLT,
            QkvLT,
            QkvLT,
            QkvLT,
            pool_ring_tt.LayoutType,
            accepted_tt.LayoutType,
            window_offsets_tt.LayoutType,
            ring_tt.LayoutType,
            accepted_tt.LayoutType,
            pool_ring_tt.Engine,
        ]
    ]()
    var replay_kernel = ctx.compile_function[
        gated_delta_recurrence_fwd_gpu[
            WORK_DTYPE,
            STATE_DTYPE,
            KEY_HEAD_DIM,
            VALUE_HEAD_DIM,
            QkvLT,
            QkvLT,
            QkvLT,
            QkvLT,
            pool_replay_tt.LayoutType,
            accepted_tt.LayoutType,
            replay_offsets_tt.LayoutType,
            pool_replay_tt.Engine,
        ]
    ]()
    var fold_kernel = ctx.compile_function[
        gated_delta_state_fold_gpu[
            STATE_DTYPE,
            RING_DTYPE,
            KEY_HEAD_DIM,
            VALUE_HEAD_DIM,
            RING_LEN,
            pool_ring_tt.LayoutType,
            state_table_tt.LayoutType,
            ring_tt.LayoutType,
            ring_table_tt.LayoutType,
            accepted_tt.LayoutType,
            pool_ring_tt.Engine,
        ]
    ]()

    # Both arms run one launch per layer on that layer's row of every pool.
    ctx.synchronize()
    for l in range(num_layers):
        var slot_col = TileTensor(
            state_table_dev.create_sub_buffer[DType.uint32](
                l * batch_size, batch_size
            ),
            row_major(batch_size),
        )
        ctx.enqueue_function(
            verify_kernel,
            Int32(batch_size),
            Int32(nv),
            Int32(nk),
            Int32(key_dim),
            TileTensor(
                out_ring_dev.create_sub_buffer[WORK_DTYPE](
                    l * window_rows * value_dim, window_rows * value_dim
                ),
                row_major(window_rows, value_dim),
            ),
            pool_ring_tt,
            slot_col,
            TileTensor(
                qkv_dev.create_sub_buffer[WORK_DTYPE](l * qkv_layer, qkv_layer),
                row_major(window_rows, conv_dim),
            ),
            TileTensor(
                decay_dev.create_sub_buffer[WORK_DTYPE](
                    l * gate_layer, gate_layer
                ),
                row_major(window_rows, nv),
            ),
            TileTensor(
                beta_dev.create_sub_buffer[WORK_DTYPE](
                    l * gate_layer, gate_layer
                ),
                row_major(window_rows, nv),
            ),
            window_offsets_tt,
            ring_tt,
            TileTensor(
                ring_table_dev.create_sub_buffer[DType.uint32](
                    l * batch_size, batch_size
                ),
                row_major(batch_size),
            ),
            Int32(record_stride),
            UInt32(conv_dim),
            UInt32(1),
            UInt32(nv),
            UInt32(1),
            UInt32(value_dim),
            UInt32(1),
            grid_dim=(batch_size * nv,),
            block_dim=(VALUE_HEAD_DIM,),
        )
        ctx.enqueue_function(
            replay_kernel,
            Int32(batch_size),
            Int32(nv),
            Int32(nk),
            Int32(key_dim),
            TileTensor(
                out_replay_dev.create_sub_buffer[WORK_DTYPE](
                    l * replay_rows * value_dim, replay_rows * value_dim
                ),
                row_major(replay_rows, value_dim),
            ),
            pool_replay_tt,
            slot_col,
            TileTensor(
                replay_qkv_dev.create_sub_buffer[WORK_DTYPE](
                    l * replay_qkv_layer, replay_qkv_layer
                ),
                row_major(replay_rows, conv_dim),
            ),
            TileTensor(
                replay_decay_dev.create_sub_buffer[WORK_DTYPE](
                    l * replay_gate_layer, replay_gate_layer
                ),
                row_major(replay_rows, nv),
            ),
            TileTensor(
                replay_beta_dev.create_sub_buffer[WORK_DTYPE](
                    l * replay_gate_layer, replay_gate_layer
                ),
                row_major(replay_rows, nv),
            ),
            replay_offsets_tt,
            UInt32(conv_dim),
            UInt32(1),
            UInt32(nv),
            UInt32(1),
            UInt32(value_dim),
            UInt32(1),
            grid_dim=(batch_size * nv,),
            block_dim=(VALUE_HEAD_DIM,),
        )

    ctx.enqueue_function(
        fold_kernel,
        Int32(batch_size),
        Int32(num_layers),
        Int32(nv),
        Int32(nk),
        pool_ring_tt,
        state_table_tt,
        ring_tt,
        ring_table_tt,
        Int32(record_stride),
        accepted_tt,
        grid_dim=(batch_size * num_layers * nv,),
        block_dim=(VALUE_HEAD_DIM,),
    )

    var pool_ring_after = ctx.enqueue_create_host_buffer[STATE_DTYPE](
        state_elems
    )
    var pool_replay_after = ctx.enqueue_create_host_buffer[STATE_DTYPE](
        state_elems
    )
    var out_ring_after = ctx.enqueue_create_host_buffer[WORK_DTYPE](
        num_layers * window_rows * value_dim
    )
    var out_replay_after = ctx.enqueue_create_host_buffer[WORK_DTYPE](
        num_layers * replay_rows * value_dim
    )
    var ring_after = ctx.enqueue_create_host_buffer[RING_DTYPE](ring_elems)
    with ctx.push_context():
        ctx.enqueue_copy(pool_ring_after.unsafe_ptr(), pool_ring_dev)
        ctx.enqueue_copy(pool_replay_after.unsafe_ptr(), pool_replay_dev)
        ctx.enqueue_copy(out_ring_after.unsafe_ptr(), out_ring_dev)
        ctx.enqueue_copy(out_replay_after.unsafe_ptr(), out_replay_dev)
        ctx.enqueue_copy(ring_after.unsafe_ptr(), ring_dev)
    ctx.synchronize()

    for i in range(ring_elems):
        if i % record_stride >= record_elems:
            assert_equal(
                Float32(ring_after[i]),
                POISON,
                "record padding was written at element " + String(i),
            )

    # A row too long for the ring must leave its ring rows poisoned.
    if len(ring_silent_rows) > 0:
        var any_written = False
        for i in range(len(ring_silent_rows)):
            for l in range(num_layers):
                var row = ring_table[l * batch_size + ring_silent_rows[i]]
                for e in range(ring_row_elems):
                    if Float32(ring_after[row * ring_row_elems + e]) != POISON:
                        any_written = True
        if expect_ring_silent:
            assert_equal(
                any_written,
                False,
                "a row too long for the ring still appended to it",
            )
        else:
            assert_equal(
                any_written,
                True,
                (
                    "negative control: the row was short enough for the ring"
                    " to capture and the poison still survived, so the check"
                    " proves nothing"
                ),
            )

    if not expect_match:
        var differs = False
        for i in range(state_elems):
            if pool_ring_after[i] != pool_replay_after[i]:
                differs = True
                break
        assert_equal(
            differs,
            True,
            (
                "negative control: the fold was given a different acceptance"
                " count and the pools still matched"
            ),
        )
        return

    for i in range(state_elems):
        assert_equal(
            Float64(pool_ring_after[i]),
            Float64(pool_replay_after[i]),
            "state pool diverged at element " + String(i),
        )

    for l in range(num_layers):
        for b in range(batch_size):
            for t in range(spec.accepted[b]):
                var ring_token = l * window_rows + window_offsets[b] + t
                var replay_token = l * replay_rows + replay_offsets[b] + t
                for c in range(value_dim):
                    assert_equal(
                        Float64(out_ring_after[ring_token * value_dim + c]),
                        Float64(out_replay_after[replay_token * value_dim + c]),
                        "readout diverged at layer "
                        + String(l)
                        + " batch "
                        + String(b)
                        + " token "
                        + String(t),
                    )

    # Rows no request owns must still hold their pre-verify bytes.
    var row_is_owned = List[Bool](length=state_rows, fill=False)
    for i in range(table_size):
        row_is_owned[state_table[i]] = True
    for r in range(state_rows):
        if row_is_owned[r]:
            continue
        for e in range(row_elems):
            assert_equal(
                Float64(pool_ring_after[r * row_elems + e]),
                Float64(pool_initial_h[r * row_elems + e]),
                "unowned pool row " + String(r) + " was written",
            )


def run_readout_neutrality[
    KEY_HEAD_DIM: Int,
    VALUE_HEAD_DIM: Int,
    RING_LEN: Int,
](spec: RingCase, ctx: DeviceContext) raises:
    """Asserts the ring verify matches `gated_delta_recurrence_fwd_gpu`'s
    readout over the whole window.

    A row that fits the ring leaves its pool row unchanged, and a longer row
    lands the pool row the plain kernel does.
    """
    var batch_size = spec.batch_size
    var nv = spec.num_value_heads
    var nk = spec.num_key_heads
    var key_dim = nk * KEY_HEAD_DIM
    var value_dim = nv * VALUE_HEAD_DIM
    var conv_dim = key_dim * 2 + value_dim
    var window_rows = 0
    var offsets = List[Int]()
    var slots = List[Int]()
    offsets.append(0)
    for b in range(batch_size):
        window_rows += spec.seq_lengths[b]
        offsets.append(window_rows)
        slots.append(b)

    var state_rows = batch_size + 1
    var ring_rows = batch_size + 1
    var state_elems = state_rows * nv * KEY_HEAD_DIM * VALUE_HEAD_DIM

    seed(0x5EED)
    var qkv_h = ctx.enqueue_create_host_buffer[WORK_DTYPE](
        window_rows * conv_dim
    )
    rand[WORK_DTYPE](
        qkv_h.unsafe_ptr(), window_rows * conv_dim, min=-1.0, max=1.0
    )
    var decay_h = ctx.enqueue_create_host_buffer[WORK_DTYPE](window_rows * nv)
    rand[WORK_DTYPE](decay_h.unsafe_ptr(), window_rows * nv, min=0.05, max=0.99)
    var beta_h = ctx.enqueue_create_host_buffer[WORK_DTYPE](window_rows * nv)
    rand[WORK_DTYPE](beta_h.unsafe_ptr(), window_rows * nv, min=0.05, max=0.99)
    var pool_seed_h = ctx.enqueue_create_host_buffer[WORK_DTYPE](state_elems)
    rand[WORK_DTYPE](pool_seed_h.unsafe_ptr(), state_elems, min=-1.0, max=1.0)
    var pool_h = ctx.enqueue_create_host_buffer[STATE_DTYPE](state_elems)
    for i in range(state_elems):
        pool_h[i] = Scalar[STATE_DTYPE](pool_seed_h[i])

    var qkv_dev = ctx.enqueue_create_buffer[WORK_DTYPE](window_rows * conv_dim)
    var decay_dev = ctx.enqueue_create_buffer[WORK_DTYPE](window_rows * nv)
    var beta_dev = ctx.enqueue_create_buffer[WORK_DTYPE](window_rows * nv)
    var pool_ring_dev = ctx.enqueue_create_buffer[STATE_DTYPE](state_elems)
    var pool_plain_dev = ctx.enqueue_create_buffer[STATE_DTYPE](state_elems)
    var out_ring_dev = ctx.enqueue_create_buffer[WORK_DTYPE](
        window_rows * value_dim
    )
    var out_plain_dev = ctx.enqueue_create_buffer[WORK_DTYPE](
        window_rows * value_dim
    )
    var record_stride = (
        gated_delta_ring_record_elements[KEY_HEAD_DIM, VALUE_HEAD_DIM](nv // nk)
        + RECORD_PADDING
    )
    var ring_dev = ctx.enqueue_create_buffer[RING_DTYPE](
        ring_rows * nk * RING_LEN * record_stride
    )
    with ctx.push_context():
        ctx.enqueue_copy(qkv_dev, qkv_h.unsafe_ptr())
        ctx.enqueue_copy(decay_dev, decay_h.unsafe_ptr())
        ctx.enqueue_copy(beta_dev, beta_h.unsafe_ptr())
        ctx.enqueue_copy(pool_ring_dev, pool_h.unsafe_ptr())
        ctx.enqueue_copy(pool_plain_dev, pool_h.unsafe_ptr())
    ctx.synchronize()
    var offsets_dev = _upload_table(ctx, offsets)
    var slot_dev = _upload_table(ctx, slots)
    var ring_slot_dev = _upload_table(ctx, slots)

    var qkv_tt = TileTensor(qkv_dev, row_major(window_rows, conv_dim))
    var decay_tt = TileTensor(decay_dev, row_major(window_rows, nv))
    var beta_tt = TileTensor(beta_dev, row_major(window_rows, nv))
    var offsets_tt = TileTensor(offsets_dev, row_major(batch_size + 1))
    var slot_tt = TileTensor(slot_dev, row_major(batch_size))
    var ring_slot_tt = TileTensor(ring_slot_dev, row_major(batch_size))
    var pool_ring_tt = TileTensor(
        pool_ring_dev, row_major(state_rows, nv, KEY_HEAD_DIM, VALUE_HEAD_DIM)
    )
    var pool_plain_tt = TileTensor(
        pool_plain_dev, row_major(state_rows, nv, KEY_HEAD_DIM, VALUE_HEAD_DIM)
    )
    var out_ring_tt = TileTensor(
        out_ring_dev, row_major(window_rows, value_dim)
    )
    var out_plain_tt = TileTensor(
        out_plain_dev, row_major(window_rows, value_dim)
    )
    var ring_tt = TileTensor(
        ring_dev, row_major(ring_rows, nk, RING_LEN, record_stride)
    )

    var ring_kernel = ctx.compile_function[
        gated_delta_recurrence_verify_ring_gpu[
            WORK_DTYPE,
            STATE_DTYPE,
            RING_DTYPE,
            KEY_HEAD_DIM,
            VALUE_HEAD_DIM,
            RING_LEN,
            out_ring_tt.LayoutType,
            qkv_tt.LayoutType,
            decay_tt.LayoutType,
            beta_tt.LayoutType,
            pool_ring_tt.LayoutType,
            slot_tt.LayoutType,
            offsets_tt.LayoutType,
            ring_tt.LayoutType,
            ring_slot_tt.LayoutType,
            out_ring_tt.Engine,
        ]
    ]()
    var plain_kernel = ctx.compile_function[
        gated_delta_recurrence_fwd_gpu[
            WORK_DTYPE,
            STATE_DTYPE,
            KEY_HEAD_DIM,
            VALUE_HEAD_DIM,
            out_plain_tt.LayoutType,
            qkv_tt.LayoutType,
            decay_tt.LayoutType,
            beta_tt.LayoutType,
            pool_plain_tt.LayoutType,
            slot_tt.LayoutType,
            offsets_tt.LayoutType,
            out_plain_tt.Engine,
        ]
    ]()

    with ctx.push_context():
        ctx.enqueue_function(
            ring_kernel,
            Int32(batch_size),
            Int32(nv),
            Int32(nk),
            Int32(key_dim),
            out_ring_tt,
            pool_ring_tt,
            slot_tt,
            qkv_tt,
            decay_tt,
            beta_tt,
            offsets_tt,
            ring_tt,
            ring_slot_tt,
            Int32(record_stride),
            UInt32(conv_dim),
            UInt32(1),
            UInt32(nv),
            UInt32(1),
            UInt32(value_dim),
            UInt32(1),
            grid_dim=(batch_size * nv,),
            block_dim=(VALUE_HEAD_DIM,),
        )
        ctx.enqueue_function(
            plain_kernel,
            Int32(batch_size),
            Int32(nv),
            Int32(nk),
            Int32(key_dim),
            out_plain_tt,
            pool_plain_tt,
            slot_tt,
            qkv_tt,
            decay_tt,
            beta_tt,
            offsets_tt,
            UInt32(conv_dim),
            UInt32(1),
            UInt32(nv),
            UInt32(1),
            UInt32(value_dim),
            UInt32(1),
            grid_dim=(batch_size * nv,),
            block_dim=(VALUE_HEAD_DIM,),
        )
    var out_ring_h = ctx.enqueue_create_host_buffer[WORK_DTYPE](
        window_rows * value_dim
    )
    var out_plain_h = ctx.enqueue_create_host_buffer[WORK_DTYPE](
        window_rows * value_dim
    )
    var pool_after_h = ctx.enqueue_create_host_buffer[STATE_DTYPE](state_elems)
    var pool_plain_h = ctx.enqueue_create_host_buffer[STATE_DTYPE](state_elems)
    with ctx.push_context():
        ctx.enqueue_copy(out_ring_h.unsafe_ptr(), out_ring_dev)
        ctx.enqueue_copy(out_plain_h.unsafe_ptr(), out_plain_dev)
        ctx.enqueue_copy(pool_after_h.unsafe_ptr(), pool_ring_dev)
        ctx.enqueue_copy(pool_plain_h.unsafe_ptr(), pool_plain_dev)
    ctx.synchronize()

    for i in range(window_rows * value_dim):
        assert_equal(
            Float64(out_ring_h[i]),
            Float64(out_plain_h[i]),
            "readout diverged from the plain kernel at element " + String(i),
        )
    var row_elems = nv * KEY_HEAD_DIM * VALUE_HEAD_DIM
    for i in range(state_elems):
        var slot = i // row_elems
        var commits = slot < batch_size and spec.seq_lengths[slot] > RING_LEN
        assert_equal(
            Float64(pool_after_h[i]),
            Float64(pool_plain_h[i]) if commits else Float64(pool_h[i]),
            "the ring verify landed the wrong pool row at element " + String(i),
        )


def _deep_ring_arm[
    ring_row_elems: Int
](
    ctx: DeviceContext,
    pool_dev: DeviceBuffer[STATE_DTYPE],
    ring_dev: DeviceBuffer[DType.uint8],
    ring_rows: Int,
    ring_row: Int,
    qkv_dev: DeviceBuffer[WORK_DTYPE],
    gate_dev: DeviceBuffer[WORK_DTYPE],
) raises:
    """Verifies a two-token window of one-element heads into `ring_row`, then
    folds both records into pool row 0."""
    comptime RING_LEN = 2
    comptime record_stride = ring_row_elems // RING_LEN
    comptime conv_dim = 3
    var out_dev = ctx.enqueue_create_buffer[WORK_DTYPE](RING_LEN)
    var offsets_dev = _upload_table(ctx, [0, RING_LEN])
    var slot_dev = _upload_table(ctx, [0])
    var ring_slot_dev = _upload_table(ctx, [ring_row])
    var accepted_dev = _upload_table(ctx, [RING_LEN])

    var out_tt = TileTensor(out_dev, row_major(RING_LEN, 1))
    var qkv_tt = TileTensor(qkv_dev, row_major(RING_LEN, conv_dim))
    var gate_tt = TileTensor(gate_dev, row_major(RING_LEN, 1))
    var pool_tt = TileTensor(pool_dev, row_major(1, 1, 1, 1))
    var ring_tt = TileTensor(
        ring_dev, row_major(ring_rows, 1, RING_LEN, record_stride)
    )
    var offsets_tt = TileTensor(offsets_dev, row_major(2))
    var slot_tt = TileTensor(slot_dev, row_major(1))
    var ring_slot_tt = TileTensor(ring_slot_dev, row_major(1))
    var table_tt = TileTensor(slot_dev, row_major(1, 1))
    var ring_table_tt = TileTensor(ring_slot_dev, row_major(1, 1))
    var accepted_tt = TileTensor(accepted_dev, row_major(1))

    var verify_kernel = ctx.compile_function[
        gated_delta_recurrence_verify_ring_gpu[
            WORK_DTYPE,
            STATE_DTYPE,
            DType.uint8,
            1,
            1,
            RING_LEN,
            out_tt.LayoutType,
            qkv_tt.LayoutType,
            gate_tt.LayoutType,
            gate_tt.LayoutType,
            pool_tt.LayoutType,
            slot_tt.LayoutType,
            offsets_tt.LayoutType,
            ring_tt.LayoutType,
            ring_slot_tt.LayoutType,
            out_tt.Engine,
        ]
    ]()
    var fold_kernel = ctx.compile_function[
        gated_delta_state_fold_gpu[
            STATE_DTYPE,
            DType.uint8,
            1,
            1,
            RING_LEN,
            pool_tt.LayoutType,
            table_tt.LayoutType,
            ring_tt.LayoutType,
            ring_table_tt.LayoutType,
            accepted_tt.LayoutType,
            pool_tt.Engine,
            KEY_DIM_TILE=1,
        ]
    ]()
    ctx.enqueue_function(
        verify_kernel,
        Int32(1),
        Int32(1),
        Int32(1),
        Int32(1),
        out_tt,
        pool_tt,
        slot_tt,
        qkv_tt,
        gate_tt,
        gate_tt,
        offsets_tt,
        ring_tt,
        ring_slot_tt,
        Int32(record_stride),
        UInt32(conv_dim),
        UInt32(1),
        UInt32(1),
        UInt32(1),
        UInt32(1),
        UInt32(1),
        grid_dim=(1,),
        block_dim=(1,),
    )
    ctx.enqueue_function(
        fold_kernel,
        Int32(1),
        Int32(1),
        Int32(1),
        Int32(1),
        pool_tt,
        table_tt,
        ring_tt,
        ring_table_tt,
        Int32(record_stride),
        accepted_tt,
        grid_dim=(1,),
        block_dim=(1,),
    )
    ctx.synchronize()


def run_deep_ring_row(ctx: DeviceContext) raises:
    """Checks a ring row past 2**32 elements is written and folded in place.

    The ring is `uint8` with one-element heads, so a row at 2**32 elements
    fits a 4 GiB pool. The same window runs against a deep row and against a
    shallow row of a second ring, and the two must agree while the deep
    ring's front row keeps its sentinel. The value is five times the key and
    the gates stay below one half, so every record is a nonnegative number
    that `uint8` holds exactly after truncation.
    """
    comptime ring_row_elems = 8
    var deep_row = (1 << 32) // ring_row_elems
    var sentinel = Scalar[DType.uint8](200)
    var initial = Scalar[STATE_DTYPE](0.75)

    var qkv_h = ctx.enqueue_create_host_buffer[WORK_DTYPE](6)
    for token in range(2):
        qkv_h[3 * token] = 1.0
        qkv_h[3 * token + 1] = 1.0
        qkv_h[3 * token + 2] = 5.0
    var gate_h = ctx.enqueue_create_host_buffer[WORK_DTYPE](2)
    gate_h[0] = 0.25
    gate_h[1] = 0.5
    var pool_h = ctx.enqueue_create_host_buffer[STATE_DTYPE](1)
    pool_h[0] = initial

    var qkv_dev = ctx.enqueue_create_buffer[WORK_DTYPE](6)
    var gate_dev = ctx.enqueue_create_buffer[WORK_DTYPE](2)
    var deep_pool_dev = ctx.enqueue_create_buffer[STATE_DTYPE](1)
    var shallow_pool_dev = ctx.enqueue_create_buffer[STATE_DTYPE](1)
    var deep_ring_dev = ctx.enqueue_create_buffer[DType.uint8](
        (deep_row + 1) * ring_row_elems
    )
    var shallow_ring_dev = ctx.enqueue_create_buffer[DType.uint8](
        2 * ring_row_elems
    )
    ctx.enqueue_memset(deep_ring_dev, sentinel)
    ctx.enqueue_memset(shallow_ring_dev, sentinel)
    with ctx.push_context():
        ctx.enqueue_copy(qkv_dev, qkv_h.unsafe_ptr())
        ctx.enqueue_copy(gate_dev, gate_h.unsafe_ptr())
        ctx.enqueue_copy(deep_pool_dev, pool_h.unsafe_ptr())
        ctx.enqueue_copy(shallow_pool_dev, pool_h.unsafe_ptr())
    ctx.synchronize()

    _deep_ring_arm[ring_row_elems](
        ctx,
        deep_pool_dev,
        deep_ring_dev,
        deep_row + 1,
        deep_row,
        qkv_dev,
        gate_dev,
    )
    _deep_ring_arm[ring_row_elems](
        ctx, shallow_pool_dev, shallow_ring_dev, 2, 1, qkv_dev, gate_dev
    )

    var deep_pool_h = ctx.enqueue_create_host_buffer[STATE_DTYPE](1)
    var shallow_pool_h = ctx.enqueue_create_host_buffer[STATE_DTYPE](1)
    var front_h = ctx.enqueue_create_host_buffer[DType.uint8](ring_row_elems)
    var deep_h = ctx.enqueue_create_host_buffer[DType.uint8](ring_row_elems)
    var shallow_h = ctx.enqueue_create_host_buffer[DType.uint8](ring_row_elems)
    with ctx.push_context():
        ctx.enqueue_copy(deep_pool_h.unsafe_ptr(), deep_pool_dev)
        ctx.enqueue_copy(shallow_pool_h.unsafe_ptr(), shallow_pool_dev)
        ctx.enqueue_copy(
            front_h.unsafe_ptr(),
            deep_ring_dev.create_sub_buffer[DType.uint8](0, ring_row_elems),
        )
        ctx.enqueue_copy(
            deep_h.unsafe_ptr(),
            deep_ring_dev.create_sub_buffer[DType.uint8](
                deep_row * ring_row_elems, ring_row_elems
            ),
        )
        ctx.enqueue_copy(
            shallow_h.unsafe_ptr(),
            shallow_ring_dev.create_sub_buffer[DType.uint8](
                ring_row_elems, ring_row_elems
            ),
        )
    ctx.synchronize()

    for e in range(ring_row_elems):
        assert_equal(
            front_h[e], sentinel, "the deep row aliased onto the front row"
        )
        assert_equal(
            deep_h[e],
            shallow_h[e],
            "the deep row's records differ from the shallow row's",
        )
    assert_equal(
        Float64(deep_pool_h[0]),
        Float64(shallow_pool_h[0]),
        "the fold of the deep row diverged",
    )
    assert_equal(
        Float64(shallow_pool_h[0]) != Float64(initial),
        True,
        "the fold left the pool unchanged, so the comparison proves nothing",
    )


def test_ring_verify_readout_is_neutral() raises:
    # Row 0 is longer than the ring.
    with DeviceContext() as ctx:
        run_readout_neutrality[32, 32, 4](
            RingCase(1, 6, 2, [6, 4, 1], [0, 0, 0]), ctx
        )


def test_full_accept_is_bit_exact() raises:
    with DeviceContext() as ctx:
        run_ring_case[32, 32, 4](RingCase(2, 4, 2, [4, 4], [4, 4]), ctx)


def test_zero_accept_keeps_the_pool() raises:
    with DeviceContext() as ctx:
        run_ring_case[32, 32, 4](RingCase(2, 4, 2, [4, 4], [0, 0]), ctx)


def test_mixed_accept_across_the_batch() raises:
    with DeviceContext() as ctx:
        run_ring_case[32, 32, 4](
            RingCase(2, 6, 2, [4, 4, 4, 4], [0, 1, 2, 4]), ctx
        )


def test_ring_longer_than_the_window() raises:
    with DeviceContext() as ctx:
        run_ring_case[32, 32, 8](RingCase(2, 4, 2, [4, 4], [3, 1]), ctx)


def test_the_shortest_ring() raises:
    with DeviceContext() as ctx:
        run_ring_case[32, 32, 2](RingCase(3, 4, 2, [2, 2], [1, 2]), ctx)


def test_ragged_windows() raises:
    with DeviceContext() as ctx:
        run_ring_case[32, 32, 4](RingCase(2, 6, 3, [1, 4, 2], [1, 3, 0]), ctx)


def test_production_head_dims() raises:
    with DeviceContext() as ctx:
        run_ring_case[128, 128, 4](RingCase(2, 12, 4, [4, 4], [2, 4]), ctx)


def test_production_head_dims_shortest_ring() raises:
    with DeviceContext() as ctx:
        run_ring_case[128, 128, 2](RingCase(2, 12, 4, [2, 2], [1, 2]), ctx)


def test_production_head_dims_longest_ring() raises:
    with DeviceContext() as ctx:
        run_ring_case[128, 128, 8](
            RingCase(2, 12, 4, [8, 8, 5], [8, 3, 5]), ctx
        )


def test_row_longer_than_the_ring_commits_itself() raises:
    """A row longer than `RING_LEN` records nothing and lands its whole window.

    The fold is given the row's full length, which exceeds the ring.
    """
    with DeviceContext() as ctx:
        run_ring_case[32, 32, 4](
            RingCase(2, 4, 2, [6], [6]),
            ctx,
            ring_silent_rows=[0],
        )


def test_long_and_short_rows_share_a_launch() raises:
    """A row longer than the ring and a recorded row share one launch.

    Row 0 records nothing and lands its window, and row 1 folds 3 of its 4
    records.
    """
    with DeviceContext() as ctx:
        run_ring_case[32, 32, 4](
            RingCase(2, 4, 2, [6, 4], [6, 3]),
            ctx,
            ring_silent_rows=[0],
        )


def test_long_row_route_discriminates() raises:
    """With `RING_LEN` 8, row 0's six records overwrite its poison."""
    with DeviceContext() as ctx:
        run_ring_case[32, 32, 8](
            RingCase(2, 4, 2, [6, 4], [0, 3]),
            ctx,
            ring_silent_rows=[0],
            expect_ring_silent=False,
        )


def test_ring_row_past_a_32_bit_offset() raises:
    with DeviceContext() as ctx:
        run_deep_ring_row(ctx)


def test_accept_count_discriminates() raises:
    with DeviceContext() as ctx:
        run_ring_case[32, 32, 4](
            RingCase(2, 4, 2, [4, 4], [3, 3]),
            ctx,
            fold_accepted_override=[2, 3],
            expect_match=False,
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
