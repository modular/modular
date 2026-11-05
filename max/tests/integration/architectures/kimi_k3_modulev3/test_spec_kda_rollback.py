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
"""A KDA verify plus rollback lands exactly where a plain decode would.

Every arm runs the two-layer KDA stack twice from identically-initialized
pools: once as a speculative verify over each request's whole ``K + 1``
window followed by the accepted-prefix rollback, and once as the plain
forward over exactly the accepted tokens, one decode step at a time -- the
arm non-spec decoding itself runs (a multi-row ``kda_decode`` call routes
to the chunk pipeline on gfx950, whose rounding is not the decode arm's).
The live recurrent pool and the conv window must match bit for bit
afterwards -- the greedy-matching contract the K3 pipeline owes its
non-spec decoding.

Parametrized over the mesh: TP1 runs the ragged batch (accepted 0, 1, 3
and 7 drafts in one verify, so every per-request fold count lands in one
graph) and a committed ``K = 0`` window; TP2 runs the ragged batch through
the production ``PerShard`` ring, per-device conv shards and real signal
buffers. Bit-identity at TP>1 is only provable inside one execution --
cross-run comparisons read whatever uninitialized memory a run holds -- so
both arms of a TP2 case are values of one realization.

The head width is K3's own -- 128, the one the KDA kernels are compiled
for. The rest of the geometry is the test's: dyadic weights for float32
exactness, and two heads so at TP2 each rank's beta projection is an
N==1 GEMV.

RED controls: a fold one record past the accepted count (at both degrees,
asserted on both pools -- the conv window slides with the same corrupted
count) and a live/ring row-table swap must both come off the
accepted-only reference, proving the bit-identity oracle can fail.
"""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import Accelerator
from max.dtype import DType
from max.graph import DeviceRef, Dim, TensorValue
from max.nn.kv_cache import (
    RecurrentLeafInputs,
    RecurrentStateInputsPerDevice,
)
from max.pipelines.architectures.kimi_k3_modulev3.layers.kda_attention import (
    KimiK3KDAAttention,
    tensor_parallel_kda_attention,
)
from max.pipelines.architectures.kimi_k3_modulev3.spec_state import (
    KimiK3RecurrentState,
    fold_kda_state_pools,
)
from max.pipelines.architectures.kimi_k3_modulev3.state_cache import (
    VERIFY_RING_LEAF_ID,
    kda_state_regions,
    verify_ring_leaf,
)
from max.pipelines.architectures.kimi_k3_modulev3.weight_adapters import (
    interleave_for_tp,
)
from max.pipelines.speculative.gated_delta_rollback import accepted_lengths
from max.pipelines.speculative.ragged_token_merger import shape_to_scalar
from max.tensor import Tensor, default_device
from max.tensor.realization_context import (
    EagerRealizationContext,
    current_realization_context,
    ensure_context,
)
from max.tensor.sharding import DeviceMesh, NamedMapping

_NUM_LAYERS = 2
_HIDDEN = 64
_HEADS = 2
"""Two heads: one per rank at TP2, where each rank's beta projection is
an N==1 GEMV."""

_HEAD_DIM = 128
"""K3's own head width, the one the KDA kernels are compiled for."""

_CONV_KERNEL = 4
_INNER = _HEADS * _HEAD_DIM
_CONV_DIM = 3 * _INNER
_BATCH = 4
_WINDOW = 8
"""The verify window: one anchor row plus K3's seven drafts, the ring
length the pipeline declares. A committed ``K = 0`` window is one row."""

_POOL_ROWS = _NUM_LAYERS * _BATCH + 3
_RING_ROWS_ALLOC = 2 * _POOL_ROWS
"""Rows the ring pool is allocated: the ring row table offsets each row
past every live one, so the ring needs rows well beyond ``_POOL_ROWS``."""

_POISON = np.float32(-7.75e18)
"""Ring fill no request writes, so a mis-folded record shows."""


def _sparse(
    rng: np.random.Generator, out_dim: int, in_dim: int, scale: float
) -> np.ndarray:
    """Returns a weight with four entries of ``+-scale`` per output row, whose
    products and sums no float32 matmul rounding touches at any row count."""
    weight = np.zeros((out_dim, in_dim), dtype=np.float32)
    for row in range(out_dim):
        cols = rng.choice(in_dim, size=4, replace=False)
        weight[row, cols] = rng.choice([-scale, scale], size=4)
    return weight


def _ring_row_shape(num_devices: int) -> tuple[int, ...]:
    """The ring row the cache declares at ``num_devices``, stride included."""
    (ring,) = (
        region
        for region in kda_state_regions(
            num_kda_layers=_NUM_LAYERS,
            conv_dim=_CONV_DIM,
            conv_kernel_size=_CONV_KERNEL,
            num_heads=_HEADS,
            head_dim=_HEAD_DIM,
            conv_dtype=DType.float32,
            recurrent_dtype=DType.float32,
            num_devices=num_devices,
            ring_len=_WINDOW,
        )
        if region.leaf_id == VERIFY_RING_LEAF_ID
    )
    return ring.row_shape


def _build_layers(mesh: DeviceMesh) -> list[KimiK3KDAAttention]:
    """Two KDA layers with distinct dyadic weights, sharded over the mesh."""
    num_devices = mesh.num_devices
    rng = np.random.default_rng(20261103)
    qkv = [_sparse(rng, _INNER, _HIDDEN, 0.5) for _ in range(3)]
    conv_weight = (
        rng.integers(-4, 5, size=(_CONV_DIM, _CONV_KERNEL)) / 8
    ).astype(np.float32)
    layers = []
    for _ in range(_NUM_LAYERS):
        # The plain-decode arm of the bit-identity oracle is the PINNED
        # decode a speculative build runs: its verify-ring and fold
        # kernels spell their multiply-adds as explicit `fma`s, so the
        # decode they must equal is pinned the same way.
        layer = KimiK3KDAAttention(
            hidden_size=_HIDDEN,
            num_heads=_HEADS,
            head_dim=_HEAD_DIM,
            conv_kernel_size=_CONV_KERNEL,
            gate_lower_bound=-5.0,
            pin_fma=True,
        )
        # The merged projection is laid down as [q ; k ; v] rows at a
        # tensor-parallel degree of one, and interleaved per rank the way
        # the weight adapter writes them at a degree of two, so the TP pass
        # below can split each rank's slice of every q/k/v block.
        layer.qkv_proj.weight = Tensor(
            (
                np.concatenate(qkv, axis=0)
                if num_devices == 1
                else interleave_for_tp(qkv, num_devices)
            ),
            dtype=DType.float32,
        )
        layer.conv_weight = Tensor(
            (
                conv_weight
                if num_devices == 1
                else interleave_for_tp(np.split(conv_weight, 3), num_devices)
            ),
            dtype=DType.float32,
        )
        layer.f_a_proj.weight = Tensor(
            _sparse(rng, _HEAD_DIM, _HIDDEN, 0.25), dtype=DType.float32
        )
        layer.f_b_proj.weight = Tensor(
            _sparse(rng, _INNER, _HEAD_DIM, 0.25), dtype=DType.float32
        )
        layer.b_proj.weight = Tensor(
            _sparse(rng, _HEADS, _HIDDEN, 0.5), dtype=DType.float32
        )
        layer.g_proj.weight = Tensor(
            _sparse(rng, _INNER, _HIDDEN, 0.5), dtype=DType.float32
        )
        layer.o_proj.weight = Tensor(
            _sparse(rng, _HIDDEN, _INNER, 0.5), dtype=DType.float32
        )
        layer.o_norm.weight = Tensor(
            np.ones(_HEAD_DIM, dtype=np.float32), dtype=DType.float32
        )
        layer.a_log = Tensor(
            (rng.integers(-4, 5, size=(_HEADS, _HEAD_DIM)) / 8).astype(
                np.float32
            ),
            dtype=DType.float32,
        )
        layer.dt_bias = Tensor(
            (rng.integers(-8, 9, size=(_HEADS, _HEAD_DIM)) / 16).astype(
                np.float32
            ),
            dtype=DType.float32,
        )
        if num_devices > 1:
            layer = tensor_parallel_kda_attention(layer, mesh)
        layers.append(layer)
    return layers


def _row_tables() -> tuple[np.ndarray, np.ndarray]:
    """The live and ring ``[num_layers, batch]`` row tables.

    One block per request, layer-major, so row ``(layer, request)`` is
    ``request * num_layers + layer`` -- the allocation
    ``RecurrentStateRegion.rows_of`` describes. The ring rows offset past
    every live row, so a live/ring mix-up reads or writes the wrong rows.
    """
    live = np.array(
        [
            [request * _NUM_LAYERS + layer for request in range(_BATCH)]
            for layer in range(_NUM_LAYERS)
        ],
        dtype=np.uint32,
    )
    return live, live + _POOL_ROWS


def _row_table_tensors(table: np.ndarray) -> list[Tensor]:
    """Returns the row table's per-layer slices, as tensors."""
    return [
        Tensor(table[position], dtype=DType.uint32)
        for position in range(_NUM_LAYERS)
    ]


class _ArmPools:
    """One arm's pools, sharded on the channel axis like the heads they hold.

    The layer calls take the distributed tensors; the state leaves and the
    pool readback take each device's ``local_shards`` entry, which wraps the
    same per-device buffer, so the kernels' in-place writes reach both.
    """

    def __init__(
        self,
        conv_init: np.ndarray,
        rec_init: np.ndarray,
        mesh: DeviceMesh,
        ring: bool,
    ) -> None:
        num_devices = mesh.num_devices
        live, ring_rows = _row_tables()
        self.conv = Tensor(conv_init, dtype=DType.float32).to(
            NamedMapping(mesh, (None, "tp", None))
        )
        self.recurrent = Tensor(rec_init, dtype=DType.float32).to(
            NamedMapping(mesh, (None, "tp", None, None))
        )
        self.rows = [
            Tensor(live, dtype=DType.uint32).to(device)
            for device in mesh.devices
        ]
        self.ring_rows = [
            Tensor(ring_rows, dtype=DType.uint32).to(device)
            for device in mesh.devices
        ]
        self.ring = (
            Tensor(
                np.full(
                    (
                        _RING_ROWS_ALLOC,
                        _HEADS,
                        *_ring_row_shape(num_devices)[1:],
                    ),
                    _POISON,
                    dtype=np.float32,
                ),
                dtype=DType.float32,
            ).to(NamedMapping(mesh, (None, "tp", None, None)))
            if ring
            else None
        )

    def state(self) -> list[RecurrentStateInputsPerDevice[Tensor, Tensor]]:
        """The per-device state leaves, the ring declared last."""
        assert self.ring is not None, "only the speculative arm holds a ring"
        conv = self.conv.local_shards
        recurrent = self.recurrent.local_shards
        ring = self.ring.local_shards
        return [
            RecurrentStateInputsPerDevice(
                leaves=(
                    RecurrentLeafInputs(
                        pool=conv[d], live_row_ids=self.rows[d]
                    ),
                    RecurrentLeafInputs(
                        pool=recurrent[d], live_row_ids=self.rows[d]
                    ),
                    RecurrentLeafInputs(
                        pool=ring[d], live_row_ids=self.ring_rows[d]
                    ),
                )
            )
            for d in range(len(conv))
        ]


def _pool_inits() -> tuple[np.ndarray, np.ndarray]:
    """The pre-verify pools both arms start from, identically."""
    rng = np.random.default_rng(20261102)
    conv = rng.standard_normal(
        (_POOL_ROWS, _CONV_DIM, _CONV_KERNEL - 1)
    ).astype(np.float32)
    rec = (
        rng.standard_normal((_POOL_ROWS, _HEADS, _HEAD_DIM, _HEAD_DIM)) * 0.1
    ).astype(np.float32)
    return conv, rec


def _window_inputs(
    rng: np.random.Generator, window: int
) -> tuple[np.ndarray, np.ndarray]:
    """Returns the packed verify window and its per-request offsets: the
    hidden states are -1, 0 and 1 values in float32."""
    x = rng.integers(-1, 2, size=(_BATCH * window, _HIDDEN)).astype(np.float32)
    offsets = np.arange(_BATCH + 1, dtype=np.uint32) * np.uint32(window)
    return x, offsets


def _decode_steps(
    x: np.ndarray, live: np.ndarray, keep: list[int], window: int
) -> list[tuple[Tensor, Tensor, list[Tensor]]]:
    """Returns one plain-decode step per token position: the requests
    still advancing at that position, their hidden states, their
    per-request offsets, and their per-layer row tables."""
    steps = []
    for t in range(max(keep)):
        members = [b for b in range(_BATCH) if keep[b] > t]
        steps.append(
            (
                Tensor(
                    x[[b * window + t for b in members]],
                    dtype=DType.float32,
                ),
                Tensor(
                    np.arange(len(members) + 1, dtype=np.uint32),
                    dtype=DType.uint32,
                ),
                _row_table_tensors(live[:, members]),
            )
        )
    return steps


@pytest.mark.compile_on_gpu(
    "the verify, fold and conv replay mutate their pools in place"
)
@pytest.mark.parametrize(
    ("num_devices", "window", "accepted"),
    [
        pytest.param(1, _WINDOW, [0, 3, 7, 1], id="tp1_ragged"),
        pytest.param(1, 1, [0] * _BATCH, id="tp1_committed"),
        pytest.param(2, _WINDOW, [0, 3, 7, 1], id="tp2_ragged"),
    ],
)
def test_rollback_lands_the_pools_on_the_accepted_prefix(
    num_devices: int,
    window: int,
    accepted: list[int],
) -> None:
    """Verify + fold + conv replay == the plain decode over exactly the
    accepted tokens, bit for bit, on each device of the mesh."""
    keep = [1 + a for a in accepted]
    rng = np.random.default_rng(20261104)
    x, window_offsets = _window_inputs(rng, window)
    live, _ = _row_tables()

    devices = [Accelerator(i) for i in range(num_devices)]
    mesh = DeviceMesh(
        devices=tuple(devices), mesh_shape=(num_devices,), axis_names=("tp",)
    )
    with default_device(mesh):
        layers = _build_layers(mesh)
        # Both arms start from identically-initialized pools.
        conv_init, rec_init = _pool_inits()
        spec_pools = _ArmPools(conv_init, rec_init, mesh, ring=True)
        ref_pools = _ArmPools(conv_init, rec_init, mesh, ring=False)
        offsets = [
            Tensor(window_offsets, dtype=DType.uint32).to(device)
            for device in devices
        ]
        num_accepted = Tensor(
            np.array(accepted, dtype=np.int64), dtype=DType.int64
        ).to(devices[0])
        # The calls' operands, realized before the context opens so the
        # dispatch hands each device its copy the way a served step does.
        x_window = Tensor(x, dtype=DType.float32)
        window_offsets_t = Tensor(window_offsets, dtype=DType.uint32)
        live_rows = _row_table_tensors(live)
        # The plain-decode reference: the accepted tokens, one per request
        # per call, the way non-spec decode advances.
        steps = _decode_steps(x, live, keep, window)

        # Both arms are values of one realization: a TP>1 run's pools
        # cannot be compared against another run's.
        with ensure_context():
            spec_state = KimiK3RecurrentState(layers)
            state = spec_pools.state()
            with spec_state.capturing(Dim(window - 1), state):
                for position, layer in enumerate(layers):
                    layer(
                        x_window,
                        spec_pools.conv,
                        live_rows[position],
                        spec_pools.recurrent,
                        live_rows[position],
                        window_offsets_t,
                    )
            if num_devices > 1:
                ctx = current_realization_context()
                assert isinstance(ctx, EagerRealizationContext)
                signal_buffers = ctx.ensure_signal_buffers(mesh)
                assert signal_buffers is not None, (
                    "TP2 needs real signal buffers"
                )
                merged_offsets_per_dev = offsets
            else:
                signal_buffers = []
                merged_offsets_per_dev = [offsets[0]]
            spec_state.roll_forward(
                state,
                merged_offsets=offsets[0],
                merged_offsets_per_dev=merged_offsets_per_dev,
                num_accepted=num_accepted,
                signal_buffers=signal_buffers,
                device=DeviceRef.GPU(0),
            )
            # The rollback spends the capture: a later build that skips
            # `capturing` must not replay this one's values.
            assert spec_state.verify_width is None
            assert not spec_state.captures
            assert spec_state.num_draft_tokens is None

            for x_t, step_offsets, step_rows in steps:
                for position, layer in enumerate(layers):
                    layer(
                        x_t,
                        ref_pools.conv,
                        step_rows[position],
                        ref_pools.recurrent,
                        step_rows[position],
                        step_offsets,
                    )

        for d, device in enumerate(devices):
            np.testing.assert_array_equal(
                spec_pools.conv.local_shards[d].to_numpy(),
                ref_pools.conv.local_shards[d].to_numpy(),
                err_msg=f"conv pool, device {device}",
            )
            np.testing.assert_array_equal(
                spec_pools.recurrent.local_shards[d].to_numpy(),
                ref_pools.recurrent.local_shards[d].to_numpy(),
                err_msg=f"recurrent pool, device {device}",
            )


@pytest.mark.compile_on_gpu(
    "the verify, fold and conv replay mutate their pools in place"
)
def test_fold_past_the_accepted_count_diverges() -> None:
    """RED control: one record past the accepted count leaves a trace.

    Proves the bit-identity oracle above can fail, so a green there means
    something. The count the kernel clamps to the window still moves every
    request that accepted a partial window -- and the conv replay slides
    with the same corrupted count, so both pools must come off the
    reference.
    """
    num_drafts, accepted = 7, [0, 3, 7, 1]
    window = num_drafts + 1
    rng = np.random.default_rng(20261105)
    x, window_offsets = _window_inputs(rng, window)
    device = Accelerator()
    mesh = DeviceMesh((device,), (1,), ("tp",))
    live, _ = _row_tables()
    with default_device(mesh):
        layers = _build_layers(mesh)
        conv_init, rec_init = _pool_inits()

        def run(over_count: list[int]) -> tuple[np.ndarray, np.ndarray]:
            pools = _ArmPools(conv_init, rec_init, mesh, ring=True)
            spec_state = KimiK3RecurrentState(layers)
            offsets = Tensor(window_offsets, dtype=DType.uint32).to(device)
            # Realized before the context opens, like the oracle above.
            x_window = Tensor(x, dtype=DType.float32)
            live_rows = _row_table_tensors(live)
            state = pools.state()
            with ensure_context():
                with spec_state.capturing(Dim(num_drafts), state):
                    for position, layer in enumerate(layers):
                        layer(
                            x_window,
                            pools.conv,
                            live_rows[position],
                            pools.recurrent,
                            live_rows[position],
                            offsets,
                        )
                spec_state.roll_forward(
                    state,
                    merged_offsets=offsets,
                    merged_offsets_per_dev=[offsets],
                    num_accepted=Tensor(
                        np.array(over_count, dtype=np.int64),
                        dtype=DType.int64,
                    ).to(device),
                    signal_buffers=[],
                    device=DeviceRef.GPU(),
                )
            return (
                pools.conv.local_shards[0].to_numpy(),
                pools.recurrent.local_shards[0].to_numpy(),
            )

        reference = run(accepted)
        over = run([a + 1 for a in accepted])

    assert not np.array_equal(over[1], reference[1]), (
        "a fold one record past the accepted count must diverge (recurrent)"
    )
    assert not np.array_equal(over[0], reference[0]), (
        "the conv replay fed the same over-count must diverge too"
    )


@pytest.mark.compile_on_gpu(
    "the verify, fold and conv replay mutate their pools in place"
)
def test_tp2_fold_past_the_accepted_count_diverges() -> None:
    """The same RED control at TP2, both arms of one realization.

    The fold is fed one record past each request's accepted count through
    the production per-device signal buffers; every device's pools must
    come off the reference the true counts land on, or the TP2 oracle
    above cannot tell an accepted count from another.
    """
    accepted = [0, 3, 7, 1]
    window = _WINDOW
    rng = np.random.default_rng(20261106)
    x, window_offsets = _window_inputs(rng, window)
    live, _ = _row_tables()
    keep = [1 + a for a in accepted]

    devices = [Accelerator(0), Accelerator(1)]
    mesh = DeviceMesh(
        devices=tuple(devices), mesh_shape=(2,), axis_names=("tp",)
    )
    with default_device(mesh):
        layers = _build_layers(mesh)
        conv_init, rec_init = _pool_inits()
        spec_pools = _ArmPools(conv_init, rec_init, mesh, ring=True)
        ref_pools = _ArmPools(conv_init, rec_init, mesh, ring=False)
        offsets = [
            Tensor(window_offsets, dtype=DType.uint32).to(device)
            for device in devices
        ]
        x_window = Tensor(x, dtype=DType.float32)
        window_offsets_t = Tensor(window_offsets, dtype=DType.uint32)
        live_rows = _row_table_tensors(live)
        steps = _decode_steps(x, live, keep, window)

        with ensure_context():
            spec_state = KimiK3RecurrentState(layers)
            state = spec_pools.state()
            with spec_state.capturing(Dim(window - 1), state):
                for position, layer in enumerate(layers):
                    layer(
                        x_window,
                        spec_pools.conv,
                        live_rows[position],
                        spec_pools.recurrent,
                        live_rows[position],
                        window_offsets_t,
                    )
            ctx = current_realization_context()
            assert isinstance(ctx, EagerRealizationContext)
            signal_buffers = ctx.ensure_signal_buffers(mesh)
            assert signal_buffers is not None, "TP2 needs real signal buffers"
            spec_state.roll_forward(
                state,
                merged_offsets=offsets[0],
                merged_offsets_per_dev=offsets,
                num_accepted=Tensor(
                    np.array([a + 1 for a in accepted], dtype=np.int64),
                    dtype=DType.int64,
                ).to(devices[0]),
                signal_buffers=signal_buffers,
                device=DeviceRef.GPU(0),
            )

            for x_t, step_offsets, step_rows in steps:
                for position, layer in enumerate(layers):
                    layer(
                        x_t,
                        ref_pools.conv,
                        step_rows[position],
                        ref_pools.recurrent,
                        step_rows[position],
                        step_offsets,
                    )

        for d, device in enumerate(devices):
            assert not np.array_equal(
                spec_pools.conv.local_shards[d].to_numpy(),
                ref_pools.conv.local_shards[d].to_numpy(),
            ), (
                f"a TP2 fold past the accepted count went unseen (conv, {device})"
            )
            assert not np.array_equal(
                spec_pools.recurrent.local_shards[d].to_numpy(),
                ref_pools.recurrent.local_shards[d].to_numpy(),
            ), (
                f"a TP2 fold past the accepted count went unseen"
                f" (recurrent, {device})"
            )


@pytest.mark.compile_on_gpu(
    "the verify, fold and conv replay mutate their pools in place"
)
def test_live_ring_row_swap_diverges() -> None:
    """RED control: folding under swapped live/ring rows lands elsewhere.

    The row tables are the only thing that distinguishes the roles at the
    fold, so swapping them must write the live pool at the ring rows and
    leave the live rows where the verify left them, at their pre-verify
    values.
    """
    num_drafts, accepted = 7, [0, 3, 7, 1]
    window = num_drafts + 1
    rng = np.random.default_rng(20261107)
    x, window_offsets = _window_inputs(rng, window)
    device = Accelerator()
    mesh = DeviceMesh((device,), (1,), ("tp",))
    live, _ = _row_tables()
    with default_device(mesh):
        layers = _build_layers(mesh)
        conv_init, rec_init = _pool_inits()
        reference = _ArmPools(conv_init, rec_init, mesh, ring=False)
        offsets = Tensor(window_offsets, dtype=DType.uint32).to(device)
        num_accepted = Tensor(
            np.array(accepted, dtype=np.int64), dtype=DType.int64
        ).to(device)
        x_window = Tensor(x, dtype=DType.float32)
        # The reference decode: the accepted tokens, one per request per
        # call.
        keep = [1 + a for a in accepted]
        with ensure_context():
            for t in range(max(keep)):
                members = [b for b in range(_BATCH) if keep[b] > t]
                step_x = Tensor(
                    x[[b * window + t for b in members]],
                    dtype=DType.float32,
                )
                step_offsets = Tensor(
                    np.arange(len(members) + 1, dtype=np.uint32),
                    dtype=DType.uint32,
                )
                step_rows = [
                    Tensor(live[position][members], dtype=DType.uint32)
                    for position in range(_NUM_LAYERS)
                ]
                for position, layer in enumerate(layers):
                    layer(
                        step_x,
                        reference.conv,
                        step_rows[position],
                        reference.recurrent,
                        step_rows[position],
                        step_offsets,
                    )

        swapped = _ArmPools(conv_init, rec_init, mesh, ring=True)
        spec_state = KimiK3RecurrentState(layers)
        # The capture reads each layer's own live rows; only the fold below
        # is sabotaged.
        capture_rows = _row_table_tensors(live)
        with ensure_context():
            with spec_state.capturing(Dim(num_drafts), swapped.state()):
                for position, layer in enumerate(layers):
                    layer(
                        x_window,
                        swapped.conv,
                        capture_rows[position],
                        swapped.recurrent,
                        capture_rows[position],
                        offsets,
                    )
            assert spec_state.verify_width is not None
            assert swapped.ring is not None, "the verify arm holds a ring"
            fold_kda_state_pools(
                [swapped.recurrent],
                [swapped.ring_rows[0]],
                [swapped.ring],
                [swapped.rows[0]],
                accepted_lengths(
                    TensorValue(offsets),
                    TensorValue(num_accepted),
                    shape_to_scalar(Dim(num_drafts), DeviceRef.GPU()),
                ),
                [offsets],
                [],
                spec_state.verify_width,
            )

        assert not np.array_equal(
            swapped.recurrent.local_shards[0].to_numpy(),
            reference.recurrent.local_shards[0].to_numpy(),
        ), "a live/ring row-table swap must diverge"


def test_the_ring_role_is_asserted_by_its_shape() -> None:
    """A live leaf read as the ring is refused, not folded through.

    The leaves are positional, so the ring is told by its record axis: a
    third leaf shaped like the live recurrent pool is a wiring bug the
    bind refuses -- production reads the ring through
    ``KimiK3RecurrentState.capturing``, which routes every device's
    leaves through this same check.
    """
    rows = Tensor(
        np.zeros((_NUM_LAYERS, _BATCH), dtype=np.uint32), dtype=DType.uint32
    )

    def pool(shape: tuple[int, ...]) -> Tensor:
        return Tensor.zeros(shape, dtype=DType.float32)

    state: list[RecurrentStateInputsPerDevice[Tensor, Tensor]] = [
        RecurrentStateInputsPerDevice(
            leaves=(
                RecurrentLeafInputs(
                    pool=pool((_POOL_ROWS, _CONV_DIM, _CONV_KERNEL - 1)),
                    live_row_ids=rows,
                ),
                RecurrentLeafInputs(
                    pool=pool((_POOL_ROWS, _HEADS, _HEAD_DIM, _HEAD_DIM)),
                    live_row_ids=rows,
                ),
                # A recurrent-shaped pool in the ring's slot.
                RecurrentLeafInputs(
                    pool=pool((_POOL_ROWS, _HEADS, _HEAD_DIM, _HEAD_DIM)),
                    live_row_ids=rows,
                ),
            )
        )
    ]
    with pytest.raises(AssertionError, match="not the verify ring"):
        verify_ring_leaf(state[0])


def test_capturing_refuses_a_cache_with_no_ring() -> None:
    """A verify against a ring-less cache fails loudly at bind time."""
    device = Accelerator()
    mesh = DeviceMesh((device,), (1,), ("tp",))
    with default_device(mesh):
        layers = _build_layers(mesh)
    rows = Tensor(
        np.zeros((_NUM_LAYERS, _BATCH), dtype=np.uint32), dtype=DType.uint32
    )

    def pool(shape: tuple[int, ...]) -> Tensor:
        return Tensor.zeros(shape, dtype=DType.float32)

    state: list[RecurrentStateInputsPerDevice[Tensor, Tensor]] = [
        RecurrentStateInputsPerDevice(
            leaves=(
                RecurrentLeafInputs(
                    pool=pool((_POOL_ROWS, _CONV_DIM, _CONV_KERNEL - 1)),
                    live_row_ids=rows,
                ),
                RecurrentLeafInputs(
                    pool=pool((_POOL_ROWS, _HEADS, _HEAD_DIM, _HEAD_DIM)),
                    live_row_ids=rows,
                ),
            )
        )
    ]
    spec_state = KimiK3RecurrentState(layers)
    with pytest.raises(AssertionError, match="verify ring"):
        with spec_state.capturing(Dim(7), state):
            pass


def test_roll_forward_refuses_a_verify_that_captured_nothing() -> None:
    """An empty capture has no rows to replay, and is refused by assertion.

    ``capturing`` enters and leaves cleanly around a verify whose forward
    never ran, so the rollback's contract check must fire rather than
    index into the empty per-device capture.
    """
    device = Accelerator()
    mesh = DeviceMesh((device,), (1,), ("tp",))
    with default_device(mesh):
        layers = _build_layers(mesh)
    rows = Tensor(
        np.zeros((_NUM_LAYERS, _BATCH), dtype=np.uint32), dtype=DType.uint32
    )

    def pool(shape: tuple[int, ...]) -> Tensor:
        return Tensor.zeros(shape, dtype=DType.float32)

    state: list[RecurrentStateInputsPerDevice[Tensor, Tensor]] = [
        RecurrentStateInputsPerDevice(
            leaves=(
                RecurrentLeafInputs(
                    pool=pool((_POOL_ROWS, _CONV_DIM, _CONV_KERNEL - 1)),
                    live_row_ids=rows,
                ),
                RecurrentLeafInputs(
                    pool=pool((_POOL_ROWS, _HEADS, _HEAD_DIM, _HEAD_DIM)),
                    live_row_ids=rows,
                ),
                RecurrentLeafInputs(
                    pool=pool((_POOL_ROWS, *_ring_row_shape(1))),
                    live_row_ids=rows,
                ),
            )
        )
    ]
    spec_state = KimiK3RecurrentState(layers)
    with ensure_context():
        with spec_state.capturing(Dim(7), state):
            pass
        offsets = Tensor(
            np.arange(_BATCH + 1, dtype=np.uint32) * np.uint32(_WINDOW),
            dtype=DType.uint32,
        )
        with pytest.raises(AssertionError, match="captured"):
            spec_state.roll_forward(
                state,
                merged_offsets=offsets,
                merged_offsets_per_dev=[offsets],
                num_accepted=Tensor(
                    np.zeros(_BATCH, dtype=np.int64), dtype=DType.int64
                ),
                signal_buffers=[],
                device=DeviceRef.GPU(),
            )
