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

"""Multi-GPU execution test for the fused allreduce + RMSNorm graph op.

Runs the Mojo handler against an independent numpy fp64 reference of the
semantics of the dispatch arm it takes on this target: on AMD the payloads
below fuse, and the norm consumes the exact float32 peer sum; on other
targets the op is today's unfused chain, whose allreduce rounds the sum to
bfloat16 before the norm reads it. The gamma cast order is selected by
``multiply_before_cast`` either way. The row counts cross the kernel's
dispatch arms at two ranks and the production column count: 1 and 8 rows
take the 1-stage kernel, 40 rows exceeds the 128 KB per-rank threshold and
takes the 2-stage kernel.

The data is rank-distinct and signed, and (in the multi-row cases) the
last row nearly cancels across ranks so the RMS is epsilon-dominated:
dropping or duplicating a rank, or ignoring ``epsilon``, changes the
reference far past the tolerance.

The norm weight is the same replicated gamma on every device -- the
production form, since the 2-stage dispatch arm normalizes each row shard
on its owner rank and then all-gathers the result, so the arm is only
correct for a shared gamma. The gamma is not 1.0, so an op that drops the
gamma still fails on every arm.
"""

from __future__ import annotations

import ml_dtypes
import numpy as np
import pytest
from max.driver import (
    CPU,
    Accelerator,
    Buffer,
    accelerator_api,
    accelerator_count,
)
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import BufferType, DeviceRef, Graph, TensorType, ops
from max.nn.legacy import Signals

NUM_GPUS = 2
COLS = 3584  # Kimi K3's routed-expert latent width; bf16 payload.
EPS = 1e-5  # Kimi K3's config rms_norm_eps.
WEIGHT_OFFSET = 0.0  # K3's RMSNorm -> ops.rms_norm default.
# bf16 has an 8-bit mantissa: 1 ULP is a 2**-8 relative step. The kernel's
# float32 pipeline (peer sum, mean square, rsqrt) sits ~2**-20 from an fp64
# reference, so a correct op is at most a rounding-boundary flip, 1 ULP,
# away; 2 ULPs is the gate.
MAX_BF16_ULPS = 2
# Fraction of elements allowed to sit 1 ULP off the reference: a rounding
# flip at an f32/bf16 boundary, not a systematically different cast order.
MAX_FLIPPED_FRACTION = 0.02
# The dispatch's fuse threshold (AR_NORM_FUSE_THRESHOLD in
# comm/allreduce_residual_rmsnorm.mojo): 512 rows x 3584 cols of bfloat16.
# A payload at or below it fuses on an AMD host with P2P; above it -- and
# everywhere on other targets, where the threshold is 0 -- the op runs the
# two-launch chain.
FUSE_THRESHOLD_BYTES = 512 * 3584 * 2
# The fused kernel tiles at most 8192 columns with 8-wide vectors, or 16384
# with 16-wide vectors when the width is a multiple of 16. Wider or
# unaligned-wide column counts must take the two-launch arm instead of
# raising.
UNTILEABLE_COLS = [16400, 8200]


def _fuses_on_target(rows: int, cols: int = COLS) -> bool:
    """Whether the dispatch fuses this payload on this target.

    Assumes peer access between the test's GPUs; without it the op always
    takes the two-launch arm.
    """
    return (
        accelerator_api() == "hip"
        and rows * cols * 2 <= FUSE_THRESHOLD_BYTES
        and (cols <= 8192 or (cols <= 16384 and cols % 16 == 0))
    )


def _to_device_bf16(arr: np.ndarray, device: Accelerator) -> Buffer:
    """Host float array -> bf16 device buffer (uint16 view: DLPack-safe)."""
    bits = np.ascontiguousarray(arr.astype(ml_dtypes.bfloat16).view(np.uint16))
    return Buffer.from_numpy(bits).view(DType.bfloat16).to(device)


def _from_device_bf16(buf: Buffer) -> np.ndarray:
    """bf16 device buffer -> host float32 (via the uint16 view)."""
    out = buf.copy(device=CPU())
    return (
        out.view(DType.uint16)
        .to_numpy()
        .view(ml_dtypes.bfloat16)
        .astype(np.float32)
    )


def _bf16(arr: np.ndarray) -> np.ndarray:
    """Rounds to bfloat16, once, returning the exact values as float32."""
    return arr.astype(ml_dtypes.bfloat16).astype(np.float32)


def _bf16_ulp_distance(got: np.ndarray, want: np.ndarray) -> np.ndarray:
    """Per-element ULP distance between two bfloat16-valued arrays.

    Maps the sign-magnitude bit pattern to a monotone integer key first, so
    negatives compare correctly (-0 is next to +0, not 0x8000 steps away).
    """

    def keys(values: np.ndarray) -> np.ndarray:
        bits = (
            np.ascontiguousarray(values)
            .astype(ml_dtypes.bfloat16)
            .view(np.uint16)
            .astype(np.int32)
        )
        return np.where(bits & 0x8000, 0xFFFF - (bits & 0xFFFF), bits | 0x8000)

    return np.abs(keys(got) - keys(want))


def _ar_rms_norm_graph(
    signals: Signals,
    cols: int,
    multiply_before_cast: bool,
    output_summed: bool = False,
) -> Graph:
    """Builds the op graph with a symbolic row count, fed as data."""
    devices = signals.devices
    input_types: list[TensorType | BufferType] = [
        TensorType(dtype=DType.bfloat16, shape=["rows", cols], device=device)
        for device in devices
    ]
    input_types += [
        TensorType(dtype=DType.bfloat16, shape=[cols], device=device)
        for device in devices
    ]
    input_types += signals.input_types()
    with Graph("allreduce_rms_norm", input_types=input_types) as graph:
        inputs = [v.tensor for v in graph.inputs[:NUM_GPUS]]
        gammas = [v.tensor for v in graph.inputs[NUM_GPUS : 2 * NUM_GPUS]]
        sigs = [v.buffer for v in graph.inputs[2 * NUM_GPUS :]]
        normed, summed = ops.allreduce_rms_norm(
            inputs=inputs,
            signal_buffers=sigs,
            gammas=gammas,
            epsilon=EPS,
            weight_offset=WEIGHT_OFFSET,
            multiply_before_cast=multiply_before_cast,
        )
        if output_summed:
            graph.output(*normed, *summed)
        else:
            # K3's production form: the sum result is dead, so only the
            # norm is carried out of the graph.
            graph.output(*normed)
        return graph


def _inputs_and_gammas(
    rows: int, devices: list[Accelerator], cols: int = COLS
) -> tuple[list[Buffer], list[Buffer], list[np.ndarray], list[np.ndarray]]:
    """Rank-distinct signed inputs plus one replicated gamma on every device.

    In the multi-row cases the last row nearly cancels across ranks (peer
    values are negatives of each other up to a ~1e-3 offset), so the row's
    RMS is epsilon-dominated and the reference hinges on ``epsilon`` being
    applied; the other rows carry full-magnitude data. The single-row case
    -- K3's decode shape -- keeps the full-magnitude data: with only one
    row there is no room for both, and the production shape must exercise
    the whole data path, not a row of zeros.
    """
    tensor_inputs: list[Buffer] = []
    gamma_inputs: list[Buffer] = []
    host_shards: list[np.ndarray] = []
    host_gammas: list[np.ndarray] = []
    positions = np.arange(rows * cols).reshape(rows, cols)
    for i, device in enumerate(devices):
        # Rank-distinct values: each rank's residue run starts where the
        # previous one ended, so a rank dropped or read twice changes the
        # sum at every position.
        offset = i * rows * cols
        idx = (positions + offset) % 251 + 1
        arr = idx.astype(np.float32)
        # Signed: some blocks peak on a negative element, so an absolute
        # value in place of the sum is detectable.
        arr = np.where(positions % 3 == 0, -arr, arr)
        if rows > 1:
            # The epsilon-dominated row: peers cancel to a 1e-3..7e-3
            # offset, a mean square at the scale of EPS, so the norm factor
            # is epsilon-set. The cancel values stay below 0.5, where
            # bfloat16's spacing is at most 2**-9, so the offsets survive
            # the input rounding instead of leaving a row of exact zeros
            # that no data-path bug could touch.
            cancel = ((np.arange(cols) % 251) + 1).astype(np.float32) / 502.0
            arr[-1] = (
                cancel
                if i == 0
                else -cancel + ((np.arange(cols) % 7) + 1) * 1e-3
            )
        host_shards.append(_bf16(arr))
        tensor_inputs.append(_to_device_bf16(arr, device))

        # One replicated gamma: 0.5 plus a small ripple, far from 1.0 so a
        # dropped gamma is visible on every arm.
        gamma = ((np.arange(cols) % 7) * 0.01) + 0.5
        host_gammas.append(_bf16(gamma.astype(np.float32)))
        gamma_inputs.append(_to_device_bf16(gamma, device))
    return tensor_inputs, gamma_inputs, host_shards, host_gammas


def _peer_sum(host_shards: list[np.ndarray]) -> np.ndarray:
    """The exact fp64 sum of every rank's shard."""
    sums = np.zeros_like(host_shards[0], dtype=np.float64)
    for shard in host_shards:
        sums += shard.astype(np.float64)
    return sums


def _reference(
    host_shards: list[np.ndarray],
    host_gammas: list[np.ndarray],
    multiply_before_cast: bool,
    round_sum: bool = False,
) -> list[np.ndarray]:
    """fp64 reference of the op, one entry per device.

    The peer sum is exact (bfloat16 inputs), computed in fp64; the mean
    square, the rsqrt and the gamma cast order follow the fused kernel, with
    one bfloat16 rounding per kernel rounding. ``round_sum`` models the
    two-launch arm instead, whose separate norm reads the sum rounded back
    to bfloat16 (today's allreduce + rms_norm chain).
    """
    sums = _peer_sum(host_shards)
    if round_sum:
        sums = _bf16(sums)
    references = []
    for gamma in host_gammas:
        m2 = np.mean(sums * sums, axis=-1, keepdims=True)
        norm_factor = 1.0 / np.sqrt(m2 + EPS)
        offset_gamma = np.float64(WEIGHT_OFFSET) + gamma.astype(np.float64)
        if multiply_before_cast:
            # Gamma in fp32 (fp64 here), one final bfloat16 rounding.
            references.append(_bf16(sums * norm_factor * offset_gamma))
        else:
            # The normalized value rounds to bfloat16 first, then the
            # bfloat16 gamma multiply rounds again.
            normalized = _bf16(sums * norm_factor)
            references.append(_bf16(normalized * _bf16(offset_gamma)))
    return references


def _check(
    outputs: list[Buffer],
    host_shards: list[np.ndarray],
    host_gammas: list[np.ndarray],
    multiply_before_cast: bool,
    label: str,
    round_sum: bool,
) -> None:
    references = _reference(
        host_shards, host_gammas, multiply_before_cast, round_sum=round_sum
    )
    for dev_idx in range(NUM_GPUS):
        got = _from_device_bf16(outputs[dev_idx])
        want = references[dev_idx]
        assert got.shape == want.shape, (
            f"{label}: output shape on GPU {dev_idx}: {got.shape} != "
            f"{want.shape}"
        )
        ulps = _bf16_ulp_distance(got, want)
        max_ulps = int(ulps.max())
        assert max_ulps <= MAX_BF16_ULPS, (
            f"{label}: normed mismatch on GPU {dev_idx} at "
            f"multiply_before_cast={multiply_before_cast}: max "
            f"{max_ulps} bf16 ULPs (gate {MAX_BF16_ULPS}) at "
            f"np.unravel_index({int(ulps.argmax())}, {got.shape})"
        )
        # The two gamma cast orders differ by at most ~1 ULP per element, so a
        # ULP gate alone cannot tell them apart: an op that computes the
        # wrong cast order still lands inside the gate. It can be told apart
        # by HOW OFTEN it crosses a rounding boundary: the correct order
        # flips only where the f32 pipeline sits on a boundary (~0.01% of
        # elements), while the wrong order disagrees on ~20% of them.
        flipped = float((ulps > 0).mean())
        assert flipped < MAX_FLIPPED_FRACTION, (
            f"{label}: {flipped:.1%} of elements on GPU {dev_idx} sit 1+ ULP "
            f"from the multiply_before_cast={multiply_before_cast} "
            f"reference (gate {MAX_FLIPPED_FRACTION:.1%}); the wrong gamma "
            "cast order disagrees on ~20%"
        )

    # The collective replicates its output: both devices must hold the
    # same bytes. The 2-stage arm computes each row shard once on its
    # owner rank and gathers, so this also pins that a copied row is
    # bit-identical to a locally computed one.
    out0 = _from_device_bf16(outputs[0])
    out1 = _from_device_bf16(outputs[1])
    assert np.array_equal(out0, out1), (
        f"{label}: the two GPUs hold different allreduced outputs"
    )


def _check_two_launch(
    outputs: list[Buffer],
    host_shards: list[np.ndarray],
    host_gammas: list[np.ndarray],
    multiply_before_cast: bool,
    label: str,
) -> None:
    """Checks the two-launch arm's two results against numpy.

    The norm reads the bfloat16-rounded peer sum, under the same flip gate
    that tells the two gamma cast orders apart, and ``summed`` holds that
    rounded sum itself.
    """
    _check(
        outputs,
        host_shards,
        host_gammas,
        multiply_before_cast,
        label,
        round_sum=True,
    )

    # `summed` holds the raw bfloat16-rounded peer sum, on both devices.
    want_sum = _bf16(_peer_sum(host_shards))
    for dev_idx in range(NUM_GPUS):
        got = _from_device_bf16(outputs[NUM_GPUS + dev_idx])
        ulps = _bf16_ulp_distance(got, want_sum)
        assert int(ulps.max()) <= MAX_BF16_ULPS, (
            f"{label}: summed mismatch on GPU {dev_idx}: max "
            f"{int(ulps.max())} bf16 ULPs (gate {MAX_BF16_ULPS})"
        )


class _GraphCache:
    """Compiles each distinct graph structure once, for every case to share."""

    def __init__(self) -> None:
        self.signals = Signals(
            devices=[DeviceRef.GPU(id=i) for i in range(NUM_GPUS)]
        )
        self.devices = [Accelerator(n) for n in range(NUM_GPUS)]
        self.session = InferenceSession(devices=[CPU(), *self.devices])
        self._models: dict[tuple[int, bool, bool], Model] = {}

    def model(
        self, cols: int, multiply_before_cast: bool, output_summed: bool
    ) -> Model:
        key = (cols, multiply_before_cast, output_summed)
        if key not in self._models:
            self._models[key] = self.session.load(
                _ar_rms_norm_graph(
                    self.signals, cols, multiply_before_cast, output_summed
                )
            )
        return self._models[key]


@pytest.fixture(scope="module")
def graphs() -> _GraphCache:
    if NUM_GPUS > accelerator_count():
        pytest.skip(f"Not enough GPUs ({NUM_GPUS}) for allreduce_rms_norm.")
    return _GraphCache()


def _run(
    model: Model,
    signals: Signals,
    devices: list[Accelerator],
    rows: int,
    cols: int = COLS,
) -> tuple[list[Buffer], list[np.ndarray], list[np.ndarray]]:
    """Executes a compiled op graph; returns outputs and the host data."""
    tensor_inputs, gamma_inputs, host_shards, host_gammas = _inputs_and_gammas(
        rows, devices, cols
    )
    outputs: list[Buffer] = []
    for out in model.execute(*tensor_inputs, *gamma_inputs, *signals.buffers()):
        assert isinstance(out, Buffer)
        outputs.append(out)
    return outputs, host_shards, host_gammas


def _execute(
    graphs: _GraphCache,
    rows: int,
    multiply_before_cast: bool,
    output_summed: bool = False,
    cols: int = COLS,
) -> tuple[list[Buffer], list[np.ndarray], list[np.ndarray]]:
    model = graphs.model(cols, multiply_before_cast, output_summed)
    return _run(model, graphs.signals, graphs.devices, rows, cols)


@pytest.mark.parametrize("multiply_before_cast", [False, True])
@pytest.mark.parametrize("rows", [1, 8, 40])
def test_allreduce_rms_norm_execution(
    graphs: _GraphCache, rows: int, multiply_before_cast: bool
) -> None:
    """Both dispatch arms, both gamma cast orders, against an fp64 reference.

    The reference must follow the arm the dispatch takes on this target:
    these payloads all sit under the fuse threshold, so they fuse on AMD
    (the norm reads the unrounded float32 sum) and run the two-launch chain
    elsewhere (the norm reads the bfloat16-rounded sum).
    """
    outputs, host_shards, host_gammas = _execute(
        graphs, rows, multiply_before_cast
    )
    label = f"rows={rows} (mbc={multiply_before_cast})"
    _check(
        outputs,
        host_shards,
        host_gammas,
        multiply_before_cast,
        label,
        round_sum=not _fuses_on_target(rows),
    )


@pytest.mark.parametrize("multiply_before_cast", [False, True])
@pytest.mark.parametrize("rows", [3, 31, 37, 129, 511, 513, 600])
def test_allreduce_rms_norm_non_divisible_rows(
    graphs: _GraphCache, rows: int, multiply_before_cast: bool
) -> None:
    """Row counts that never split evenly across the two ranks.

    Real prompt lengths are arbitrary, but the only row count that had
    reached the row-sharding 2-stage arm was 40, which splits evenly, so a
    rows % ngpus defect would have slipped through. The sweep crosses all
    three arms with rows that divide by two for no rank: on AMD, 3-31 take
    the fused 1-stage kernel, 37-511 the fused 2-stage kernel (whose rank
    shard is ceildiv(rows, 2) rows, so an odd batch leaves one rank owning a
    partial shard and its last scratch slot unused), and 513 and 600 run
    above the fuse gate as today's two-launch chain.
    """
    # Off AMD the op always runs the two-launch chain.
    two_launch = not _fuses_on_target(rows)
    outputs, host_shards, host_gammas = _execute(
        graphs, rows, multiply_before_cast, output_summed=two_launch
    )
    label = f"rows={rows} non-divisible (mbc={multiply_before_cast})"

    if not two_launch:
        _check(
            outputs,
            host_shards,
            host_gammas,
            multiply_before_cast,
            label,
            round_sum=False,
        )
        return

    _check_two_launch(
        outputs, host_shards, host_gammas, multiply_before_cast, label
    )


@pytest.mark.parametrize("multiply_before_cast", [False, True])
def test_allreduce_rms_norm_two_launch_arm(
    graphs: _GraphCache, multiply_before_cast: bool
) -> None:
    """Above the fuse threshold the op is today's two-launch chain.

    576 rows at 3584 cols is a 4.1 MB payload, past the 512-row fuse
    threshold, so the dispatch hands off to a standalone allreduce into the
    op's `summed` result followed by a separate norm. That arm is the only
    one that writes `summed`; the graph carries both outputs here (the
    production call site drops the sum), and each is checked against its
    own semantics: the norm reads the bfloat16-rounded sum, and `summed`
    is that rounded sum itself.
    """
    rows = 576
    outputs, host_shards, host_gammas = _execute(
        graphs, rows, multiply_before_cast, output_summed=True
    )
    label = f"rows={rows} two-launch (mbc={multiply_before_cast})"
    _check_two_launch(
        outputs, host_shards, host_gammas, multiply_before_cast, label
    )


def test_allreduce_rms_norm_rank_is_the_list_position() -> None:
    """The op's rank is the input list position, not the device ordinal.

    The kernel takes its rank from the op and only falls back to the
    physical device id when the caller passes none. Every other test here
    runs on devices 0 and 1, where the two coincide, so that fallback is
    indistinguishable; this test's device pair carries ids outside
    [0, num_devices), where the fallback is rejected outright.

    A pair like (1, 0) would not do: with two devices, a swap is a
    consistent relabeling of the barrier and reduce protocols, so the
    collective still computes the right answer from the wrong rank source.
    """
    if accelerator_count() < 4:
        pytest.skip(
            "Distinguishing a list-position rank from a device-id rank"
            " needs both device ids outside [0, 2), where the device-id"
            " fallback is rejected on every device; fewer visible devices"
            " cannot supply such a pair."
        )

    first = accelerator_count() - 2
    devices = [Accelerator(first), Accelerator(first + 1)]
    signals = Signals(
        devices=[DeviceRef.GPU(id=first), DeviceRef.GPU(id=first + 1)]
    )
    rows = 8
    model = InferenceSession(devices=[CPU(), *devices]).load(
        _ar_rms_norm_graph(signals, COLS, False)
    )
    outputs, host_shards, host_gammas = _run(model, signals, devices, rows)
    _check(
        outputs,
        host_shards,
        host_gammas,
        False,
        f"rows={rows} devices=({first}, {first + 1})",
        round_sum=not _fuses_on_target(rows),
    )


def test_allreduce_rms_norm_empty_batch(graphs: _GraphCache) -> None:
    """A zero-row batch must not launch (the grid would be empty)."""
    outputs, _, _ = _execute(graphs, rows=0, multiply_before_cast=False)
    assert len(outputs) == NUM_GPUS
    for out in outputs:
        assert _from_device_bf16(out).shape[0] == 0


@pytest.mark.parametrize("cols", UNTILEABLE_COLS)
@pytest.mark.parametrize("multiply_before_cast", [False, True])
@pytest.mark.parametrize("rows", [1, 3])
def test_allreduce_rms_norm_untileable_cols(
    graphs: _GraphCache, rows: int, multiply_before_cast: bool, cols: int
) -> None:
    """Widths the fused kernel cannot warp-tile fall back to two-launch.

    A few rows of these widths sit far under the byte threshold, so a
    payload-only gate would fuse and the kernel would raise.
    """
    assert not _fuses_on_target(rows, cols)
    outputs, host_shards, host_gammas = _execute(
        graphs, rows, multiply_before_cast, output_summed=True, cols=cols
    )
    _check_two_launch(
        outputs,
        host_shards,
        host_gammas,
        multiply_before_cast,
        f"rows={rows} cols={cols} (mbc={multiply_before_cast})",
    )


@pytest.mark.parametrize("multiply_before_cast", [False, True])
def test_allreduce_rms_norm_vendor_ccl(multiply_before_cast: bool) -> None:
    """With the vendor CCL define the op reduces through RCCL/NCCL.

    The fused arm is skipped at every row count, so the vendor allreduce
    writes `summed` and the separate norm reads the bfloat16-rounded sum,
    the same semantics as the two-launch arm.
    """
    if NUM_GPUS > accelerator_count():
        pytest.skip(f"Not enough GPUs ({NUM_GPUS}) for allreduce_rms_norm.")

    signals = Signals(devices=[DeviceRef.GPU(id=i) for i in range(NUM_GPUS)])
    devices = [Accelerator(n) for n in range(NUM_GPUS)]
    session = InferenceSession(devices=[CPU(), *devices])
    session._use_vendor_ccl("true")
    model = session.load(
        _ar_rms_norm_graph(signals, COLS, multiply_before_cast, True)
    )
    for rows in (1, 8, 40, 600):
        outputs, host_shards, host_gammas = _run(model, signals, devices, rows)
        _check_two_launch(
            outputs,
            host_shards,
            host_gammas,
            multiply_before_cast,
            f"rows={rows} vendor CCL (mbc={multiply_before_cast})",
        )
