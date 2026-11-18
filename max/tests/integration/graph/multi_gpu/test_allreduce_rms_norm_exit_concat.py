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

"""Multi-GPU test of the fused MoE-exit collective against the chain it replaces.

One graph stages both: ``allreduce_rms_norm_exit_concat``, and the unfused
chain -- ``allreduce.sum``, ``rms_norm`` with its defaults, this device's
column shard cast to bfloat16, and the shared expert's SiTU-and-mul beside
it. Both read the same partials, so the two outputs must agree on every
device: the normed shard to within a rounding flip of the mean square's
reduction order, the activation to within the transcendental's rounding.
"""

from __future__ import annotations

import ml_dtypes
import numpy as np
import pytest
from max.driver import CPU, Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import BufferType, DeviceRef, Graph, TensorType, TensorValue, ops
from max.nn.legacy import Signals

NUM_GPUS = 2
COLS = 3584  # Kimi K3's routed-expert latent width.
HALF = 256  # The shared expert's per-device half width.
EPS = 1e-5
BETA = 1.5
LINEAR_BETA = 1.5
# A rounding flip moves a bfloat16 by one ULP; anything more is a bug.
MAX_BF16_ULPS = 1
# Fraction of elements allowed one ULP off the chain.
MAX_FLIPPED_FRACTION = 0.02


def _to_device(arr: np.ndarray, dtype: DType, device: Accelerator) -> Buffer:
    if dtype == DType.float32:
        return Buffer.from_numpy(arr.astype(np.float32)).to(device)
    assert dtype == DType.bfloat16, f"no host packing for {dtype}"
    bits = np.ascontiguousarray(arr.astype(ml_dtypes.bfloat16).view(np.uint16))
    return Buffer.from_numpy(bits).view(DType.bfloat16).to(device)


def _bf16_bits(buf: Buffer) -> np.ndarray:
    return buf.copy(device=CPU()).view(DType.uint16).to_numpy().astype(np.int32)


def _ulps(got: np.ndarray, want: np.ndarray) -> np.ndarray:
    """bfloat16 ULP distance between two arrays of same-signed bit patterns."""

    def ordered(bits: np.ndarray) -> np.ndarray:
        return np.where(bits & 0x8000, 0x8000 - (bits & 0x7FFF), bits + 0x8000)

    return np.abs(ordered(got) - ordered(want))


def _situ_and_mul(gate_up: TensorValue) -> TensorValue:
    half = int(gate_up.shape[1]) // 2
    gate = ops.cast(gate_up[:, :half], DType.float32)
    up = ops.cast(gate_up[:, half:], DType.float32)
    situ = BETA * ops.tanh(gate / BETA) * ops.sigmoid(gate)
    up = LINEAR_BETA * ops.tanh(up / LINEAR_BETA)
    return ops.cast(situ * up, DType.bfloat16)


@pytest.mark.parametrize(
    "dtype, gamma_dtype, rows",
    [
        # The production pair carries the row sweep: 1 is the degenerate batch,
        # 32 K3's decode step, 37 an odd batch past a tile, and 1024 a prefill
        # chunk above the fuse gate, where the op's launcher runs its chain arm
        # instead (a plain allreduce, then the norm on this device's local sum).
        # The rows only choose an arm and hit a tile edge, so they do not need
        # crossing with every dtype pair.
        (DType.float32, DType.bfloat16, 1),
        (DType.float32, DType.bfloat16, 32),
        (DType.float32, DType.bfloat16, 37),
        (DType.float32, DType.bfloat16, 1024),
        # The other two pairs pin their own dispatch at one batch each, plus the
        # arm their dtypes have to keep: bfloat16 at the prefill width that runs
        # the chain arm, float32 in the 20000-row signal-buffer case below.
        (DType.float32, DType.float32, 32),
        (DType.bfloat16, DType.bfloat16, 32),
        (DType.bfloat16, DType.bfloat16, 1024),
    ],
    # Each case is its own static-shape graph, recorded once per flavor by the
    # package's `precompile_mefs`, so the matrix is that recording's cost. What
    # the cross product used to add over this -- rows 7, and the float32 and
    # bfloat16 pairs at the other rows -- is swept at the same widths by
    # `//max/kernels/test/gpu/comm:test_allreduce_rmsnorm_exit_concat.mojo.test`
    # at 2, 4 and 8 GPUs.
)
def test_exit_concat_matches_unfused_chain(
    dtype: DType, gamma_dtype: DType, rows: int
) -> None:
    """The fused op equals allreduce + rms_norm + slice/cast + SiTU concat."""
    _check_exit_concat(dtype, gamma_dtype, rows)


def test_chain_arm_fits_the_plain_allreduce_signal_buffer() -> None:
    """A batch past the old full-width-sum cap runs the chain arm.

    The chain arm used to stage its whole ``[rows, cols]`` float32 sum in the
    signal buffer's payload, which capped a float32 call at about 16k rows of
    the K3 latent. It now reduces into a workspace of its own, so the payload
    only has to hold the plain allreduce's partition: this batch needs 410 MiB
    staged by the old rule, more than the 256 MiB payload, and 137 MiB by the
    new one.
    """
    _check_exit_concat(DType.float32, DType.float32, 20000)


def _check_exit_concat(dtype: DType, gamma_dtype: DType, rows: int) -> None:
    """Runs the fused op and the chain on one shape and compares them."""
    if NUM_GPUS > accelerator_count():
        pytest.skip(f"Not enough GPUs ({NUM_GPUS}).")

    refs = [DeviceRef.GPU(id=i) for i in range(NUM_GPUS)]
    signals = Signals(devices=refs)
    input_types: list[TensorType | BufferType] = []
    input_types += [
        TensorType(dtype=dtype, shape=[rows, COLS], device=d) for d in refs
    ]
    input_types += [
        TensorType(dtype=gamma_dtype, shape=[COLS], device=d) for d in refs
    ]
    input_types += [
        TensorType(dtype=DType.bfloat16, shape=[rows, 2 * HALF], device=d)
        for d in refs
    ]
    input_types += signals.input_types()
    slice_cols = COLS // NUM_GPUS
    with Graph("exit_concat", input_types=input_types) as graph:
        partials = [v.tensor for v in graph.inputs[:NUM_GPUS]]
        gammas = [v.tensor for v in graph.inputs[NUM_GPUS : 2 * NUM_GPUS]]
        gate_ups = [v.tensor for v in graph.inputs[2 * NUM_GPUS : 3 * NUM_GPUS]]
        sigs = [v.buffer for v in graph.inputs[3 * NUM_GPUS :]]
        fused = ops.allreduce_rms_norm_exit_concat(
            partials, sigs, gammas, gate_ups, EPS, BETA, LINEAR_BETA
        )
        summed = ops.allreduce.sum(partials, sigs)
        chain = []
        for rank in range(NUM_GPUS):
            normed = ops.rms_norm(summed[rank], gammas[rank], EPS, 0.0, False)
            shard = normed[:, rank * slice_cols : (rank + 1) * slice_cols]
            chain.append(
                ops.concat(
                    [
                        ops.cast(shard, DType.bfloat16),
                        _situ_and_mul(gate_ups[rank]),
                    ],
                    axis=1,
                )
            )
        graph.output(*fused, *chain)

    devices = [Accelerator(i) for i in range(NUM_GPUS)]
    model = InferenceSession(devices=[CPU(), *devices]).load(graph)

    rng = np.random.default_rng(rows * 2 + (dtype == DType.bfloat16))
    shards = [rng.standard_normal((rows, COLS)) for _ in range(NUM_GPUS)]
    gamma = ((np.arange(COLS) % 7) * 0.01) + 0.5
    gate_up = [
        rng.standard_normal((rows, 2 * HALF)) * 2.0 for _ in range(NUM_GPUS)
    ]
    outputs = model.execute(
        *[
            _to_device(s, dtype, d)
            for s, d in zip(shards, devices, strict=True)
        ],
        *[_to_device(gamma, gamma_dtype, d) for d in devices],
        *[
            _to_device(g, DType.bfloat16, d)
            for g, d in zip(gate_up, devices, strict=True)
        ],
        *signals.buffers(),
    )
    for rank in range(NUM_GPUS):
        got_buf, want_buf = outputs[rank], outputs[NUM_GPUS + rank]
        assert isinstance(got_buf, Buffer) and isinstance(want_buf, Buffer)
        got, want = _bf16_bits(got_buf), _bf16_bits(want_buf)
        assert got.shape == (rows, slice_cols + HALF)
        assert want.shape == got.shape
        for name, cols in (
            ("normed shard", slice(0, slice_cols)),
            ("activation", slice(slice_cols, None)),
        ):
            ulps = _ulps(got[:, cols], want[:, cols])
            flipped = float(np.mean(ulps > 0))
            print(
                f"rows={rows} {dtype} gamma {gamma_dtype} rank {rank} {name}: "
                f"max {ulps.max()} "
                f"ULP, {flipped:.4%} not byte-equal"
            )
            assert ulps.max() <= MAX_BF16_ULPS, (
                f"rank {rank} {name}: {ulps.max()} ULPs off the chain"
            )
            assert flipped <= MAX_FLIPPED_FRACTION, (
                f"rank {rank} {name}: {flipped:.2%} of elements flipped"
            )
