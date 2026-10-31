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
"""The fused collective + RMS norm functionals on their real kernels.

Needs 2 real, distinct devices: on a mesh that repeats a device there are no
signal buffers, and the functionals take their simulated path instead.
"""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import CPU, Accelerator, Device
from max.dtype import DType
from max.functional import (
    allgather_rms_norm,
    allreduce_rms_norm,
    reduce_scatter_rms_norm,
    transfer_to,
)
from max.tensor import Tensor, default_device
from max.tensor.sharding import (
    DeviceMapping,
    DeviceMesh,
    Partial,
    Replicated,
    Sharded,
)
from max.tensor.sharding.placements import even_shard_sizes

MESH_2 = DeviceMesh(
    devices=(Accelerator(0), Accelerator(1)),
    mesh_shape=(2,),
    axis_names=("tp",),
)
EPS = 1e-6
COLS = 256
# bfloat16 has an 8-bit mantissa.
TOL = 3e-2


def _rms_norm(x: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + EPS)
    return x / rms * gamma


def _bf16_on(values: np.ndarray, device: Device) -> Tensor:
    return Tensor(values).to(device).cast(DType.bfloat16)


def _gamma(rng: np.random.Generator) -> tuple[np.ndarray, Tensor]:
    values = rng.standard_normal(COLS).astype(np.float32)
    return values, transfer_to(
        Tensor(values), DeviceMapping(MESH_2, (Replicated(),))
    ).cast(DType.bfloat16)


def _assert_close(actual: Tensor, expected: np.ndarray) -> None:
    np.testing.assert_allclose(
        actual.cast(DType.float32).to_numpy(), expected, rtol=TOL, atol=TOL
    )


def _fragments(
    rng: np.random.Generator, rows: int, mesh: DeviceMesh
) -> list[np.ndarray]:
    return [
        rng.standard_normal((rows, COLS)).astype(np.float32)
        for _ in mesh.devices
    ]


def _partial_over(fragments: list[np.ndarray], mesh: DeviceMesh) -> Tensor:
    return Tensor._from_shards(
        tuple(
            _bf16_on(fragment, device).driver_tensor
            for fragment, device in zip(fragments, mesh.devices, strict=True)
        ),
        DeviceMapping(mesh, (Partial(),)),
    )


def _assert_replicated_rms_norm(
    normed: Tensor,
    fragments: list[np.ndarray],
    gamma_values: np.ndarray,
    mesh: DeviceMesh,
) -> None:
    assert normed.placements == (Replicated(),)
    total = np.sum(fragments, axis=0)
    reference = _rms_norm(total, gamma_values)
    for i in range(mesh.num_devices):
        _assert_close(normed.local_shards[i], reference)
    # Replicated: every device holds the same normed sum.
    assert np.array_equal(
        normed.local_shards[0].cast(DType.float32).to_numpy(),
        normed.local_shards[1].cast(DType.float32).to_numpy(),
    )


@pytest.mark.parametrize("rows", [8, 7])
def test_reduce_scatter_rms_norm(rows: int) -> None:
    rng = np.random.default_rng(0)
    fragments = [
        rng.standard_normal((rows, COLS)).astype(np.float32)
        for _ in MESH_2.devices
    ]
    gamma_values, gamma = _gamma(rng)
    partial = Tensor._from_shards(
        tuple(
            _bf16_on(fragment, device).driver_tensor
            for fragment, device in zip(fragments, MESH_2.devices, strict=True)
        ),
        DeviceMapping(MESH_2, (Partial(),)),
    )

    normed, summed = reduce_scatter_rms_norm(partial, gamma, EPS)

    assert normed.placements == (Sharded(0),)
    assert summed.placements == (Sharded(0),)
    total = np.sum(fragments, axis=0)
    start = 0
    for i, size in enumerate(even_shard_sizes(rows, MESH_2.num_devices)):
        expected = total[start : start + size]
        start += size
        _assert_close(summed.local_shards[i], expected)
        _assert_close(normed.local_shards[i], _rms_norm(expected, gamma_values))


@pytest.mark.parametrize("rows", [8, 7])
def test_allreduce_rms_norm(rows: int) -> None:
    rng = np.random.default_rng(2)
    fragments = _fragments(rng, rows, MESH_2)
    gamma_values, gamma = _gamma(rng)
    partial = _partial_over(fragments, MESH_2)

    normed = allreduce_rms_norm(partial, gamma, EPS)

    _assert_replicated_rms_norm(normed, fragments, gamma_values, MESH_2)


@pytest.mark.parametrize("rows", [8, 7])
def test_allreduce_rms_norm_without_signal_buffers(rows: int) -> None:
    """The no-signal-buffer fallback: reduce on one device, norm beside it.

    A mesh that repeats one device allocates no signal buffers, so the
    wrapper sums on the first device and runs the norm there -- with the
    gamma, which reaches it whole from the host, moved onto that device --
    then hands every mesh slot its copy. Signed data, a gamma far from
    ones, against numpy.
    """
    rng = np.random.default_rng(4)
    simulated = DeviceMesh(
        devices=(MESH_2.devices[0], MESH_2.devices[0]),
        mesh_shape=(2,),
        axis_names=("tp",),
    )
    fragments = _fragments(rng, rows, simulated)
    gamma_values = (rng.standard_normal(COLS).astype(np.float32) * 0.5) + 0.5
    with default_device(CPU()):
        gamma = Tensor(gamma_values).cast(DType.bfloat16)
    partial = _partial_over(fragments, simulated)

    normed = allreduce_rms_norm(partial, gamma, EPS)

    _assert_replicated_rms_norm(normed, fragments, gamma_values, simulated)


@pytest.mark.parametrize("rows", [8, 7])
def test_allreduce_rms_norm_host_gamma(rows: int) -> None:
    """A host gamma rides the fused op on the real mesh.

    A gamma that is not distributed reaches the wrapper whole, on the
    host; the fused op reads one gamma per device on that device, so the
    wrapper must place a copy on every mesh device before building the op.
    Signed data, a gamma far from ones, against numpy.
    """
    rng = np.random.default_rng(5)
    fragments = _fragments(rng, rows, MESH_2)
    gamma_values = (rng.standard_normal(COLS).astype(np.float32) * 0.5) + 0.5
    with default_device(CPU()):
        gamma = Tensor(gamma_values).cast(DType.bfloat16)
    partial = _partial_over(fragments, MESH_2)

    normed = allreduce_rms_norm(partial, gamma, EPS)

    _assert_replicated_rms_norm(normed, fragments, gamma_values, MESH_2)


@pytest.mark.parametrize("rows", [8, 7])
def test_allreduce_rms_norm_fp32_falls_back(rows: int) -> None:
    """A non-bfloat16 input takes the two-launch form and still normalizes.

    The fused kernel is bfloat16-only, so the wrapper must reduce and norm
    separately for float32 inputs (the form float32 TP tests exercise).
    """
    rng = np.random.default_rng(3)
    fragments = _fragments(rng, rows, MESH_2)
    gamma_values = rng.standard_normal(COLS).astype(np.float32)
    gamma = transfer_to(
        Tensor(gamma_values), DeviceMapping(MESH_2, (Replicated(),))
    )
    partial = Tensor._from_shards(
        tuple(
            Tensor(fragment).to(device).driver_tensor
            for fragment, device in zip(fragments, MESH_2.devices, strict=True)
        ),
        DeviceMapping(MESH_2, (Partial(),)),
    )

    normed = allreduce_rms_norm(partial, gamma, EPS)

    assert normed.placements == (Replicated(),)
    total = np.sum(fragments, axis=0)
    reference = _rms_norm(total, gamma_values)
    for i in range(MESH_2.num_devices):
        np.testing.assert_allclose(
            normed.local_shards[i].to_numpy(),
            reference,
            rtol=1e-4,
            atol=1e-4,
        )


@pytest.mark.parametrize("rows", [8, 7])
def test_allgather_rms_norm(rows: int) -> None:
    rng = np.random.default_rng(1)
    full = rng.standard_normal((rows, COLS)).astype(np.float32)
    gamma_values, gamma = _gamma(rng)
    sizes = even_shard_sizes(rows, MESH_2.num_devices)
    sharded = Tensor._from_shards(
        tuple(
            _bf16_on(
                full[sum(sizes[:i]) : sum(sizes[: i + 1])], device
            ).driver_tensor
            for i, device in enumerate(MESH_2.devices)
        ),
        DeviceMapping(MESH_2, (Sharded(0),)),
    )

    normed, gathered = allgather_rms_norm(sharded, gamma, EPS)

    assert normed.placements == (Replicated(),)
    assert gathered.placements == (Replicated(),)
    for i in range(MESH_2.num_devices):
        _assert_close(gathered.local_shards[i], full)
        _assert_close(normed.local_shards[i], _rms_norm(full, gamma_values))
