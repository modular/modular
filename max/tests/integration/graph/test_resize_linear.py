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
"""Tests for resize_linear: the CPU kernel against a float64 reference
(including symbolic output sizes and a rank-3 input), and GPU against CPU."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
from max.driver import Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from numpy.typing import NDArray

# Two axes change, so the GPU kernel runs two passes through a scratch buffer.
INPUT_SHAPE = [1, 3, 5, 7]
OUTPUT_SHAPE = [1, 3, 9, 3]


def _source_coord(out: int, in_size: int, out_size: int, mode: int) -> float:
    """Maps one output coordinate to an input coordinate (mode 0-3)."""
    scale = in_size / out_size
    if mode == 1:  # align_corners
        return out * (in_size - 1) / (out_size - 1) if out_size > 1 else 0.0
    if mode == 2:  # asymmetric
        return out * scale
    if mode == 3 and out_size == 1:  # half_pixel_1D
        return 0.0
    # half_pixel (0), and half_pixel_1D (3) on every other axis.
    return (out + 0.5) * scale - 0.5


def _resize_axis(
    arr: NDArray[np.float64],
    out_size: int,
    dim: int,
    coord_mode: int,
    antialias: bool,
) -> NDArray[np.float64]:
    """Resizes ``arr`` along ``dim`` with the kernel's linear interpolation."""
    in_size = arr.shape[dim]
    if in_size == out_size:
        return arr

    moved = np.moveaxis(arr, dim, 0)
    out = np.zeros((out_size, *moved.shape[1:]), dtype=np.float64)
    scale = in_size / out_size
    trailing = (1,) * (moved.ndim - 1)

    for i in range(out_size):
        src = _source_coord(i, in_size, out_size, coord_mode)
        if antialias and scale > 1.0:
            # Widened tent filter: taps within [src - scale, src + scale].
            lo = max(int(np.floor(src - scale + 1e-6)), 0)
            hi = min(int(np.ceil(src + scale - 1e-6)), in_size - 1)
            weights = np.maximum(
                0.0, 1.0 - np.abs(np.arange(lo, hi + 1) - src) / scale
            )
            out[i] = (moved[lo : hi + 1] * weights.reshape(-1, *trailing)).sum(
                axis=0
            ) / weights.sum()
        else:
            src = min(max(src, 0.0), in_size - 1.0)
            i0 = int(np.floor(src))
            i1 = min(i0 + 1, in_size - 1)
            w = src - i0
            out[i] = (1.0 - w) * moved[i0] + w * moved[i1]

    return np.moveaxis(out, 0, dim)


def _reference(
    data: NDArray[np.float32],
    out_shape: Sequence[int],
    coord_mode: int = 0,
    antialias: bool = False,
) -> NDArray[np.float32]:
    """Separable float64 linear resize, matching the kernel's definition."""
    out = data.astype(np.float64)
    for dim in range(2, data.ndim):
        out = _resize_axis(out, out_shape[dim], dim, coord_mode, antialias)
    return out.astype(np.float32)


def _execute(
    session: InferenceSession,
    device: DeviceRef,
    input_type: TensorType,
    out_shape: Sequence[int | str],
    data: NDArray[np.float32],
    coord_mode: int = 0,
    antialias: bool = False,
) -> NDArray[np.float32]:
    """Executes resize_linear on one graph and returns the host result."""
    with Graph("test_resize_linear", input_types=[input_type]) as graph:
        graph.output(
            ops.resize_linear(
                graph.inputs[0].tensor,
                out_shape,
                coordinate_transform_mode=coord_mode,
                antialias=antialias,
            )
        )
    model = session.load(graph)
    result = model.execute(Buffer.from_numpy(data).to(model.input_devices[0]))[
        0
    ]
    assert isinstance(result, Buffer)
    assert result.device == model.input_devices[0]
    return result.to_numpy()


def _run(
    session: InferenceSession,
    device: DeviceRef,
    data: NDArray[np.float32],
    coord_mode: int,
    antialias: bool,
) -> NDArray[np.float32]:
    input_type = TensorType(DType.float32, INPUT_SHAPE, device=device)
    return _execute(
        session, device, input_type, OUTPUT_SHAPE, data, coord_mode, antialias
    )


@pytest.mark.skipif(accelerator_count() == 0, reason="No GPU available")
@pytest.mark.parametrize("coord_mode", range(4))
@pytest.mark.parametrize("antialias", [False, True])
def test_resize_linear_gpu_matches_cpu(
    session: InferenceSession, coord_mode: int, antialias: bool
) -> None:
    rng = np.random.default_rng(0)
    data = rng.standard_normal(INPUT_SHAPE).astype(np.float32)

    cpu = _run(session, DeviceRef.CPU(), data, coord_mode, antialias)
    gpu = _run(session, DeviceRef.GPU(), data, coord_mode, antialias)

    assert gpu.shape == tuple(OUTPUT_SHAPE)
    np.testing.assert_allclose(gpu, cpu, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(accelerator_count() == 0, reason="No GPU available")
def test_resize_linear_gpu_same_shape_copies(
    session: InferenceSession,
) -> None:
    """An unchanged shape takes the copy path, which must still write."""
    rng = np.random.default_rng(1)
    data = rng.standard_normal(INPUT_SHAPE).astype(np.float32)

    input_type = TensorType(DType.float32, INPUT_SHAPE, device=DeviceRef.GPU())
    with Graph("test_resize_linear_same", input_types=[input_type]) as graph:
        graph.output(ops.resize_linear(graph.inputs[0].tensor, INPUT_SHAPE))
    model = session.load(graph)
    result = model.execute(Buffer.from_numpy(data).to(model.input_devices[0]))[
        0
    ]
    assert isinstance(result, Buffer)
    np.testing.assert_array_equal(result.to_numpy(), data)


@pytest.mark.parametrize("coord_mode", range(4))
@pytest.mark.parametrize("antialias", [False, True])
def test_resize_linear_cpu_matches_reference(
    session: InferenceSession, coord_mode: int, antialias: bool
) -> None:
    """The CPU kernel matches a float64 separable reference."""
    rng = np.random.default_rng(2)
    data = rng.standard_normal(INPUT_SHAPE).astype(np.float32)

    out = _run(session, DeviceRef.CPU(), data, coord_mode, antialias)

    assert out.shape == tuple(OUTPUT_SHAPE)
    np.testing.assert_allclose(
        out,
        _reference(data, OUTPUT_SHAPE, coord_mode, antialias),
        rtol=1e-3,
        atol=1e-3,
    )


def test_resize_linear_symbolic_dims(
    session: InferenceSession,
) -> None:
    """A symbolic batch dim flows through to a dynamic output shape."""
    rng = np.random.default_rng(3)
    data = rng.standard_normal((2, 3, 5, 7)).astype(np.float32)
    input_type = TensorType(
        DType.float32, ["batch", 3, 5, 7], device=DeviceRef.CPU()
    )

    out = _execute(
        session, DeviceRef.CPU(), input_type, ["batch", 3, 9, 3], data
    )

    assert out.shape == (2, 3, 9, 3)
    np.testing.assert_allclose(
        out, _reference(data, [2, 3, 9, 3]), rtol=1e-3, atol=1e-3
    )


def test_resize_linear_rank3_cpu(
    session: InferenceSession,
) -> None:
    """A rank-3 (NCW) input resizes along its one spatial axis."""
    rng = np.random.default_rng(4)
    data = rng.standard_normal((1, 4, 8)).astype(np.float32)
    out_shape = [1, 4, 16]

    input_type = TensorType(DType.float32, [1, 4, 8], device=DeviceRef.CPU())
    out = _execute(session, DeviceRef.CPU(), input_type, out_shape, data)

    assert out.shape == (1, 4, 16)
    np.testing.assert_allclose(
        out, _reference(data, out_shape), rtol=1e-3, atol=1e-3
    )
