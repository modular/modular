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
import torch
from max._core.dialects import builtin, kgen, mo, rmo
from max.driver import Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from max.graph.ops import CoordinateTransformMode
from numpy.typing import NDArray

# bfloat16 has no numpy dtype, so 16-bit graph inputs go through torch.
_DTYPE_TO_TORCH = {
    DType.float16: torch.float16,
    DType.bfloat16: torch.bfloat16,
}

# Two axes change, so the GPU kernel runs two passes through a scratch buffer.
INPUT_SHAPE = [1, 3, 5, 7]
OUTPUT_SHAPE = [1, 3, 9, 3]


def _source_coord(
    out: int, in_size: int, out_size: int, mode: CoordinateTransformMode
) -> float:
    """Maps one output coordinate to an input coordinate."""
    scale = in_size / out_size
    if mode == CoordinateTransformMode.ALIGN_CORNERS:
        return out * (in_size - 1) / (out_size - 1) if out_size > 1 else 0.0
    if mode == CoordinateTransformMode.ASYMMETRIC:
        return out * scale
    if mode == CoordinateTransformMode.HALF_PIXEL_1D and out_size == 1:
        return 0.0
    # HALF_PIXEL, and HALF_PIXEL_1D on every other axis.
    return (out + 0.5) * scale - 0.5


def _resize_axis(
    arr: NDArray[np.float64],
    out_size: int,
    dim: int,
    coord_mode: CoordinateTransformMode,
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
    coord_mode: CoordinateTransformMode = CoordinateTransformMode.HALF_PIXEL,
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
    coord_mode: CoordinateTransformMode = CoordinateTransformMode.HALF_PIXEL,
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
    coord_mode: CoordinateTransformMode,
    antialias: bool,
) -> NDArray[np.float32]:
    input_type = TensorType(DType.float32, INPUT_SHAPE, device=device)
    return _execute(
        session, device, input_type, OUTPUT_SHAPE, data, coord_mode, antialias
    )


@pytest.mark.skipif(accelerator_count() == 0, reason="No GPU available")
@pytest.mark.parametrize("coord_mode", list(CoordinateTransformMode))
@pytest.mark.parametrize("antialias", [False, True])
def test_resize_linear_gpu_matches_cpu(
    session: InferenceSession,
    coord_mode: CoordinateTransformMode,
    antialias: bool,
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


@pytest.mark.parametrize("coord_mode", list(CoordinateTransformMode))
@pytest.mark.parametrize("antialias", [False, True])
def test_resize_linear_cpu_matches_reference(
    session: InferenceSession,
    coord_mode: CoordinateTransformMode,
    antialias: bool,
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


@pytest.mark.skipif(accelerator_count() == 0, reason="No GPU available")
def test_resize_linear_gpu_rank3(session: InferenceSession) -> None:
    """A rank-3 (NCW) input resizes on the GPU along its one spatial axis.

    The accelerated eager warm builds only rank-4 graphs (MXF-841), so a
    rank-3 graph also exercises the rank outside the warm set in a real
    compiled graph.
    """
    rng = np.random.default_rng(5)
    data = rng.standard_normal((1, 4, 8)).astype(np.float32)
    out_shape = [1, 4, 16]

    input_type = TensorType(DType.float32, [1, 4, 8], device=DeviceRef.GPU())
    out = _execute(session, DeviceRef.GPU(), input_type, out_shape, data)

    assert out.shape == (1, 4, 16)
    np.testing.assert_allclose(
        out, _reference(data, out_shape), rtol=1e-4, atol=1e-4
    )


@pytest.mark.skipif(accelerator_count() == 0, reason="No GPU available")
def test_resize_linear_gpu_dynamic_sizes(session: InferenceSession) -> None:
    """One loaded model serves two different output spatial sizes.

    `size` is a runtime operand (`MO_SingleDeviceWithHostOperands`), so the
    graph takes it as a second input and the same compiled model resizes
    5x7 to both 9x3 and 4x11 — the way the eager family cache reuses one
    graph per (device, dtype, rank, variant) for every output size.
    """
    rng = np.random.default_rng(6)
    data = rng.standard_normal((2, 3, 5, 7)).astype(np.float32)
    input_type = TensorType(
        DType.float32, ["batch", 3, 5, 7], device=DeviceRef.GPU()
    )
    size_type = TensorType(DType.int64, [4], device=DeviceRef.CPU())

    with Graph(
        "test_resize_linear_dynamic",
        input_types=[input_type, size_type],
    ) as graph:
        x, size = (value.tensor for value in graph.inputs)
        graph.output(
            Graph.current._add_op_generated(
                rmo.MoResizeLinearOp,
                result=TensorType(
                    DType.float32,
                    ["batch", 3, "out_h", "out_w"],
                    device=DeviceRef.GPU(),
                ),
                input=x,
                size=size,
                coordinate_transform_mode=mo.CoordinateTransformModeAttr(
                    mo.CoordinateTransformMode(0)
                ),
                antialias=builtin.BoolAttr(False),
                output_param_decls=kgen.ParamDeclArrayAttr([]),
            )[0].tensor
        )
    model = session.load(graph)

    for out_shape in ([2, 3, 9, 3], [2, 3, 4, 11]):
        result = model.execute(
            Buffer.from_numpy(data).to(model.input_devices[0]),
            Buffer.from_numpy(np.array(out_shape, dtype=np.int64)),
        )[0]
        assert isinstance(result, Buffer)
        assert result.shape == tuple(out_shape)
        np.testing.assert_allclose(
            result.to_numpy(),
            _reference(data, out_shape),
            rtol=1e-4,
            atol=1e-4,
        )


@pytest.mark.skipif(accelerator_count() == 0, reason="No GPU available")
@pytest.mark.parametrize("dtype", [DType.float16, DType.bfloat16])
@pytest.mark.parametrize("antialias", [False, True])
def test_resize_linear_gpu_dtypes(
    session: InferenceSession,
    dtype: DType,
    antialias: bool,
) -> None:
    """16-bit GPU inputs match the float64 reference.

    `test_resize_linear_gpu_matches_cpu` covers every coordinate mode.
    The mode does not depend on the dtype, so this test does not repeat
    them: each mode would only add a graph compile.

    float64 is excluded: it is not a GPU float dtype (MSTDL-2711). The
    kernel accumulates in float32 for every 16-bit input, so the
    tolerance widens only as far as the final store rounds.
    """
    rng = np.random.default_rng(7)
    data = rng.standard_normal(INPUT_SHAPE).astype(np.float32)

    input_type = TensorType(dtype, INPUT_SHAPE, device=DeviceRef.GPU())
    with Graph("test_resize_linear_dtype", input_types=[input_type]) as graph:
        graph.output(
            ops.resize_linear(
                graph.inputs[0].tensor,
                OUTPUT_SHAPE,
                antialias=antialias,
            )
        )
    model = session.load(graph)
    # bfloat16 has no numpy dtype, so 16-bit inputs go through torch.
    x_t = torch.from_numpy(data).to(_DTYPE_TO_TORCH[dtype])
    result = model.execute(Buffer.from_dlpack(x_t).to(model.input_devices[0]))[
        0
    ]
    assert isinstance(result, Buffer)

    # The kernel accumulates in float32 and rounds once on store.
    np.testing.assert_allclose(
        torch.from_dlpack(result).float().cpu().numpy(),
        _reference(data, OUTPUT_SHAPE, antialias=antialias),
        rtol=2e-2,
        atol=2e-2,
    )


@pytest.mark.skipif(accelerator_count() == 0, reason="No GPU available")
@pytest.mark.parametrize(
    "in_shape, out_shape, antialias",
    [
        # Latent decode upsamples: (H, W) -> (4H, 4W), as VAE decoders do.
        ([1, 8, 64, 64], [1, 8, 256, 256], False),
        # Latent upsample: (H, W) -> (2H, 2W).
        ([2, 4, 32, 32], [2, 4, 64, 64], False),
        # A non-integer-scale downsample widens the tent filter (antialias).
        ([1, 8, 64, 64], [1, 8, 50, 40], True),
    ],
)
def test_resize_linear_gpu_upsample_shapes(
    session: InferenceSession,
    in_shape: Sequence[int],
    out_shape: Sequence[int],
    antialias: bool,
) -> None:
    """Representative latent/VAE-style shapes run on the GPU.

    The in-repo VAE decoders upsample with `resize.nearest`, so `linear`
    has no model-level consumer yet; these shapes pin the sizes a
    diffusion/VAE pipeline would feed a bilinear upsample.
    """
    rng = np.random.default_rng(8)
    data = rng.standard_normal(in_shape).astype(np.float32)

    input_type = TensorType(DType.float32, in_shape, device=DeviceRef.GPU())
    out = _execute(
        session,
        DeviceRef.GPU(),
        input_type,
        out_shape,
        data,
        antialias=antialias,
    )

    assert out.shape == tuple(out_shape)
    np.testing.assert_allclose(
        out,
        _reference(data, out_shape, antialias=antialias),
        rtol=1e-2,
        atol=1e-2,
    )
