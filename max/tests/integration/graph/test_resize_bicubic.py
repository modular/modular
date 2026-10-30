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
"""Test resize_bicubic operation execution."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
import torch
import torchvision.transforms as T
from max._core.dialects import builtin, kgen, rmo
from max.driver import Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, ops
from numpy.typing import NDArray
from PIL import Image
from torchvision.transforms.functional import InterpolationMode


def torchvision_bicubic_resize_nchw(
    input_array: NDArray[np.float32], height: int, width: int
) -> NDArray[np.float32]:
    """Reference implementation using torchvision bicubic resize for NCHW format.

    Args:
        input_array: Input array of shape (batch, channels, height, width)
        height: Target height
        width: Target width

    Returns:
        Resized array using torchvision's bicubic interpolation in NCHW format
    """
    # Convert numpy array to torch tensor
    input_tensor = torch.from_numpy(input_array)

    # Create resize transform
    resize_transform = T.Resize(
        (height, width),
        interpolation=InterpolationMode.BICUBIC,
        antialias=False,
    )

    # Apply resize to each image in the batch
    batch_size = input_tensor.shape[0]
    output_list = []
    for b in range(batch_size):
        resized = resize_transform(input_tensor[b])
        output_list.append(resized)

    # Stack results back into a batch
    output_tensor = torch.stack(output_list)

    # Convert back to numpy
    return output_tensor.numpy()


@pytest.mark.parametrize("device", [DeviceRef.CPU(), DeviceRef.GPU()])
@pytest.mark.parametrize(
    "input_shape,output_shape",
    [
        # Upscale 2x (NCHW format)
        ([1, 3, 224, 224], [1, 3, 448, 448]),
        # Downscale 2x
        ([1, 3, 448, 448], [1, 3, 224, 224]),
        # Non-square input
        ([1, 3, 336, 224], [1, 3, 448, 448]),
        # Different channels
        ([2, 1, 128, 128], [2, 1, 256, 256]),
    ],
)
def test_resize_bicubic_execution(
    session: InferenceSession,
    device: DeviceRef,
    input_shape: Sequence[int],
    output_shape: Sequence[int],
) -> None:
    """Test resize_bicubic compilation and execution."""
    # Skip GPU tests if no GPU is available
    if device.device_type == "gpu" and accelerator_count() == 0:
        pytest.skip("No GPU available")
    # Phase 1: Graph construction
    input_type = TensorType(
        dtype=DType.float32, shape=input_shape, device=device
    )

    with Graph("test_resize_bicubic", input_types=[input_type]) as graph:
        # Resize operates on NCHW format directly
        resized = ops.resize(
            graph.inputs[0].tensor,
            output_shape,
            interpolation=ops.InterpolationMode.BICUBIC,
        )
        graph.output(resized)

    # Phase 2: Compilation
    model = session.load(graph)

    # Phase 3: Execution
    # Create test input (NCHW format)
    np.random.seed(42)
    input_data = np.random.rand(*input_shape).astype(np.float32)

    # Get reference output from torchvision in NCHW format
    _, _, out_h, out_w = output_shape
    expected = torchvision_bicubic_resize_nchw(input_data, out_h, out_w)

    # Execute MAX graph
    result = model.execute(
        Buffer.from_numpy(input_data).to(model.input_devices[0])
    )[0]
    assert isinstance(result, Buffer)
    result_np = result.to_numpy()

    # Verify shape and values
    assert result_np.shape == tuple(output_shape)
    # Torchvision and our implementation may have small differences
    np.testing.assert_allclose(result_np, expected, rtol=0.01, atol=0.01)


@pytest.mark.parametrize("device", [DeviceRef.CPU(), DeviceRef.GPU()])
def test_resize_bicubic_identity(
    session: InferenceSession, device: DeviceRef
) -> None:
    """Test identity transformation (same size)."""
    # Skip GPU tests if no GPU is available
    if device.device_type == "gpu" and accelerator_count() == 0:
        pytest.skip("No GPU available")
    shape = [1, 3, 256, 256]  # NCHW format

    input_type = TensorType(dtype=DType.float32, shape=shape, device=device)

    with Graph(
        "test_resize_bicubic_identity", input_types=[input_type]
    ) as graph:
        # Resize to same shape
        resized = ops.resize(
            graph.inputs[0].tensor,
            shape,
            interpolation=ops.InterpolationMode.BICUBIC,
        )
        graph.output(resized)

    # Compile and execute
    model = session.load(graph)

    input_data = np.random.rand(*shape).astype(np.float32)
    result = model.execute(
        Buffer.from_numpy(input_data).to(model.input_devices[0])
    )[0]

    # For identity transformation, output should match input closely
    assert isinstance(result, Buffer)
    np.testing.assert_allclose(
        result.to_numpy(), input_data, rtol=1e-5, atol=1e-6
    )


@pytest.fixture(scope="module")
def antialiased_resize(session: InferenceSession) -> Model:
    """A CPU graph that resizes any NCHW float32 tensor, with antialiasing,
    to a size given at runtime."""
    input_type = TensorType(
        DType.float32, ["n", "c", "h", "w"], device=DeviceRef.CPU()
    )
    size_type = TensorType(DType.int64, [4], device=DeviceRef.CPU())
    with Graph(
        "test_resize_bicubic_antialias", input_types=[input_type, size_type]
    ) as graph:
        x, size = (v.tensor for v in graph.inputs)
        graph.output(
            ops.resize_bicubic(
                x,
                size,
                antialias=True,
                out_dims=["n", "c", "h_out", "w_out"],
            )
        )
    return session.load(graph)


@pytest.mark.parametrize(
    "input_shape,output_hw",
    [
        # Downscales, where the stretched filter reads more taps.
        ((1, 3, 448, 448), (224, 224)),
        ((2, 3, 301, 517), (97, 203)),
        # An upscale, where it matches the plain a = -0.5 cubic.
        ((1, 1, 37, 53), (112, 140)),
    ],
)
def test_antialiased_resize_bicubic_matches_torch(
    antialiased_resize: Model,
    input_shape: tuple[int, int, int, int],
    output_hw: tuple[int, int],
) -> None:
    """One compiled graph resizes to every runtime size, as torch's CPU
    antialiased bicubic does to float32 rounding."""
    rng = np.random.default_rng(7)
    input_data = rng.random(input_shape, dtype=np.float32) * 255
    out_h, out_w = output_hw
    size = np.array([input_shape[0], input_shape[1], out_h, out_w], np.int64)

    result = antialiased_resize.execute(
        Buffer.from_numpy(input_data), Buffer.from_numpy(size)
    )[0]
    assert isinstance(result, Buffer)

    expected = torch.nn.functional.interpolate(
        torch.from_numpy(input_data),
        size=output_hw,
        mode="bicubic",
        antialias=True,
        align_corners=False,
    ).numpy()
    assert result.to_numpy().shape == expected.shape
    # A few float32 ulps of the 0-255 pixel range.
    np.testing.assert_allclose(result.to_numpy(), expected, rtol=0, atol=1e-4)


@pytest.fixture(scope="module")
def uint8_and_float32_resizes(session: InferenceSession) -> Model:
    """A CPU graph that resizes NCHW uint8 pixels with antialiasing, to a
    size given at runtime, twice: uint8 to uint8, and through float32 with a
    round and clamp after."""
    input_type = TensorType(
        DType.uint8, ["n", "c", "h", "w"], device=DeviceRef.CPU()
    )
    size_type = TensorType(DType.int64, [4], device=DeviceRef.CPU())
    out_dims = ["n", "c", "h_out", "w_out"]
    with Graph(
        "test_resize_bicubic_uint8", input_types=[input_type, size_type]
    ) as graph:
        x, size = (v.tensor for v in graph.inputs)
        resized = ops.resize_bicubic(
            x.cast(DType.float32), size, antialias=True, out_dims=out_dims
        )
        cpu = DeviceRef.CPU()
        levels = ops.min(
            ops.max(ops.round(resized), ops.constant(0, DType.float32, cpu)),
            ops.constant(255, DType.float32, cpu),
        )
        graph.output(
            ops.resize_bicubic(x, size, antialias=True, out_dims=out_dims),
            levels.cast(DType.uint8),
        )
    return session.load(graph)


@pytest.mark.parametrize(
    "input_shape,output_hw",
    [
        ((1, 3, 301, 517), (97, 203)),
        ((2, 3, 37, 53), (112, 140)),
        ((1, 3, 64, 64), (96, 64)),
        # A 2x upscale of one side weighs taps in 128ths, so a whole-level
        # input often lands on an exact half.
        ((1, 3, 64, 32), (64, 64)),
        ((1, 3, 40, 40), (40, 40)),
    ],
)
def test_uint8_antialiased_resize_bicubic_rounds_the_float32_resize(
    uint8_and_float32_resizes: Model,
    input_shape: tuple[int, int, int, int],
    output_hw: tuple[int, int],
) -> None:
    """A uint8 resize is the float32 resize of the same pixels, rounded half
    to even and clamped, bit for bit. Noise overshoots [0, 255], so the
    clamp counts, and the tie break counts where a 2x upscale lands on
    exact halves."""
    rng = np.random.default_rng(input_shape[2] * output_hw[0])
    image = rng.integers(0, 256, input_shape, dtype=np.uint8)
    size = np.array([*input_shape[:2], *output_hw], np.int64)

    got, want = uint8_and_float32_resizes.execute(
        Buffer.from_numpy(image), Buffer.from_numpy(size)
    )
    assert isinstance(got, Buffer) and isinstance(want, Buffer)
    assert got.dtype == DType.uint8
    np.testing.assert_array_equal(got.to_numpy(), want.to_numpy())


@pytest.mark.parametrize(
    "dtype,device",
    [
        (DType.bfloat16, DeviceRef.CPU()),
        (DType.int32, DeviceRef.CPU()),
        (DType.float32, DeviceRef.GPU()),
    ],
)
def test_antialiased_resize_bicubic_rejects_unsupported_input(
    dtype: DType, device: DeviceRef
) -> None:
    input_type = TensorType(dtype, [1, 3, 8, 8], device=device)
    with Graph("bad_antialias", input_types=[input_type]) as graph:
        with pytest.raises(ValueError, match="float32 or uint8 input on"):
            ops.resize_bicubic(
                graph.inputs[0].tensor, [1, 3, 4, 4], antialias=True
            )


@pytest.mark.parametrize(
    "out_dims", [["n", 1, "h_out", "w_out"], [2, "c", "h_out", "w_out"]]
)
def test_resize_bicubic_rejects_out_dims_that_change_planes(
    out_dims: list[str | int],
) -> None:
    """The kernel resizes each of the input's N * C planes into the output's,
    so out_dims can't declare a different batch or channel count."""
    input_type = TensorType(
        DType.float32, ["n", "c", "h", "w"], device=DeviceRef.CPU()
    )
    size_type = TensorType(DType.int64, [4], device=DeviceRef.CPU())
    with Graph("bad_out_dims", input_types=[input_type, size_type]) as graph:
        x, size = (v.tensor for v in graph.inputs)
        with pytest.raises(ValueError, match="batch and channel"):
            ops.resize_bicubic(x, size, antialias=True, out_dims=out_dims)


@pytest.fixture(
    scope="module", params=[False, True], ids=["plain", "antialias"]
)
def free_output_resize(
    session: InferenceSession, request: pytest.FixtureRequest
) -> Model:
    """A CPU float32 bicubic resize whose every output dim is free, so a
    runtime size can name other batch or channel counts than the input's.
    ``ops.resize_bicubic`` refuses such out_dims, so the graph declares the
    op directly, as the eager interpreter's resize graphs do."""
    input_type = TensorType(
        DType.float32, ["n", "c", "h", "w"], device=DeviceRef.CPU()
    )
    size_type = TensorType(DType.int64, [4], device=DeviceRef.CPU())
    with Graph(
        "free_output_resize", input_types=[input_type, size_type]
    ) as graph:
        x, size = (v.tensor for v in graph.inputs)
        out_type = TensorType(
            DType.float32, ["o0", "o1", "o2", "o3"], device=DeviceRef.CPU()
        )
        graph.output(
            Graph.current._add_op_generated(
                rmo.MoResizeBicubicOp,
                result=out_type,
                input=x,
                size=size,
                antialias=builtin.BoolAttr(request.param),
                output_param_decls=kgen.ParamDeclArrayAttr([]),
            )[0].tensor
        )
    return session.load(graph)


@pytest.mark.parametrize("size", [[1, 1, 4, 4], [1, 6, 4, 4], [2, 3, 4, 4]])
def test_resize_bicubic_rejects_a_runtime_size_that_changes_planes(
    free_output_resize: Model, size: list[int]
) -> None:
    """The kernel resizes the input's N * C planes into the output's, so it
    raises on an output whose batch or channels differ."""
    image = np.zeros((1, 3, 8, 8), np.float32)
    with pytest.raises(Exception, match="batch and channel"):
        free_output_resize.execute(
            Buffer.from_numpy(image),
            Buffer.from_numpy(np.array(size, np.int64)),
        )


def _natural_image(
    rng: np.random.Generator, height: int, width: int
) -> NDArray[np.uint8]:
    """Returns smooth RGB content with mild noise, whose resize overshoots
    [0, 255] nowhere."""
    yy, xx = np.mgrid[0:height, 0:width]
    channels = [
        128 + 100 * np.sin(xx / (7 + 3 * c) + yy / (11 + c)) for c in range(3)
    ]
    image = np.stack(channels, -1) + rng.normal(0, 8, (height, width, 3))
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


@pytest.mark.parametrize(
    "in_hw,out_hw",
    [
        # Downscales, by whole and fractional factors.
        ((448, 448), (224, 224)),
        ((301, 517), (97, 203)),
        ((500, 400), (96, 128)),
        ((1500, 2000), (672, 912)),
        # Upscales, including from a tiny image.
        ((37, 53), (112, 140)),
        ((5, 5), (48, 48)),
        ((1, 9), (3, 3)),
        # Only one side changes.
        ((64, 64), (64, 32)),
        ((64, 64), (96, 64)),
        # Both change in opposite directions, or to one pixel.
        ((9, 13), (40, 5)),
        ((300, 200), (1, 1)),
        # Extreme aspect ratios.
        ((3, 3000), (3, 100)),
        ((10, 3000), (48, 48)),
        ((1000, 9), (100, 40)),
        ((549, 5), (81, 758)),
        ((851, 4), (2, 816)),
        # The size is unchanged.
        ((100, 100), (100, 100)),
    ],
)
def test_antialiased_resize_bicubic_is_near_pillows(
    antialiased_resize: Model,
    in_hw: tuple[int, int],
    out_hw: tuple[int, int],
) -> None:
    """A uint8 image resized in float32, then rounded and clamped, is within a
    level or two of Pillow's BICUBIC.

    Pillow resamples uint8 in fixed point and rounds and clamps to uint8
    between its two passes; the float32 resize does neither, so it differs
    from Pillow by an occasional level. Where a sharp edge overshoots
    [0, 255] between the passes, Pillow's clamp moves a few pixels further,
    so the bound holds for smooth content, not for noise or hard edges.
    """
    rng = np.random.default_rng(in_hw[0] * in_hw[1] + out_hw[0])
    image = _natural_image(rng, *in_hw)
    nchw = np.ascontiguousarray(image.transpose(2, 0, 1)[None], np.float32)
    size = np.array([1, 3, *out_hw], np.int64)

    result = antialiased_resize.execute(
        Buffer.from_numpy(nchw), Buffer.from_numpy(size)
    )[0]
    assert isinstance(result, Buffer)

    got = np.clip(np.rint(result.to_numpy()), 0, 255)
    expected = np.asarray(
        Image.fromarray(image).resize(
            (out_hw[1], out_hw[0]), Image.Resampling.BICUBIC
        )
    ).transpose(2, 0, 1)[None]
    diff = np.abs(got - expected.astype(np.float32))
    assert diff.max() <= 2
    assert diff.mean() <= 0.3
