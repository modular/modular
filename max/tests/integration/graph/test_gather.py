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

from collections.abc import Sequence

import numpy as np
import pytest
import torch
from max.driver import Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType
from max.graph.ops import cast, gather


def test_gather_valid_indices(session: InferenceSession) -> None:
    input_shape = [3, 2]
    index_shape = [5, 1]
    axis = 0
    device_to_use = DeviceRef.from_device(session.devices[0])
    with Graph(
        "conv2d",
        input_types=[
            TensorType(DType.int64, input_shape, device=device_to_use),
            TensorType(DType.int64, index_shape, device=device_to_use),
        ],
    ) as graph:
        input, index = graph.inputs

        output = gather(input.tensor, index.tensor, axis=axis)
        graph.output(output)

    model = session.load(graph)

    inputs = torch.Tensor([[0, 1], [2, 3], [4, 5]]).to(torch.int64)
    index = torch.Tensor([[0], [1], [2], [1], [1]]).to(torch.int64)

    actual_tensor = model(
        Buffer.from_dlpack(inputs).to(model.input_devices[0]),
        Buffer.from_dlpack(index).to(model.input_devices[1]),
    )[0]
    assert isinstance(actual_tensor, Buffer)
    actual = actual_tensor.to_numpy()
    expected = torch.take_along_dim(inputs, index, dim=0).numpy()
    np.testing.assert_equal(actual.reshape(5, 2), expected)


@pytest.mark.skipif(
    accelerator_count() > 0, reason="Bounds checking only runs on CPU"
)
def test_gather_invalid_indices(session: InferenceSession) -> None:
    input_shape = [3, 2]
    index_shape = [5, 1]
    axis = 0
    device_to_use = DeviceRef.from_device(session.devices[0])
    with Graph(
        "conv2d",
        input_types=[
            TensorType(DType.int64, input_shape, device=device_to_use),
            TensorType(DType.int64, index_shape, device=device_to_use),
        ],
    ) as graph:
        input, index = graph.inputs

        output = gather(input.tensor, index.tensor, axis=axis)
        graph.output(output)

    model = session.load(graph)

    inputs = torch.Tensor([[0, 1], [2, 3], [4, 5]]).to(torch.int64)
    # Should raise an error since indices must be between [0,3).
    index = torch.Tensor([[0], [1], [2], [3], [4]]).to(torch.int64)

    with pytest.raises(ValueError):
        model(
            Buffer.from_dlpack(inputs).to(model.input_devices[0]),
            Buffer.from_dlpack(index).to(model.input_devices[1]),
        )


def _load_gather_last_axis(
    session: InferenceSession,
    data_dtype: DType,
    index_dtype: DType,
    index_shape: Sequence[int | str],
) -> Model:
    """Compiles a CPU `gather` along the last axis of a [3, 5, 17] input."""
    device = DeviceRef.CPU()
    with Graph(
        "gather_last_axis",
        input_types=[
            TensorType(data_dtype, [3, 5, 17], device=device),
            TensorType(index_dtype, index_shape, device=device),
        ],
    ) as graph:
        data, index = graph.inputs
        graph.output(gather(data.tensor, index.tensor, axis=-1))
    return session.load(graph)


def _run_gather(
    model: Model, data: np.ndarray, index: np.ndarray
) -> np.ndarray:
    result = model(
        Buffer.from_numpy(data).to(model.input_devices[0]),
        Buffer.from_numpy(index).to(model.input_devices[1]),
    )[0]
    assert isinstance(result, Buffer)
    return result.to_numpy()


@pytest.mark.parametrize("data_dtype", [DType.uint8, DType.float32])
@pytest.mark.parametrize("index_dtype", [DType.int32, DType.int64])
def test_gather_last_axis(
    session: InferenceSession, data_dtype: DType, index_dtype: DType
) -> None:
    """Gathering the last axis on CPU matches `np.take`, including negative
    indices and index counts that leave a SIMD remainder.
    """
    model = _load_gather_last_axis(session, data_dtype, index_dtype, ["n"])
    rng = np.random.default_rng(0)
    data = rng.integers(0, 256, [3, 5, 17]).astype(data_dtype.to_numpy())
    for n in [1, 7, 64, 65]:
        index = rng.integers(-17, 17, [n]).astype(index_dtype.to_numpy())
        np.testing.assert_array_equal(
            _run_gather(model, data, index), np.take(data, index, axis=-1)
        )


def test_gather_last_axis_2d_indices(session: InferenceSession) -> None:
    """Multi-dimensional indices gather along the last axis."""
    model = _load_gather_last_axis(session, DType.float32, DType.int32, [4, 33])
    rng = np.random.default_rng(0)
    data = rng.standard_normal([3, 5, 17]).astype(np.float32)
    index = rng.integers(-17, 17, [4, 33]).astype(np.int32)
    np.testing.assert_array_equal(
        _run_gather(model, data, index), np.take(data, index, axis=-1)
    )


@pytest.mark.parametrize(
    "image_shape",
    [
        # The innermost extent is below any SIMD width, so every element takes
        # the scalar path.
        [9, 71, 3],
        # Whole vectors and a remainder along the innermost axis, with the
        # uint8 index loads fused with the cast.
        [9, 5, 67],
    ],
)
def test_gather_lut_from_uint8_image(
    session: InferenceSession, image_shape: list[int]
) -> None:
    """A uint8 image indexes a float32 lookup table through a fused cast."""
    device = DeviceRef.CPU()
    with Graph(
        "gather_lut",
        input_types=[
            TensorType(DType.float32, [256], device=device),
            TensorType(DType.uint8, image_shape, device=device),
        ],
    ) as graph:
        lut, image = graph.inputs
        graph.output(
            gather(lut.tensor, cast(image.tensor, DType.int32), axis=0)
        )
    model = session.load(graph)

    rng = np.random.default_rng(1)
    lut_np = rng.standard_normal(256).astype(np.float32)
    image_np = rng.integers(0, 256, image_shape).astype(np.uint8)
    np.testing.assert_array_equal(
        _run_gather(model, lut_np, image_np), lut_np[image_np]
    )


def test_gather_last_axis_invalid_indices(session: InferenceSession) -> None:
    """An out-of-range index raises wherever it falls in a vector."""
    model = _load_gather_last_axis(session, DType.float32, DType.int64, [41])
    data = np.zeros([3, 5, 17], np.float32)
    for bad_index in [17, -18]:
        for bad_lane in [0, 3, 40]:
            index = np.arange(41, dtype=np.int64) % 17
            index[bad_lane] = bad_index
            with pytest.raises(ValueError, match="out of bounds"):
                _run_gather(model, data, index)
