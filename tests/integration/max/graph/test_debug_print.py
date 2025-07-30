# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
"""Test value printing and debug print options."""

import platform
from pathlib import Path

import numpy as np
import pytest
import torch
from max.driver import Tensor, accelerator_count
from max.driver.tensor import load_max_tensor
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    Graph,
    TensorType,
    TensorValue,
)


@pytest.fixture(scope="module")
def compiled_model(session: InferenceSession) -> Model:
    def print_input(x: TensorValue) -> TensorValue:
        x.print("test_x_value")
        return x

    g = Graph(
        "test_print",
        print_input,
        [TensorType(DType.float32, ["dim1"], device=DeviceRef.CPU())],
    )
    return session.load(g)


@pytest.fixture(scope="module")
def compiled_buffer_model(session: InferenceSession):
    def print_input(x: BufferValue) -> BufferValue:
        x.print("test_x_value")
        return x

    g = Graph(
        "test_print",
        print_input,
        [BufferType(DType.float32, ["dim1"], device=DeviceRef.CPU())],
    )
    return session.load(g)


def test_debug_print_options(session: InferenceSession, tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="Invalid debug print style"):
        session.set_debug_print_options("NOTVALID")

    session.set_debug_print_options("COMPACT")
    session.set_debug_print_options("NONE")

    session.set_debug_print_options("FULL")
    session.set_debug_print_options("FULL", precision=5)
    with pytest.raises(TypeError, match="precision must be an int"):
        session.set_debug_print_options("FULL", precision=5.0)  # type: ignore[arg-type]

    session.set_debug_print_options("BINARY", output_directory=tmp_path)
    session.set_debug_print_options("BINARY", output_directory=str(tmp_path))
    with pytest.raises(ValueError, match="output directory cannot be empty"):
        session.set_debug_print_options("BINARY")


def test_debug_print_compact(
    compiled_model: Model,
    session: InferenceSession,
    capfd: pytest.CaptureFixture,
) -> None:
    session.set_debug_print_options("COMPACT")
    _ = compiled_model.execute(
        Tensor.from_numpy(np.full([20], 1.1234567, np.float32)).to(
            compiled_model.input_devices[0]
        )
    )
    captured = capfd.readouterr()
    assert (
        "test_x_value = tensor([[1.1235, 1.1235, 1.1235, ..., 1.1235,"
        " 1.1235, 1.1235]], dtype=f32, shape=[20])" in captured.out
    )


def test_debug_print_buffer(
    compiled_buffer_model: Model,
    session: InferenceSession,
    capfd: pytest.CaptureFixture,
) -> None:
    session.set_debug_print_options("COMPACT")
    _ = compiled_buffer_model.execute(
        Tensor.from_numpy(np.full([20], 1.1234567, np.float32)).to(
            compiled_buffer_model.input_devices[0]
        )
    )
    captured = capfd.readouterr()
    assert (
        "test_x_value = tensor([[1.1235, 1.1235, 1.1235, ..., 1.1235,"
        " 1.1235, 1.1235]], dtype=f32, shape=[20])" in captured.out
    )


def test_debug_print_full(
    compiled_model: Model,
    session: InferenceSession,
    capfd: pytest.CaptureFixture,
) -> None:
    session.set_debug_print_options("FULL", 2)
    _ = compiled_model.execute(
        Tensor.from_numpy(np.full([20], 1.1234567, np.float32)).to(
            compiled_model.input_devices[0]
        )
    )
    captured = capfd.readouterr()
    assert (
        "test_x_value = tensor<20xf32> [1.12e+00, 1.12e+00, 1.12e+00,"
        " 1.12e+00, 1.12e+00, 1.12e+00, 1.12e+00, 1.12e+00, 1.12e+00, 1.12e+00,"
        " 1.12e+00, 1.12e+00, 1.12e+00, 1.12e+00, 1.12e+00, 1.12e+00, 1.12e+00,"
        " 1.12e+00, 1.12e+00, 1.12e+00]" in captured.out
    )

    session.set_debug_print_options("FULL", 6)
    _ = compiled_model.execute(
        Tensor.from_numpy(np.full([20], 1.1234567, np.float32)).to(
            compiled_model.input_devices[0]
        )
    )
    captured = capfd.readouterr()
    assert (
        "test_x_value = tensor<20xf32> [1.123457e+00, 1.123457e+00,"
        " 1.123457e+00, 1.123457e+00, 1.123457e+00, 1.123457e+00, 1.123457e+00,"
        " 1.123457e+00, 1.123457e+00, 1.123457e+00, 1.123457e+00, 1.123457e+00,"
        " 1.123457e+00, 1.123457e+00, 1.123457e+00, 1.123457e+00, 1.123457e+00,"
        " 1.123457e+00, 1.123457e+00, 1.123457e+00]" in captured.out
    )


def test_debug_print_none(
    compiled_model: Model,
    session: InferenceSession,
    capfd: pytest.CaptureFixture,
) -> None:
    session.set_debug_print_options("NONE")
    _ = compiled_model.execute(
        Tensor.from_numpy(np.full([20], 1.1234567, np.float32)).to(
            compiled_model.input_devices[0]
        )
    )
    captured = capfd.readouterr()
    assert "test_x_value" not in captured.out


def test_debug_print_binary(
    compiled_model: Model,
    session: InferenceSession,
    capfd: pytest.CaptureFixture,
    tmp_path: Path,
) -> None:
    session.set_debug_print_options("BINARY", output_directory=tmp_path)
    input = np.full([20], 1.1234567, np.float32)
    _ = compiled_model.execute(
        Tensor.from_numpy(input).to(compiled_model.input_devices[0])
    )
    captured = capfd.readouterr()
    assert "test_x_value" not in captured.out
    assert (tmp_path / "test_x_value").exists()
    from_file = np.fromfile(tmp_path / "test_x_value", np.float32)
    assert (input == from_file).all()


def test_debug_print_binary_max(
    compiled_model: Model,
    session: InferenceSession,
    capfd: pytest.CaptureFixture,
    tmp_path: Path,
) -> None:
    session.set_debug_print_options(
        "BINARY_MAX_CHECKPOINT", output_directory=tmp_path
    )
    input = np.random.uniform(size=[15]).astype(np.float32)
    _ = compiled_model.execute(
        Tensor.from_numpy(input).to(compiled_model.input_devices[0])
    )
    captured = capfd.readouterr()
    assert "test_x_value" not in captured.out

    max_path = tmp_path / "test_x_value.max"
    assert max_path.exists()
    from_file = load_max_tensor(max_path).to_numpy()
    assert (input == from_file).all()


@pytest.mark.skipif(
    platform.machine() in ["arm64", "aarch64"],
    reason="BF16 is not supported on ARM CPU architecture",
)
def test_debug_print_binary_max_bf16(
    session: InferenceSession, capfd: pytest.CaptureFixture, tmp_path: Path
) -> None:
    def print_input(x: TensorValue) -> TensorValue:
        x.print("test_x_value")
        return x

    g = Graph(
        "test_print",
        print_input,
        [TensorType(DType.bfloat16, ["dim1"], device=DeviceRef.CPU())],
    )
    compiled_model = session.load(g)

    session.set_debug_print_options(
        "BINARY_MAX_CHECKPOINT", output_directory=tmp_path
    )
    dim = 15
    input = Tensor(DType.bfloat16, [dim])
    for i in range(dim):
        input[i] = np.random.uniform()

    _ = compiled_model.execute(input.to(compiled_model.input_devices[0]))
    captured = capfd.readouterr()
    assert "test_x_value" not in captured.out

    max_path = tmp_path / "test_x_value.max"
    assert max_path.exists()
    from_file = load_max_tensor(max_path)

    for i in range(dim):
        assert from_file[i].item() == input[i].item()


@pytest.mark.skipif(
    platform.machine() in ["arm64", "aarch64"],
    reason="BF16 is not supported on ARM CPU architecture",
)
@pytest.mark.parametrize("shape", [(), (5,), (2, 3), (2, 3, 4)])
def test_debug_print_binary_max_bf16_shapes(
    session: InferenceSession,
    capfd: pytest.CaptureFixture,
    tmp_path: Path,
    shape: tuple[int, ...],
) -> None:
    """Test bfloat16 tensor loading with various shapes including scalar"""

    def print_input(x: TensorValue) -> TensorValue:
        x.print("test_x_value")
        return x

    # Create graph with specified shape
    g = Graph(
        f"test_print_bf16_{len(shape)}d",
        print_input,
        [TensorType(DType.bfloat16, shape, device=DeviceRef.CPU())],
    )
    compiled_model = session.load(g)

    # Generate test data.
    input = Tensor(DType.bfloat16, shape)
    rng = np.random.default_rng()

    if shape:
        # Generate and set values for non-scalar tensors.
        np_arr = rng.uniform(size=shape).astype(np.float32)
        for idx in np.ndindex(shape):
            input[idx] = np_arr[idx]
    else:
        # Special handling for scalar.
        input[()] = rng.uniform()

    # Execute and save.
    session.set_debug_print_options(
        "BINARY_MAX_CHECKPOINT", output_directory=tmp_path
    )
    _ = compiled_model.execute(input.to(compiled_model.input_devices[0]))

    # Verify saved file.
    max_path = tmp_path / "test_x_value.max"
    assert max_path.exists()

    # Load and validate.
    loaded = load_max_tensor(max_path)
    if shape:
        assert loaded.shape == shape
    else:
        # TODO(bduke): `max.driver.Tensor` saves/loads bfloat16 scalars as a tensor due to using a
        # 2-byte uint8 tensor as an intermediate format.
        assert len(loaded.shape) == 1 and loaded.shape[0] == 1
    assert loaded.dtype == DType.bfloat16

    # Element-wise comparison.
    if shape:
        for idx in np.ndindex(shape):
            assert loaded[idx].item() == input[idx].item()
    else:
        assert loaded.item() == input[()].item()


@pytest.mark.parametrize("dtype", list(DType))
def test_save_load_all_dtypes(
    session: InferenceSession, tmp_path: Path, dtype: DType
) -> None:
    """Verify round-trip save/load works for all supported DType values."""
    # Skip unsupported configurations.
    if dtype.is_half() and platform.machine() in ["arm64", "aarch64"]:
        pytest.skip("half-precision types not supported on ARM")

    if dtype == DType.float16 and accelerator_count() == 0:
        pytest.skip("F16 type is not supported on cpu")

    if dtype.is_float8():
        pytest.skip("F8 types are not supported by dlpack")

    # Create test graph.
    def print_input(x: TensorValue) -> TensorValue:
        x.print("test_x_value")
        return x

    shape = (2,)
    compiled_model = session.load(
        Graph(
            f"test_{dtype.name}",
            print_input,
            [TensorType(dtype, shape, device=DeviceRef.CPU())],
        )
    )

    # Generate test data based on dtype using PyTorch.
    device = "cuda" if dtype.is_float8() else "cpu"
    torch_dtype = getattr(torch, dtype.name)

    if dtype.is_float():
        if dtype.is_float8():
            # Generate FP8 data with appropriate scaling.
            torch_data = torch.rand(shape, dtype=torch.float32, device=device)
            torch_data = torch_data.to(torch_dtype)
        else:
            torch_data = torch.rand(shape, dtype=torch_dtype, device=device)
    elif dtype.is_integral():
        torch_data = torch.randint(
            0, 100, shape, dtype=torch_dtype, device=device
        )
    else:
        pytest.skip(f"No test data generation implemented for {dtype}")

    input_tensor = Tensor.from_dlpack(torch_data)

    # Save and load.
    session.set_debug_print_options(
        "BINARY_MAX_CHECKPOINT", output_directory=tmp_path
    )
    _ = compiled_model.execute(input_tensor.to(compiled_model.input_devices[0]))

    max_path = tmp_path / "test_x_value.max"
    loaded_tensor = load_max_tensor(max_path)

    # Validate using PyTorch
    assert loaded_tensor.dtype == dtype
    assert loaded_tensor.shape == shape

    loaded_torch = torch.from_dlpack(loaded_tensor)
    if dtype.is_float():
        # Convert FP8 back to float32 for comparison.
        if dtype.is_float8():
            loaded_torch = loaded_torch.float()
            torch_data = torch_data.float()
        torch.testing.assert_close(loaded_torch.cpu(), torch_data.cpu())
    else:
        torch.testing.assert_close(loaded_torch.cpu(), torch_data.cpu())
