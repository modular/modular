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
"""Tests for acos operation."""

import platform

import numpy as np
import pytest
import torch
from max.driver import Tensor
from max.dtype import DType
from max.engine.api import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops


@pytest.mark.parametrize("dtype", [DType.float32, DType.bfloat16])
def test_acos(session: InferenceSession, dtype: DType) -> None:
    """Test acos with random values in the valid domain [-1, 1]."""
    if dtype == DType.bfloat16 and platform.machine() in ["arm64", "aarch64"]:
        pytest.skip("BF16 is not supported on ARM CPU architecture")

    input_type = TensorType(
        dtype, [1024], device=DeviceRef.from_device(session.devices[0])
    )

    with Graph(f"acos_{dtype}", input_types=[input_type]) as graph:
        out = ops.acos(graph.inputs[0].tensor)
        graph.output(out.cast(DType.float32))

    model = session.load(graph)

    # Set fixed seed for reproducibility
    torch.manual_seed(42)

    torch_dtype = torch.float32 if dtype == DType.float32 else torch.bfloat16

    # Generate random values in the valid domain [-1, 1]
    input_data = torch.rand(1024, dtype=torch_dtype) * 2.0 - 1.0

    output = model(Tensor.from_dlpack(input_data).to(model.input_devices[0]))[0]
    assert isinstance(output, Tensor)
    max_result = output.to_numpy()

    torch_result = torch.acos(input_data).to(dtype=torch.float32).cpu().numpy()

    np.testing.assert_allclose(
        max_result,
        torch_result,
        rtol=1e-5,
        atol=1e-5,
        verbose=True,
    )


def test_acos_special_values(session: InferenceSession) -> None:
    """Test acos with special mathematical values."""
    input_type = TensorType(
        DType.float32, [5], device=DeviceRef.from_device(session.devices[0])
    )

    with Graph("acos_special", input_types=[input_type]) as graph:
        out = ops.acos(graph.inputs[0].tensor)
        graph.output(out)

    model = session.load(graph)

    # Test known mathematical values:
    # acos(1.0) = 0.0
    # acos(0.5) ≈ 1.0472 (π/3)
    # acos(0.0) = π/2 ≈ 1.5708
    # acos(-0.5) ≈ 2.0944 (2π/3)
    # acos(-1.0) = π ≈ 3.1416
    input_data = torch.tensor([1.0, 0.5, 0.0, -0.5, -1.0], dtype=torch.float32)

    output = model(Tensor.from_dlpack(input_data).to(model.input_devices[0]))[0]
    assert isinstance(output, Tensor)
    max_result = output.to_numpy()

    torch_result = torch.acos(input_data).cpu().numpy()

    np.testing.assert_allclose(
        max_result,
        torch_result,
        rtol=1e-6,
        atol=1e-6,
        verbose=True,
    )


def test_acos_2d_tensor(session: InferenceSession) -> None:
    """Test acos with 2D tensor."""
    input_type = TensorType(
        DType.float32,
        [10, 10],
        device=DeviceRef.from_device(session.devices[0]),
    )

    with Graph("acos_2d", input_types=[input_type]) as graph:
        out = ops.acos(graph.inputs[0].tensor)
        graph.output(out)

    model = session.load(graph)

    torch.manual_seed(123)
    input_data = torch.rand(10, 10, dtype=torch.float32) * 2.0 - 1.0

    output = model(Tensor.from_dlpack(input_data).to(model.input_devices[0]))[0]
    assert isinstance(output, Tensor)
    max_result = output.to_numpy()

    torch_result = torch.acos(input_data).cpu().numpy()

    np.testing.assert_allclose(
        max_result,
        torch_result,
        rtol=1e-6,
        atol=1e-6,
        verbose=True,
    )


def test_acos_edge_domain_values(session: InferenceSession) -> None:
    """Test acos with values near domain boundaries."""
    input_type = TensorType(
        DType.float32, [6], device=DeviceRef.from_device(session.devices[0])
    )

    with Graph("acos_edge", input_types=[input_type]) as graph:
        out = ops.acos(graph.inputs[0].tensor)
        graph.output(out)

    model = session.load(graph)

    # Test values very close to -1 and 1, and at the boundaries
    input_data = torch.tensor(
        [-1.0, -0.999, -0.99, 0.99, 0.999, 1.0], dtype=torch.float32
    )

    output = model(Tensor.from_dlpack(input_data).to(model.input_devices[0]))[0]
    assert isinstance(output, Tensor)
    max_result = output.to_numpy()

    torch_result = torch.acos(input_data).cpu().numpy()

    np.testing.assert_allclose(
        max_result,
        torch_result,
        rtol=1e-5,
        atol=1e-5,
        verbose=True,
    )


def test_acos_clamping(session: InferenceSession) -> None:
    """Test that acos properly clamps values outside [-1, 1]."""
    input_type = TensorType(
        DType.float32, [4], device=DeviceRef.from_device(session.devices[0])
    )

    with Graph("acos_clamp", input_types=[input_type]) as graph:
        out = ops.acos(graph.inputs[0].tensor)
        graph.output(out)

    model = session.load(graph)

    # Values outside the valid domain should be clamped
    # acos implementation clamps to [-1, 1] internally
    input_data = torch.tensor([-1.5, -1.0, 1.0, 1.5], dtype=torch.float32)

    output = model(Tensor.from_dlpack(input_data).to(model.input_devices[0]))[0]
    assert isinstance(output, Tensor)
    max_result = output.to_numpy()

    # Expected: clamped values should give acos(-1) or acos(1)
    expected = torch.tensor(
        [np.pi, np.pi, 0.0, 0.0], dtype=torch.float32
    ).numpy()

    np.testing.assert_allclose(
        max_result,
        expected,
        rtol=1e-5,
        atol=1e-5,
        verbose=True,
    )
