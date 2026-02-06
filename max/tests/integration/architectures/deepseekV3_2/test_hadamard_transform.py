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
"""Test comparing MAX Hadamard transform implementation to PyTorch."""

from __future__ import annotations

import math
from typing import Any

import pytest
import scipy.linalg
import torch
import torch.nn.functional as F
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Dim, Graph, TensorType
from max.pipelines.architectures.deepseekV3_2.layers import (
    HadamardTransform,
)
from torch.utils.dlpack import from_dlpack

# Check if outputs are close with appropriate tolerance
RTOL = 2 * torch.finfo(torch.bfloat16).eps
ATOL = 8 * torch.finfo(torch.bfloat16).eps


def hadamard_transform_ref(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """
    Reference implementation with padding logic.
    [Source](https://github.com/Dao-AILab/fast-hadamard-transform/blob/master/fast_hadamard_transform/fast_hadamard_transform_interface.py#L156)
    Arguments:
        x: (..., dim)
        scale: float. Multiply the output by this number.
    Returns:
        out: (..., dim)

    Multiply each row of x by the Hadamard transform matrix.
    Equivalent to F.linear(x, torch.tensor(scipy.linalg.hadamard(dim))) * scale.
    If dim is not a power of 2, we implicitly pad x with zero so that dim is the next power of 2.
    """
    x_shape = x.shape
    dim = x.shape[-1]
    x = x.reshape(-1, dim)
    log_dim = math.ceil(math.log2(dim))
    dim_padded = 2**log_dim
    if dim != dim_padded:
        x = F.pad(x, (0, dim_padded - dim))

    hadamard_weight = torch.tensor(
        scipy.linalg.hadamard(dim_padded, dtype=float),
        dtype=x.dtype,
        device=x.device,
    )
    out = F.linear(x, hadamard_weight)
    out = out * scale
    return out[..., :dim].reshape(*x_shape)


def generate_max_hadamard_transform(
    x: torch.Tensor,
    shape: tuple[Dim],
    scale: float,
    dtype: DType = DType.bfloat16,
) -> torch.Tensor:
    """Generate output using MAX Hadamard transform.

    Args:
        x: Input tensor of shape (..., dim).
        dim: Dimension of the Hadamard matrix.
        scale: Scale factor to apply after the transform.
        dtype: Data type for MAX computation.

    Returns:
        Transformed tensor using MAX matmul operation.
    """
    cuda = Accelerator()
    session = InferenceSession(devices=[cuda])

    with Graph(
        "hadamard_transform_test",
        input_types=[
            TensorType(
                dtype=dtype,
                shape=shape,
                device=DeviceRef.GPU(),
            )
        ],
    ) as graph:
        inputs = graph.inputs[0].tensor
        hadamard_transform = HadamardTransform(scale)
        result = hadamard_transform(inputs)
        graph.output(result)

    model = session.load(graph)
    x_buffer = Buffer.from_dlpack(x).to(cuda)
    max_output = model.execute(x_buffer)[0]

    return max_output


@pytest.mark.parametrize(
    "input_shape,scale",
    [
        # ((), 1.0), # will raise (correctly)
        # ((1,), 1.0),
        ((1, 2), 1.0),
        # ((1, 3), 1.0),
        ((2, 6), 0.5),
        # ((4, 16), 2.0),
        ((2, 1, 3), 1.0),
        # ((3, 3, 4, 2), 1.0), # times out
    ],
)
@torch.no_grad()
def test_hadamard_transform(
    input_shape: tuple, scale: float, torch_dtype: Any = torch.bfloat16
) -> None:
    """Test Hadamard transform comparing MAX vs PyTorch implementation.

    Args:
        batch_size: Batch size for input tensor.
        dim: Dimension of the Hadamard matrix (must be power of 2).
        scale: Scale factor to apply after transform.
        dtype: Data type for MAX computation.
    """
    assert len(input_shape), "Need at least one dim."

    # Create random input tensor
    dim = input_shape[-1]
    torch.manual_seed(42)
    input_tensor = torch.randn(*input_shape, dtype=torch_dtype, device="cuda")

    # Generate PyTorch output
    torch_output = hadamard_transform_ref(input_tensor, scale)

    # Generate MAX output
    max_output = generate_max_hadamard_transform(
        input_tensor, input_tensor.shape, scale
    )
    # Convert MAX output to torch for comparison
    max_output_torch = from_dlpack(max_output)

    # Verify shapes match
    assert torch_output.shape == max_output_torch.shape, (
        f"Shape mismatch: {torch_output.shape} vs {max_output_torch.shape}"
    )

    torch.testing.assert_close(
        torch_output,
        max_output_torch,
        rtol=RTOL,
        atol=ATOL,
    )
