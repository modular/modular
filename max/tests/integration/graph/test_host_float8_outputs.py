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
"""Host-resident FP8 and E8M0 tensors as graph outputs. DISTINF-836."""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from numpy.typing import NDArray

# Raw byte patterns, since NumPy has no FP8 or E8M0 dtype.
_CODES = {
    DType.float8_e4m3fn: np.array([0x00, 0x38, 0x7E, 0xB8], dtype=np.uint8),
    DType.float8_e8m0fnu: np.array([0x7F, 0x80, 0x7E, 0x00], dtype=np.uint8),
}


def _host_tensor(codes: NDArray[np.uint8], dtype: DType) -> Buffer:
    """Returns a host buffer holding `codes` as the raw bytes of `dtype`."""
    return Buffer.from_numpy(codes).view(dtype, list(codes.shape))


def _result_bytes(result: Buffer, length: int) -> NDArray[np.uint8]:
    """Returns the raw bytes of an 8-bit-wide float result."""
    return np.asarray(
        np.from_dlpack(result.view(DType.uint8, [length])), dtype=np.uint8
    )


@pytest.fixture(scope="module")
def host_session() -> InferenceSession:
    """Returns a session whose default device is the host."""
    return InferenceSession(devices=[CPU()])


@pytest.mark.parametrize("dtype", list(_CODES), ids=lambda dtype: dtype.name)
def test_constant_output(host_session: InferenceSession, dtype: DType) -> None:
    """A host constant of an unsupported float dtype is a valid graph output."""
    codes = _CODES[dtype]
    with Graph("host_float8_constant", input_types=[]) as graph:
        constant = ops.constant(
            _host_tensor(codes, dtype), dtype, device=DeviceRef.CPU()
        )
        graph.output(constant)

    (result,) = host_session.load(graph).execute()
    assert result.dtype == dtype
    np.testing.assert_array_equal(_result_bytes(result, len(codes)), codes)


@pytest.mark.parametrize("dtype", list(_CODES), ids=lambda dtype: dtype.name)
def test_input_output(host_session: InferenceSession, dtype: DType) -> None:
    """A host tensor of an unsupported float dtype passes through the graph."""
    codes = _CODES[dtype]
    tensor_type = TensorType(dtype, codes.shape, device=DeviceRef.CPU())
    with Graph("host_float8_passthrough", input_types=[tensor_type]) as graph:
        graph.output(graph.inputs[0].tensor)

    model = host_session.load(graph)
    (result,) = model.execute(_host_tensor(codes, dtype).to(CPU()))
    assert result.dtype == dtype
    np.testing.assert_array_equal(_result_bytes(result, len(codes)), codes)
