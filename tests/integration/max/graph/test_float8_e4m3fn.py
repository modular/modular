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
"""Test float8_e4m3fn dtype support."""

import numpy as np
import pytest
import torch
import torch.utils.dlpack
from max.driver import Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, Weight, ops
from test_common.graph_utils import is_h100_h200


@pytest.mark.skipif(not is_h100_h200(), reason="float8 requires H100 or H200")
@pytest.mark.parametrize("cast_dtype", [DType.float32, DType.bfloat16])
def test_f8_upcast(session: InferenceSession, cast_dtype: DType) -> None:
    with Graph(
        "f8",
        input_types=[
            TensorType(
                dtype=DType.float8_e4m3fn,
                shape=["x", 2],
                device=DeviceRef.GPU(),
            ),
        ],
    ) as graph:
        x = graph.inputs[0].tensor
        out = x.cast(cast_dtype)
        graph.output(out)

    model = session.load(graph)

    x_torch = (
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]).to(torch.float8_e4m3fn).cuda()
    )
    x_tensor = Tensor.from_dlpack(x_torch.view(torch.uint8)).view(
        DType.float8_e4m3fn
    )
    result = model.execute(x_tensor)[0]
    assert isinstance(result, Tensor)
    out_torch = torch.utils.dlpack.from_dlpack(result)

    expected = x_torch.to(torch.float32).cpu()
    np.testing.assert_allclose(
        out_torch.to(torch.float32).cpu(), expected, rtol=0.06
    )


@pytest.mark.skipif(not is_h100_h200(), reason="float8 requires H100 or H200")
@pytest.mark.parametrize("cast_dtype", [DType.float32, DType.bfloat16])
def test_f8_downcast(session: InferenceSession, cast_dtype: DType) -> None:
    with Graph(
        "f8",
        input_types=[
            TensorType(
                dtype=cast_dtype, shape=["x", 2], device=DeviceRef.GPU()
            ),
        ],
    ) as graph:
        x = graph.inputs[0].tensor
        out = x.cast(DType.float8_e4m3fn)
        graph.output(out)

    model = session.load(graph)

    x_torch = (
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]).to(cast_dtype.to_torch()).cuda()
    )
    x_tensor = Tensor.from_dlpack(x_torch)
    result = model.execute(x_tensor)[0]
    assert isinstance(result, Tensor)
    out_torch = torch.utils.dlpack.from_dlpack(result.view(DType.uint8)).view(
        torch.float8_e4m3fn
    )

    expected = x_torch.to(torch.float32).cpu()
    np.testing.assert_allclose(
        out_torch.to(torch.float32).cpu(), expected, rtol=0.06
    )


@pytest.mark.skipif(not is_h100_h200(), reason="float8 requires H100 or H200")
def test_f8_matmul(session: InferenceSession) -> None:
    with Graph(
        "f8",
        input_types=[
            TensorType(
                dtype=DType.float8_e4m3fn,
                shape=["x", 2],
                device=DeviceRef.GPU(),
            )
        ],
    ) as graph:
        x = graph.inputs[0].tensor
        y = ops.constant(
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            dtype=DType.float8_e4m3fn,
            device=DeviceRef.GPU(),
        )
        out = x @ y
        graph.output(out)

    model = session.load(graph)

    x_torch = (
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]).to(torch.float8_e4m3fn).cuda()
    )
    y_torch = (
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]).to(torch.float8_e4m3fn).cuda()
    )
    x_tensor = Tensor.from_dlpack(x_torch.view(torch.uint8)).view(
        DType.float8_e4m3fn
    )
    result = model.execute(x_tensor)[0]
    assert isinstance(result, Tensor)
    out_torch = torch.utils.dlpack.from_dlpack(result.view(DType.uint8)).view(
        torch.float8_e4m3fn
    )

    # Torch does not support matmul of float8_e4m3fn.
    # Instead, run in bfloat16 and increase the tolerances.
    expected = (
        (x_torch.to(torch.bfloat16) @ y_torch.to(torch.bfloat16))
        .to(torch.float32)
        .cpu()
    )
    np.testing.assert_allclose(
        out_torch.to(torch.float32).cpu(), expected, rtol=0.06
    )


@pytest.mark.skipif(not is_h100_h200(), reason="float8 requires H100 or H200")
def test_f8_constant(session: InferenceSession) -> None:
    y_data = np.array([5.0, 6.0])
    with Graph(
        "f8",
        input_types=[
            TensorType(
                dtype=DType.float8_e4m3fn,
                shape=["x", 2],
                device=DeviceRef.GPU(),
            ),
        ],
    ) as graph:
        x = graph.inputs[0].tensor
        y = ops.constant(
            y_data, dtype=DType.float8_e4m3fn, device=DeviceRef.GPU()
        )
        out = x @ y
        graph.output(out)

    model = session.load(graph)

    x_torch = (
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]).to(torch.float8_e4m3fn).cuda()
    )
    y_torch = torch.tensor(y_data).to(torch.float8_e4m3fn).cuda()
    x_tensor = Tensor.from_dlpack(x_torch.view(torch.uint8)).view(
        DType.float8_e4m3fn
    )
    result = model.execute(x_tensor)[0]
    assert isinstance(result, Tensor)
    out_torch = torch.utils.dlpack.from_dlpack(result.view(DType.uint8)).view(
        torch.float8_e4m3fn
    )

    # Torch does not support matmul of float8_e4m3fn.
    # Instead, run in bfloat16 and increase the tolerances.
    expected = (
        (x_torch.to(torch.bfloat16) @ y_torch.to(torch.bfloat16))
        .to(torch.float32)
        .cpu()
    )
    np.testing.assert_allclose(
        out_torch.to(torch.float32).cpu(), expected, rtol=0.06
    )


@pytest.mark.skipif(not is_h100_h200(), reason="float8 requires H100 or H200")
def test_f8_weight_cpu(session: InferenceSession) -> None:
    y_data = np.array([5.0, 6.0])
    with Graph(
        "f8",
        input_types=[
            TensorType(
                dtype=DType.float8_e4m3fn,
                shape=["x", 2],
                device=DeviceRef.GPU(),
            ),
        ],
    ) as graph:
        x = graph.inputs[0].tensor
        y = Weight(
            "y",
            dtype=DType.float8_e4m3fn,
            shape=[2],
            device=DeviceRef.CPU(),
        )
        out = x @ y.to(DeviceRef.GPU())
        graph.output(out)

    y_torch = torch.tensor(y_data).to(torch.float8_e4m3fn).cuda()
    y_weight = Tensor.from_dlpack(y_torch.cpu().view(torch.uint8)).view(
        DType.float8_e4m3fn
    )
    model = session.load(graph, weights_registry={"y": y_weight})

    x_torch = (
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]).to(torch.float8_e4m3fn).cuda()
    )
    x_tensor = Tensor.from_dlpack(x_torch.view(torch.uint8)).view(
        DType.float8_e4m3fn
    )
    result = model.execute(x_tensor)[0]
    assert isinstance(result, Tensor)
    out_torch = torch.utils.dlpack.from_dlpack(result.view(DType.uint8)).view(
        torch.float8_e4m3fn
    )

    # Torch does not support matmul of float8_e4m3fn.
    # Instead, run in bfloat16 and increase the tolerances.
    expected = (
        (x_torch.to(torch.bfloat16) @ y_torch.to(torch.bfloat16))
        .to(torch.float32)
        .cpu()
    )
    np.testing.assert_allclose(
        out_torch.to(torch.float32).cpu(), expected, rtol=0.06
    )


@pytest.mark.skipif(not is_h100_h200(), reason="float8 requires H100 or H200")
def test_f8_weight_gpu(session: InferenceSession) -> None:
    y_data = np.array([5.0, 6.0])
    with Graph(
        "f8",
        input_types=[
            TensorType(
                dtype=DType.float8_e4m3fn,
                shape=["x", 2],
                device=DeviceRef.GPU(),
            ),
        ],
    ) as graph:
        x = graph.inputs[0].tensor
        y = Weight(
            "y",
            dtype=DType.float8_e4m3fn,
            shape=[2],
            device=DeviceRef.GPU(),
        )
        out = x @ y
        graph.output(out)

    y_torch = torch.tensor(y_data).to(torch.float8_e4m3fn).cuda()
    y_weight = Tensor.from_dlpack(y_torch.cpu().view(torch.uint8)).view(
        DType.float8_e4m3fn
    )
    model = session.load(graph, weights_registry={"y": y_weight})

    x_torch = (
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]).to(torch.float8_e4m3fn).cuda()
    )
    x_tensor = Tensor.from_dlpack(x_torch.view(torch.uint8)).view(
        DType.float8_e4m3fn
    )
    result = model.execute(x_tensor)[0]
    assert isinstance(result, Tensor)
    out_torch = torch.utils.dlpack.from_dlpack(result.view(DType.uint8)).view(
        torch.float8_e4m3fn
    )

    # Torch does not support matmul of float8_e4m3fn.
    # Instead, run in bfloat16 and increase the tolerances.
    expected = (
        (x_torch.to(torch.bfloat16) @ y_torch.to(torch.bfloat16))
        .to(torch.float32)
        .cpu()
    )
    np.testing.assert_allclose(
        out_torch.to(torch.float32).cpu(), expected, rtol=0.06
    )
