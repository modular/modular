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

import pytest
from max.driver import CPU, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Dim, Graph, TensorType


@pytest.mark.skip("MAXPLAT-332: parameter with no declaration")
def test_rebind__new_parameter_expression__not_divisible_by_4(
    session: InferenceSession,
) -> None:
    input = Tensor(DType.float32, [7, 4], device=CPU())
    input_type = TensorType(DType.float32, ["batch", 4], device=DeviceRef.CPU())

    with Graph("reshape", input_types=[input_type]) as graph:
        (x,) = graph.inputs
        x = x.tensor.rebind([Dim("n_patches_over_4") * 4, 4])
        n_patches, _ = x.shape
        graph.output(x.reshape([n_patches // 4, 4, 4]))

    model = session.load(graph)
    with pytest.raises(Exception):
        model.execute(input)


@pytest.mark.skip("MAXPLAT-332: parameter with no declaration")
def test_rebind__new_parameter_expression__divisible_by_4(
    session: InferenceSession,
) -> None:
    input = Tensor(DType.float32, [8, 4], device=CPU())
    input_type = TensorType(DType.float32, ["batch", 4], device=DeviceRef.CPU())

    with Graph("reshape", input_types=[input_type]) as graph:
        (x,) = graph.inputs
        x = x.tensor.rebind([Dim("n_patches_over_4") * 4, 4])
        n_patches, _ = x.shape
        graph.output(x.reshape([n_patches // 4, 4, 4]))

    model = session.load(graph)
    result = model.execute(input)[0]
    assert isinstance(result, Tensor)
    assert result.shape == (2, 4, 4)


def test_rebind__no_new_parameter__not_divisible_by_4(
    session: InferenceSession,
) -> None:
    input = Tensor(DType.float32, [7, 4], device=CPU())
    input_type = TensorType(DType.float32, ["batch", 4], device=DeviceRef.CPU())

    with Graph("reshape", input_types=[input_type]) as graph:
        (x,) = graph.inputs
        n_patches, _ = x.tensor.shape
        x = x.tensor.rebind([(n_patches // 4) * 4, 4])
        graph.output(x.reshape([n_patches // 4, 4, 4]))

    model = session.load(graph)
    with pytest.raises(Exception):
        model.execute(input)


def test_rebind__no_new_parameter__divisible_by_4(
    session: InferenceSession,
) -> None:
    input = Tensor(DType.float32, [8, 4], device=CPU())
    input_type = TensorType(DType.float32, ["batch", 4], device=DeviceRef.CPU())

    with Graph("reshape", input_types=[input_type]) as graph:
        (x,) = graph.inputs
        n_patches, _ = x.tensor.shape
        x = x.tensor.rebind([(n_patches // 4) * 4, 4])
        graph.output(x.reshape([n_patches // 4, 4, 4]))

    model = session.load(graph)
    result = model.execute(input)[0]
    assert isinstance(result, Tensor)
    assert result.shape == (2, 4, 4)
