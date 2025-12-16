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


def test_reshape__minus_one__not_divisible_raises(
    session: InferenceSession,
) -> None:
    """Test that -1 preserves runtime divisibility checks with symbolic dims."""
    # batch=7, 7*4 elements cannot reshape to [-1, 4, 4]
    input = Tensor(DType.float32, [7, 4], device=CPU())
    input_type = TensorType(DType.float32, ["batch", 4], device=DeviceRef.CPU())
    with Graph("reshape_minus_one_bad", input_types=[input_type]) as graph:
        (x,) = graph.inputs
        batch, _ = x.tensor.shape
        # Rebind to assert batch is divisible by 4.
        x_rebound = x.tensor.rebind([(batch // 4) * 4, 4])
        # Use -1 in reshape; it should compute to batch // 4.
        y = x_rebound.reshape([-1, 4, 4])
        graph.output(y)
    model = session.load(graph)
    with pytest.raises(Exception):
        # At runtime with batch=7: rebind asserts 7 == (7//4)*4 = 4, which fails.
        model.execute(input)


def test_reshape__minus_one__static_divisible(
    session: InferenceSession,
) -> None:
    """Test that -1 works with static divisible shapes."""
    # 8*4 elements reshape to [-1, 4, 4] => 2,4,4
    input = Tensor(DType.float32, [8, 4], device=CPU())
    input_type = TensorType(DType.float32, [8, 4], device=DeviceRef.CPU())
    with Graph("reshape_minus_one_ok", input_types=[input_type]) as graph:
        (x,) = graph.inputs
        y = x.tensor.reshape([-1, 4, 4])
        graph.output(y)
    model = session.load(graph)
    result = model.execute(input)[0]
    assert isinstance(result, Tensor)
    assert result.shape == (2, 4, 4)
