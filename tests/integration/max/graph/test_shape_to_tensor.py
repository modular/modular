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


import numpy as np
from max.driver import Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, TensorValue


def test_shape_to_tensor_static(session: InferenceSession) -> None:
    input_type = TensorType(
        dtype=DType.float32, shape=[2, 4], device=DeviceRef.CPU()
    )
    with Graph("input_shape", input_types=(input_type,)) as graph:
        shape = graph.inputs[0].tensor.shape
        graph.output(TensorValue(shape))

    compiled = session.load(graph)

    x = np.ones((2, 4)).astype(np.float32)
    output = compiled.execute(
        Tensor.from_numpy(x).to(compiled.input_devices[0])
    )
    assert isinstance(output[0], Tensor)

    np.testing.assert_equal(output[0].to_numpy(), np.array([2, 4]))


def test_shape_to_tensor_dynamic(session: InferenceSession) -> None:
    input_type = TensorType(
        dtype=DType.float32, shape=["batch", "channels"], device=DeviceRef.CPU()
    )
    with Graph("input_shape", input_types=(input_type,)) as graph:
        shape = graph.inputs[0].tensor.shape
        graph.output(TensorValue(shape))

    compiled = session.load(graph)

    x = np.ones((7, 3)).astype(np.float32)
    output = compiled.execute(
        Tensor.from_numpy(x).to(compiled.input_devices[0])
    )
    assert isinstance(output[0], Tensor)

    np.testing.assert_equal(output[0].to_numpy(), np.array([7, 3]))


def test_shape_to_tensor_solo_dim(session: InferenceSession) -> None:
    input_type = TensorType(
        dtype=DType.float32, shape=["batch", "channels"], device=DeviceRef.CPU()
    )
    with Graph("input_shape", input_types=(input_type,)) as graph:
        shape = graph.inputs[0].tensor.shape
        graph.output(TensorValue(shape[1]))

    compiled = session.load(graph)

    x = np.ones((7, 3)).astype(np.float32)
    output = compiled.execute(
        Tensor.from_numpy(x).to(compiled.input_devices[0])
    )
    assert isinstance(output[0], Tensor)

    # Output is only a scalar
    assert output[0].shape == ()
    np.testing.assert_equal(output[0].to_numpy(), np.array([3]))
