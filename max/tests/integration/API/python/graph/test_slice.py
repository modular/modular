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

from collections.abc import Sequence

import numpy as np
import pytest
from max.driver import Tensor, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, StaticDim, TensorType, TensorValue
from test_common.graph_utils import are_all_tensors_sequence

device_ref = DeviceRef.GPU() if accelerator_count() > 0 else DeviceRef.CPU()

PARAMS = [
    # x[1:]
    (
        TensorType(DType.float32, shape=["dim0"], device=device_ref),
        (slice(1, None),),
    ),
    (
        TensorType(DType.float32, shape=["dim0", "dim1"], device=device_ref),
        (slice(1, None),),
    ),
    # x[:-1]
    (
        TensorType(DType.float32, shape=["dim0"], device=device_ref),
        (slice(None, -1)),
    ),
    # x[-1:]
    (
        TensorType(DType.float32, shape=["dim0"], device=device_ref),
        (slice(-1, None)),
    ),
    # x[::2]
    (
        TensorType(DType.float32, shape=["dim0"], device=device_ref),
        (slice(None, None, 2),),
    ),
    # x[::-1]
    # TODO(AIPIPE-109): allow negative step after improving rmo.slice.
    # (TensorType(DType.float32, shape=["dim0"]), (slice(None, None, -1),)),
    # x[:, None, :]
    (
        TensorType(DType.float32, shape=["dim0", "dim1"], device=device_ref),
        (slice(None), None, slice(None)),
    ),
    # x[..., None]
    (
        TensorType(DType.float32, shape=["dim0", "dim1"], device=device_ref),
        (Ellipsis, None),
    ),
    # x[..., 1]
    (
        TensorType(
            DType.float32,
            shape=["dim0", "dim1", "dim2"],
            device=device_ref,
        ),
        (Ellipsis, 1),
    ),
    # x[Ellipsis, 1:]
    (
        TensorType(DType.float32, shape=["dim0", "dim1"], device=device_ref),
        (Ellipsis, slice(1, None)),
    ),
    # x[1, ..., ::-1]
    # TODO(AIPIPE-109): allow negative step after improving rmo.slice.
    # (
    #     TensorType(DType.float32, shape=["dim0", "dim1", "dim2"]),
    #     (1, Ellipsis, slice(None, None, -1)),
    # ),
    # x[:, -1]
    (
        TensorType(
            DType.float32,
            shape=["dim0", "dim1", "dim2"],
            device=device_ref,
        ),
        (slice(None), -1),
    ),
]


@pytest.fixture(scope="module")
def input_arrays() -> list[np.ndarray]:
    inputs = []
    for tensor_type, _ in PARAMS:
        input_shape = [
            idx * 3 + 7 if not isinstance(dim, StaticDim) else dim.dim
            for idx, dim in enumerate(tensor_type.shape)
        ]
        input_array = np.random.randn(*input_shape).astype(
            tensor_type.dtype.to_numpy()
        )
        inputs.append(input_array)
    return inputs


@pytest.fixture(scope="module")
def graph_outputs(
    session: InferenceSession, input_arrays: list[np.ndarray]
) -> Sequence[Tensor]:
    with Graph(
        "slice",
        input_types=[t[0] for t in PARAMS],
        output_types=[t[0] for t in PARAMS],
    ) as graph:
        outputs = []
        for i, (_, indices) in enumerate(PARAMS):
            x: TensorValue = graph.inputs[i].tensor
            outputs.append(x[indices])
        graph.output(*outputs)

    # Execute the graph.
    model = session.load(graph)
    results = model.execute(
        *[Tensor.from_numpy(a).to(model.input_devices[0]) for a in input_arrays]
    )
    assert are_all_tensors_sequence(results)
    return results


@pytest.mark.parametrize(("tensor_type", "indices"), PARAMS)
def test_slice_numpy(
    graph_outputs: list[Tensor],
    input_arrays: list[np.ndarray],
    tensor_type: TensorType,
    indices: tuple[slice],
) -> None:
    """Tests end-to-end slice lowering and execution."""
    i = PARAMS.index((tensor_type, indices))
    actual = graph_outputs[i].to_numpy()

    # Verify that the max.graph slicing matches NumPy.
    expected = input_arrays[i][indices]
    np.testing.assert_array_equal(actual, expected)
