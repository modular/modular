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
import pytest
from max.driver import Tensor, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from max.graph.quantization import QuantizationEncoding


@pytest.mark.skipif(
    accelerator_count() > 0,
    reason="Quantization only supported on cpu currently",
)
def test_qmatmul(session: InferenceSession) -> None:
    graph = Graph(
        "qmatmul",
        input_types=[
            TensorType(DType.float32, (5, 32), device=DeviceRef.CPU()),
            TensorType(DType.uint8, (32, 18), device=DeviceRef.CPU()),
        ],
        output_types=[
            TensorType(DType.float32, (5, 32), device=DeviceRef.CPU())
        ],
    )

    with graph:
        graph.output(
            ops.qmatmul(
                QuantizationEncoding.Q4_0,
                None,
                *[x.tensor for x in graph.inputs],
            )
        )

    compiled = session.load(graph)
    # This is a pretty bad test -- the inputs and outputs here are all zeroes.
    # But it's better than nothing -- at least we don't crash.  Also qmatmul
    # does not validate its tensor shapes all (except that the second input's
    # first dimension is a multiple of 32) so even if this were wrong we would
    # not be able to tell.
    generated = compiled.execute(
        np.zeros((5, 32), dtype="float32"),
        np.zeros((32, 18), dtype="uint8"),
    )[0]
    assert isinstance(generated, Tensor)
    expected = np.zeros((5, 32))
    np.testing.assert_equal(generated.to_numpy(), expected)


@pytest.mark.skipif(
    True,  # TODO(KERN-1881): Re-enable for CPU.
    reason="Quantization only supported on cpu currently",
)
def test_dequantize(session: InferenceSession) -> None:
    graph = Graph(
        "dequantize",
        input_types=[TensorType(DType.uint8, (1, 18), device=DeviceRef.CPU())],
        output_types=[
            TensorType(DType.float32, (1, 32), device=DeviceRef.CPU())
        ],
    )

    with graph:
        graph.output(
            ops.dequantize(
                QuantizationEncoding.Q4_0, *[x.tensor for x in graph.inputs]
            )
        )

    compiled = session.load(graph)
    # TODO: This is more of a smoke test than anything; we should really add a
    # test that uses some non-zero inputs and outputs (MSDK-820).
    generated = compiled.execute(np.zeros((1, 18), dtype="uint8"))[0]
    assert isinstance(generated, Tensor)
    expected = np.zeros((1, 32))
    np.testing.assert_equal(generated.to_numpy(), expected)
