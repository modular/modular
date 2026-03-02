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
"""Tests for max.nn.sampling"""

from typing import cast

import numpy as np
import pytest
from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.nn.sampling import MinPSampler


# NOTE THAT ONLY RANK 2 TENSORS
# ARE CURRENTLY SUPPORTED
@pytest.mark.parametrize(
    ("input_shape", "min_p", "temperature"),
    [
        ((5, 5), 0.0, 0.0),
        ((4, 3), 0.01, 0.1),
        ((2, 3), 0.05, 0.2),
        ((6, 2), 0.08, 0.25),
        ((3, 5), 0.1, 0.3),
    ],
)
def test_min_p_execution(
    session: InferenceSession,
    input_shape: tuple[int, ...],
    min_p: float,
    temperature: float,
) -> None:
    """Tests end-to-end MinPSampling lowering and execution."""
    with Graph(
        "min_p_test",
        input_types=[
            TensorType(DType.float32, shape=input_shape, device=DeviceRef.CPU())
        ],
    ) as graph:
        inputs, *_ = graph.inputs
        sampler = MinPSampler(DType.float32, input_shape, temperature)
        out = sampler(inputs.tensor, min_p)
        graph.output(out)

        # Compile and execute the graph.
        model = session.load(graph)

        # Generate random input data.
        np_input = np.random.randn(*input_shape).astype(np.float32)

        # Execute MAX model.
        min_p_output, *_ = model.execute(np_input)
        cast(Buffer, min_p_output).to_numpy()


def test_min_p_known_inputs_outputs(session: InferenceSession) -> None:
    """Tests MinP sampling with known inputs and expected behavior."""
    batch_size = 5
    vocab_size = 4
    input_shape = (batch_size, vocab_size)
    temperature = 1.0  # Set to 1 for predictable softmax behavior

    with Graph(
        "min_p_known_test",
        input_types=[
            TensorType(
                DType.float32, shape=input_shape, device=DeviceRef.CPU()
            ),
            TensorType(
                DType.float32, shape=(batch_size,), device=DeviceRef.CPU()
            ),
        ],
    ) as graph:
        inputs, min_p_tensor, *_ = graph.inputs
        sampler = MinPSampler(DType.float32, input_shape, temperature)
        out = sampler(inputs.tensor, min_p_tensor.tensor)
        graph.output(out)
        model = session.load(graph)

        np_input = np.array(
            [
                [-1.0, 1.0, 2.0, 0.0],
                [0.0, 0.0, 0.0, 3.0],
                [1.0, 1.0, 1.1, 1.0],
                [0.0, 2.0, 4.0, 1.0],
                [0.0, 0.0, 2.0, 1.0],
            ],
            dtype=np.float32,
        )

        min_p_array = np.array([0.1, 0.1, 0.26, 0.1, 0.1], dtype=np.float32)
        min_p_output, *_ = model.execute(np_input, min_p_array)
        result = cast(Buffer, min_p_output).to_numpy()
        assert (
            result[2, 0] == 2
        )  # 1.1 is the only logit that is greater than 0.26 after softmax
