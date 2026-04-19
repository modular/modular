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

"""Tests for the Model.kernel_summaries property."""

from __future__ import annotations

import numpy as np
from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops


def test_kernel_summaries_returns_strings(session: InferenceSession) -> None:
    """kernel_summaries returns a list of strings after loading a graph."""
    with Graph(
        "simple_add",
        input_types=[
            TensorType(DType.float32, [4], device=DeviceRef.CPU()),
            TensorType(DType.float32, [4], device=DeviceRef.CPU()),
        ],
    ) as graph:
        a, b = graph.inputs[0].tensor, graph.inputs[1].tensor
        graph.output(a + b)

    model = session.load(graph)
    summaries = model.kernel_summaries
    assert isinstance(summaries, list)
    assert len(summaries) > 0
    assert all(isinstance(s, str) for s in summaries)
    combined = " ".join(summaries)
    assert "add" in combined.lower(), (
        f"Expected 'add' in kernel summaries for add graph, got: {summaries}"
    )


def test_kernel_summaries_correctness(session: InferenceSession) -> None:
    """kernel_summaries content is consistent with model execution."""
    with Graph(
        "matmul_relu",
        input_types=[
            TensorType(DType.float32, [2, 4], device=DeviceRef.CPU()),
            TensorType(DType.float32, [4, 3], device=DeviceRef.CPU()),
        ],
    ) as graph:
        a, b = graph.inputs[0].tensor, graph.inputs[1].tensor
        graph.output(ops.relu(a @ b))

    model = session.load(graph)
    summaries = model.kernel_summaries

    # The model should compile and have kernel summaries.
    assert len(summaries) > 0
    combined = " ".join(summaries).lower()
    assert "matmul" in combined or "dot" in combined, (
        f"Expected matmul-related kernel in summaries, got: {summaries}"
    )

    # Verify the model still executes correctly.
    input_a = np.ones((2, 4), dtype=np.float32)
    input_b = np.ones((4, 3), dtype=np.float32)
    result = model.execute(
        Buffer.from_numpy(input_a).to(model.input_devices[0]),
        Buffer.from_numpy(input_b).to(model.input_devices[1]),
    )
    assert result[0].to_numpy().shape == (2, 3)
