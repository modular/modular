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
"""Tests for reduce-add operation."""

import numpy as np
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops


def test_reduce_add(session: InferenceSession) -> None:
    # Use symbolic dimensions for the graph
    batch = "batch"
    dim = "dim"
    input_type = TensorType(DType.float32, [batch, dim], device=DeviceRef.CPU())

    with Graph("reduce_add", input_types=[input_type]) as graph:
        x = graph.inputs[0].tensor
        # Reduce along axis 1 (the 'dim' dimension)
        reduced = ops.sum(x, axis=1)
        # Squeeze to remove the size-1 dimension
        result = ops.squeeze(reduced, axis=1)
        graph.output(result)

    model = session.load(graph)

    batch_sizes = st.integers(min_value=1, max_value=16)
    reduction_dims = st.integers(min_value=1, max_value=64)

    @settings(max_examples=100, deadline=None)
    @given(batch_size=batch_sizes, reduction_dim=reduction_dims)
    def check_reduce_add(batch_size: int, reduction_dim: int) -> None:
        # Generate random input data
        input_data = torch.randn(
            (batch_size, reduction_dim), dtype=torch.float32
        )

        # Run through MAX
        max_input = Buffer.from_dlpack(input_data).to(model.input_devices[0])
        max_result = model(max_input)[0]
        assert isinstance(max_result, Buffer)
        max_result_np = max_result.to_numpy()

        expected = torch.sum(input_data, dim=1).numpy()

        # A float32 sum of up to 64 N(0, 1) terms rounds by about 1e-5 in
        # any add order, and a row sum can land near zero where a relative
        # tolerance alone does not cover that.
        np.testing.assert_allclose(
            max_result_np,
            expected,
            rtol=1e-5,
            atol=1e-4,
        )

    check_reduce_add()
