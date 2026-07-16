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
"""Tests for integer indexing on a tensor (`x[i]`).

Integer indexing lowers to slice + reshape that squeezes a leading 1-dim. A
graph compiler pass that swims reshapes past slices used to produce malformed
IR for this pattern; these tests guard the compile + execution path.
"""

import numpy as np
import torch
from max import engine
from max.driver import CPU, Buffer
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, TensorValue


def test_int_index_returns_first_row() -> None:
    """End-to-end: `x[0]` on shape (2,3) should yield the first row."""
    input_type = TensorType(
        dtype=DType.int32, shape=(2, 3), device=DeviceRef.CPU()
    )
    with Graph("int_index_first_dim", input_types=(input_type,)) as graph:
        x = graph.inputs[0]
        assert isinstance(x, TensorValue)
        graph.output(x[0])

    session = engine.InferenceSession(devices=[CPU()])
    model = session.load(graph)

    arr = torch.arange(6, dtype=torch.int32).reshape(2, 3)
    actual = model(arr)[0]
    assert isinstance(actual, Buffer)
    np.testing.assert_array_equal(actual.to_numpy(), arr[0].cpu().numpy())
