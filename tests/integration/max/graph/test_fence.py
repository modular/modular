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
"""Integration tests for .fence end-to-end compilation.

These tests exercise full graph compilation (MO → MGP) and execution with
`ops.fence` present in the IR.
"""

from __future__ import annotations

import numpy as np
from max.driver import Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops


def _input_type() -> TensorType:
    return TensorType(DType.float32, [2], device=DeviceRef.CPU())


def test_fence_is_staged_in_ir() -> None:
    """Staging a fence should materialize a `mo.fence` op in the graph IR."""
    with Graph(
        "fence_ir_staging", input_types=(_input_type(), _input_type())
    ) as graph:
        x, y = graph.inputs
        z = ops.add(x, y)
        _ = ops.fence(z)
        graph.output(z)

    ir_text = str(graph._module)
    assert "mo.fence" in ir_text


def test_fence_full_compile_and_execute(session: InferenceSession) -> None:
    """Full compilation with a fence in the IR should succeed and execute.

    This validates that the MO→MGP pipeline tolerates a `mo.fence` op in the
    module (either by lowering or canonicalization) without changing behavior.
    """
    with Graph(
        "fence_compile", input_types=(_input_type(), _input_type())
    ) as graph:
        x, y = graph.inputs
        z = ops.add(x, y)
        result = ops.fence(z)
        graph.output(*result)

    # Compile the graph and execute on CPU.
    compiled = session.load(graph)
    a_np = np.array([1.0, 2.0], dtype=np.float32)
    b_np = np.array([3.0, 4.0], dtype=np.float32)

    a = Tensor.from_numpy(a_np).to(compiled.input_devices[0])
    b = Tensor.from_numpy(b_np).to(compiled.input_devices[1])

    out = compiled.execute(a, b)
    assert isinstance(out[0], Tensor)
    np.testing.assert_allclose(out[0].to_numpy(), a_np + b_np)
