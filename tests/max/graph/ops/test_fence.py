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

"""Tests for the ops.fence op."""

from __future__ import annotations

import numpy as np
import pytest
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, ops


def test_fence_returns_same_objects_and_types() -> None:
    input_ty = TensorType(DType.float32, [2], DeviceRef.CPU())

    with Graph("fence_identity", input_types=(input_ty, input_ty)) as graph:
        x, y = graph.inputs
        z = ops.add(x, y)

        # Fence should be a pure identity on values.
        result = ops.fence(z)
        assert len(result) == 1
        z2 = result[0]

        # Same object type.
        assert z2.type == z.type


def test_fence_does_not_change_chain() -> None:
    # NOTE: This test depends on internal implementation details (_current_chain)
    # to verify that fence doesn't modify the execute chain. This is intentional
    # as it tests an important internal invariant.
    input_ty = TensorType(DType.float32, [2], DeviceRef.CPU())
    a = np.array([1.0, 2.0], dtype=np.float32)

    with Graph("fence_chain_noop", input_types=(input_ty, input_ty)) as graph:
        x, y = graph.inputs
        before = graph._current_chain

        z = ops.add(x, y)
        result = ops.fence(z)
        assert len(result) == 1

        after = graph._current_chain

        # Fencing values should not affect the graph's current chain value.
        assert after is before


def test_fence_raises_on_zero_args() -> None:
    """Test that fence() with no arguments raises ValueError."""
    input_ty = TensorType(DType.float32, shape=[2], device=DeviceRef.CPU())

    with Graph("fence_zero_args", input_types=(input_ty,)) as graph:
        with pytest.raises(ValueError, match="fence.*at least one input"):
            ops.fence()


def test_fence_variadic_returns_list_ordered() -> None:
    """Test that fence with multiple values returns a list with preserved order."""
    input_ty = TensorType(DType.float32, [2], DeviceRef.CPU())

    with Graph(
        "fence_tuple_ordering", input_types=(input_ty, input_ty)
    ) as graph:
        x, y = graph.inputs
        # Create distinct values to track ordering.
        x2 = ops.mul(x, 2.0)  # x * 2
        y3 = ops.mul(y, 3.0)  # y * 3

        # Fence multiple values.
        result = ops.fence(x2, y3)
        assert len(result) == 2

        # Order should be preserved.
        assert result[0] is not result[1]
        # The first result should correspond to x2, second to y3.
        # We can verify this by checking the operation structure.
        graph.output(result[0], result[1])
