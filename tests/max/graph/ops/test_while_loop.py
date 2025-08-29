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
"""Test while loop operations."""

from typing import NoReturn

from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, TensorValue, ops


def test_while_loop_basic() -> None:
    """Test basic while loop functionality."""
    with Graph(
        "while_loop_basic",
        input_types=[TensorType(DType.int32, [], device=DeviceRef.CPU())],
    ) as graph:
        x = graph.inputs[0]

        def pred(x: TensorValue) -> TensorValue:
            return x < 10

        def body(x: TensorValue) -> TensorValue:
            return x + 1

        results = ops.while_loop(x, pred, body)
        graph.output(results[0])

    # Verify MLIR contains while op and expected structure
    mlir_str = str(graph)
    assert "mo.while" in mlir_str
    assert " do " in mlir_str


def test_while_loop_multiple_args() -> None:
    """Test while loop with multiple arguments."""
    with Graph(
        "while_loop_multiple_args",
        input_types=[
            TensorType(DType.int32, [], device=DeviceRef.CPU()),
            TensorType(DType.int32, [], device=DeviceRef.CPU()),
        ],
    ) as graph:
        x, y = graph.inputs

        def pred(x: TensorValue, y: TensorValue) -> TensorValue:
            return x < 10 and y < 10

        def body(x: TensorValue, y: TensorValue) -> list[TensorValue]:
            return [x + 1, y + 1]

        results = ops.while_loop((x, y), pred, body)
        graph.output(results[0], results[1])

    # Verify MLIR contains while op with multiple args
    mlir_str = str(graph)
    assert "mo.while" in mlir_str
    assert " do " in mlir_str


def test_while_loop_empty_init() -> None:
    """Test while loop with empty initial values raises error."""
    with Graph("while_loop_empty_init", input_types=()) as graph:
        try:
            ops.while_loop([], lambda: True, lambda: [])
        except ValueError as e:
            assert "While loops must have at least one iteration value" in str(
                e
            )


def test_while_loop_type_check() -> None:
    """Test type checking in while loop."""
    with Graph(
        "while_loop_type_check",
        input_types=[TensorType(DType.int32, [], device=DeviceRef.CPU())],
    ) as graph:
        x = graph.inputs[0]

        def pred(x: TensorValue) -> TensorValue:
            return x < 10

        def body(x: TensorValue) -> TensorValue:
            # Return wrong type
            return ops.cast(x + 1, DType.float32)

        try:
            ops.while_loop(x, pred, body)
        except TypeError as e:
            assert "Results don't match expected types" in str(e)

        graph.output()

    graph._mlir_op.verify()


def test_while_loop_with_raising() -> None:
    with Graph(
        "while_loop_with_raising",
        input_types=[TensorType(DType.int32, [], device=DeviceRef.CPU())],
    ) as graph:
        x = graph.inputs[0]
        chain = graph._current_chain

        def pred(x: TensorValue) -> TensorValue:
            return x < 10

        def body(x: TensorValue) -> NoReturn:
            raise Exception("raising")

        try:
            ops.while_loop(x, pred, body)
        except Exception as e:
            assert "raising" in str(e)

        assert graph._current_chain == chain
        graph.output()
    graph._mlir_op.verify()


def test_while_loop_with_pred_block_chain_mutation() -> None:
    with Graph(
        "while_loop_with_pred_block_chain_mutation",
        input_types=[TensorType(DType.int32, [], device=DeviceRef.CPU())],
    ) as graph:
        x = graph.inputs[0]

        def pred(x: TensorValue) -> TensorValue:
            # print mutates the chain
            x.print()
            return x < 10

        def body(x: TensorValue) -> TensorValue:
            return x + 1

        try:
            ops.while_loop(x, pred, body)
        except Exception as e:
            assert "Chain mutation detected" in str(e)

        graph.output()
    graph._mlir_op.verify()
