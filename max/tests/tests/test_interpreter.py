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
"""Tests for MO graph interpreter module."""

from typing import Any

import numpy as np
import pytest
from max._interpreter import MOInterpreter
from max._interpreter_ops import _MO_OP_HANDLERS, register_op_handler
from max._realization_context import EagerRealizationContext
from max.driver import CPU, Buffer
from max.dtype import DType
from max.graph import Graph, TensorType, ops


class TestMOInterpreter:
    """Tests for MOInterpreter class."""

    def test_init(self) -> None:
        """Test interpreter initialization."""
        interp = MOInterpreter()
        assert interp is not None

    def test_validate_inputs_wrong_count(self) -> None:
        """Test that validate_inputs catches input count mismatch."""
        interp = MOInterpreter()

        class MockGraph:
            @property
            def inputs(self) -> list[int]:
                return [1, 2, 3]  # 3 inputs expected

        graph: Any = MockGraph()
        inputs: Any = [object()]
        with pytest.raises(ValueError, match="Expected 3 inputs, got 1"):
            interp._validate_inputs(graph, inputs)


class TestCanExecute:
    """Tests for MOInterpreter.can_execute() pre-flight check."""

    def test_can_execute_supported_graph(self) -> None:
        """Test can_execute returns True for a graph with only supported ops."""
        with Graph("supported", input_types=[]) as graph:
            a = ops.constant([1.0, 2.0], dtype=DType.float32, device=CPU())
            b = ops.constant([3.0, 4.0], dtype=DType.float32, device=CPU())
            c = ops.add(a, b)
            graph.output(c)

        interp = MOInterpreter()
        assert interp.can_execute(graph) is True

    def test_can_execute_returns_false_for_compilation_required_op(
        self,
    ) -> None:
        """Test can_execute returns False when graph contains a
        compilation-required op (e.g. mo.CustomOp)."""
        interp = MOInterpreter()

        # Temporarily add ConstantOp to _COMPILATION_REQUIRED_OP_NAMES to
        # simulate a compilation-required op without needing a real custom
        # kernel.  ConstantOp is used because it's always present in simple
        # graphs and its name is stable.
        orig = MOInterpreter._COMPILATION_REQUIRED_OP_NAMES
        MOInterpreter._COMPILATION_REQUIRED_OP_NAMES = ("ConstantOp",)
        try:
            with Graph("custom", input_types=[]) as graph:
                c = ops.constant([1.0, 1.0], dtype=DType.float32, device=CPU())
                graph.output(c)

            assert interp.can_execute(graph) is False
        finally:
            MOInterpreter._COMPILATION_REQUIRED_OP_NAMES = orig


class TestOpHandlerRegistry:
    """Tests for op handler registration."""

    def test_register_handler(self) -> None:
        """Test handler registration decorator."""

        class MockOpType:
            pass

        op_type: Any = MockOpType

        @register_op_handler(op_type)
        def test_handler(op: Any, inputs: Any) -> list[Any]:
            return []

        assert op_type in _MO_OP_HANDLERS
        assert _MO_OP_HANDLERS[op_type] is test_handler

        # Cleanup
        del _MO_OP_HANDLERS[op_type]

    def test_register_handler_overwrites(self) -> None:
        """Test that registering same op type overwrites previous handler."""

        class MockOpType:
            pass

        op_type: Any = MockOpType

        @register_op_handler(op_type)
        def handler1(op: Any, inputs: Any) -> list[Any]:
            return [1]

        @register_op_handler(op_type)
        def handler2(op: Any, inputs: Any) -> list[Any]:
            return [2]

        assert _MO_OP_HANDLERS[op_type] is handler2

        # Cleanup
        del _MO_OP_HANDLERS[op_type]


class TestGraphExecution:
    """Integration tests that build MO graphs and run them through the interpreter."""

    def test_constant_only_graph(self) -> None:
        """Test executing a graph with only a constant output."""
        # Create a graph that just outputs a constant
        with Graph("constant_graph", input_types=[]) as graph:
            c = ops.constant([1.0, 2.0, 3.0], dtype=DType.float32, device=CPU())
            graph.output(c)

        # Execute through interpreter
        interp = MOInterpreter()
        outputs = interp.execute(graph, [])

        # Verify output
        assert len(outputs) == 1
        result = outputs[0]
        assert isinstance(result, Buffer)

        expected = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_graph_with_input(self) -> None:
        """Test executing a graph with an input tensor."""
        input_type = TensorType(DType.float32, [2, 3], CPU())
        with Graph("input_graph", input_types=[input_type]) as graph:
            x = graph.inputs[0]
            c = ops.constant(
                [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                dtype=DType.float32,
                device=CPU(),
            )
            y = ops.add(x, c)
            graph.output(y)

        # Create input buffer
        input_np = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32
        )
        input_buffer = Buffer.from_numpy(input_np)

        interp = MOInterpreter()
        outputs = interp.execute(graph, [input_buffer])

        assert len(outputs) == 1
        result = outputs[0]
        assert isinstance(result, Buffer)

        expected = np.array(
            [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]], dtype=np.float32
        )
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_chained_operations(self) -> None:
        """Test a graph with multiple chained operations."""
        with Graph("chained_ops", input_types=[]) as graph:
            a = ops.constant([1.0, 2.0, 3.0], dtype=DType.float32, device=CPU())
            b = ops.constant([2.0, 2.0, 2.0], dtype=DType.float32, device=CPU())
            # (a + b) * 3 - 1
            c = ops.add(a, b)  # [3, 4, 5]
            three = ops.constant(
                [3.0, 3.0, 3.0], dtype=DType.float32, device=CPU()
            )
            d = ops.mul(c, three)  # [9, 12, 15]
            one = ops.constant(
                [1.0, 1.0, 1.0], dtype=DType.float32, device=CPU()
            )
            e = ops.sub(d, one)  # [8, 11, 14]
            graph.output(e)

        interp = MOInterpreter()
        outputs = interp.execute(graph, [])

        assert len(outputs) == 1
        result = outputs[0]
        assert isinstance(result, Buffer)

        expected = np.array([8.0, 11.0, 14.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_multiple_outputs(self) -> None:
        """Test a graph with multiple outputs."""
        with Graph("multi_output", input_types=[]) as graph:
            a = ops.constant([1.0, 2.0], dtype=DType.float32, device=CPU())
            b = ops.constant([3.0, 4.0], dtype=DType.float32, device=CPU())
            sum_ab = ops.add(a, b)
            prod_ab = ops.mul(a, b)
            graph.output(sum_ab, prod_ab)

        interp = MOInterpreter()
        outputs = interp.execute(graph, [])

        assert len(outputs) == 2
        sum_result = outputs[0]
        assert isinstance(sum_result, Buffer)
        prod_result = outputs[1]
        assert isinstance(prod_result, Buffer)

        np.testing.assert_array_almost_equal(
            sum_result.to_numpy(), np.array([4.0, 6.0], dtype=np.float32)
        )
        np.testing.assert_array_almost_equal(
            prod_result.to_numpy(), np.array([3.0, 8.0], dtype=np.float32)
        )


class TestRealizationContextIntegration:
    """Tests for interpreter integration with EagerRealizationContext."""

    def test_default_does_not_use_interpreter(self, monkeypatch: Any) -> None:
        """Test that interpreter is disabled by default when env var not set."""

        # Ensure env var is not set
        monkeypatch.delenv("MAX_USE_EAGER_INTERPRETER", raising=False)

        # Just test the attribute without creating operations
        # to avoid cleanup issues with graph state
        ctx = EagerRealizationContext()
        assert ctx._use_interpreter is False
        # Note: We don't enter/exit the context here since we're just
        # testing the constructor attribute. The context holds a Graph
        # that has its own lifecycle.

    def test_can_enable_interpreter(self) -> None:
        """Test that interpreter can be enabled."""

        ctx = EagerRealizationContext(use_interpreter=True)
        assert ctx._use_interpreter is True

    def test_env_var_enables_interpreter(self, monkeypatch: Any) -> None:
        """Test that MAX_USE_EAGER_INTERPRETER=1 enables interpreter by default."""
        monkeypatch.setenv("MAX_USE_EAGER_INTERPRETER", "1")

        ctx = EagerRealizationContext()
        assert ctx._use_interpreter is True

    def test_env_var_true_enables_interpreter(self, monkeypatch: Any) -> None:
        """Test that MAX_USE_EAGER_INTERPRETER=true enables interpreter by default."""
        monkeypatch.setenv("MAX_USE_EAGER_INTERPRETER", "true")

        ctx = EagerRealizationContext()
        assert ctx._use_interpreter is True

    def test_explicit_false_overrides_env_var(self, monkeypatch: Any) -> None:
        """Test that explicit use_interpreter=False overrides env var."""
        monkeypatch.setenv("MAX_USE_EAGER_INTERPRETER", "1")

        ctx = EagerRealizationContext(use_interpreter=False)
        assert ctx._use_interpreter is False
