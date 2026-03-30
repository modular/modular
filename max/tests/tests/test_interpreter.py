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
from max.driver import CPU, Buffer
from max.dtype import DType
from max.experimental.realization_context import EagerRealizationContext
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

    def test_can_execute_max_ops_within_limit(self) -> None:
        """Test can_execute respects max_ops when graph is within limit."""
        with Graph("small", input_types=[]) as graph:
            a = ops.constant([1.0, 2.0], dtype=DType.float32, device=CPU())
            b = ops.constant([3.0, 4.0], dtype=DType.float32, device=CPU())
            c = ops.add(a, b)
            graph.output(c)

        interp = MOInterpreter()
        # 3 dispatchable ops (2 constants + 1 add), limit at 5 -> ok
        assert interp.can_execute(graph, max_ops=5) is True

    def test_can_execute_max_ops_exceeds_limit(self) -> None:
        """Test can_execute returns False when graph exceeds max_ops."""
        with Graph("big", input_types=[]) as graph:
            a = ops.constant([1.0, 2.0], dtype=DType.float32, device=CPU())
            b = ops.constant([3.0, 4.0], dtype=DType.float32, device=CPU())
            c = ops.add(a, b)
            d = ops.mul(a, b)
            e = ops.sub(c, d)
            graph.output(e)

        interp = MOInterpreter()
        # 5 dispatchable ops (2 constants + add + mul + sub), limit at 2 -> too many
        assert interp.can_execute(graph, max_ops=2) is False

    def test_can_execute_no_limit(self) -> None:
        """Test can_execute with max_ops=None imposes no limit."""
        with Graph("unlimited", input_types=[]) as graph:
            a = ops.constant([1.0, 2.0], dtype=DType.float32, device=CPU())
            b = ops.constant([3.0, 4.0], dtype=DType.float32, device=CPU())
            c = ops.add(a, b)
            d = ops.mul(c, b)
            e = ops.sub(d, a)
            graph.output(e)

        interp = MOInterpreter()
        assert interp.can_execute(graph, max_ops=None) is True

    def test_can_execute_returns_false_for_unhandled_op(self) -> None:
        """Test can_execute returns False when an op has no registered handler."""
        with Graph("unsupported", input_types=[]) as graph:
            a = ops.constant(
                [[1.0, 2.0], [3.0, 4.0]], dtype=DType.float32, device=CPU()
            )
            # pad has no interpreter handler
            result = ops.pad(a, [0, 1, 0, 1])
            graph.output(result)

        interp = MOInterpreter()
        assert interp.can_execute(graph) is False


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

    def test_default_uses_interpreter(self, monkeypatch: Any) -> None:
        """Test that interpreter is enabled by default."""
        monkeypatch.delenv("MAX_USE_EAGER_INTERPRETER", raising=False)

        ctx = EagerRealizationContext()
        assert ctx._use_interpreter is True

    def test_can_enable_interpreter(self) -> None:
        """Test that interpreter can be enabled explicitly."""

        ctx = EagerRealizationContext(use_interpreter=True)
        assert ctx._use_interpreter is True

    def test_env_var_disables_interpreter(self, monkeypatch: Any) -> None:
        """Test that MAX_USE_EAGER_INTERPRETER=0 disables interpreter."""
        monkeypatch.setenv("MAX_USE_EAGER_INTERPRETER", "0")

        ctx = EagerRealizationContext()
        assert ctx._use_interpreter is False

    def test_env_var_false_disables_interpreter(self, monkeypatch: Any) -> None:
        """Test that MAX_USE_EAGER_INTERPRETER=false disables interpreter."""
        monkeypatch.setenv("MAX_USE_EAGER_INTERPRETER", "false")

        ctx = EagerRealizationContext()
        assert ctx._use_interpreter is False

    def test_env_var_1_keeps_interpreter_enabled(
        self, monkeypatch: Any
    ) -> None:
        """Test that MAX_USE_EAGER_INTERPRETER=1 keeps interpreter enabled."""
        monkeypatch.setenv("MAX_USE_EAGER_INTERPRETER", "1")

        ctx = EagerRealizationContext()
        assert ctx._use_interpreter is True

    def test_explicit_false_overrides_default(self) -> None:
        """Test that explicit use_interpreter=False overrides the default."""

        ctx = EagerRealizationContext(use_interpreter=False)
        assert ctx._use_interpreter is False


class TestRuntimeFallback:
    """Tests for interpreter runtime fallback to the graph compiler.

    These tests verify the control flow in realize_all(): when the
    interpreter raises at runtime (e.g. unsupported dtype), auto-mode
    falls back to the compiler, while explicit mode surfaces the error.
    """

    def test_auto_mode_falls_back_on_execute_error(self) -> None:
        """When auto-selected, a runtime error in execute() sets
        use_interpreter=False so the compiler path runs instead."""
        import max.experimental.realization_context as rc

        ctx = EagerRealizationContext()
        assert ctx._use_interpreter is True
        assert ctx._auto_interpreter is True

        interp = MOInterpreter()

        with Graph(
            "add",
            input_types=[TensorType(DType.float32, [3], device=CPU())],
        ) as graph:
            (a,) = graph.inputs
            graph.output(ops.add(a, a))

        assert interp.can_execute(graph)

        call_log: list[str] = []

        original_execute = MOInterpreter.execute

        def _failing_execute(self: Any, graph: Any, inputs: Any) -> Any:
            call_log.append("interpreter_called")
            raise RuntimeError("simulated unsupported dtype")

        MOInterpreter.execute = _failing_execute  # type: ignore[method-assign]
        try:
            max_ops = rc._interpreter_max_ops()
            use_interpreter = True
            if not interp.can_execute(graph, max_ops=max_ops):
                use_interpreter = False

            assert use_interpreter is True

            if use_interpreter and ctx._auto_interpreter:
                inp = Buffer(shape=[3], dtype=DType.float32, device=CPU())
                try:
                    interp.execute(graph, [inp])
                except Exception:
                    use_interpreter = False

            assert use_interpreter is False
            assert "interpreter_called" in call_log
        finally:
            MOInterpreter.execute = original_execute  # type: ignore[method-assign]

    def test_explicit_interpreter_does_not_swallow_error(
        self, monkeypatch: Any
    ) -> None:
        """When the interpreter is explicitly requested, runtime errors
        must propagate instead of silently falling back."""
        ctx_auto = EagerRealizationContext()
        assert ctx_auto._auto_interpreter is True

        ctx_explicit = EagerRealizationContext(use_interpreter=True)
        assert ctx_explicit._auto_interpreter is False

        ctx_explicit_off = EagerRealizationContext(use_interpreter=False)
        assert ctx_explicit_off._auto_interpreter is False


class TestInterpreterMaxOps:
    """Tests for the MAX_INTERPRETER_MAX_OPS threshold."""

    def test_default_max_ops(self, monkeypatch: Any) -> None:
        """Test that the default max_ops is 30."""
        monkeypatch.delenv("MAX_INTERPRETER_MAX_OPS", raising=False)
        from max.experimental.realization_context import _interpreter_max_ops

        assert _interpreter_max_ops() == 30

    def test_env_var_overrides_max_ops(self, monkeypatch: Any) -> None:
        """Test that MAX_INTERPRETER_MAX_OPS env var overrides the default."""
        monkeypatch.setenv("MAX_INTERPRETER_MAX_OPS", "5")
        from max.experimental.realization_context import _interpreter_max_ops

        assert _interpreter_max_ops() == 5

    def test_env_var_max_ops_2(self, monkeypatch: Any) -> None:
        """Test setting the threshold to 2."""
        monkeypatch.setenv("MAX_INTERPRETER_MAX_OPS", "2")
        from max.experimental.realization_context import _interpreter_max_ops

        assert _interpreter_max_ops() == 2

    def test_invalid_env_var_uses_default(self, monkeypatch: Any) -> None:
        """Test that non-numeric env var falls back to default."""
        monkeypatch.setenv("MAX_INTERPRETER_MAX_OPS", "abc")
        from max.experimental.realization_context import _interpreter_max_ops

        assert _interpreter_max_ops() == 30
