# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, ops


def test_conditional_no_results() -> None:
    with Graph("conditional", input_types=()) as graph:
        cond = ops.constant(True, dtype=DType.bool, device=DeviceRef.CPU())

        def then_fn():
            ops.print("then")

        def else_fn():
            ops.print("else")

        ops.cond(cond, None, then_fn, else_fn)
        graph.output()

    # Verify both branches are present in MLIR
    mlir_str = str(graph._mlir_op)
    assert "then" in mlir_str
    assert "else" in mlir_str


def test_conditional_with_results() -> None:
    # Test conditional with return values
    with Graph("conditional_with_returns", input_types=()) as graph:
        cond = ops.constant(True, dtype=DType.bool, device=DeviceRef.CPU())

        def then_fn():
            return ops.constant(1, DType.int32, device=DeviceRef.CPU())

        def else_fn():
            return ops.constant(0, DType.int32, device=DeviceRef.CPU())

        result = ops.cond(
            cond,
            [TensorType(DType.int32, shape=[], device=DeviceRef.CPU())],
            then_fn,
            else_fn,
        )
        graph.output(result[0])

    assert "1" in str(graph._mlir_op)
    assert "0" in str(graph._mlir_op)


def test_conditional_type_check() -> None:
    # Test type checking between branches
    with Graph("conditional_type_check", input_types=()) as graph:
        cond = ops.constant(False, dtype=DType.bool, device=DeviceRef.CPU())

        def then_fn():
            return ops.constant(1.0, DType.float32, device=DeviceRef.CPU())

        def else_fn():
            return ops.constant(0, DType.int32, device=DeviceRef.CPU())

        try:
            ops.cond(
                cond,
                [TensorType(DType.float32, shape=[], device=DeviceRef.CPU())],
                then_fn,
                else_fn,
            )
        except TypeError as e:
            assert "Results don't match expected types" in str(e)

        graph.output()

    graph._mlir_op.verify()


def test_conditional_with_raising() -> None:
    with Graph("conditional_with_chain", input_types=()) as graph:
        chain = graph._current_chain
        cond = ops.constant(True, dtype=DType.bool, device=DeviceRef.CPU())

        def then_fn():
            return

        def else_fn():
            raise Exception("else")

        try:
            result = ops.cond(cond, None, then_fn, else_fn)
        except Exception as e:
            assert "else" in str(e)

        assert graph._current_chain == chain
        graph.output()
    graph._mlir_op.verify()
