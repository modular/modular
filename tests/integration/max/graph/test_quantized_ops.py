# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import numpy as np
import pytest
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph, SymbolicDim, TensorType, ops
from max.graph.quantization import QuantizationEncoding


def test_qmatmul():
    graph = Graph(
        "qmatmul",
        input_types=[
            TensorType(DType.float32, (5, 32)),
            TensorType(DType.uint8, (32, 18)),
        ],
        output_types=[TensorType(DType.float32, (5, 32))],
    )

    with graph:
        graph.output(
            ops.qmatmul(QuantizationEncoding.Q4_0, None, *graph.inputs)
        )

    session = InferenceSession()
    compiled = session.load(graph)
    # This is a pretty bad test -- the inputs and outputs here are all zeroes.
    # But it's better than nothing -- at least we don't crash.  Also qmatmul
    # does not validate its tensor shapes all (except that the second input's
    # first dimension is a multiple of 32) so even if this were wrong we would
    # not be able to tell.
    generated = compiled.execute_legacy(
        input0=np.zeros((5, 32), dtype="float32"),
        input1=np.zeros((32, 18), dtype="uint8"),
    )
    expected = np.zeros((5, 32))
    np.testing.assert_equal(generated["output0"], expected)


def test_dequantize():
    graph = Graph(
        "dequantize",
        input_types=[TensorType(DType.uint8, (1, 18))],
        output_types=[TensorType(DType.float32, (1, 32))],
    )

    with graph:
        graph.output(ops.dequantize(QuantizationEncoding.Q4_0, *graph.inputs))

    session = InferenceSession()
    compiled = session.load(graph)
    # TODO: This is more of a smoke test than anything; we should really add a
    # test that uses some non-zero inputs and outputs (MSDK-820).
    generated = compiled.execute_legacy(input0=np.zeros((1, 18), dtype="uint8"))
    expected = np.zeros((1, 32))
    np.testing.assert_equal(generated["output0"], expected)


def test_dequantize_nondivisible_error():
    graph = Graph(
        "dequantize",
        input_types=[TensorType(DType.uint8, (1, 19))],
        output_types=[TensorType(DType.float32, (1, 32))],
    )

    with graph:
        with pytest.raises(
            ValueError,
            match=(
                r"last dimension \(.*19.*\) not divisible by block size \(18\)"
            ),
        ):
            ops.dequantize(QuantizationEncoding.Q4_0, *graph.inputs)


def test_dequantize_nonstatic_last_dim_error():
    graph = Graph(
        "dequantize",
        input_types=[TensorType(DType.uint8, (1, SymbolicDim("x")))],
        output_types=[TensorType(DType.float32, (1, 32))],
    )

    with graph:
        with pytest.raises(
            TypeError,
            match="dequantize only supported with static last dimension",
        ):
            ops.dequantize(QuantizationEncoding.Q4_0, *graph.inputs)
