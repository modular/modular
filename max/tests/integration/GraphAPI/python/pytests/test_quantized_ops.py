# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import numpy as np
from max.engine import InferenceSession
from max.graph import DType, Graph, TensorType, ops
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
        graph.output(ops.qmatmul(QuantizationEncoding.Q4_0, *graph.inputs))

    session = InferenceSession()
    compiled = session.load(graph)
    # This is a pretty bad test -- the inputs and outputs here are all zeroes.
    # But it's better than nothing -- at least we don't crash.  Also qmatmul
    # does not validate its tensor shapes all (except that the second input's
    # first dimension is a multiple of 32) so even if this were wrong we would
    # not be able to tell.
    generated = compiled.execute(
        input0=np.zeros((5, 32), dtype="float32"),
        input1=np.zeros((32, 18), dtype="uint8"),
    )
    expected = np.zeros((5, 32))
    np.testing.assert_equal(generated["output0"], expected)
