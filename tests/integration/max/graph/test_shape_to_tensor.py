# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


import os
import tempfile

import numpy as np
from max.graph import DType, Graph, GraphValue, TensorType, ops


def test_shape_to_tensor_static(session):
    input_type = TensorType(dtype=DType.float32, shape=[2, 4])
    with Graph("input_shape", input_types=(input_type,)) as graph:
        shape = graph.inputs[0].shape
        graph.output(GraphValue.from_shape(shape))

    compiled = session.load(graph)

    x = np.ones((2, 4)).astype(np.float32)
    output = compiled.execute(input0=x)

    np.testing.assert_equal(output["output0"], np.array([2, 4]))


def test_shape_to_tensor_dynamic(session):
    input_type = TensorType(dtype=DType.float32, shape=["batch", "channels"])
    with Graph("input_shape", input_types=(input_type,)) as graph:
        shape = graph.inputs[0].shape
        graph.output(GraphValue.from_shape(shape))

    compiled = session.load(graph)

    x = np.ones((7, 3)).astype(np.float32)
    output = compiled.execute(input0=x)

    np.testing.assert_equal(output["output0"], np.array([7, 3]))


def test_shape_to_tensor_solo_dim(session):
    input_type = TensorType(dtype=DType.float32, shape=["batch", "channels"])
    with Graph("input_shape", input_types=(input_type,)) as graph:
        shape = graph.inputs[0].shape
        graph.output(GraphValue.from_dim(shape[1]))

    compiled = session.load(graph)

    x = np.ones((7, 3)).astype(np.float32)
    output = compiled.execute(input0=x)

    # Output is only a scalar
    assert output["output0"].shape == ()
    np.testing.assert_equal(output["output0"], np.array([3]))
