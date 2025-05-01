# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


import numpy as np
from max.driver import Tensor
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, TensorValue


def test_shape_to_tensor_static(session):
    input_type = TensorType(
        dtype=DType.float32, shape=[2, 4], device=DeviceRef.CPU()
    )
    with Graph("input_shape", input_types=(input_type,)) as graph:
        shape = graph.inputs[0].tensor.shape
        graph.output(TensorValue(shape))

    compiled = session.load(graph)

    x = np.ones((2, 4)).astype(np.float32)
    output = compiled.execute(
        Tensor.from_numpy(x).to(compiled.input_devices[0])
    )

    np.testing.assert_equal(output[0].to_numpy(), np.array([2, 4]))


def test_shape_to_tensor_dynamic(session):
    input_type = TensorType(
        dtype=DType.float32, shape=["batch", "channels"], device=DeviceRef.CPU()
    )
    with Graph("input_shape", input_types=(input_type,)) as graph:
        shape = graph.inputs[0].tensor.shape
        graph.output(TensorValue(shape))

    compiled = session.load(graph)

    x = np.ones((7, 3)).astype(np.float32)
    output = compiled.execute(
        Tensor.from_numpy(x).to(compiled.input_devices[0])
    )

    np.testing.assert_equal(output[0].to_numpy(), np.array([7, 3]))


def test_shape_to_tensor_solo_dim(session):
    input_type = TensorType(
        dtype=DType.float32, shape=["batch", "channels"], device=DeviceRef.CPU()
    )
    with Graph("input_shape", input_types=(input_type,)) as graph:
        shape = graph.inputs[0].tensor.shape
        graph.output(TensorValue(shape[1]))

    compiled = session.load(graph)

    x = np.ones((7, 3)).astype(np.float32)
    output = compiled.execute(
        Tensor.from_numpy(x).to(compiled.input_devices[0])
    )

    # Output is only a scalar
    assert output[0].shape == ()
    np.testing.assert_equal(output[0].to_numpy(), np.array([3]))
