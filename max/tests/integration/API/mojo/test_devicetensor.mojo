# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s %S/mo_unnamed.mlir

# COM: Test with mojo build
# RUN: mkdir -p %t
# RUN: rm -rf %t/engine-device-tensor
# RUN: %mojo-build %s -o %t/engine-device-tensor
# RUN: %t/engine-device-tensor %S/mo_unnamed.mlir

from pathlib import Path
from sys import argv

from max.driver import cpu
from max.engine import InferenceSession, SessionOptions
from max.graph import Graph, Symbol, TensorType, Type
from max.tensor import TensorSpec
from testing import assert_equal, assert_true


fn test_model_device_tensor() raises:
    var args = argv()
    assert_equal(len(args), 2)
    assert_true("mo_unnamed.mlir" in args[1])

    var model_path = args[1]
    var device = cpu()

    var options = SessionOptions()
    options._set_device(device)
    var session = InferenceSession(options)
    var model = session.load(Path(model_path))

    var dt = device.allocate(TensorSpec(DType.float32, 5))
    var input_tensor = dt^.to_tensor[DType.float32, 1]()
    for i in range(5):
        input_tensor[i] = 1.0

    var outputs = model.execute(input_tensor^)

    assert_equal(len(outputs), 1)
    var dm_back = outputs[0].take().to_device_tensor()
    var output_tensor = dm_back^.to_tensor[DType.float32, 1]()

    assert_equal(output_tensor[0], 4.0)
    assert_equal(output_tensor[1], 2.0)
    assert_equal(output_tensor[2], -5.0)
    assert_equal(output_tensor[3], 3.0)
    assert_equal(output_tensor[4], 6.0)


fn test_device_graph() raises:
    # Build graph that adds a single constant tensor to its input.
    var g = Graph("add_constant", List[Type](TensorType(DType.float32, 1)))
    var x = g.scalar[DType.float32](1.0, rank=1)
    var y = g.op("mo.add", List[Symbol](g[0], x), TensorType(DType.float32, 1))
    g.output(y)
    g.verify()

    var device = cpu()
    var options = SessionOptions()
    options._set_device(device)
    var session = InferenceSession(options)
    var model = session.load(g)

    var dt = device.allocate(TensorSpec(DType.float32, 1))
    var input_tensor = dt^.to_tensor[DType.float32, 1]()
    input_tensor[0] = 1.0

    var outputs = model.execute(input_tensor^)

    assert_equal(len(outputs), 1)
    var dm_back = outputs[0].take().to_device_tensor()
    var output_tensor = dm_back^.to_tensor[DType.float32, 1]()
    assert_equal(output_tensor[0], 2.0)


fn main() raises:
    test_model_device_tensor()
    test_device_graph()
