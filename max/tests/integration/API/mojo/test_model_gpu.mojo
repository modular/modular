# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: windows
# REQUIRES: cuda
# RUN: %mojo %s %S/mo.mlir

from sys import argv
from testing import assert_equal, assert_true
from pathlib import Path

from max.engine import InferenceSession, SessionOptions
from max.engine._context import _Device
from tensor import Tensor, TensorShape


fn test_model_metadata() raises:
    var args = argv()

    assert_equal(len(args), 2)

    assert_true("mo.mlir" in args[1])

    var model_path = args[1]
    var options = SessionOptions()
    options._set_device(_Device.CUDA)
    var session = InferenceSession(options)
    var compiled_model = session.load(Path(model_path))
    assert_equal(compiled_model.num_model_inputs(), 1)

    var input_names = compiled_model.get_model_input_names()
    for name in input_names:
        assert_equal(name[], "input")

    assert_equal(input_names[0], "input")

    assert_equal(compiled_model.num_model_outputs(), 1)

    var output_names = compiled_model.get_model_output_names()
    for name in output_names:
        assert_equal(name[], "output")

    assert_equal(output_names[0], "output")


fn test_model() raises:
    var args = argv()

    assert_equal(len(args), 2)

    assert_true("mo.mlir" in args[1])

    var model_path = args[1]

    var options = SessionOptions()
    options._set_device(_Device.CUDA)
    var session = InferenceSession(options)
    var model = session.load(Path(model_path))
    var input_tensor = Tensor[DType.float32](5)

    for i in range(5):
        input_tensor[i] = 1.0

    # TODO: Hide tensor map from user
    var input_map = session.new_tensor_map()
    input_map.borrow("input", input_tensor)
    var outputs = model.execute(input_map)
    _ = input_tensor^  # Keep inputs alive
    var output_tensor = outputs.get[DType.float32]("output")

    assert_equal(str(output_tensor.spec()), "5xfloat32")

    var expected_output = Tensor[DType.float32](
        TensorShape(5), 4.0, 2.0, -5.0, 3.0, 6.0
    )
    assert_equal(expected_output, output_tensor)


fn main() raises:
    test_model_metadata()
    test_model()
