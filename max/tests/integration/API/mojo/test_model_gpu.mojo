# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: windows
# REQUIRES: cuda
# RUN: %mojo -I %engine_pkg_dir -I %test_utils_pkg_dir %s %S/mo.mlir | FileCheck %s

from max.engine import (
    InferenceSession,
    TensorMap,
    EngineTensorView,
    SessionOptions,
)
from max.engine._context import _Device
from sys import argv
from tensor import Tensor, TensorShape
from closed_source_test_utils import linear_fill
from pathlib import Path


fn test_model_metadata() raises:
    # CHECK: test_model_metadata
    print("====test_model_metadata")

    var args = argv()

    # CHECK: 2
    print(len(args))

    # CHECK: mo.mlir
    print(args[1])

    var model_path = args[1]
    var options = SessionOptions()
    options._set_device(_Device.CUDA)
    var session = InferenceSession(options)
    var compiled_model = session.load_model(Path(model_path))
    # CHECK: 1
    print(compiled_model.num_model_inputs())

    var input_names = compiled_model.get_model_input_names()
    # CHECK: input
    for name in input_names:
        print(name[])

    # CHECK: input
    print(input_names[0])

    # CHECK: 1
    print(compiled_model.num_model_outputs())

    var output_names = compiled_model.get_model_output_names()
    # CHECK: output
    for name in output_names:
        print(name[])

    # CHECK: output
    print(output_names[0])


fn test_model() raises:
    # CHECK: test_model
    print("====test_model")

    var args = argv()

    # CHECK: 2
    print(len(args))

    # CHECK: mo.mlir
    print(args[1])

    var model_path = args[1]

    var options = SessionOptions()
    options._set_device(_Device.CUDA)
    var session = InferenceSession(options)
    var model = session.load_model(Path(model_path))
    var input_tensor = Tensor[DType.float32](5)

    for i in range(5):
        input_tensor[i] = 1.0

    # TODO: Hide tensor map from user
    var input_map = session.new_tensor_map()
    input_map.borrow("input", input_tensor)
    var outputs = model.execute(input_map)
    _ = input_tensor ^  # Keep inputs alive
    var output_tensor = outputs.get[DType.float32]("output")

    # CHECK: 5xfloat32
    print(output_tensor.spec().__str__())

    var expected_output = Tensor[DType.float32](5)
    linear_fill(expected_output, 4.0, 2.0, -5.0, 3.0, 6.0)
    # CHECK: True
    print(expected_output == output_tensor)


fn main() raises:
    test_model_metadata()
    test_model()
