# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: windows
# REQUIRES: mtorch
# RUN: %mojo %s %torchscript_relu_model | FileCheck %s

from max.engine import (
    InferenceSession,
    TensorMap,
    EngineTensorView,
    LoadOptions,
)
from sys import argv
from tensor import TensorSpec
from collections import Optional
from testing import assert_equal


fn test_pytorch_model() raises:
    # CHECK: test_pytorch_model
    print("====test_pytorch_model")

    let args = argv()
    let model_path = args[1]

    let session = InferenceSession()
    var config = LoadOptions()
    config.add_input_spec("x", TensorSpec(DType.float32, 3, 100, 100))
    let compiled_model = session.load_model(Path(model_path), config)

    # CHECK: 1
    print(compiled_model.num_model_inputs())

    let input_names = compiled_model.get_model_input_names()
    # CHECK: x
    for name in input_names:
        print(name)

    # CHECK: x
    print(input_names[0])

    # CHECK: 1
    print(compiled_model.num_model_outputs())


fn test_pytorch_model2() raises:
    # CHECK: test_pytorch_model2
    print("====test_pytorch_model2")

    let args = argv()
    let model_path = args[1]

    let session = InferenceSession()
    var config = LoadOptions()
    var shape = DynamicVector[Optional[Int64]]()
    shape.push_back(Int64(3))
    shape.push_back(Int64(100))
    shape.push_back(Int64(100))
    config.add_input_spec("x", shape, DType.float32)
    let compiled_model = session.load_model(Path(model_path), config)

    # CHECK: 1
    print(compiled_model.num_model_inputs())

    let input_names = compiled_model.get_model_input_names()
    # CHECK: x
    for name in input_names:
        print(name)

    # CHECK: x
    print(input_names[0])

    # CHECK: 1
    print(compiled_model.num_model_outputs())


fn test_model_execute() raises:
    # CHECK: test_pytorch_model2
    print("====test_pytorch_model2")

    let args = argv()
    let model_path = args[1]

    let session = InferenceSession()
    var config = LoadOptions()
    config.add_input_spec("x", TensorSpec(DType.float32, 3, 100, 100))
    let model = session.load_model(Path(model_path), config)
    var input_tensor = Tensor[DType.float32](3, 100, 100)
    input_tensor._to_buffer().fill(-1)
    let outputs = model.execute("x", input_tensor)
    let output_tensor = outputs.get[DType.float32]("result0")
    var expected_output = Tensor[DType.float32](3, 100, 100)
    expected_output._to_buffer().fill(0)
    assert_equal(output_tensor, expected_output)


fn main() raises:
    test_pytorch_model()
    test_pytorch_model2()
    test_model_execute()
