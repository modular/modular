# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: windows
# REQUIRES: mtorch
# RUN: %mojo -debug-level full %s %torchscript_relu_model

from max.engine import (
    InferenceSession,
    TensorMap,
    EngineTensorView,
    LoadOptions,
    ShapeElement,
)
from sys import argv
from tensor import Tensor, TensorSpec
from collections import Optional
from testing import assert_equal, assert_raises
from pathlib import Path


fn test_pytorch_model() raises:
    var args = argv()
    var model_path = args[1]

    var session = InferenceSession()
    var config = LoadOptions()
    config.add_input_spec(TensorSpec(DType.float32, 1, 3, 100, 100))
    var compiled_model = session.load(Path(model_path), config)

    assert_equal(compiled_model.num_model_inputs(), 1)

    var input_names = compiled_model.get_model_input_names()
    for name in input_names:
        assert_equal(name[], "x")

    assert_equal(input_names[0], "x")

    assert_equal(compiled_model.num_model_outputs(), 1)


fn test_pytorch_model2() raises:
    var args = argv()
    var model_path = args[1]

    var session = InferenceSession()
    var config = LoadOptions()
    var shape = List[Optional[Int64]]()
    shape.append(Int64(1))
    shape.append(Int64(3))
    shape.append(Int64(100))
    shape.append(Int64(100))
    config.add_input_spec(shape, DType.float32)
    var compiled_model = session.load(Path(model_path), config)

    assert_equal(compiled_model.num_model_inputs(), 1)

    var input_names = compiled_model.get_model_input_names()
    for name in input_names:
        assert_equal(name[], "x")

    assert_equal(input_names[0], "x")

    assert_equal(compiled_model.num_model_outputs(), 1)


fn test_named_input_dims() raises:
    # CHECK-LABEL: ====test_named_input_dims
    print("====test_named_input_dims")

    var model_path = Path(argv()[1])
    var session = InferenceSession()
    var config = LoadOptions()
    var shape = List[ShapeElement]()
    shape.append("batch")
    shape.append(3)
    shape.append(100)
    shape.append(100)
    config.add_input_spec(shape, DType.float32)
    _ = session.load(model_path, config)

    shape.clear()
    config = LoadOptions()
    shape.append("1two3")
    shape.append(3)
    shape.append(100)
    shape.append(100)
    with assert_raises():
        config.add_input_spec(shape, DType.float32)
        _ = session.load(model_path, config)


fn test_model_execute() raises:
    var args = argv()
    var model_path = args[1]

    var session = InferenceSession()
    var config = LoadOptions()
    config.add_input_spec(TensorSpec(DType.float32, 1, 3, 100, 100))
    var model = session.load(Path(model_path), config)
    var input_tensor = Tensor[DType.float32](1, 3, 100, 100)
    input_tensor._to_buffer().fill(-1)
    var outputs = model.execute("x", input_tensor)
    var output_tensor = outputs.get[DType.float32]("result0")
    var expected_output = Tensor[DType.float32](1, 3, 100, 100)
    expected_output._to_buffer().fill(0)
    assert_equal(output_tensor, expected_output)


fn main() raises:
    test_pytorch_model()
    test_pytorch_model2()
    test_named_input_dims()
    test_model_execute()
