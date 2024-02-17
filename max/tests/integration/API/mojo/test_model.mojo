# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: windows
# RUN: %mojo -I %engine_pkg_dir -I %test_utils_pkg_dir %s %S/mo.model %S/model_different_input_output.mlir | FileCheck %s

from max.engine import (
    InferenceSession,
    TensorMap,
    EngineTensorView,
)
from sys import argv
from tensor import Tensor, TensorShape
from test_utils import linear_fill
from pathlib import Path


fn test_model_metadata() raises:
    # CHECK: test_model_metadata
    print("====test_model_metadata")

    let args = argv()
    let model_path = args[1]

    let session = InferenceSession()
    let compiled_model = session.load_model(Path(model_path))
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


fn test_model_mismatched_input_output_count() raises:
    # CHECK: test_model_mismatched_input_output_count
    print("====test_model_mismatched_input_output_count")

    let args = argv()
    let model_path = args[2]

    let session = InferenceSession()
    let compiled_model = session.load_model(Path(model_path))
    # CHECK: 2
    print(compiled_model.num_model_inputs())

    var input_names = compiled_model.get_model_input_names()

    # CHECK: 2
    print(len(input_names))

    # CHECK: input0
    # CHECK: input1
    for name in input_names:
        print(name[])

    # CHECK: input1
    print(input_names[1])

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

    let args = argv()
    let model_path = args[1]

    let session = InferenceSession()
    let model = session.load_model(Path(model_path))
    var input_tensor = Tensor[DType.float32](5)

    for i in range(5):
        input_tensor[i] = 1.0

    # TODO: Hide tensor map from user
    let input_map = session.new_tensor_map()
    input_map.borrow("input", input_tensor)
    let outputs = model.execute(input_map)
    _ = input_tensor ^  # Keep inputs alive
    let output_tensor = outputs.get[DType.float32]("output")

    # CHECK: 5xfloat32
    print(output_tensor.spec().__str__())

    var expected_output = Tensor[DType.float32](5)
    linear_fill(expected_output, 4.0, 2.0, -5.0, 3.0, 6.0)
    # CHECK: True
    print(expected_output == output_tensor)


fn test_model_tuple_input() raises:
    # CHECK: test_model_tuple_input
    print("====test_model_tuple_input")

    let args = argv()
    let model_path = args[1]

    var input_tensor = Tensor[DType.float32](5)

    for i in range(5):
        input_tensor[i] = 1.0

    let session = InferenceSession()
    let model = session.load_model(Path(model_path))
    let outputs = model.execute(("input", EngineTensorView(input_tensor)))
    _ = input_tensor ^
    let output_tensor = outputs.get[DType.float32]("output")

    # CHECK: 5xfloat32
    print(output_tensor.spec().__str__())

    var expected_output = Tensor[DType.float32](5)
    linear_fill(expected_output, 4.0, 2.0, -5.0, 3.0, 6.0)
    # CHECK: True
    print(expected_output == output_tensor)


fn test_model_tuple_input_dynamic() raises:
    # CHECK: test_model_tuple_input_dynamic
    print("====test_model_tuple_input_dynamic")

    let args = argv()
    let model_path = args[1]

    var input_tensor = Tensor[DType.float32](5)

    for i in range(5):
        input_tensor[i] = 1.0

    let session = InferenceSession()
    let model = session.load_model(Path(model_path))
    let tensor_name: String = "input"
    # TODO: Remove use of StringRef from `model.execute` APIs
    # once we support std::Tuple on memory-only types.
    # See https://github.com/modularml/modular/issues/30576
    let outputs = model.execute(
        (tensor_name._strref_dangerous(), EngineTensorView(input_tensor))
    )
    _ = input_tensor ^
    let output_tensor = outputs.get[DType.float32]("output")

    # CHECK: 5xfloat32
    print(str(output_tensor.spec()))

    var expected_output = Tensor[DType.float32](5)
    linear_fill(expected_output, 4.0, 2.0, -5.0, 3.0, 6.0)
    # CHECK: True
    print(expected_output == output_tensor)
    tensor_name._strref_keepalive()


fn main() raises:
    test_model_metadata()
    test_model_mismatched_input_output_count()
    test_model()
    test_model_tuple_input()
    test_model_tuple_input_dynamic()
