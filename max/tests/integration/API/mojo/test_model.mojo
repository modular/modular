# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: windows
# RUN: %mojo -I %engine_pkg_dir -I %test_utils_pkg_dir %s %S/mo.model %S/model_different_input_output.mlir %S/model_different_dtypes.mlir | FileCheck %s

from max.engine import (
    InferenceSession,
    TensorMap,
    EngineTensorView,
    NamedTensor,
)
from sys import argv
from tensor import Tensor, TensorShape
from test_utils import linear_fill
from pathlib import Path


fn test_model_num_io_and_names() raises:
    # CHECK: test_model_num_io_and_names
    print("====test_model_num_io_and_names")

    var args = argv()
    var model_path = args[1]

    var session = InferenceSession()
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


fn test_model_metadata() raises:
    # CHECK: test_model_metadata
    print("====test_model_metadata")

    var args = argv()
    var model_path = args[1]

    var session = InferenceSession()
    var compiled_model = session.load_model(Path(model_path))

    var input_metadata = compiled_model.get_model_input_metadata()
    var num_inputs = len(input_metadata)

    # CHECK: 1
    print(num_inputs)

    # CHECK: input
    # CHECK: float32
    for input in input_metadata:
        print(input[].get_name())
        print(input[].get_dtype())

    var output_metadata = compiled_model.get_model_output_metadata()
    var num_outputs = len(output_metadata)

    # CHECK: 1
    print(num_outputs)

    # CHECK: output
    # CHECK: float32
    for output in output_metadata:
        print(output[].get_name())
        print(output[].get_dtype())


fn test_model_mismatched_input_output_count() raises:
    # CHECK: test_model_mismatched_input_output_count
    print("====test_model_mismatched_input_output_count")

    var args = argv()
    var model_path = args[2]

    var session = InferenceSession()
    var compiled_model = session.load_model(Path(model_path))
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

    var args = argv()
    var model_path = args[1]

    var session = InferenceSession()
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


fn test_model_tuple_input() raises:
    # CHECK: test_model_tuple_input
    print("====test_model_tuple_input")

    var args = argv()
    var model_path = args[1]

    var input_tensor = Tensor[DType.float32](5)

    for i in range(5):
        input_tensor[i] = 1.0

    var session = InferenceSession()
    var model = session.load_model(Path(model_path))
    var outputs = model.execute(NamedTensor("input", input_tensor ^))
    var output_tensor = outputs.get[DType.float32]("output")

    # CHECK: 5xfloat32
    print(output_tensor.spec().__str__())

    var expected_output = Tensor[DType.float32](5)
    linear_fill(expected_output, 4.0, 2.0, -5.0, 3.0, 6.0)
    # CHECK: True
    print(expected_output == output_tensor)


fn test_model_tuple_input_different_dtypes() raises:
    # CHECK: test_model_tuple_input_different_dtypes
    print("====test_model_tuple_input_different_dtypes")

    var args = argv()
    var model_path = args[3]

    var input_tensor_float = Tensor[DType.float32](5)
    var input_tensor_int = Tensor[DType.int32](5)

    for i in range(5):
        input_tensor_float[i] = 1.0
        input_tensor_int[i] = i

    var session = InferenceSession()
    var model = session.load_model(Path(model_path))
    var outputs = model.execute(
        NamedTensor("input0", input_tensor_float ^),
        NamedTensor("input1", input_tensor_int),
    )
    var output_tensor = outputs.get[DType.int32]("output")

    # CHECK: 5xint32
    print(output_tensor.spec())

    # CHECK: True
    print(input_tensor_int == output_tensor)


fn test_model_tuple_input_dynamic() raises:
    # CHECK: test_model_tuple_input_dynamic
    print("====test_model_tuple_input_dynamic")

    var args = argv()
    var model_path = args[1]

    var input_tensor = Tensor[DType.float32](5)

    for i in range(5):
        input_tensor[i] = 1.0

    var session = InferenceSession()
    var model = session.load_model(Path(model_path))
    var tensor_name: String = "input"

    var outputs = model.execute(NamedTensor(tensor_name, input_tensor ^))
    var output_tensor = outputs.get[DType.float32]("output")

    # CHECK: 5xfloat32
    print(str(output_tensor.spec()))

    var expected_output = Tensor[DType.float32](5)
    linear_fill(expected_output, 4.0, 2.0, -5.0, 3.0, 6.0)
    # CHECK: True
    print(expected_output == output_tensor)


fn main() raises:
    test_model_num_io_and_names()
    test_model_metadata()
    test_model_mismatched_input_output_count()
    test_model()
    test_model_tuple_input()
    test_model_tuple_input_different_dtypes()
    test_model_tuple_input_dynamic()
