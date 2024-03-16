# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: windows
# RUN: mojo package %mojo_user_pkg -o %t.mojopkg
# RUN: %mojo -I %engine_pkg_dir -I %test_utils_pkg_dir %s %S/mo.mlir %t.mojopkg | FileCheck %s

from max.engine import (
    InferenceSession,
    TensorMap,
    EngineTensorView,
    LoadOptions,
)
from sys import argv
from tensor import Tensor, TensorShape
from pathlib import Path


fn test_model_metadata() raises:
    var args = argv()
    var model_path = args[1]
    var user_defined_ops_path = args[2]

    var session = InferenceSession()
    var config = LoadOptions()
    config.set_custom_ops_path(Path(user_defined_ops_path))
    var compiled_model = session.load_model(Path(model_path), config)
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
    var args = argv()
    print(args[1])

    var model_path = args[1]
    var user_defined_ops_path = args[2]

    var session = InferenceSession()
    var config = LoadOptions()
    config.set_custom_ops_path(Path(user_defined_ops_path))
    var model = session.load_model(Path(model_path), config ^)
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

    # Verify our custom add op got replaced and called during execution
    # Our custom add op does x + y + 100 instead of the typical x + y.
    var expected_output = Tensor[DType.float32](
        TensorShape(5), 104.0, 102.0, 95.0, 103.0, 106.0
    )

    # CHECK: True
    print(expected_output == output_tensor)


fn main() raises:
    test_model_metadata()
    test_model()
