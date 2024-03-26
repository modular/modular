# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: windows
# RUN: mojo package %mojo_user_pkg -o %t.mojopkg
# RUN: %mojo -debug-level full %s %S/mo.mlir %t.mojopkg

from max.engine import (
    InferenceSession,
    TensorMap,
    EngineTensorView,
    LoadOptions,
)
from sys import argv
from tensor import Tensor, TensorShape
from testing import assert_equal
from pathlib import Path


fn test_model_metadata() raises:
    var args = argv()
    var model_path = args[1]
    var user_defined_ops_path = args[2]

    var session = InferenceSession()
    var config = LoadOptions()
    config.set_custom_ops_path(Path(user_defined_ops_path))
    var compiled_model = session.load_model(Path(model_path), config)
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

    var model_path = args[1]
    var user_defined_ops_path = args[2]

    var session = InferenceSession()
    var config = LoadOptions()
    config.set_custom_ops_path(Path(user_defined_ops_path))
    var model = session.load_model(Path(model_path), config^)
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

    # Verify our custom add op got replaced and called during execution
    # Our custom add op does x + y + 100 instead of the typical x + y.
    var expected_output = Tensor[DType.float32](
        TensorShape(5), 104.0, 102.0, 95.0, 103.0, 106.0
    )

    assert_equal(expected_output, output_tensor)


fn main() raises:
    test_model_metadata()
    test_model()
