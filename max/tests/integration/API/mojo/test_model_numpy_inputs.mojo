# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: windows
# UNSUPPORTED: address
# REQUIRES: numpy
# RUN: %mojo -I %engine_pkg_dir -I %test_utils_pkg_dir %s %S/mo.mlir | FileCheck %s

from max.engine import (
    InferenceSession,
    TensorMap,
    EngineNumpyView,
)
from sys import argv
from tensor import Tensor, TensorShape
from collections import List
from python import Python


fn test_model_numpy_input() raises:
    # CHECK: test_model_numpy_input
    print("====test_model_numpy_input")

    var args = argv()

    # CHECK: 2
    print(len(args))

    # CHECK: mo.mlir
    print(args[1])

    var model_path = args[1]
    var session = InferenceSession()
    var model = session.load_model(Path(model_path))

    var expected_output = Tensor[DType.float32](
        TensorShape(5), 4.0, 2.0, -5.0, 3.0, 6.0
    )

    var np = Python.import_module("numpy")
    var input_np_tensor = np.ones((5,)).astype(np.float32)

    var np_outputs = model.execute(("input", EngineNumpyView(input_np_tensor)))
    _ = input_np_tensor ^
    var output_np_tensor = np_outputs.get[DType.float32]("output")

    # CHECK: 5xfloat32
    print(output_np_tensor.spec().__str__())

    # CHECK: True
    print(expected_output == output_np_tensor)


fn main() raises:
    test_model_numpy_input()
