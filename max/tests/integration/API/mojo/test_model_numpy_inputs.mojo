# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: windows
# UNSUPPORTED: address
# REQUIRES: numpy
# RUN: %mojo -debug-level full %s %S/mo.mlir

from max.engine import (
    InferenceSession,
    TensorMap,
    EngineNumpyView,
)
from pathlib import Path
from sys import argv
from tensor import Tensor, TensorShape
from testing import assert_equal
from collections import List
from python import Python


fn test_model_numpy_input() raises:
    var args = argv()

    assert_equal(len(args), 2)

    assert_equal(args[1], "mo.mlir")

    var model_path = args[1]
    var session = InferenceSession()
    var model = session.load_model(Path(model_path))

    var expected_output = Tensor[DType.float32](
        TensorShape(5), 4.0, 2.0, -5.0, 3.0, 6.0
    )

    var np = Python.import_module("numpy")
    var input_np_tensor = np.ones((5,)).astype(np.float32)

    var np_outputs = model.execute(("input", EngineNumpyView(input_np_tensor)))
    _ = input_np_tensor^
    var output_np_tensor = np_outputs.get[DType.float32]("output")

    assert_equal(str(output_np_tensor.spec()), "5xfloat32")

    assert_equal(expected_output, output_np_tensor)


fn main() raises:
    test_model_numpy_input()
