# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s %S/mo.mlir

from pathlib import Path
from sys import argv

from max.engine import EngineNumpyView, InferenceSession
from max.tensor import Tensor, TensorShape
from python import Python
from testing import assert_equal


fn test_model_numpy_input() raises:
    var args = argv()

    assert_equal(len(args), 2)

    var model_path = args[1]
    var session = InferenceSession()
    var model = session.load(Path(model_path))

    var expected_output = Tensor[DType.float32](
        TensorShape(5), 4.0, 2.0, -5.0, 3.0, 6.0
    )

    var np = Python.import_module("numpy")
    var input_np_tensor = np.ones((5,)).astype(np.float32)

    var np_outputs = model.execute(("input", EngineNumpyView(input_np_tensor)))
    _ = input_np_tensor^
    var output_np_tensor = np_outputs.get[DType.float32]("output")

    assert_equal(String(output_np_tensor.spec()), "5xfloat32")

    assert_equal(expected_output, output_np_tensor)


fn main() raises:
    test_model_numpy_input()
