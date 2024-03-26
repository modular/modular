# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: windows
# Mojo requires that the "python3" executable on PATH be usable to determine
# the location of site-packages.  We don't know that the user is using the
# autovenv, so we can't just prepend the autovenv bin to the PATH, and we
# wouldn't want all the executables in there anyway.  Symbolic link also fails
# because Python can't find its pyvenv.cfg file.  So we need to create a stub
# bin directory containing a shell script causing python3 to be re-exec'ed with
# its original argv0, just so Mojo's Python can find the site-packages so that
# the Engine's NumPy import works.
# RUN: rm -rf %t/bin
# RUN: mkdir -p %t/bin
# RUN: printf '#!/bin/bash\nexec "%%s" "$@"\n' %pyexe > %t/bin/python3
# RUN: chmod +x %t/bin/python3
# RUN: env "PATH=%t/bin:$PATH" %mojo -debug-level full %s %S/mo.mlir %S/model_different_input_output.mlir %S/model_different_dtypes.mlir

from max.engine import (
    InferenceSession,
    TensorMap,
    EngineTensorView,
    NamedTensor,
)
from sys import argv
from tensor import Tensor, TensorShape
from testing import assert_equal, assert_false, assert_true
from collections import List
from pathlib import Path
from python import Python


fn test_model_num_io_and_names() raises:
    var args = argv()
    var model_path = args[1]

    var session = InferenceSession()
    var compiled_model = session.load_model(Path(model_path))
    assert_equal(compiled_model.num_model_inputs(), 1)

    var input_names = compiled_model.get_model_input_names()
    assert_equal(len(input_names), 1)

    for name in input_names:
        assert_equal(name[], "input")

    assert_equal(input_names[0], "input")

    assert_equal(compiled_model.num_model_outputs(), 1)

    var output_names = compiled_model.get_model_output_names()
    assert_equal(len(output_names), 1)

    for name in output_names:
        assert_equal(name[], "output")

    assert_equal(output_names[0], "output")


fn test_model_metadata() raises:
    var args = argv()
    var model_path = args[1]

    var session = InferenceSession()
    var compiled_model = session.load_model(Path(model_path))

    var input_metadata = compiled_model.get_model_input_metadata()
    var num_inputs = len(input_metadata)

    assert_equal(num_inputs, 1)

    for input in input_metadata:
        assert_equal(input[].get_name(), "input")
        assert_equal(str(input[].get_dtype()), "float32")

    var output_metadata = compiled_model.get_model_output_metadata()
    var num_outputs = len(output_metadata)

    assert_equal(num_outputs, 1)

    for output in output_metadata:
        assert_equal(output[].get_name(), "output")
        assert_equal(str(output[].get_dtype()), "float32")


fn test_model_mismatched_input_output_count() raises:
    var args = argv()
    var model_path = args[2]

    var session = InferenceSession()
    var compiled_model = session.load_model(Path(model_path))
    assert_equal(compiled_model.num_model_inputs(), 2)

    var input_names = compiled_model.get_model_input_names()

    assert_equal(len(input_names), 2)

    var count = 0
    for name in input_names:
        assert_equal(name[], "input" + str(count))
        count += 1

    assert_equal(input_names[1], "input1")

    assert_equal(compiled_model.num_model_outputs(), 1)

    var output_names = compiled_model.get_model_output_names()
    for name in output_names:
        assert_equal(name[], "output")

    assert_equal(output_names[0], "output")


fn test_model() raises:
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

    assert_equal(str(output_tensor.spec()), "5xfloat32")

    var expected_output = Tensor[DType.float32](
        TensorShape(5), 4.0, 2.0, -5.0, 3.0, 6.0
    )
    assert_equal(expected_output, output_tensor)


fn test_model_tuple_input() raises:
    var args = argv()
    var model_path = args[1]

    var input_tensor = Tensor[DType.float32](5)

    for i in range(5):
        input_tensor[i] = 1.0

    var session = InferenceSession()
    var model = session.load_model(Path(model_path))
    var outputs = model.execute(NamedTensor("input", input_tensor ^))
    var output_tensor = outputs.get[DType.float32]("output")

    assert_equal(str(output_tensor.spec()), "5xfloat32")

    var expected_output = Tensor[DType.float32](
        TensorShape(5), List[Float32](4.0, 2.0, -5.0, 3.0, 6.0)
    )
    assert_equal(expected_output, output_tensor)


fn test_model_tuple_input_different_dtypes() raises:
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

    assert_equal(output_tensor.spec(), "5xint32")

    assert_equal(input_tensor_int, output_tensor)


fn test_model_tuple_input_dynamic() raises:
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

    assert_equal(str(output_tensor.spec()), "5xfloat32")

    var expected_output = Tensor[DType.float32](
        TensorShape(5), List[Float32](4.0, 2.0, -5.0, 3.0, 6.0)
    )
    assert_equal(expected_output, output_tensor)


fn test_model_py_dict_execute() raises:
    var model_path = Path(argv()[1])
    var session = InferenceSession()
    var model = session.load_model(model_path)
    var inputs = Python.evaluate(
        "{'input': __import__('numpy').arange(5).astype('float32')}"
    )
    var outputs = model.execute(inputs)
    var output_tensor = outputs.get[DType.float32]("output")

    assert_equal(str(output_tensor.spec()), "5xfloat32")

    var expected_output = Tensor[DType.float32](
        TensorShape(5), List[Float32](3.0, 2.0, -4.0, 5.0, 9.0)
    )
    assert_equal(expected_output, output_tensor)


fn main() raises:
    test_model_num_io_and_names()
    test_model_metadata()
    test_model_mismatched_input_output_count()
    test_model()
    test_model_tuple_input()
    test_model_tuple_input_different_dtypes()
    test_model_tuple_input_dynamic()
    test_model_py_dict_execute()
