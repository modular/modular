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
# RUN: env "PATH=%t/bin:$PATH" %mojo -I %engine_pkg_dir -I %test_utils_pkg_dir %s %S/mo.model %S/model_different_input_output.mlir %S/model_different_dtypes.mlir | FileCheck %s

from max.engine import (
    InferenceSession,
    TensorMap,
    EngineTensorView,
    NamedTensor,
)
from sys import argv
from tensor import Tensor, TensorShape
from closed_source_test_utils import linear_fill
from pathlib import Path
from python import Python


# CHECK-LABEL: ==== test_model_num_io_and_names
fn test_model_num_io_and_names() raises:
    print("==== test_model_num_io_and_names")

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


# CHECK-LABEL: ==== test_model_metadata
fn test_model_metadata() raises:
    print("==== test_model_metadata")

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


# CHECK-LABEL: ==== test_model_mismatched_input_output_count
fn test_model_mismatched_input_output_count() raises:
    print("==== test_model_mismatched_input_output_count")

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


# CHECK-LABEL: ==== test_model
fn test_model() raises:
    print("==== test_model")

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


# CHECK-LABEL: ==== test_model_tuple_input
fn test_model_tuple_input() raises:
    print("==== test_model_tuple_input")

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


# CHECK-LABEL: ==== test_model_tuple_input_different_dtypes
fn test_model_tuple_input_different_dtypes() raises:
    print("==== test_model_tuple_input_different_dtypes")

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


# CHECK-LABEL: ==== test_model_tuple_input_dynamic
fn test_model_tuple_input_dynamic() raises:
    print("==== test_model_tuple_input_dynamic")

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


# CHECK-LABEL: ==== test_model_py_dict_execute
fn test_model_py_dict_execute() raises:
    print("==== test_model_py_dict_execute")

    var model_path = Path(argv()[1])
    var session = InferenceSession()
    var model = session.load_model(model_path)
    var inputs = Python.evaluate(
        "{'input': __import__('numpy').arange(5).astype('float32')}"
    )
    var outputs = model.execute(inputs)
    var output_tensor = outputs.get[DType.float32]("output")

    # CHECK: 5xfloat32
    print(str(output_tensor.spec()))

    var expected_output = Tensor[DType.float32](5)
    linear_fill(expected_output, 3.0, 2.0, -4.0, 5.0, 9.0)
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
    test_model_py_dict_execute()
