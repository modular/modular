# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# COM: See #34373 - flaky test
# REQUIRES: disabled
# UNSUPPORTED: windows
# NOTE: This file is a temporary copy of test_model.mojo to debug #34373. This
# file should be deleted if / when we resolve it. Any applicable changes should
# be moved to test_model.mojo
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


# This is intentionally duplicated to simulate a check assert that fires right
# after running test_model_metadata(). We want to see if it was a cleanup issue
# causing CHECKs to fail in test_model_mismatched_input_output_count().
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


fn main() raises:
    test_model_metadata()
    test_model_mismatched_input_output_count()
