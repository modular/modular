# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: windows
# RUN: %mojo -I %engine_pkg_dir %s | FileCheck %s

from aiengine import (
    InferenceSession,
    EngineDType,
    EngineTensorSpec,
)
from tensor import TensorSpec


fn test_tensor_spec() raises:
    # CHECK: test_tensor_spec
    print("====test_tensor_spec")

    let session = InferenceSession()
    let spec = TensorSpec(DType.float32, 1, 2, 3)
    let engine_spec = session.get_as_engine_tensor_spec("tensor", spec)

    # CHECK: 1
    print(engine_spec[0])

    # CHECK: tensor
    print(engine_spec.get_name())

    # CHECK: {name=tensor, spec=1x2x3xfloat32}
    print(engine_spec.__str__())

    # CHECK: True
    print(engine_spec.get_as_tensor_spec() == spec)


fn main() raises:
    test_tensor_spec()
