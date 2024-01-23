# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: windows
# RUN: %mojo -I %engine_pkg_dir %s | FileCheck %s

from max.engine import (
    InferenceSession,
    EngineTensorSpec,
)
from tensor import TensorSpec
from collections.optional import Optional


fn test_tensor_spec() raises:
    # CHECK: test_tensor_spec
    print("====test_tensor_spec")

    let session = InferenceSession()
    let spec = TensorSpec(DType.float32, 1, 2, 3)
    let engine_spec = session.get_as_engine_tensor_spec("tensor", spec)

    # CHECK: 1
    print(engine_spec[0].value())

    # CHECK: tensor
    print(engine_spec.get_name())

    # CHECK: {name=tensor, spec=1x2x3xfloat32}
    print(str(engine_spec))

    # CHECK: True
    print(engine_spec.get_as_tensor_spec() == spec)

    var dynamic_dim_shape = DynamicVector[Optional[Int64]]()
    dynamic_dim_shape.push_back(None)
    dynamic_dim_shape.push_back(Int64(1))
    dynamic_dim_shape.push_back(Int64(2))
    let dynamic_dim_spec = session.get_as_engine_tensor_spec(
        "tensor", dynamic_dim_shape, DType.float32
    )

    # CHECK: False
    print(dynamic_dim_spec[0].__bool__())

    # CHECK: 3
    print(dynamic_dim_spec.rank().value())

    # CHECK: True
    print(dynamic_dim_spec.has_rank())

    # CHECK: {name=tensor, spec=-1x1x2xfloat32}
    print(str(dynamic_dim_spec))

    let dynamic_rank_spec = session.get_as_engine_tensor_spec(
        "tensor", None, DType.float32
    )

    # CHECK: False
    print(dynamic_rank_spec.rank().__bool__())

    # CHECK: False
    print(dynamic_rank_spec.has_rank())

    # CHECK: {name=tensor, spec=None x float32}
    print(str(dynamic_rank_spec))


fn main() raises:
    test_tensor_spec()
