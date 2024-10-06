# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

from sys import sizeof

from tensor_internal.tensor import TensorSpec, _serialize_as_tensor


# CHECK: test_serialize
fn test_serialize():
    print("== test_serialize")

    var spec = TensorSpec(DType.float32, 1)
    var bytes = _serialize_as_tensor(spec)
    # CHECK: True
    print(bytes.num_elements() == sizeof[TensorSpec]())

    # CHECK: True
    print(TensorSpec.from_bytes(bytes.unsafe_ptr()) == spec)


def main():
    test_serialize()
