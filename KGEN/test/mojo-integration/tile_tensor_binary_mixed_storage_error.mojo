# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# TileTensor's in-place binary operators require both operands to use the
# same storage class: a `PointerStorage`-backed destination rejects a
# `StaticOffsetStorage`-backed rhs at compile time. Mixed-policy ops are
# still available through the storage-level TensorOps API (see
# tile_tensor_binary_mixed_storage.mojo).

# RUN: not %mojo %s 2>&1 | FileCheck %s

from layout import TileTensor, row_major
from layout.tensor_storage import StaticOffsetStorage


def main():
    var a_data = InlineArray[Float32, 4](fill=1.0)
    var b_data = InlineArray[Float32, 4](fill=1.0)
    var a = TileTensor(a_data, row_major[2, 2]())
    var b = TileTensor[
        DType.float32,
        type_of(row_major[2, 2]()),
        origin_of(b_data),
        Storage=StaticOffsetStorage[static_offset=0],
    ](b_data.unsafe_ptr(), row_major[2, 2]())

    # CHECK: in-place binary ops require operands with the same storage class
    a += b
