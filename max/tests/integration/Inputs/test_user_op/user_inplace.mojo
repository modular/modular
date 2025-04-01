# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import compiler
from tensor import ManagedTensorSlice
from tensor_internal.managed_tensor_slice import (
    _MutableInputTensor as MutableInputTensor,
)


@compiler.register("mutable_test_op")
struct MutableTestOp:
    @staticmethod
    fn execute(in_place_tensor: MutableInputTensor) raises:
        x = in_place_tensor._ptr.load(0)
        x += 1
        in_place_tensor._ptr.store(0, x)


@compiler.register("foo")
struct FooKernel:
    @staticmethod
    fn execute(in_place_tensor: MutableInputTensor) raises:
        in_place_tensor._ptr.store(0, 0)


@compiler.register("bar")
struct BarKernel:
    @staticmethod
    fn execute(in_place_tensor: MutableInputTensor) raises:
        in_place_tensor._ptr.store(0, 0)


@compiler.register("baz")
struct BazKernel:
    @staticmethod
    fn execute(in_place_tensor: MutableInputTensor) raises:
        in_place_tensor._ptr.store(0, 0)
