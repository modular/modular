# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate -import-mojo -mojo-enable-prebuilt-packages %s | not kgen-opt -mogg-annotate 2>&1 | FileCheck %s

from compiler.directives import register, specsof
from tensor_utils.managed_tensor_slice import ManagedTensorSlice
from utils import StaticIntTuple


# CHECK: Struct based extensibility cannot have execute and initialize_output op!
@register("mo.matmul")
struct MyKernel:
    @staticmethod
    fn execute[](
        inout c: ManagedTensorSlice,
        a: ManagedTensorSlice,
        b: ManagedTensorSlice,
    ):
        pass

    @staticmethod
    fn initialize_output(
        a: ManagedTensorSlice, b: ManagedTensorSlice
    ) -> StaticIntTuple[2]:
        return StaticIntTuple[2]()
