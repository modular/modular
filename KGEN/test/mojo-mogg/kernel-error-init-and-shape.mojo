# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate -import-mojo -mojo-enable-prebuilt-packages %s | not kgen-opt -mogg-annotate 2>&1 | FileCheck %s

from compiler.directives import register, specsof
from tensor_utils.managed_tensor_slice import ManagedTensorSlice
from utils import IndexList


# CHECK: Struct based extensibility cannot have initialize_output and shape op!
@register("mo.matmul")
struct MyKernel:
    @staticmethod
    fn initialize_output(
        a: ManagedTensorSlice, b: ManagedTensorSlice
    ) -> IndexList[2]:
        return IndexList[2]()

    @staticmethod
    fn shape(a: ManagedTensorSlice, b: ManagedTensorSlice) -> IndexList[2]:
        return IndexList[2]()
