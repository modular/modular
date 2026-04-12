# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s -split-input-file | FileCheck %s


from std.builtin.variadics import *


# CHECK-LABEL:  lit.fn @"foldable
def foldable(t0: Tuple[Int, Int, Int], t1: Tuple[FloatDyn, FloatDyn, FloatDyn]):
    comptime zipped = Variadic.zip_types[
        # CHECK: [!Int, !FloatDyn], [!Int, !FloatDyn], [!Int, !FloatDyn]
        type_of(t0).element_types.values,
        type_of(t1).element_types.values,
    ]
    pass
