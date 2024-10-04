# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: not mojo  %s 2>&1 | FileCheck %s

from tensor_internal.tensor_shape import TensorShape


def main():
    # CHECK: constraint failed: shape should consist of integer values
    var weights: TensorShape = (-0.5, 2.2)
