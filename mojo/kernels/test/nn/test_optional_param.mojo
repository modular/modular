# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s | FileCheck %s

from buffer.dimlist import Dim, DimList
from nn._optional_param import OptionalParamInt, OptionalParamInts
from testing import *

from utils import StaticIntTuple


# CHECK-LABEL: test_opt_param_int
fn test_opt_param_int():
    print("=== test_opt_param_int")
    alias dp0 = Dim(0)
    # CHECK: 0
    print(OptionalParamInt[dp0](1).get())

    alias dp1 = Dim()
    # CHECK: 1
    print(OptionalParamInt[dp1](1).get())


# CHECK-LABEL: test_opt_param_ints
fn test_opt_param_ints():
    print("=== test_opt_param_ints")
    alias dp0 = DimList(0, 0)
    var d0 = OptionalParamInts[2, dp0](StaticIntTuple[2](1, 1))
    # CHECK: 0
    print(d0.at[0]())
    # CHECK: 0
    print(d0.at[1]())

    alias dp1 = DimList(Dim(), 0)
    var d1 = OptionalParamInts[2, dp1](StaticIntTuple[2](1, 1))
    # CHECK: 1
    print(d1.at[0]())
    # CHECK: 0
    print(d1.at[1]())


def main():
    test_opt_param_int()
    test_opt_param_ints()
