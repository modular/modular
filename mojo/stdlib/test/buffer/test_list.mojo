# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s

from utils.index import StaticIntTuple
from utils.list import Dim, DimList
from utils._optional_param import OptionalParamInt, OptionalParamInts


# CHECK-LABEL: test_dim_list
fn test_dim_list():
    print("== test_dim_list")

    let lst = DimList(1, 2, 3, 4)

    # CHECK: [1, 2, 3, 4]
    print[4](lst)

    # CHECK: 24
    print(lst.product[4]().get())


# CHECK-LABEL: test_dim
fn test_dim():
    print("== test_dim")

    let dim0 = Dim(8)
    # CHECK: True
    print(dim0.is_multiple[4]())

    let dim1 = Dim()
    # CHECK: False
    print(dim1.is_multiple[4]())


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
    let d0 = OptionalParamInts[2, dp0](StaticIntTuple[2](1, 1))
    # CHECK: 0
    print(d0.at[0]())
    # CHECK: 0
    print(d0.at[1]())

    alias dp1 = DimList(Dim(), 0)
    let d1 = OptionalParamInts[2, dp1](StaticIntTuple[2](1, 1))
    # CHECK: 1
    print(d1.at[0]())
    # CHECK: 0
    print(d1.at[1]())


fn main():
    test_dim_list()
    test_dim()
    test_opt_param_int()
    test_opt_param_ints()
