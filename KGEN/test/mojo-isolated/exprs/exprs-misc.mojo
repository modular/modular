# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s -verify-diagnostics | FileCheck %s

struct Unmovable:
  fn __init__(inout self): pass

##===----------------------------------------------------------------------===##
# Test return slot optimization
##===----------------------------------------------------------------------===##

# NOTE: Don't remove this argument, this was defeating return slot opzn.
fn getUnmovable(a: Unmovable) -> Unmovable:
  return Unmovable()

# This can only be codegen'd directly into x.
# CHECK-LABEL: lit.func @"testUnmovable
fn testUnmovable(a: Unmovable):
   # CHECK-NEXT: %x = lit.var.decl "x"
   # CHECK-NEXT: lit.call {{.*}}(%a, %x)
   var x : Unmovable = getUnmovable(a)

##===----------------------------------------------------------------------===##
# __type_of
##===----------------------------------------------------------------------===##

alias index = __mlir_type.index

# CHECK-LABEL: lit.func @"simple_typeof_return(
# CHECK: __mlir_type.index)"(%x: index) -> index
fn simple_typeof_return(x: index) -> __type_of(x):
    return x


# CHECK-LABEL: lit.func @"typeof_arg(
# CHECK: __mlir_type.index,__mlir_type.index)"(%x: index, %y: index) -> index
fn typeof_arg(x: index, y: __type_of(x)) -> index:
    var z : __type_of(x) = y
    return z

# CHECK-LABEL: lit.func @"typeof_dynval_in_param(
fn typeof_dynval_in_param(x: index):
    # CHECK-NEXT:  %y = lit.var.decl
    # CHECK-NEXT: lit.call {{.*}}String::@"__init__
    var y = String()

    # CHECK-NEXT: lit.alias.decl *"a`1": type = <index>
    alias a = __type_of(x)
    # CHECK-NEXT: lit.alias.decl *"b`2": !mt_Int = <!Int>
    alias b = __type_of(y.__len__())


##===----------------------------------------------------------------------===##
# __lifetime_of
##===----------------------------------------------------------------------===##

# CHECK-LABEL: lit.func @"lifetime_of
fn lifetime_of(x: Unmovable, y: Unmovable, inout z: Unmovable):
    # CHECK-NEXT: lifetime<1> = <#lit.lifetime>
    alias lt0 = __lifetime_of()
    # CHECK-NEXT: lifetime<0> = <*"x`">
    alias lt1 = __lifetime_of(x)
    # CHECK-NEXT: lifetime<0> = <{*"x`", *"y`1"}>
    alias lt2 = __lifetime_of(x, y)
    # CHECK-NEXT: lifetime<1> = <*"z`2">
    alias lt3 = __lifetime_of(z)
    # CHECK-NEXT: lifetime<0> = <{*"x`", (mutcast mut *"z`2")}>
    alias lt4 = __lifetime_of(x, z)

##===----------------------------------------------------------------------===##
# in / not in
##===----------------------------------------------------------------------===##

# CHECK-LABEL: lit.func @"test_in
fn test_in(a: String, b: String):
    # CHECK-NEXT: lit.call {{.*}}__contains__{{.*}}(%b, %a)
    _ = a in b
    # CHECK-NEXT: [[RES:%.*]] = lit.call {{.*}}__contains__{{.*}}(%b, %a)
    # CHECK-NEXT: [[RESB:%.*]] = lit.call {{.*}}__bool__{{.*}}([[RES]])
    # CHECK-NEXT: = lit.call {{.*}}__invert__{{.*}}([[RESB]])
    _ = a not in b
