# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s -verify-diagnostics | FileCheck %s
struct Unmovable:
    fn __init__(out self):
        pass


fn throwing_fn() raises -> Int:
    return 0


fn literal_promotion[cond: Bool]():
    # This needs to coerce to the materialization type of float literal
    alias a = 2.0 if cond else 3



##===----------------------------------------------------------------------===##
# Test return slot optimization
##===----------------------------------------------------------------------===##


# NOTE: Don't remove this argument, this was defeating return slot opzn.
fn getUnmovable(a: Unmovable) -> Unmovable:
    return Unmovable()


# This can only be codegen'd directly into x.
# CHECK-LABEL: lit.fn @"testUnmovable
fn testUnmovable(a: Unmovable):
    # CHECK-NEXT: %x = lit.var.decl "x"
    # CHECK-NEXT: lit.call {{.*}}(%a, %x)
    var x: Unmovable = getUnmovable(a)


##===----------------------------------------------------------------------===##
# __type_of
##===----------------------------------------------------------------------===##

alias index = __mlir_type.index


# CHECK-LABEL: lit.fn @"simple_typeof_return(
# CHECK: __mlir_type.index)"(%x: index) -> index
fn simple_typeof_return(x: index) -> __type_of(x):
    return x


# CHECK-LABEL: lit.fn @"typeof_arg(
# CHECK: __mlir_type.index,__mlir_type.index)"(%x: index, %y: index) -> index
fn typeof_arg(x: index, y: __type_of(x)) -> index:
    var z: __type_of(x) = y
    return z


# CHECK-LABEL: lit.fn @"typeof_dynval_in_param(
fn typeof_dynval_in_param(x: index):
    # CHECK-NEXT:  %y = lit.var.decl
    # CHECK-NEXT: lit.call {{.*}}String::@"__init__
    var y = String()

    # CHECK-NEXT: lit.alias.decl *"a`1": type = <index>
    alias a = __type_of(x)
    # CHECK-NEXT: lit.alias.decl *"b`2": !mt_Int = <!Int>
    alias b = __type_of(y.__len__())

    # CHECK-NEXT: lit.alias.decl *"c`3": !mt_Int = <!Int>
    alias c = __type_of(throwing_fn())


##===----------------------------------------------------------------------===##
# __origin_of
##===----------------------------------------------------------------------===##


# CHECK-LABEL: lit.fn @"lifetime_of
fn lifetime_of(x: Unmovable, y: Unmovable, mut z: Unmovable):
    # CHECK-NEXT: origin<0> = <{}>
    alias lt0 = __origin_of()
    # CHECK-NEXT: origin<0> = <*"x`">
    alias lt1 = __origin_of(x)
    # CHECK-NEXT: origin<0> = <{*"x`", *"y`1"}>
    alias lt2 = __origin_of(x, y)
    # CHECK-NEXT: origin<1> = <*"z`2">
    alias lt3 = __origin_of(z)
    # CHECK-NEXT: origin<0> = <{*"x`", (mutcast mut *"z`2")}>
    alias lt4 = __origin_of(x, z)


##===----------------------------------------------------------------------===##
# in / not in
##===----------------------------------------------------------------------===##


# CHECK-LABEL: lit.fn @"test_in
fn test_in(a: String, b: String):
    # CHECK-NEXT: lit.call {{.*}}__contains__{{.*}}(%b, %a)
    _ = a in b
    # CHECK-NEXT: [[RES:%.*]] = lit.call {{.*}}__contains__{{.*}}(%b, %a)
    # CHECK-NEXT: [[RESB:%.*]] = lit.call {{.*}}__bool__{{.*}}([[RES]])
    # CHECK-NEXT: = lit.call {{.*}}__invert__{{.*}}([[RESB]])
    _ = a not in b


##===----------------------------------------------------------------------===##
# String literals
##===----------------------------------------------------------------------===##

# CHECK-LABEL: lit.fn @"test_string_literal1
fn test_string_literal1(cond: Bool):
  _ = 4

  # String literals should be fine at start of expression.
  # expected-warning @+1 {{'Bool' value is unused}}
  "a" == "abc" 


##===----------------------------------------------------------------------===##
# MergeWith
##===----------------------------------------------------------------------===##

@register_passable("trivial")
struct TypeA:
    fn __merge_with__[other_type: __type_of(TypeB)](self) -> TypeB:
        pass
    fn __merge_with__[other_type: __type_of(TypeC)](self) -> Int:
        pass

@register_passable("trivial")
struct TypeB:
    fn __merge_with__[other_type: __type_of(Int)](self) -> Int:
        pass

@register_passable("trivial")
struct TypeC:
    fn __merge_with__[other_type: __type_of(TypeA)](self) -> Int:
        pass
    fn __merge_with__[other_type: __type_of(TypeD)](self) -> TypeE:
        pass


@register_passable("trivial")
struct TypeD:
    fn __merge_with__[other_type: __type_of(TypeA)](self) -> Int:
        pass

@register_passable("trivial")
struct TypeE:
    @implicit
    fn __init__(out self, other: TypeD):
        pass




# CHECK-LABEL: lit.fn @"test_mergewith
fn test_mergewith(cond: __mlir_type.i1, a: TypeA, b: TypeB, c: TypeC, d: TypeD):

  # One merges to the other.
  _ = a if cond else b
  # CHECK: hlcf.if %cond
  # CHECK-NEXT:   [[ARES:%.*]] = lit.call {{.*}}TypeA::@"__merge_with__
  # CHECK-NEXT:   hlcf.yield [[ARES]]
  # CHECK-NEXT: } else {
  # CHECK-NEXT:   hlcf.yield %b
  # CHECK-NEXT: }
  
  # This merge with two merge_with
  _ = a if cond else c
  # CHECK: hlcf.if %cond
  # CHECK:   [[ARES:%.*]] = lit.call {{.*}}TypeA::@"__merge_with__
  # CHECK:   hlcf.yield [[ARES]]
  # CHECK: } else {
  # CHECK:   [[CRES:%.*]] = lit.call {{.*}}TypeC::@"__merge_with__
  # CHECK:   hlcf.yield [[CRES]]
  # CHECK: }

  # One merge and one implicit conversion.
  _ = c if cond else d
  # CHECK: hlcf.if %cond
  # CHECK:   [[CRES:%.*]] = lit.call {{.*}}TypeC::@"__merge_with__
  # CHECK:   hlcf.yield [[CRES]]
  # CHECK: } else {
  # CHECK:   [[ARES:%.*]] = lit.call {{.*}}TypeE::@"__init__
  # CHECK:   hlcf.yield [[ARES]]
  # CHECK: }
