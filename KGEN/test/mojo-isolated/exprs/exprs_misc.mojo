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
# Assignment operator
##===----------------------------------------------------------------------===##

fn test_rhs_inference():
    var a: List[Int]

    # None of these should be ambiguous.
    a = []
    (a) = []


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
    # CHECK-NEXT: [[SLICE:%.*]] = lit.call {{.*}}StringSlice::@"__init__{{.*}}(%a)
    # CHECK-NEXT: lit.call {{.*}}__contains__{{.*}}(%b, [[SLICE]])
    _ = a in b
    # CHECK-NEXT: [[SLICE:%.*]] = lit.call {{.*}}StringSlice::@"__init__{{.*}}(%a)
    # CHECK-NEXT: [[RES:%.*]] = lit.call {{.*}}__contains__{{.*}}(%b, [[SLICE]])
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

  # String literals should merge.
  var _ss: StaticString = "T" if cond else "F"

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

##===----------------------------------------------------------------------===##
# Chained comparisons.
##===----------------------------------------------------------------------===##

# CHECK-LABEL: lit.fn @"chained_cmp
fn chained_cmp(a: Int, b: Int, c: Int, d: Int, e: Int):
    # CHECK-NEXT: %res = lit.var.decl "res"
    # CHECK:      [[CMP_A_B:%.*]] = lit.call @{{.*}}__lt__{{.*}}(%a, %b)
    # CHECK-NEXT: %[[CMP_A_B_I1:.*]] = lit.call @{{.*}}__mlir_i1__{{.*}}([[CMP_A_B]])
    # CHECK-NEXT: %[[IF_A_B:.*]] = hlcf.if %[[CMP_A_B_I1]]
    # CHECK-NEXT:   %[[CMP_B_C:.*]] = lit.call @{{.*}}__lt__{{.*}}(%b, %c)
    # CHECK:        %[[IF_B_C:.*]] = hlcf.if
    # CHECK-NEXT:     %[[CMP_C_D:.*]] = lit.call @{{.*}}__lt__{{.*}}(%c, %d)
    # CHECK-NEXT:     hlcf.yield %[[CMP_C_D]]
    # CHECK-NEXT:   } else {
    # CHECK-NEXT:     hlcf.yield %[[CMP_B_C]]
    # CHECK-NEXT:   }
    # CHECK-NEXT:   hlcf.yield %[[IF_B_C]]
    # CHECK-NEXT: } else {
    # CHECK-NEXT:   hlcf.yield [[CMP_A_B]]
    # CHECK-NEXT: }
    # CHECK-NEXT: lit.ref.store %[[IF_A_B]], %res
    var res = a < b < c < d

    # COM: This checks the parsing precedence between `<` and `and`.
    # CHECK:      %[[CMP_A_B:.*]] = lit.call @{{.*}}__lt__{{.*}}(%a, %b)
    # CHECK:       %[[CMP_A_B_I1:.*]] = lit.call @{{.*}}__mlir_i1__{{.*}}(%[[CMP_A_B]])
    # CHECK-NEXT: %[[IF_A_B:.*]] = hlcf.if %[[CMP_A_B_I1]]
    # CHECK:   %[[CMP_B_C:.*]] = lit.call @{{.*}}__lt__{{.*}}(
    # CHECK-NEXT:   hlcf.yield %[[CMP_B_C]]
    # CHECK-NEXT: } else {
    # CHECK-NEXT:   hlcf.yield %[[CMP_A_B]]
    # CHECK-NEXT: }
    # CHECK-NEXT: %[[CMP_I1:.*]] = lit.call @{{.*}}__mlir_i1__{{.*}}(%[[IF_A_B]])
    # CHECK-NEXT: %[[IF:.*]] = hlcf.if %[[CMP_I1]]
    # CHECK-NEXT:   %[[CMP_D_E:.*]] = lit.call @{{.*}}__lt__{{.*}}(%d, %e)
    # CHECK-NEXT:   hlcf.yield %[[CMP_D_E]]
    # CHECK-NEXT: } else {
    # CHECK-NEXT:   hlcf.yield %[[IF_A_B]]
    # CHECK-NEXT: }
    # CHECK-NEXT: lit.ref.store %[[IF]], %res
    res = a < b < c and d < e

# Test chained comparison op in parameter domain for issue
# https://github.com/modularml/modular/issues/22050
# CHECK: lit.alias.decl *"chainedCmpAlias1{{.*}}": !Bool ={{.*}}{:i1 0}
alias chainedCmpAlias1 = 1 == 2 == 3 == 4 == 5
# CHECK: lit.alias.decl *"chainedCmpAlias2{{.*}}": !Bool ={{.*}}{:i1 1}
alias chainedCmpAlias2 = 1 <= 2 <= 3 <= 4 <= 5
# CHECK: lit.alias.decl *"chainedCmpAlias3{{.*}}": !Bool ={{.*}}{:i1 0}
alias chainedCmpAlias3 = 1 <= 2 <= 9 <= 4 <= 5
fn chainedCmpSemiDyn(x: Int, a: Int, b: Int, c: Int):
  # CHECK: [[XCMP:%.*]] = lit.var.decl "xCmp"
  # CHECK-NEXT: [[IFCOND:%.*]] = kgen.param.constant: i1 = <1>
  # CHECK-NEXT: [[FINALRESULT:%.*]] = hlcf.if [[IFCOND]] -> !Bool {
  # CHECK-NEXT:   [[PV:%.*]] = {{.*}}constant{{.*}}77
  # CHECK-NEXT:   [[CMPRESULT1:%.*]] = {{.*}}__lt__{{.*}}([[PV]], %x)
  # CHECK-NEXT:   [[IFCOND:%.*]] = {{.*}}__mlir_i1__{{.*}}([[CMPRESULT1]])
  # CHECK-NEXT:   [[INNERRESULT:%.*]] = hlcf.if [[IFCOND]] -> !Bool {
  # CHECK-NEXT:     [[PV:%.*]] = {{.*}}constant{{.*}}105
  # CHECK-NEXT:     [[CMPRESULT2:%.*]] = {{.*}}__lt__{{.*}}(%x, [[PV]])
  # CHECK-NEXT:     [[IFCOND:%.*]] = {{.*}}__mlir_i1__{{.*}}([[CMPRESULT2]])
  # CHECK-NEXT:     [[MOSTINNERRESULT:%.*]] = hlcf.if [[IFCOND]] -> !Bool {
  # CHECK-NEXT:       [[TRUEPARAM:%.*]] = kgen.param.constant: !Bool = {{.*}}{:i1 1}
  # CHECK-NEXT:       hlcf.yield [[TRUEPARAM]]
  # CHECK-NEXT:     } else {
  # CHECK-NEXT:       hlcf.yield [[CMPRESULT2]]
  # CHECK-NEXT:     }
  # CHECK-NEXT:     hlcf.yield [[MOSTINNERRESULT]]
  # CHECK-NEXT:   } else {
  # CHECK-NEXT:     hlcf.yield [[CMPRESULT1]]
  # CHECK-NEXT:   }
  # CHECK-NEXT:   hlcf.yield [[INNERRESULT]]
  # CHECK-NEXT: } else {
  # CHECK-NEXT:   [[TRUEPARAM:%.*]] = kgen.param.constant: !Bool = {{.*}}{:i1 1}
  # CHECK-NEXT:   hlcf.yield [[TRUEPARAM]]
  # CHECK-NEXT: }
  # CHECK-NEXT: lit.ref.store [[FINALRESULT]], [[XCMP]]
  var xCmp = 5 < 77 < x < 105 < 177
  # A fully deep check of this would be a lot of work, but this at least
  # shows that its not choking during parsing on a mix of dynamic and
  # parameter comparisons.  It required some care with the interaction
  # between recursive calls of emitNextCmp calls to get this to work.
  var mixedChain = 0 < 1 < a < 10 < 11 < b < 20 < 21 < c < 30 < 31

##===----------------------------------------------------------------------===##
# or/and
##===----------------------------------------------------------------------===##


# MOCO-1987: Parser error when temporary PythonObject appears in or expression
@register_passable
struct RPType(Copyable, Movable):
  fn __init__(out self): pass

  fn __bool__(self) -> Bool:
      return Bool()

# CHECK-LABEL: lit.fn @"test_rp_and_or
fn test_rp_and_or():
  # Evaluate the LHS, but materialize the rvalue into a memory slot.

  # CHECK-NEXT: [[LHS:%.*]] = lit.call {{.*}}RPType::@"__init__()
  # CHECK-NEXT: [[TMPMEM:%.*]] = lit.var.decl "anonymous
  # CHECK-NEXT: lit.ref.store [[LHS]], [[TMPMEM]]

  # CHECK-NEXT: [[IMMTMP:%.*]] = lit.ref.immut [[TMPMEM]]
  # CHECK-NEXT: lit.call {{.*}}RPType::@"__bool__{{.*}}([[IMMTMP]])
  # CHECK:      hlcf.if
  # CHECK-NEXT:     [[LHS:%.*]] = lit.load.consume [[TMPMEM]]
  # CHECK-NEXT:     hlcf.yield [[LHS]] : !RPType

  _ = RPType() or RPType()
