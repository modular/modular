# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Test more advanced reference cases.

# RUN: kgen-translate -import-mojo -mojo-experimental-lifetimes %s --mlir-print-debuginfo -o %t.mlir
# RUN: kgen-opt %t.mlir -lower-semantic-cf -check-lifetimes -verify-diagnostics | FileCheck %s

# TODO: should autoimport some day.
from memory.unsafe import Reference

# ===----------------------------------------------------------------------=== #
# Parsing of references
# ===----------------------------------------------------------------------=== #

# CHECK-LABEL: lit.struct.decl @MemExample
struct MemExample:
  fn __init__(inout self): pass
  fn __moveinit__(inout self, owned existing: Self): pass
  fn __copyinit__(inout self, existing: Self): pass
  fn __del__(owned self): pass
  fn noop(self): pass
  fn mutate(inout self): pass

# CHECK-LABEL: lit.func @"borrow{{.*}}"<
# CHECK-SAME: [[LT:.*_lt]][lt]: lifetime>(%a: !lit.ref<!MemExample, [[LT]]> borrow)
fn borrow[lt: Lifetime](a: ref[lt] MemExample):
  pass

# CHECK-LABEL: lit.func @"mutate{{.*}}"<
# CHECK-SAME: [[LT:.*_lt]][lt]: lifetime>(%a: !lit.ref<mut !MemExample, [[LT]]> borrow)
fn mutate[lt: Lifetime](a: mutref[lt] MemExample):
  pass

# CHECK-LABEL: lit.func @"implicit_borrow
fn implicit_borrow(a: MemExample):
  pass

# CHECK-LABEL: lit.func @"implicit_inout
fn implicit_inout(inout a: MemExample):
  pass

# CHECK-LABEL: lit.func @"implicit_owned
fn implicit_owned(owned a: MemExample):
  pass

# CHECK-LABEL: lit.func @"addrSpaces
fn addrSpaces[lt: Lifetime, as1: __mlir_type.index]():
  # CHECK: lit.varlet.decl "ref1" {{.*}} !lit.ref<mut !MemExample, {{.*}}_lt, {{.*}}_as1>
  let ref1 : mutref[lt, as1] MemExample

  # CHECK: lit.alias.decl {{.*}}_as2: !Int = <#lit.struct<{value = 42}>>
  alias as2 : Int = 42

  # CHECK: lit.varlet.decl "ref2" {{.*}}!lit.ref<!MemExample, {{.*}}_lt, apply(:!lit.signature<("self": !Int borrow) -> index> @"$stdlib"::@"$builtin"::@"$int"::@Int::@"__mlir_index__($stdlib::$builtin::$int::Int)", apply(:!lit.signature<("self": !Int borrow) -> !Int> @"$stdlib"::@"$builtin"::@"$int"::@Int::@"__index__($stdlib::$builtin::$int::Int)", {{.*}}_as2))>
  let ref2 : ref[lt, as2] MemExample

# This preserves reference mutability
# CHECK-LABEL: lit.func @"parametricMut
# CHECK-SAME: (%a: !lit.ref<mut={{.*}}isMut, !MemExample, {{.*}}x18_life> borrow)
# CHECK-SAME: -> !lit.ref<mut=_{{.*}}x18_isMut, !MemExample, _{{.*}}x18_life>
fn parametricMut[isMut: __mlir_type.i1,
                 life: Lifetime](a: ref[mut=isMut, life] MemExample)
   -> ref[mut=isMut, life] MemExample:
  return a

# CHECK-LABEL: lit.func @"testParametricMut
fn testParametricMut(i: MemExample, inout m: MemExample):
  # This infers an immutable reference.
  # CHECK: [[RES:%.*]] = lit.call {{.*}}parametricMut
  # CHECK-NEXT: %iRef = lit.letreg.decl "iRef" = [[RES]] : !lit.ref<!MemExample, *"`i">
  let iRef = parametricMut(Reference(i).value)

  # This infers a mutable reference.
  # CHECK: [[RES:%.*]] = lit.call {{.*}}parametricMut
  # CHECK: %mRef = lit.letreg.decl "mRef" = [[RES]] : !lit.ref<mut !MemExample, *"`m">
  let mRef = parametricMut(Reference(m).value)

##===----------------------------------------------------------------------===##
# Conditional lifetimes
##===----------------------------------------------------------------------===##

# CHECK-LABEL: lit.func @"testUseConditional
fn testUseConditional(cond: __mlir_type.i1):
  # CHECK-NOT: __del__

  # CHECK: lit.call @{{.*}}__init__{{.*}}(%a)
  let a = MemExample()

  # CHECK: lit.call @{{.*}}__init__{{.*}}(%b)
  let b = MemExample()

  let aref = Reference(a).value
  let bref = Reference(b).value

  # CHECK: %cref = lit.letreg.decl "cref"
  let cref = aref if cond else bref

  # This uses both A and B, so it needs to extend both of their lifetimes.
  Reference(cref)[].noop()
  # CHECK-NEXT: [[REF:%.*]] = lit.call @{{.*}}@Reference::@"__init__{{.*}}(%cref)
  # CHECK-NEXT: [[MREF:%.*]] = lit.call @{{.*}}__refitem__{{.*}}([[REF]])
  # CHECK-NEXT: lit.ref.immut [[MREF]]
  # CHECK-NEXT: lit.call @{{.*}}noop
  # CHECK-NEXT: lit.call @{{.*}}__del__{{.*}}(%a)
  # CHECK-NEXT: lit.call @{{.*}}__del__{{.*}}(%b)

# CHECK-LABEL: lit.func @"testDefConditional
fn testDefConditional(cond: __mlir_type.i1):
  # CHECK-NOT: __del__

  # CHECK: lit.call @{{.*}}__init__{{.*}}(%a)
  var a = MemExample()

  # CHECK: lit.call @{{.*}}__init__{{.*}}(%b)
  var b = MemExample()

  let aref = Reference(a).value
  let bref = Reference(b).value

  # CHECK: %cref = lit.letreg.decl "cref"
  let cref = aref if cond else bref

  # Mutating either of these is fine - it doesn't matter which one is mutated,
  # we know that both are live.
  Reference(cref)[].mutate()
  # CHECK-NEXT: [[REF:%.*]] = lit.call @{{.*}}@Reference::@"__init__{{.*}}(%cref)
  # CHECK-NEXT: [[MREF:%.*]] = lit.call @{{.*}}__refitem__{{.*}}([[REF]])
  # CHECK-NEXT: lit.call @{{.*}}mutate{{.*}}([[MREF]])

  # Overwriting one means that we need to immediately destroy the same reference
  # because we cannot know which one is being set.
  Reference(cref)[] = MemExample()
  # CHECK-NEXT: [[REF:%.*]] = lit.call @{{.*}}@Reference::@"__init__{{.*}}(%cref)
  # CHECK-NEXT: [[MREF:%.*]] = lit.call @{{.*}}__refitem__{{.*}}([[REF]])
  # CHECK-NEXT: lit.call @{{.*}}__del__{{.*}}([[MREF]])
  # CHECK-NEXT: lit.call @{{.*}}__init__{{.*}}([[MREF]])

  # Overwriting is eligible for copy => move optimization as well.
  let shouldBeMovedFrom = MemExample()
  Reference(cref)[] = shouldBeMovedFrom
  # CHECK: lit.call @{{.*}}__init__{{.*}}(%shouldBeMovedFrom)
  # CHECK-NEXT: [[REF:%.*]] = lit.call @{{.*}}@Reference::@"__init__{{.*}}(%cref)
  # CHECK-NEXT: [[MREF:%.*]] = lit.call @{{.*}}__refitem__{{.*}}([[REF]])
  # CHECK-NEXT: lit.ref.immut
  # CHECK-NEXT: lit.call @{{.*}}__del__{{.*}}([[MREF]])
  # CHECK-NEXT: lit.call @{{.*}}__moveinit__{{.*}}([[MREF]], %shouldBeMovedFrom)

  # The mutation above could either of A or B, so we needed to extend both of
  # their lifetimes, but now we can say goodbye.
  # CHECK-NEXT: lit.call @{{.*}}__del__{{.*}}(%a)
  # CHECK-NEXT: lit.call @{{.*}}__del__{{.*}}(%b)

# ===----------------------------------------------------------------------=== #
# Tests of the Reference type.
# ===----------------------------------------------------------------------=== #

# CHECK-LABEL: lit.func @"testUseConditionalReference

fn testUseConditionalReference(cond: __mlir_type.i1, imm: MemExample):
  # CHECK: %a = lit.varlet.decl {{.*}} : !lit.ref<mut !MemExample, *"`a0">
  # CHECK: lit.call @{{.*}}__init__{{.*}}(%a)

  var a = MemExample()

  # CHECK-NEXT: [[ARV:%.*]] = lit.call @{{.*}}@Reference::@"__init__{{.*}}(%a)

  # CHECK-NEXT: %aref = lit.letreg.decl "aref" = [[ARV]]
  let aref = Reference(a)
  # CHECK-NEXT: lit.alias.decl {{.*}}_aLifetime: lifetime = <*"`a0">
  alias aLifetime =  aref.lifetime

  # CHECK-NEXT: [[LITREF:%.*]] = lit.call {{.*}}__refitem__{{.*}}(%aref)
  aref[].noop()
  # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut [[LITREF]]
  # CHECK-NEXT: lit.call {{.*}}noop{{.*}}([[IMMREF]])

  # This is a mutable reference so go head and store through it whynot?
  # CHECK-NEXT: [[LITREF:%.*]] = lit.call {{.*}}__refitem__{{.*}}(%aref)
  aref[] = MemExample()
  # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}([[LITREF]])
  # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}([[LITREF]])

  # Ok, this was the last use of A so it can go away.
  # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}(%a)

  # The reference being alive doesn't keep the underlying stuff alive, only
  # accesses
  # CHECK-NEXT: %aref2 = lit.letreg.decl "aref2" = %aref
  let aref2 = aref

  # Reference can bind to immutable things as well, no problem.
  # CHECK-NEXT: [[IMMRV:%.*]] = lit.call @{{.*}}@Reference::@"__init__{{.*}}(%imm)
  # CHECK-NEXT: %immref = lit.letreg.decl "immref" = [[IMMRV]]
  let immref = Reference(imm)
  immref[].noop()
