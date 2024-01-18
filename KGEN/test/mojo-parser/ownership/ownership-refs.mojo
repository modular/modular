# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Test more advanced reference cases.

# RUN: kgen-translate -import-mojo %s --mlir-print-debuginfo -o %t.mlir
# RUN: kgen-opt %t.mlir -lower-semantic-cf -check-lifetimes -verify-diagnostics | FileCheck %s

# TODO: should autoimport some day.
from memory.unsafe import Reference

# A helper to spell !lit.ref types.  Parametric aliases would be nice!
struct _LITRef[
    element_type: AnyType,
    is_mutable: __mlir_type.i1,
    lifetime: Lifetime,
    addr_space: __mlir_type.index = Int(0).__mlir_index__()
]:
    alias type = __mlir_type[
        `!lit.ref<mut=`,
        is_mutable,
        `, :`,
        AnyType,
        ` `,
        element_type,
        `, `,
        lifetime,
        `, `,
        addr_space,
        `>`,
    ]

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
fn borrow[lt: Lifetime](a: _LITRef[MemExample, False.__mlir_i1__(), lt].type):
  pass

# CHECK-LABEL: lit.func @"mutate{{.*}}"<
# CHECK-SAME: [[LT:.*_lt]][lt]: lifetime>(%a: !lit.ref<mut !MemExample, [[LT]]> borrow)
fn mutate[lt: Lifetime](a: _LITRef[MemExample, True.__mlir_i1__(), lt].type):
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
  let ref1 : _LITRef[MemExample, True.__mlir_i1__(), lt, as1].type

  # CHECK: lit.alias.decl {{.*}}_as2: !Int = <#lit.struct<{value = 42}>>
  alias as2 : Int = 42

  # CHECK: lit.varlet.decl "ref2" {{.*}}!lit.ref<!MemExample, {{.*}}_lt, apply(:!lit.signature<("self": !Int borrow) -> index> {{.*}}__mlir_index__{{.*}}, {{.*}}_as2)>
  let ref2 : _LITRef[MemExample, False.__mlir_i1__(), lt, as2.__mlir_index__()].type

# This preserves reference mutability
# CHECK-LABEL: lit.func @"parametricMut
# CHECK-SAME: (%a: !lit.ref<mut={{.*}}isMut, !MemExample, {{.*}}x18_life> borrow)
# CHECK-SAME: -> !lit.ref<mut=_{{.*}}x18_isMut, !MemExample, _{{.*}}x18_life>
fn parametricMut[isMut: __mlir_type.i1,
                 life: Lifetime](a: _LITRef[MemExample, isMut, life].type)
   -> _LITRef[MemExample, isMut, life].type:
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
  var a = MemExample()

  # CHECK: lit.call @{{.*}}__init__{{.*}}(%b)
  var b = MemExample()

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

# ===----------------------------------------------------------------------=== #
# Test that we can bind self lifetime.
# ===----------------------------------------------------------------------=== #

# Need a way to get a lifetime of Self.
# https://github.com/modularml/modular/issues/29069

struct SelfRefTest:
  fn __init__(inout self): pass

  # CHECK-LABEL: lit.func @"method
  # CHECK-SAME: (%self: !lit.ref<mut={{.*}}x27_isMut, !SelfRefTest, {{.*}}x13_lt> borrow)
  fn method[lt: Lifetime, isMut: __mlir_type.i1](
     self: _LITRef[Self, isMut, lt].type) -> Reference[Self, isMut, lt]:
      return Reference(self)

# CHECK-LABEL: lit.func @"testSelfRef
fn testSelfRef(a: SelfRefTest, inout b: SelfRefTest):
  # Bind immutably to a
  # CHECK-NEXT: = lit.call {{.*}}method{{.*}}<:lifetime *"`a", :i1 0>(%a)
  let r1 = a.method()

  # Bind mutably to b
  # CHECK: = lit.call {{.*}}method{{.*}}<:lifetime *"`b", :i1 1>(%b)
  let r2 = b.method()
