# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Test more advanced reference cases.

# RUN: kgen-translate -import-mojo %s --mlir-print-debuginfo -o %t.mlir
# RUN: kgen-opt %t.mlir -lower-semantic-cf -check-lifetimes -verify-diagnostics | FileCheck %s

# TODO: should autoimport some day.
from memory.unsafe import Reference, _LITRef

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

# CHECK-LABEL: lit.func @"borrow{{.*}}"<lt: lifetime<0>>(%a: !lit.ref<!MemExample, imm lt> borrow)
fn borrow[lt: ImmLifetime](a: _LITRef[MemExample, False.__mlir_i1__(), lt].type):
  pass

# CHECK-LABEL: lit.func @"mutate{{.*}}"<lt: lifetime<1>>(%a: !lit.ref<!MemExample, mut lt> borrow)
fn mutate[lt: MutLifetime](a: _LITRef[MemExample, True.__mlir_i1__(), lt].type):
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
fn addrSpaces[lt1: MutLifetime, lt2: ImmLifetime, as1: __mlir_type.index]():
  # CHECK: lit.var.decl "ref1" {{.*}}!lit.ref<!MemExample, mut lt1, as1>
  var ref1 : _LITRef[MemExample, True.__mlir_i1__(), lt1, as1].type

  # CHECK: lit.alias.decl [[AS2:.*]]: !Int = <{42}>
  alias as2 : Int = 42

  # CHECK: lit.var.decl "ref2" {{.*}}!lit.ref<!MemExample, imm lt2, apply(:!lit.signature<("self": !Int borrow) -> index> {{.*}}__mlir_index__{{.*}}, [[AS2]])>
  var ref2 : _LITRef[MemExample, False.__mlir_i1__(), lt2, as2.__mlir_index__()].type

# This preserves reference mutability
# CHECK-LABEL: lit.func @"parametricMut
# CHECK-SAME: (%a: !lit.ref<!MemExample, mut=isMut, life> borrow)
# CHECK-SAME: -> !lit.ref<!MemExample, mut=isMut, life>
fn parametricMut[isMut: __mlir_type.i1,
                 life: AnyLifetime[isMut].type](a: _LITRef[MemExample, isMut, life].type)
   -> _LITRef[MemExample, isMut, life].type:
  return a

# CHECK-LABEL: lit.func @"testParametricMut
fn testParametricMut(i: MemExample, inout m: MemExample):
  # This infers an immutable reference.
  # CHECK:  lit.call {{.*}}parametricMut{{.*}}!lit.ref<!MemExample, imm *"i`">
  _ = parametricMut(Reference(i).value)

  # This infers a mutable reference.
  # CHECK: lit.call {{.*}}parametricMut{{.*}}!lit.ref<!MemExample, mut *"m`1">
  _ = parametricMut(Reference(m).value)

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

  var aref = Reference(a).value
  var bref = Reference(b).value

  # CHECK: %cref = lit.var.decl "cref"
  var cref = aref if cond else bref

  # This uses both A and B, so it needs to extend both of their lifetimes.
  Reference(cref)[].noop()
  # CHECK: [[CR:%.*]] = lit.ref.load %cref
  # CHECK-NEXT: [[REF:%.*]] = lit.call @{{.*}}@Reference::@"__init__{{.*}}([[CR]])
  # CHECK-NEXT: [[MREF:%.*]] = lit.call @{{.*}}__refitem__{{.*}}([[REF]])
  # CHECK-NEXT: lit.ref.immut [[MREF]]
  # CHECK-NEXT: lit.call @{{.*}}noop
  # CHECK-NEXT: lit.call @{{.*}}__del__{{.*}}(%a)
  # CHECK-NEXT: lit.call @{{.*}}__del__{{.*}}(%b)

# CHECK-LABEL: lit.func @"testDefConditional
fn testDefConditional(cond: __mlir_type.i1):
  # CHECK-NOT: lit.call {{[^)]*}}__del__

  var a = MemExample()

  var b = MemExample()

  var aref = Reference(a).value
  var bref = Reference(b).value

  # CHECK: %cref = lit.var.decl "cref"
  var cref = aref if cond else bref

  # Mutating either of these is fine - it doesn't matter which one is mutated,
  # we know that both are live.
  Reference(cref)[].mutate()
  # CHECK: [[CR:%.*]] = lit.ref.load %cref
  # CHECK-NEXT: [[REF:%.*]] = lit.call @{{.*}}@Reference::@"__init__{{.*}}([[CR]])
  # CHECK-NEXT: [[MREF:%.*]] = lit.call @{{.*}}__refitem__{{.*}}([[REF]])
  # CHECK-NEXT: lit.call @{{.*}}mutate{{.*}}([[MREF]])

  # Overwriting one means that we need to immediately destroy the same reference
  # because we cannot know which one is being set.
  Reference(cref)[] = MemExample()
  # CHECK: [[CR:%.*]] = lit.ref.load %cref
  # CHECK-NEXT: [[REF:%.*]] = lit.call @{{.*}}@Reference::@"__init__{{.*}}([[CR]])
  # CHECK-NEXT: [[MREF:%.*]] = lit.call @{{.*}}__refitem__{{.*}}([[REF]])
  # CHECK-NEXT: lit.call @{{.*}}__del__{{.*}}([[MREF]])
  # CHECK-NEXT: lit.call @{{.*}}__init__{{.*}}([[MREF]])

  # Overwriting is eligible for copy => move optimization as well.
  var shouldBeMovedFrom = MemExample()
  Reference(cref)[] = shouldBeMovedFrom
  # CHECK: lit.call @{{.*}}__init__{{.*}}(%shouldBeMovedFrom)
  # CHECK: [[CR:%.*]] = lit.ref.load %cref
  # CHECK-NEXT: [[REF:%.*]] = lit.call @{{.*}}@Reference::@"__init__{{.*}}([[CR]])
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
  # CHECK: %a = lit.var.decl {{.*}} : !lit.ref<!MemExample, mut *"a`1">
  # CHECK: lit.call @{{.*}}__init__{{.*}}(%a)

  var a = MemExample()

  # CHECK-NEXT: %aref = lit.var.decl "aref"
  # CHECK-NEXT: [[ARV:%.*]] = lit.call @{{.*}}@Reference::@"__init__{{.*}}(%a)
  # CHECK-NEXT: lit.ref.store [[ARV]], %aref
  var aref = Reference(a)
  # CHECK-NEXT: lit.alias.decl *"aLifetime{{.*}}": lifetime<1> = <*"a`1">
  alias aLifetime =  aref.lifetime

  # CHECK-NEXT: [[AR:%.*]] = lit.ref.load %aref
  # CHECK-NEXT: [[LITREF:%.*]] = lit.call {{.*}}__refitem__{{.*}}([[AR]])
  aref[].noop()
  # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut [[LITREF]]
  # CHECK-NEXT: lit.call {{.*}}noop{{.*}}([[IMMREF]])

  # This is a mutable reference so go head and store through it whynot?
  # CHECK-NEXT: [[AR:%.*]] = lit.ref.load %aref
  # CHECK-NEXT: [[LITREF:%.*]] = lit.call {{.*}}__refitem__{{.*}}([[AR]])
  aref[] = MemExample()
  # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}([[LITREF]])
  # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}([[LITREF]])

  # Ok, this was the last use of A so it can go away.
  # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}(%a)

  # The reference being alive doesn't keep the underlying stuff alive, only
  # accesses
  # CHECK-NEXT: %aref2 = lit.var.decl "aref2"
  # CHECK-NEXT: [[AR:%.*]] = lit.ref.load %aref
  # CHECK-NEXT: lit.ref.store [[AR]], %aref2
  var aref2 = aref

  # Reference can bind to immutable things as well, no problem.
  # CHECK-NEXT: %immref = lit.var.decl "immref"
  # CHECK-NEXT: [[IMMRV:%.*]] = lit.call @{{.*}}@Reference::@"__init__{{.*}}(%imm)
  # CHECK-NEXT: lit.ref.store [[IMMRV]], %immref
  var immref = Reference(imm)
  immref[].noop()

# ===----------------------------------------------------------------------=== #
# Test that we can bind self lifetime.
# ===----------------------------------------------------------------------=== #

# Need a way to get a lifetime of Self.
# https://github.com/modularml/modular/issues/29069

struct SelfRefTest:
  fn __init__(inout self): pass

  # CHECK-LABEL: lit.func @"method
  # CHECK-SAME: (%self: !lit.ref<!SelfRefTest, mut=isMut, lt> borrow)
  fn method[isMut: __mlir_type.i1, lt: AnyLifetime[isMut].type](
     self: _LITRef[Self, isMut, lt].type) -> Reference[Self, isMut, lt]:
      return Reference(self)

# CHECK-LABEL: lit.func @"testSelfRef
fn testSelfRef(a: SelfRefTest, inout b: SelfRefTest):
  # Bind immutably to a
  # CHECK: = lit.call {{.*}}method{{.*}}:lifetime<0> *"a`">(%a)
  _ = a.method()

  # Bind mutably to b
  # CHECK: = lit.call {{.*}}method{{.*}}:lifetime<1> *"b`1">(%b)
  _ = b.method()


# CHECK-LABEL: lit.func @"testLifetimeOf1
# CHECK-SAME: (%a: !lit.ref<!MemExample, imm *"a`"> borrow_in_mem) ->
# CHECK-SAME: Reference <{{.*}}, :i1 0, :lifetime<0> *"a`">
fn testLifetimeOf1(a: MemExample) ->
  Reference[MemExample, __mlir_attr.`0: i1`, __lifetime_of(a)]:
  return a

# CHECK-LABEL: lit.func @"testLifetimeOf2
# CHECK-SAME: (%a: !lit.ref<!MemExample, imm *"a`"> borrow_in_mem) ->
# CHECK-SAME: !lit.ref<!MemExample, imm *"a`">
fn testLifetimeOf2(a: MemExample) -> _LITRef[
        MemExample,  __mlir_attr.`0: i1`, __lifetime_of(a)].type:

  # CHECK: kgen.return {{.*}} : !lit.ref<!MemExample, imm *"a`">
  return Reference(a).value

# CHECK-LABEL: lit.func @"callByRefResultLifetime
fn callByRefResultLifetime(inout x: MemExample, inout y: MemExample, z: MemExample):
  # CHECK: lit.var.decl "l1" var : !lit.ref<@"ownership-refs"::@OneLifetime<:lifetime<0> (mutcast mut *"x`")>
  var l1 = returnOneArgLifetime(x)

  # CHECK: lit.var.decl "l2" var : !lit.ref<@"ownership-refs"::@TwoLifetimes<:lifetime<0> (mutcast mut *"x`"), :lifetime<0> (mutcast mut *"y`1")>
  var l2 = returnTwoArgLifetimes(x, y)
  # CHECK: %l3 = lit.var.decl "l3" var : !lit.ref<@"ownership-refs"::@TwoLifetimes<:lifetime<0> (mutcast mut *"x`"), :lifetime<0> (mutcast mut *"x`")>
  var l3 = returnTwoArgLifetimes(x, x)
  # CHECK: %l4 = lit.var.decl "l4" var : !lit.ref<@"ownership-refs"::@TwoLifetimes<:lifetime<0> *"z`2", :lifetime<0> *"z`2">
  var l4 = returnTwoArgLifetimes(z, z)

fn returnOneArgLifetime(a: MemExample)
  -> OneLifetime[__lifetime_of(a)]:
  return OneLifetime[__lifetime_of(a)]()

fn returnTwoArgLifetimes(a: MemExample, b: MemExample)
  -> TwoLifetimes[__lifetime_of(a), __lifetime_of(b)]:
  return TwoLifetimes[__lifetime_of(a), __lifetime_of(b)]()

struct OneLifetime[a_lifetime: ImmLifetime]:
  fn __init__(inout self): pass

struct TwoLifetimes[a_lifetime: ImmLifetime,
                    b_lifetime: ImmLifetime]:
  fn __init__(inout self): pass
