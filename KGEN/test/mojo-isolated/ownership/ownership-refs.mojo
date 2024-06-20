# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Test more advanced reference cases.

# RUN: %parse-mojo-isolated %s --mlir-print-debuginfo -o %t.mlir
# RUN: kgen-opt %t.mlir -lower-semantic-cf -check-lifetimes -verify-diagnostics | FileCheck %s

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
fn borrow[lt: ImmutableLifetime](a: Reference[MemExample, lt]._mlir_type):
  pass

# CHECK-LABEL: lit.func @"mutate{{.*}}"<lt: lifetime<1>>(%a: !lit.ref<!MemExample, mut lt> borrow)
fn mutate[lt: MutableLifetime](a: Reference[MemExample, lt]._mlir_type):
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
fn addrSpaces[lt1: MutableLifetime, lt2: ImmutableLifetime, as1: AddressSpace]():
  # CHECK: lit.var.decl "ref1" {{.*}}!lit.ref<!MemExample, mut lt1, #lit.struct.extract<:!Int #lit.struct.extract<:!AddressSpace as1, "_value">, "value">>
  var ref1 : Reference[MemExample, lt1, as1]._mlir_type

  # CHECK: lit.alias.decl [[AS2:.*]]: !AddressSpace = {{.*}} {42}
  alias as2: AddressSpace = AddressSpace(42)

  # CHECK: lit.var.decl "ref2" {{.*}}!lit.ref<!MemExample, imm lt2,{{.*}}!AddressSpace [[AS2]], "_value">
  var ref2 : __mlir_type[`!lit.ref<`, MemExample, `, `, lt2, `, `, as2._value.value, `>`]

# This preserves reference mutability
# CHECK-LABEL: lit.func @"parametricMut
# CHECK-SAME: (%a: !lit.ref<!MemExample, mut=#lit.struct.extract<:!Bool isMut, "value">, life> borrow)
# CHECK-SAME: -> !lit.ref<!MemExample, mut=#lit.struct.extract<:!Bool isMut, "value">, life>
fn parametricMut[isMut: Bool,
                 life: AnyLifetime[isMut].type](a: Reference[MemExample, life]._mlir_type)
   -> Reference[MemExample, life]._mlir_type:
  return a

# CHECK-LABEL: lit.func @"testParametricMut
fn testParametricMut(i: MemExample, inout m: MemExample):
  # This infers an immutable reference.
  # CHECK:  lit.call {{.*}}parametricMut{{.*}}!lit.ref<!MemExample, imm *"i`">
  _ = parametricMut(__get_mvalue_as_litref(i))

  # This infers a mutable reference.
  # CHECK: lit.call {{.*}}parametricMut{{.*}}!lit.ref<!MemExample, mut *"m`1">
  _ = parametricMut(__get_mvalue_as_litref(m))

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

  var aref = __get_mvalue_as_litref(a)
  var bref = __get_mvalue_as_litref(b)

  # CHECK: %cref = lit.var.decl "cref"
  var cref = aref if cond else bref

  # This uses both A and B, so it needs to extend both of their lifetimes.
  Reference(__get_litref_as_mvalue(cref))[].noop()
  # CHECK: [[CR:%.*]] = lit.ref.load %cref
  # CHECK: [[REFREF:%.*]] = lit.var.decl
  # CHECK-NEXT: lifetime.start [[REFREF]]
  # CHECK-NEXT: lit.call @{{.*}}@Reference::@"__init__{{.*}}([[REFREF]], [[CR]])
  # CHECK-NEXT: [[REF:%.*]] = lit.ref.load [[REFREF]]
  # CHECK-NEXT: lifetime.end [[REFREF]]
  # CHECK-NEXT: [[MREF:%.*]] = lit.call @{{.*}}__getitem__{{.*}}([[REF]])
  # CHECK-NEXT: lit.ref.immut [[MREF]]
  # CHECK-NEXT: lit.call @{{.*}}noop
  # CHECK-NEXT: lit.call @{{.*}}__del__{{.*}}(%a)
  # CHECK-NEXT: lifetime.end %a
  # CHECK-NEXT: lit.call @{{.*}}__del__{{.*}}(%b)
  # CHECK-NEXT: lifetime.end %b

# CHECK-LABEL: lit.func @"testDefConditional
fn testDefConditional(cond: __mlir_type.i1):
  # CHECK-NOT: lit.call {{[^)]*}}__del__

  var a = MemExample()

  var b = MemExample()

  var aref = __get_mvalue_as_litref(a)
  var bref = __get_mvalue_as_litref(b)

  # CHECK: %cref = lit.var.decl "cref"
  var cref = aref if cond else bref

  # Mutating either of these is fine - it doesn't matter which one is mutated,
  # we know that both are live.
  Reference(__get_litref_as_mvalue(cref))[].mutate()
  # CHECK: [[CR:%.*]] = lit.ref.load %cref
  # CHECK: [[REFREF:%.*]] = lit.var.decl
  # CHECK-NEXT: lifetime.start [[REFREF]]
  # CHECK-NEXT: lit.call @{{.*}}@Reference::@"__init__{{.*}}([[REFREF]], [[CR]])
  # CHECK-NEXT: [[REF:%.*]] = lit.ref.load [[REFREF]]
  # CHECK-NEXT: lifetime.end [[REFREF]]
  # CHECK-NEXT: [[MREF:%.*]] = lit.call @{{.*}}__getitem__{{.*}}([[REF]])
  # CHECK-NEXT: lit.call @{{.*}}mutate{{.*}}([[MREF]])

  # Overwriting one means that we need to immediately destroy the same reference
  # because we cannot know which one is being set.
  Reference(__get_litref_as_mvalue(cref))[] = MemExample()
  # CHECK: [[CR:%.*]] = lit.ref.load %cref
  # CHECK: [[REFREF:%.*]] = lit.var.decl
  # CHECK-NEXT: lifetime.start [[REFREF]]
  # CHECK-NEXT: lit.call @{{.*}}@Reference::@"__init__{{.*}}([[REFREF]], [[CR]])
  # CHECK-NEXT: [[REF:%.*]] = lit.ref.load [[REFREF]]
  # CHECK-NEXT: lifetime.end [[REFREF]]
  # CHECK-NEXT: [[MREF:%.*]] = lit.call @{{.*}}__getitem__{{.*}}([[REF]])
  # CHECK-NEXT: lit.call @{{.*}}__del__{{.*}}([[MREF]])
  # CHECK-NEXT: lit.call @{{.*}}__init__{{.*}}([[MREF]])

  # Overwriting is eligible for copy => move optimization as well.
  var shouldBeMovedFrom = MemExample()
  Reference(__get_litref_as_mvalue(cref))[] = shouldBeMovedFrom
  # CHECK: lit.call @{{.*}}__init__{{.*}}(%shouldBeMovedFrom)
  # CHECK: [[CR:%.*]] = lit.ref.load %cref
  # CHECK: [[REFREF:%.*]] = lit.var.decl
  # CHECK-NEXT: lifetime.start [[REFREF]]
  # CHECK-NEXT: lit.call @{{.*}}@Reference::@"__init__{{.*}}([[REFREF]], [[CR]])
  # CHECK-NEXT: [[REF:%.*]] = lit.ref.load [[REFREF]]
  # CHECK-NEXT: lifetime.end [[REFREF]]
  # CHECK-NEXT: [[MREF:%.*]] = lit.call @{{.*}}__getitem__{{.*}}([[REF]])
  # CHECK-NEXT: lit.ref.immut
  # CHECK-NEXT: lit.call @{{.*}}__del__{{.*}}([[MREF]])
  # CHECK-NEXT: lit.call @{{.*}}__moveinit__{{.*}}([[MREF]], %shouldBeMovedFrom)
  # CHECK-NEXT: lifetime.end %shouldBeMovedFrom

  # The mutation above could either of A or B, so we needed to extend both of
  # their lifetimes, but now we can say goodbye.
  # CHECK-NEXT: lit.call @{{.*}}__del__{{.*}}(%a)
  # CHECK-NEXT: lifetime.end %a
  # CHECK-NEXT: lit.call @{{.*}}__del__{{.*}}(%b)
  # CHECK-NEXT: lifetime.end %b

# ===----------------------------------------------------------------------=== #
# Tests of the Reference type.
# ===----------------------------------------------------------------------=== #

# CHECK-LABEL: lit.func @"testUseConditionalReference

fn testUseConditionalReference(cond: __mlir_type.i1, imm: MemExample):
  # CHECK: %a = lit.var.decl {{.*}} : !lit.ref<!MemExample, mut *"a`1">
  # CHECK: lit.call @{{.*}}__init__{{.*}}(%a)

  var a = MemExample()

  # CHECK-NEXT: %aref = lit.var.decl "aref"
  # CHECK-NEXT: lifetime.start %aref
  # CHECK-NEXT: lit.call @{{.*}}@Reference::@"__init__{{.*}}(%aref, %a)
  var aref = Reference(a)
  # CHECK-NEXT: lit.alias.decl *"aLifetime{{.*}}": lifetime<1> = <*"a`1">
  alias aLifetime =  aref.lifetime

  # CHECK-NEXT: [[AR:%.*]] = lit.ref.load %aref
  # CHECK-NEXT: [[REF:%.*]] = lit.call {{.*}}__getitem__{{.*}}([[AR]])
  aref[].noop()
  # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut [[REF]]
  # CHECK-NEXT: lit.call {{.*}}noop{{.*}}([[IMMREF]])

  # This is a mutable reference so go head and store through it whynot?
  # CHECK-NEXT: [[AR:%.*]] = lit.ref.load %aref
  # CHECK-NEXT: [[REF:%.*]] = lit.call {{.*}}__getitem__{{.*}}([[AR]])
  aref[] = MemExample()
  # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}([[REF]])
  # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}([[REF]])

  # Ok, this was the last use of A so it can go away.
  # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}(%a)
  # CHECK-NEXT: lifetime.end %a

  # The reference being alive doesn't keep the underlying stuff alive, only
  # accesses
  # CHECK-NEXT: %aref2 = lit.var.decl "aref2"
  # CHECK-NEXT: [[AR:%.*]] = lit.ref.load %aref
  # CHECK-NEXT: lifetime.end %aref
  # CHECK-NEXT: lifetime.start %aref2
  # CHECK-NEXT: lit.ref.store [[AR]], %aref2
  # CHECK-NEXT: lifetime.end %aref2
  var aref2 = aref

  # Reference can bind to immutable things as well, no problem.
  # CHECK-NEXT: %immref = lit.var.decl "immref"
  # CHECK-NEXT: lifetime.start %immref
  # CHECK-NEXT: [[IMMRV:%.*]] = lit.call @{{.*}}@Reference::@"__init__{{.*}}(%immref, %imm)
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
  # CHECK-SAME: (%self: !lit.ref<!SelfRefTest
  fn method(ref [_] self: Self) -> Reference[Self, __lifetime_of(self)]:
      return self

# CHECK-LABEL: lit.func @"testSelfRef
fn testSelfRef(a: SelfRefTest, inout b: SelfRefTest):
  # Bind immutably to a
  # CHECK: = lit.call {{.*}}method{{.*}}<:!Bool {:i1 0}, :!AnyType #SelfRefTest1, :lifetime<0> *"a`"
  _ = a.method()

  # Bind mutably to b
  # CHECK: = lit.call {{.*}}method{{.*}}<:!Bool {:i1 1}, :!AnyType #SelfRefTest1, :lifetime<1> *"b`1"
  _ = b.method()


# CHECK-LABEL: lit.func @"testLifetimeOf1
# CHECK-SAME: (%a: !lit.ref<!MemExample, imm *"a`"> borrow_in_mem) ->
# CHECK-SAME: Reference <{{.*}}, :lifetime<0> *"a`", :!AddressSpace {_value: !Int = {0}}>>
fn testLifetimeOf1(a: MemExample) -> Reference[MemExample, __lifetime_of(a)]:
  return a

# CHECK-LABEL: lit.func @"testLifetimeOf2
# CHECK-SAME: (%a: !lit.ref<!MemExample, imm *"a`"> borrow_in_mem) ->
# CHECK-SAME: !lit.ref<!MemExample, imm *"a`">
fn testLifetimeOf2(a: MemExample) -> Reference[MemExample, __lifetime_of(a)]._mlir_type:

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

struct OneLifetime[a_lifetime: ImmutableLifetime]:
  fn __init__(inout self): pass

struct TwoLifetimes[a_lifetime: ImmutableLifetime,
                    b_lifetime: ImmutableLifetime]:
  fn __init__(inout self): pass

# Crash converting mvalue of #lit.lifetime lifetime to Reference with specific one.
# https://github.com/modularml/mojo/issues/1921
struct SomeStruct:
  # CHECK-LABEL: lit.func @"refBindingToImmortal
  fn refBindingToImmortal(inout self, ptr: UnsafePointer[Int])
      -> Reference[Int, __lifetime_of(self)]:
    # CHECK: [[REFVAL:%.*]] = lit.call {{.*}}__getitem__{{.*}}(%ptr)
    # CHECK: [[REBIND:%.*]] = kgen.rebind [[REFVAL]]
    # CHECK-SAME : !lit.ref<!Int, mut #lit.lifetime> to !lit.ref<!Int, mut *"self`2x">
    # CHECK: [[TMP:%.*]] = lit.var.decl "anonymous*"
    # CHECK: lit.call {{.*}}__init__{{.*}}([[TMP]], [[REBIND]]
    return ptr[]


# Test that we can infer the type of 'T' in the func param invocation.
# CHECK-LABEL: CutDownVariadicPack
struct CutDownVariadicPack[
    element_trait: __mlir_type[`!lit.anytrait<`, AnyType, `>`],
    *element_types: element_trait,
]:

    # CHECK: lit.func @"each_hack
    fn each_hack[i: Int, func: fn[T: element_trait] (T) -> None](self):
        # Test that we can infer the type of 'T' from the argument.
        # CHECK-NEXT: [[REFVAL:%.*]] = lit.call {{.*}}get_element{{.*}}(%self)
        # CHECK-NEXT: [[REF:%.*]] = lit.call {{.*}}Reference::@"__getitem__{{.*}}([[REFVAL]])
        # CHECK-NEXT: [[LITREFRB:%.*]] = kgen.rebind [[REF]] : !lit.ref<:!AnyType {{.*}} to !lit.ref<:!kgen.paramref<:!lit.anytrait<!AnyType> element_trait>
        # CHECK-NEXT: lit.call{{.*}}([[LITREFRB]])
        func(self.get_element[i]()[])

    fn get_element[index: Int](self) -> Reference[
        element_types[index.value],
        __lifetime_of(self),
    ]:
       while True: pass

# Test that you can implicitly convert an immortal mutable reference (as is returned
# by UnsafePointer for example) to mortal reference with specified lifetime.
# CHECK: lit.func @"test_immortal_to_mortal
fn test_immortal_to_mortal(arg: Reference[Int, _])
    -> Reference[Int, arg.lifetime]:
  # CHECK-NEXT: [[ARGREF:%.*]] = lit.call {{.*}}Reference::@"__getitem__{{.*}}(%arg)
  # CHECK-NEXT: [[PTRVAL:%.*]] = lit.call {{.*}}UnsafePointer::@"address_of{{.*}}([[ARGREF]])
  # CHECK-NEXT: [[REF:%.*]] = lit.call {{.*}}UnsafePointer::@"__getitem__{{.*}}([[PTRVAL]])

  # CHECK-NEXT: [[ADJREFVAL:%.*]] = kgen.rebind [[REF]] : !lit.ref<!Int, mut #lit.lifetime> to !lit.ref<!Int, mut=#lit.struct.extract<:!Bool *"is_mutable`", "value">, *"lifetime`1">
  # CHECK-NEXT: [[ANON2:%.*]] = lit.var.decl "anonymous*"
  # CHECK-NEXT: lifetime.start [[ANON2]]
  # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}([[ANON2]], [[ADJREFVAL]])
  # CHECK-NEXT: [[RES:%.*]] = lit.load.consume [[ANON2:%.*]] : !lit.ref
  # CHECK-NEXT: lifetime.end [[ANON2]]
  # CHECK-NEXT: kgen.return [[RES]]
  return UnsafePointer.address_of(arg[])[]


# CHECK-LABEL: lit.func @"ref_copyability
fn ref_copyability[*element_types: Copyable](*args: *element_types):
  # CHECK: %x = lit.var.decl
  # CHECK: lit.call[{{.*}}get_type_method(:!Copyable{{.*}}__copyinit__{{.*}}(%x, %0)
  var x = args[4]

  # CHECK-NEXT: lit.call[{{.*}}get_type_method(:!Copyable{{.*}}__del__{{.*}}(%x)

# Issue #37659: Parameter inference doesn't work with force-immut lifetimes
fn thing_taking_immutable_ref[T: AnyType, value_lifetime: ImmutableLifetime](a: Reference[T, value_lifetime]): pass
fn test_passing_mutable_ref(inout i: String):
    thing_taking_immutable_ref(i)

# Verify that we can propagate parametric mutability through field accesses.
struct ThingWithFields:
  var field: Int

# CHECK-LABEL: lit.func @"parametric_mut_mbvalue
fn parametric_mut_mbvalue[
    is_mutable: __mlir_type.i1,
    lifetime: AnyLifetime[is_mutable].type,
 ](a: Reference[ThingWithFields, lifetime])
   -> Reference[Int, lifetime]:
  # CHECK: lit.ref.struct.ger
  return a[].field


# Reference directly with inferred params.
struct SomeStructWithReferenceSelfArgument:
    fn __init__(inout self): pass
    fn hello(ref [_] self: Self):
        pass

# CHECK-LABEL: lit.func @"testMethodRef
fn testMethodRef(a: SomeStructWithReferenceSelfArgument):
    # CHECK-NEXT: lit.call {{.*}}@"hello{{.*}}(%a)
    a.hello()



# CHECK-LABEL: lit.func @"variadic_inout_mems_iter
fn variadic_inout_mems_iter(inout *mems: MemExample):
  # Verify the iterator keeps the VariadicListMem alive.
  # CHECK-NEXT: %mems_0 = lit.var.decl

  # CHECK: %iter = lit.var.decl
  # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %mems_0 :
  # CHECK-NEXT: lifetime.start %iter
  # CHECK-NEXT: lit.call {{.*}}__iter__{{.*}}([[IMMREF]], %iter)
  var iter = mems.__iter__()

  # CHECK-NEXT: %x = lit.var.decl
  # CHECK-NEXT: [[ELTREF:%.*]] = lit.call {{.*}}__next__{{.*}}(%iter)

  ## FIXME: This destruction should be ordered after the destroy of the iterator
  ## Since the iterator can refer to the mems struct.
  # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}(%mems_0)
  # CHECK-NEXT: lifetime.end %mems_0

  # Iterator is destroyed as soon as we're done with it.
  # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}(%iter)
  # CHECK-NEXT: lifetime.end %iter

  # __next__ returns a Reference which needs to turn in to !lit.ref
  # CHECK-NEXT: [[ELTDEREF:%.*]] = lit.call {{.*}}__getitem__{{.*}}([[ELTREF]])
  # CHECK-NEXT: [[ELTDEREFIMM:%.*]] = lit.ref.immut [[ELTDEREF]]
  # CHECK-NEXT: lifetime.start %x
  # CHECK-NEXT: lit.call {{.*}}__copyinit__{{.*}}(%x, [[ELTDEREFIMM]])
  var x : MemExample = iter.__next__()[]
  # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}(%x)
  # CHECK-NEXT: lifetime.end %x

  # CHECK-NEXT: kgen.param.constant: none
  # CHECK-NEXT: kgen.return
