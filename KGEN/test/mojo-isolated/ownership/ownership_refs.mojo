# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Test more advanced reference cases.

# RUN: %parse-mojo-isolated %s --mlir-print-debuginfo -o %t.mlir
# RUN: kgen-opt %t.mlir -lower-semantic-cf -check-lifetimes -verify-parameters -verify-diagnostics | FileCheck %s

fn use_any[*Ts: AnyType](*args: *Ts): pass

# ===----------------------------------------------------------------------=== #
# Parsing of references
# ===----------------------------------------------------------------------=== #

# CHECK-LABEL: lit.struct.decl @MemExample
struct MemExample:
  fn __init__(out self): pass
  fn __moveinit__(out self, owned existing: Self): pass
  fn __copyinit__(out self, existing: Self): pass
  fn __del__(owned self): pass
  fn noop(self): pass
  fn mutate(mut self): pass

# CHECK-LABEL: lit.fn @"borrow{{.*}}"<lt: origin<0>>(%a: !lit.ref<!MemExample, imm lt>)
fn borrow[lt: ImmutableOrigin](a: Pointer[MemExample, lt]._mlir_type):
  pass

# CHECK-LABEL: lit.fn @"mutate{{.*}}"<lt: origin<1>>(%a: !lit.ref<!MemExample, mut lt>)
fn mutate[lt: MutableOrigin](a: Pointer[MemExample, lt]._mlir_type):
  pass

# CHECK-LABEL: lit.fn @"implicit_borrow
fn implicit_borrow(a: MemExample):
  pass

# CHECK-LABEL: lit.fn @"implicit_inout
fn implicit_inout(mut a: MemExample):
  pass

# CHECK-LABEL: lit.fn @"implicit_owned
fn implicit_owned(owned a: MemExample):
  pass

# This preserves reference mutability
# CHECK-LABEL: lit.fn @"parametricMut
# CHECK-SAME: (%a: !lit.ref<!MemExample, mut=#lit.struct.extract<:!Bool isMut, "value">, life>)
# CHECK-SAME: -> !lit.ref<!MemExample, mut=#lit.struct.extract<:!Bool isMut, "value">, life>
fn parametricMut[isMut: Bool,
                 life: Origin[isMut]._mlir_type](a: Pointer[MemExample, life]._mlir_type)
   -> Pointer[MemExample, life]._mlir_type:
  return a

# CHECK-LABEL: lit.fn @"testParametricMut
fn testParametricMut(i: MemExample, mut m: MemExample):
  # This infers an immutable reference.
  # CHECK:  lit.call {{.*}}parametricMut{{.*}}!lit.ref<!MemExample, imm *"i`">
  _ = parametricMut(__get_mvalue_as_litref(i))

  # This infers a mutable reference.
  # CHECK: lit.call {{.*}}parametricMut{{.*}}!lit.ref<!MemExample, mut *"m`1">
  _ = parametricMut(__get_mvalue_as_litref(m))

##===----------------------------------------------------------------------===##
# Conditional origins
##===----------------------------------------------------------------------===##

# CHECK-LABEL: lit.fn @"testUseConditional
fn testUseConditional(cond: __mlir_type.i1):
  # CHECK-NOT: __del__

  # CHECK: lit.call @{{.*}}__init__{{.*}}(%a)
  var a = MemExample()

  # CHECK: lit.call @{{.*}}__init__{{.*}}(%b)
  var b = MemExample()

  # CHECK: %cptr = lit.var.decl "cptr"
  var cptr = Pointer(to=a) if cond else Pointer(to=b)

  # This uses both A and B, so it needs to extend both of their origins.
  cptr[].noop()
  # CHECK: [[CV:%.*]] = lit.ref.load %cptr
  # CHECK-NEXT: lit.var.lifetime.end %cptr
  # CHECK-NEXT: [[MREF:%.*]] = lit.call {{.*}}__getitem__{{.*}}([[CV]])
  # CHECK-NEXT: lit.ref.immut [[MREF]]
  # CHECK-NEXT: lit.call @{{.*}}noop
  # CHECK-NEXT: lit.call @{{.*}}__del__{{.*}}(%a)
  # CHECK-NEXT: lifetime.end %a
  # CHECK-NEXT: lit.call @{{.*}}__del__{{.*}}(%b)
  # CHECK-NEXT: lifetime.end %b

# CHECK-LABEL: lit.fn @"testDefConditional
fn testDefConditional(cond: __mlir_type.i1):
  # CHECK-NOT: lit.call {{[^)]*}}__del__

  var a = MemExample()
  var b = MemExample()

  # CHECK: %cptr = lit.var.decl "cptr"
  var cptr = Pointer(to=a) if cond else Pointer(to=b)


  # Mutating either of these is fine - it doesn't matter which one is mutated,
  # we know that both are live.
  cptr[].mutate()
  # CHECK: [[CP:%.*]] = lit.ref.load %cptr
  # CHECK-NEXT: [[MREF:%.*]] = lit.call @{{.*}}__getitem__{{.*}}([[CP]])
  # CHECK-NEXT: lit.call @{{.*}}mutate{{.*}}([[MREF]])

  # Overwriting one means that we need to immediately destroy the same reference
  # because we cannot know which one is being set.
  cptr[] = MemExample()
  # CHECK: [[CP:%.*]] = lit.ref.load %cptr
  # CHECK-NEXT: [[MREF:%.*]] = lit.call @{{.*}}__getitem__{{.*}}([[CP]])
  # CHECK-NEXT: lit.call @{{.*}}__del__{{.*}}([[MREF]])
  # CHECK-NEXT: lit.call @{{.*}}__init__{{.*}}([[MREF]])

  # Overwriting is eligible for copy => move optimization as well.
  var shouldBeMovedFrom = MemExample()
  # CHECK: lit.call @{{.*}}__init__{{.*}}(%shouldBeMovedFrom)
  cptr[] = shouldBeMovedFrom
  # CHECK: [[CP:%.*]] = lit.ref.load %cptr
  # CHECK-NEXT: lit.var.lifetime.end %cptr
  # CHECK-NEXT: [[MREF:%.*]] = lit.call @{{.*}}__getitem__{{.*}}([[CP]])
  # CHECK-NEXT: lit.ref.immut
  # CHECK-NEXT: lit.call @{{.*}}__del__{{.*}}([[MREF]])
  # CHECK-NEXT: lit.call @{{.*}}__moveinit__{{.*}}(%shouldBeMovedFrom, [[MREF]])
  # CHECK-NEXT: lifetime.end %shouldBeMovedFrom

  # The mutation above could either of A or B, so we needed to extend both of
  # their origins, but now we can say goodbye.
  # CHECK-NEXT: lit.call @{{.*}}__del__{{.*}}(%b)
  # CHECK-NEXT: lifetime.end %b

  # A use so the assignment isn't dead.
  a.noop()
  # CHECK-NEXT: [[ATMP:%.*]] = lit.ref.immut %a
  # CHECK-NEXT: lit.call @{{.*}}noop{{.*}}([[ATMP]])
  # CHECK-NEXT: lit.call @{{.*}}__del__{{.*}}(%a)
  # CHECK-NEXT: lifetime.end %a

# ===----------------------------------------------------------------------=== #
# Tests of the Pointer type.
# ===----------------------------------------------------------------------=== #

# CHECK-LABEL: lit.fn @"testUseConditionalReference

fn testUseConditionalReference(cond: __mlir_type.i1, imm: MemExample):
  # CHECK: %a = lit.var.decl {{.*}} : !lit.ref<!MemExample, mut *"a`1">
  # CHECK: lit.call @{{.*}}__init__{{.*}}(%a)

  var a = MemExample()

  # CHECK: lit.call @stdlib::@builtin::@stubs::@Pointer::@"__init__{{.*}}(%a)
  var aref = Pointer(to=a)
  # CHECK: lit.alias.decl *"aLifetime{{.*}}": origin<1> = <*"a`1">
  alias aLifetime =  aref.origin._mlir_origin

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
  # expected-warning @+1 {{assignment to 'aref2' was never used}}
  var aref2 = aref

  # Pointer can bind to immutable things as well, no problem.
  # CHECK-NEXT: %immref = lit.var.decl "immref"
  # CHECK-NEXT: [[IMMRV:%.*]] = lit.call @stdlib::@builtin::@stubs::@Pointer::@"__init__{{.*}}(%imm)
  # CHECK: lit.ref.store [[IMMRV]], %immref
  var immref = Pointer(to=imm)
  immref[].noop()

# ===----------------------------------------------------------------------=== #
# Test that we can bind self origin.
# ===----------------------------------------------------------------------=== #

# Need a way to get a origin of Self.
# https://github.com/modularml/modular/issues/29069

struct SelfRefTest:
  fn __init__(out self): pass
  fn __del__(owned self): pass
  # CHECK-LABEL: lit.fn @"method
  # CHECK-SAME: (%self: !lit.ref<!SelfRefTest
  fn method(ref self) -> Pointer[Self, __origin_of(self)]:
      return Pointer(to=self)

# CHECK-LABEL: lit.fn @"testSelfRef
fn testSelfRef(a: SelfRefTest, mut b: SelfRefTest):
  # Bind immutably to a
  # CHECK: = lit.call {{.*}}method{{.*}}<:!Bool {:i1 0}, :!AnyType #SelfRefTest1, {{.*}}origin<0> = *"a`"
  _ = a.method()

  # Bind mutably to b
  # CHECK: = lit.call {{.*}}method{{.*}}<:!Bool {:i1 1}, :!AnyType #SelfRefTest1, {{.*}}origin<1> = *"b`1"
  _ = b.method()


# CHECK-LABEL: lit.fn @"testLifetimeOf1
# CHECK-SAME: (%a: !lit.ref<!MemExample, imm *"a`"> read_mem) ->
# CHECK-SAME: Pointer <{{.*}}origin<0> = *"a`"}, :!AddressSpace {_value: !Int = {0}}>>
fn testLifetimeOf1(a: MemExample) -> Pointer[MemExample, __origin_of(a)]:
  return Pointer(to=a)

# CHECK-LABEL: lit.fn @"testLifetimeOf2
# CHECK-SAME: (%a: !lit.ref<!MemExample, imm *"a`"> read_mem) ->
# CHECK-SAME: !lit.ref<!MemExample, imm *"a`">
fn testLifetimeOf2(a: MemExample) -> Pointer[MemExample, __origin_of(a)]._mlir_type:

  # CHECK: kgen.return {{.*}} : !lit.ref<!MemExample, imm *"a`">
  return Pointer(to=a)._value

# CHECK-LABEL: lit.fn @"callByRefResultLifetime
fn callByRefResultLifetime(mut x: MemExample, mut y: MemExample, z: MemExample):
  # CHECK: lit.var.decl "l1" var : !lit.ref<@ownership_refs::@OneLifetime<:origin<0> (mutcast mut *"x`")>
  var l1 = returnOneArgLifetime(x)

  # CHECK: lit.var.decl "l2" var : !lit.ref<@ownership_refs::@TwoLifetimes<:origin<0> (mutcast mut *"x`"), :origin<0> (mutcast mut *"y`1")>
  var l2 = returnTwoArgLifetimes(x, y)
  # CHECK: %l3 = lit.var.decl "l3" var : !lit.ref<@ownership_refs::@TwoLifetimes<:origin<0> (mutcast mut *"x`"), :origin<0> (mutcast mut *"x`")>
  var l3 = returnTwoArgLifetimes(x, x)
  # CHECK: %l4 = lit.var.decl "l4" var : !lit.ref<@ownership_refs::@TwoLifetimes<:origin<0> *"z`2", :origin<0> *"z`2">
  var l4 = returnTwoArgLifetimes(z, z)

  use_any(l1, l2, l3, l4)

fn returnOneArgLifetime(a: MemExample)
  -> OneLifetime[__origin_of(a)]:
  return OneLifetime[__origin_of(a)]()

fn returnTwoArgLifetimes(a: MemExample, b: MemExample)
  -> TwoLifetimes[__origin_of(a), __origin_of(b)]:
  return TwoLifetimes[__origin_of(a), __origin_of(b)]()

struct OneLifetime[a_origin: ImmutableOrigin]:
  fn __init__(out self): pass

struct TwoLifetimes[a_origin: ImmutableOrigin,
                    b_origin: ImmutableOrigin]:
  fn __init__(out self): pass

# Test that we can infer the type of 'T' in the func param invocation.
# CHECK-LABEL: CutDownVariadicPack
struct CutDownVariadicPack[element_trait: __type_of(AnyType),
                           *element_types: element_trait]:

    # CHECK: lit.fn @"each_hack
    fn each_hack[i: Int, func: fn[T: element_trait] (T) -> None](self):
        # Test that we can infer the type of 'T' from the argument.
        # CHECK-NEXT: [[REFVAL:%.*]] = lit.call {{.*}}get_element{{.*}}(%self)
        # CHECK-NEXT: [[REF:%.*]] = lit.call {{.*}}Pointer::@"__getitem__{{.*}}([[REFVAL]])
        # CHECK-NEXT: lit.call{{.*}} func, variadic_get(:variadic<:!lit.anytrait<!AnyType> element_trait> element_types{{.*}}([[REF]])
        func(self.get_element[i]()[])

    fn get_element[index: Int](self) -> Pointer[
        element_types[index.value],
        __origin_of(self),
    ]:
       while True: pass

# Test that you can implicitly convert an "any" mutable reference (as is returned
# by UnsafePointer for example) to mortal reference with specified origin.
# CHECK: lit.fn @"test_immortal_to_mortal
fn test_immortal_to_mortal(arg: Pointer[Int, _])
    -> Pointer[Int, arg.origin]:
  # CHECK-NEXT: [[ARGREF:%.*]] = lit.call {{.*}}Pointer::@"__getitem__{{.*}}(%arg)
  # CHECK-NEXT: [[PTRVAL:%.*]] = lit.call {{.*}}UnsafePointer::@"address_of{{.*}}([[ARGREF]])
  # CHECK-NEXT: [[REF:%.*]] = lit.call {{.*}}UnsafePointer::@"__getitem__{{.*}}([[PTRVAL]])

  # CHECK-NEXT: [[ADJREFVAL:%.*]] = kgen.rebind [[REF]] : !lit.ref<!Int, mut #lit.any.origin> to !lit.ref<!Int, mut=#lit.struct.extract<:!Bool *"mut`", "value">, #lit.struct.extract<:@stdlib::@builtin::@stubs::@Origin<:!Bool *"mut`"> *"origin`1", "_mlir_origin">>
  # CHECK-NEXT: [[RES:%.*]] = lit.call @stdlib::@builtin::@stubs::@Pointer::@"__init__{{.*}}([[ADJREFVAL]])
  # CHECK-NEXT: kgen.return [[RES]]
  return Pointer[Int, arg.origin](to=UnsafePointer.address_of(arg[])[])


# CHECK-LABEL: lit.fn @"ref_copyability
fn ref_copyability[*element_types: Copyable](*args: *element_types):
  # CHECK: %_x = lit.var.decl
  # CHECK: [[ITEM:%.*]] = lit.call @stdlib::@builtin::@stubs::@VariadicPack::@"__getitem__
  # CHECK: lit.call[{{.*}}get_vtable_entry(:!Copyable{{.*}}__copyinit__{{.*}}([[ITEM]], %_x)
  var _x = args[4]

  # CHECK-NEXT: lit.call[{{.*}}get_vtable_entry(:!Copyable{{.*}}__del__{{.*}}(%_x)

# Issue #37659: Parameter inference doesn't work with force-immut origins

# FIXME (Patch #48185): need to support implicit conversions to immutable reference.

#fn thing_taking_immutable_ref[T: AnyType, value_origin: ImmutableOrigin](a: Pointer[T, value_origin]): pass
#fn test_passing_mutable_ref(mut i: String):
#    thing_taking_immutable_ref(Pointer(to=i))

# Verify that we can propagate parametric mutability through field accesses.
struct ThingWithFields:
  var field: Int

# CHECK-LABEL: lit.fn @"parametric_mut_mbvalue
fn parametric_mut_mbvalue[
    mut: __mlir_type.i1,
    origin: Origin[mut]._mlir_type,
 ](a: Pointer[ThingWithFields, origin])
   -> Pointer[Int, __origin_of(a[].field)]:
  # CHECK: lit.ref.struct.ger
  return Pointer(to=a[].field)

# Pointer directly with inferred params.
struct SomeStructWithReferenceSelfArgument:
    fn __init__(out self): pass
    fn hello(ref self):
        pass

# CHECK-LABEL: lit.fn @"testMethodRef
fn testMethodRef(a: SomeStructWithReferenceSelfArgument):
    # CHECK-NEXT: lit.call {{.*}}@"hello{{.*}}(%a)
    a.hello()



# CHECK-LABEL: lit.fn @"variadic_inout_mems_iter
fn variadic_inout_mems_iter(mut *mems: MemExample):
  # Verify the iterator keeps the VariadicListMem alive.
  # CHECK-NEXT: %mems_0 = lit.var.decl

  # CHECK: %iter = lit.var.decl
  # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %mems_0 :
  # CHECK-NEXT: lifetime.start %iter
  # CHECK-NEXT: lit.call {{.*}}__iter__{{.*}}([[IMMREF]], %iter)
  var iter = mems.__iter__()

  # CHECK-NEXT: %x = lit.var.decl
  # CHECK-NEXT: [[ELTREF:%.*]] = lit.call {{.*}}__next__{{.*}}(%iter)

  # Iterator is destroyed as soon as we're done with it.
  # CHECK-NEXT: lifetime.end %iter

  ## NOTE: This destruction should be ordered after the destroy of the iterator
  ## Since the iterator can refer to the mems struct.
  # CHECK-NEXT: lifetime.end %mems_0

  # __next__ returns a Pointer which needs to turn in to !lit.ref
  # CHECK-NEXT: [[ELTDEREF:%.*]] = lit.call {{.*}}__getitem__{{.*}}([[ELTREF]])
  # CHECK-NEXT: [[ELTDEREFIMM:%.*]] = lit.ref.immut [[ELTDEREF]]
  # CHECK-NEXT: lifetime.start %x
  # CHECK-NEXT: lit.call {{.*}}__copyinit__{{.*}}([[ELTDEREFIMM]], %x)
  var x : MemExample = iter.__next__()[]

  # CHECK-NEXT: lit.call {{.*}}mutate{{.*}}(%x)
  x.mutate()

  # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}(%x)
  # CHECK-NEXT: lifetime.end %x

  # CHECK-NEXT: kgen.param.constant: none
  # CHECK-NEXT: kgen.return

# CHECK-LABEL: lit.fn @"test_pvalue_ref_formation
fn test_pvalue_ref_formation[a: SelfRefTest]():
  # This is invoking a method (accepting a ref) on a pvalue.  This need to
  # materialize into a temporary and use the origin of the temporary, not an
  # immortal origin.

  # CHECK: [[ANONTMP:%.*]] = lit.var.decl "anonymous*" {{.*}}!lit.ref<!SelfRefTest, mut *"anonymous*`1">
  var r = a.method()
  # The result reference should have inferred the origin of the temp
  # CHECK: lit.ref.store {{.*}}, %r : {{.*}}#SelfRefTest1, {{.*}}origin<0> = (mutcast mut *"anonymous*`1")},

  # This use of the temp should keep it alive.
  # CHECK: [[REFERENCE:%.*]] = lit.ref.load %r
  # CHECK: [[REF:%.*]] = lit.call {{.*}}Pointer::@"__getitem__{{.*}}([[REFERENCE]])
  # CHECK-NEXT: lit.call {{.*}}method{{.*}}([[REF]])
  _ = r[].method()
  # CHECK-NEXT: lit.call {{.*}}SelfRefTest::@"__del__{{.*}}([[ANONTMP]])

# MOCO-1025 - Need hierarchical origins
struct FieldRefPropagation:
  var field1 : Optional[Int]
  var field2 : Int

  fn __init__(out self):
     # Should be able to initialize field1 and use it.
     self.field1 = 42
     # Should be able to project it and assign through ref.
     self.field1.value() = 17
     # Then initialize field2
     self.field2 = 1


# Issue #3444 (nightly) Raising init causing use of uninitialized variable
# https://github.com/modular/mojo/issues/3444
struct HasRaisingInit:
  fn __init__(out self) raises: pass
  fn __moveinit__(out self, owned existing: Self): pass
  fn __copyinit__(out self, existing: Self): pass
  fn __del__(owned self): pass

struct ImmovableRaisingInit:
  fn __init__(out self) raises: pass

struct RaisingInitWrapper:
    var field: HasRaisingInit
    var immfield: ImmovableRaisingInit

    fn __init__(out self) raises:
      self.field = HasRaisingInit()
      self.immfield = ImmovableRaisingInit()

# CHECK-LABEL: lit.fn @"test_inout_raising_init
fn test_inout_raising_init(mut a: HasRaisingInit, mut b: RaisingInitWrapper) raises:
  # These init calls need a temporary instead of direct assignment into the dest
  # to avoid invalidating a value on the error path.
  # CHECK-NEXT: [[TEMP:%.*]] = lit.var.decl
  # CHECK: lit.call {{.*}}HasRaisingInit::@"__init__{{.*}}({{.*}}, [[TEMP]])
  a = HasRaisingInit()
  # EH logic.
  # CHECK: lit.call {{.*}}HasRaisingInit::@"__moveinit__{{.*}}([[TEMP]], %a)

  # CHECK: [[FIELDREF:%.*]] = lit.ref.struct.ger %b[field]
  # CHECK: [[TEMP:%.*]] = lit.var.decl
  # CHECK: lit.call {{.*}}HasRaisingInit::@"__init__{{.*}}({{.*}}, [[TEMP]])
  b.field = HasRaisingInit()
  # EH logic.
  # CHECK: lit.call {{.*}}HasRaisingInit::@"__moveinit__{{.*}}([[TEMP]], [[FIELDREF:%.*]]) :

# CHECK-LABEL: lit.fn @"test_parameter_closure_captures
fn test_parameter_closure_captures(owned x: MemExample, owned y: MemExample):
  # CHECK: lit.fn *"capture
  @parameter
  fn capture():
    _ = x^
    _ = y^

  # CHECK: lit.call[!lit.generator<:{mut *"x`{{.*}}", mut *"y`{{.*}}"}:
  # CHECK-NEXT: lit.call {{.*}}MemExample::@"__del__{{.*}}(%x)
  # CHECK-NEXT: lit.call {{.*}}MemExample::@"__del__{{.*}}(%y)
  capture()

fn higher_order_function[lts: __mlir_type.`!lit.origin.set`, //, f: fn() capturing [lts] -> None]():
  pass

# CHECK-LABEL: lit.fn @"test_higher_order_capture
fn test_higher_order_capture(owned x: MemExample, owned y: MemExample):
  # CHECK: lit.fn *"capture
  @parameter
  fn capture():
    _ = x^
    _ = y^

  # CHECK: lit.call {{.*}}higher_order_function{{.*}} !lit.generator<:{mut *"x`{{.*}}", mut *"y`{{.*}}"}
  # CHECK-NEXT: lit.call {{.*}}MemExample::@"__del__{{.*}}(%x)
  # CHECK-NEXT: lit.call {{.*}}MemExample::@"__del__{{.*}}(%y)
  higher_order_function[capture]()

# CHECK-LABEL: lit.fn @"test_origin_ref_spec
# CHECK-SAME: !lit.ref<!Int, mut #lit.struct.extract<{{.*}}@Origin<:!Bool {:i1 1}> our_origin, "_mlir_origin">> mutref)
fn test_origin_ref_spec[our_origin: Origin[True]](ref[our_origin] a: Int):
    pass

# CHECK-LABEL: lit.fn @"another_min
fn another_min[mut: Bool, //, ao: Origin[mut], bo: Origin[mut]](ref [ao]a: Int, ref [bo]b: Int) -> ref [a, b] Int:
    if a < b:
        return a
    else: # This failed due to union canonicalization problems.
        return b

struct RefResultStruct:
  var x: Int
  fn __init__(out self):  self.x = 1
  fn method(self) -> ref [self.x] Int: return self.x

# https://github.com/modular/mojo/issues/3960
# CHECK-LABEL: lit.struct.decl @FieldSensitiveUse
struct FieldSensitiveUse:
    var x: RefResultStruct
    var y: String

    # CHECK: lit.fn @"__init__
    fn __init__(out self):
        # CHECK: lit.call {{.*}}RefResultStruct::@"__init__
        self.x = RefResultStruct()
        # CHECK: [[TMP:%.*]] = lit.call {{.*}}RefResultStruct::@"method
        _ = self.x.method()
        # CHECK-NEXT: lit.ref.load [[TMP]]
        self.y = String()
        # CHECK-NEXT: [[TMP:%.*]] = lit.ref.struct.ger %self[y]
        # CHECK-NEXT: lit.call {{.*}}String::@"__init__{{.*}}([[TMP]])
