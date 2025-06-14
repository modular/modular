# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s


fn has_default_args(a: Index, b: Index = `1`, c: Index = `2`):
    pass


# CHECK-LABEL: lit.fn @"test_kw_arg_passing
fn test_kw_arg_passing(x: Index, y: Index, z: Index):
    # CHECK: %[[C2:.*]] = kgen.param.constant = <2>
    # CHECK: call {{.*}}@"has_default_args{{.*}}"(%x, %y, %[[C2]])
    has_default_args(x, b=y)

    # CHECK: call {{.*}}@"has_default_args{{.*}}"(%x, %y, %z)
    has_default_args(x, b=y, c=z)

    # CHECK: %[[C1:.*]] = kgen.param.constant = <1>
    # CHECK: call {{.*}}@"has_default_args{{.*}}"(%x, %[[C1]], %z)
    has_default_args(x, c=z)

    # CHECK: call {{.*}}@"has_default_args{{.*}}"(%x, %y, %z)
    has_default_args(x, c=z, b=y)

    # CHECK: call {{.*}}@"has_default_args{{.*}}"(%x, %y, %z)
    has_default_args(a=x, c=z, b=y)

    # CHECK: call {{.*}}@"has_default_args{{.*}}"(%x, %y, %z)
    has_default_args(c=z, b=y, a=x)


# CHECK-LABEL: lit.fn @"test_kw_arg_passing_indirect
fn test_kw_arg_passing_indirect[callee: fn(a: Index, b: Index=`1`, c: Index=`2`)->None](x: Index, y: Index, z: Index):
    # CHECK-DAG: %[[C1:.*]] = kgen.param.constant = <1>
    # CHECK-NEXT: lit.call[{{.*}}](%x, %[[C1]], %z)
    callee(x, c=z)

    # CHECK-NEXT: lit.call[{{.*}}](%x, %y, %z)
    callee(c=z, b=y, a=x)

fn has_default_params[a: Index, b: Index = `1`, c: Index = `2`]():
    pass


# CHECK-LABEL: lit.fn @"test_kw_param_passing
fn test_kw_param_passing[x: Index, y: Index, z: Index]():
    # CHECK: lit.call @{{.*}}@"has_default_params{{.*}}"<x, y, 2>
    has_default_params[x, b=y]()

    # CHECK: lit.call @{{.*}}@"has_default_params{{.*}}"<x, y, z>
    has_default_params[x, b=y, c=z]()

    # CHECK: lit.call @{{.*}}@"has_default_params{{.*}}"<x, 1, z>
    has_default_params[x, c=z]()

    # CHECK: lit.call @{{.*}}@"has_default_params{{.*}}"<x, y, z>
    has_default_params[x, c=z, b=y]()

    # CHECK: lit.call @{{.*}}@"has_default_params{{.*}}"<x, y, z>
    has_default_params[a=x, c=z, b=y]()

    # CHECK: lit.call @{{.*}}@"has_default_params{{.*}}"<x, y, z>
    has_default_params[c=z, b=y, a=x]()


# CHECK-LABEL: lit.fn @"test_kw_param_passing_indirect
fn test_kw_param_passing_indirect[x: Index, y: Index, z: Index,
                                  callee: fn[a: Index, b: Index=`1`, c: Index=`2`]()->None]():

    # CHECK: call{{.*}}bind_params(:!lit.generator<{{.*}}> callee, x, 1, z)]()
    callee[x, c=z]()

    # CHECK: call{{.*}}bind_params(:!lit.generator<{{.*}}> callee, x, y, z)]()
    callee[c=z, b=y, a=x]()


@fieldwise_init
struct MyCallable:
    fn __call__(self, m: Index, n: Index = `2`):
        pass


# CHECK-LABEL: lit.fn @"test_callable_object
fn test_callable_object(x: Index, y: Index):
    # CHECK: %[[CALLABLE:.*]] = lit.var.decl {{.*}}: !lit.ref<!MyCallable
    var callable = MyCallable()

    # CHECK-DAG: %[[IMMREF:.*]] = lit.ref.immut %[[CALLABLE]]
    # CHECK-DAG: %[[C2:.*]] = kgen.param.constant = <2>
    # CHECK-NEXT: call {{.*}}@MyCallable::@"__call__{{.*}}(%[[IMMREF]], %x, %[[C2]])
    callable(x)

    # CHECK-DAG: %[[IMMREF:.*]] = lit.ref.immut %[[CALLABLE]]
    # CHECK-NEXT: call {{.*}}@MyCallable::@"__call__{{.*}}(%[[IMMREF]], %y, %x)
    callable(n=x, m=y)


fn takes_kw_only_args(a: Index, b: Index = `1`, *, c: Index, d: Index = `2`):
    pass


# CHECK-LABEL: lit.fn @"test_kw_only_args
fn test_kw_only_args(x: Index):
    # CHECK-DAG: %[[C1:.*]] = kgen.param.constant = <1>
    # CHECK-DAG: %[[C2:.*]] = kgen.param.constant = <2>
    # CHECK-NEXT: lit.call {{.*}}@"takes_kw_only_args{{.*}}"(%x, %[[C1]], %x, %[[C2]])
    takes_kw_only_args(x, c=x)

    # CHECK-DAG: %[[C1:.*]] = kgen.param.constant = <1>
    # CHECK-DAG: %[[C2:.*]] = kgen.param.constant = <2>
    # CHECK-NEXT: lit.call {{.*}}@"takes_kw_only_args{{.*}}"(%x, %[[C1]], %x, %[[C2]])
    takes_kw_only_args(c=x, a=x)

    # CHECK: %[[C2:.*]] = kgen.param.constant = <2>
    # CHECK-NEXT: lit.call {{.*}}@"takes_kw_only_args{{.*}}"(%x, %x, %x, %[[C2]])
    takes_kw_only_args(x, c=x, b=x)

    # CHECK-DAG: %[[C1:.*]] = kgen.param.constant = <1>
    # CHECK-NEXT: lit.call {{.*}}@"takes_kw_only_args{{.*}}"(%x, %[[C1]], %x, %x)
    takes_kw_only_args(x, d=x, c=x)

    # CHECK: lit.call {{.*}}@"takes_kw_only_args{{.*}}"(%x, %x, %x, %x)
    takes_kw_only_args(d=x, b=x, c=x, a=x)


# CHECK-LABEL: lit.fn @"test_kw_only_indirect
fn test_kw_only_indirect[callee: fn(a: Index, b: Index = `1`, *, c: Index, d: Index = `2`)->None](x: Index):

    # CHECK-DAG: %[[C1:.*]] = kgen.param.constant = <1>
    # CHECK-DAG: %[[C2:.*]] = kgen.param.constant = <2>
    # CHECK-NEXT: lit.call[{{.*}}](%x, %[[C1]], %x, %[[C2]])
    callee(x, c=x)

    # CHECK-DAG: %[[C1:.*]] = kgen.param.constant = <1>
    # CHECK-NEXT: lit.call[{{.*}}](%x, %[[C1]], %x, %x)
    callee(x, d=x, c=x)


fn takes_kw_only_params[a: Index, b: Index = `1`, *, c: Index, d: Index = `2`]():
    pass


# CHECK-LABEL: lit.fn @"test_kw_only_params
fn test_kw_only_params[x: Index]():
    # CHECK: call {{.*}}takes_kw_only_params{{.*}}"<x, 1, x, 2>()
    takes_kw_only_params[x, c=x]()

    # CHECK: call {{.*}}takes_kw_only_params{{.*}}"<x, 1, x, 2>()
    takes_kw_only_params[c=x, a=x]()

    # CHECK: call {{.*}}takes_kw_only_params{{.*}}"<x, x, x, 2>()
    takes_kw_only_params[x, c=x, b=x]()

    # CHECK: call {{.*}}takes_kw_only_params{{.*}}"<x, 1, x, x>()
    takes_kw_only_params[x, d=x, c=x]()

    # CHECK: call {{.*}}takes_kw_only_params{{.*}}"<x, x, x, x>()
    takes_kw_only_params[d=x, b=x, c=x, a=x]()


# CHECK-LABEL: lit.fn @"test_kw_only_params_indirect
fn test_kw_only_params_indirect[x: Index, callee: fn[a: Index, b: Index = `1`, *, c: Index, d: Index = `2`]()->None]():

    # CHECK: call{{.*}}bind_params(:!lit.generator<{{.*}}> callee, x, 1, x, 2)]()
    callee[x, c=x]()

    # CHECK: call{{.*}}bind_params(:!lit.generator<{{.*}}> callee, x, 1, x, x)]()
    callee[x, d=x, c=x]()


fn takes_variadic_and_kw_only_args(
    a: Index, b: Index, *args: Index, c: Index, d: Index = `0`
):
    pass


# CHECK-LABEL: lit.fn @"test_variadic_and_kw_only_args
fn test_variadic_and_kw_only_args(x: Index):
    # CHECK-DAG: %[[VAR:.*]] = kgen.param.constant: variadic<index> = <[]>
    # CHECK-DAG: %[[ZERO:.*]] = kgen.param.constant = <0>
    # CHECK-NEXT: lit.call {{.*}}@"takes_variadic_and_kw_only_args{{.*}}"(%x, %x, %[[VAR]], %x, %[[ZERO]])
    takes_variadic_and_kw_only_args(x, x, c=x)

    # CHECK: %[[VAR:.*]] = kgen.param.constant: variadic<index> = <[]>
    # CHECK-NEXT: lit.call {{.*}}@"takes_variadic_and_kw_only_args{{.*}}"(%x, %x, %[[VAR]], %x, %x)
    takes_variadic_and_kw_only_args(x, x, d=x, c=x)

    # CHECK-DAG: %[[VAR:.*]] = pop.variadic.splat  2, %x : !kgen.variadic<index>
    # CHECK-DAG: %[[ZERO:.*]] = kgen.param.constant = <0>
    # CHECK-NEXT: lit.call {{.*}}@"takes_variadic_and_kw_only_args{{.*}}"(%x, %x, %[[VAR]], %x, %[[ZERO]])
    takes_variadic_and_kw_only_args(x, x, x, x, c=x)


fn takes_variadic_and_kw_only_params[
    a: Index, b: Index, *args: Index, c: Index, d: Index = `0`
]():
    pass


# CHECK-LABEL: lit.fn @"test_variadic_and_kw_only_params
fn test_variadic_and_kw_only_params[x: Index]():
    # CHECK: call {{.*}}takes_variadic_and_kw_only_param{{.*}}"<x, x, :variadic<index> [], x, 0>()
    takes_variadic_and_kw_only_params[x, x, c=x]()

    # CHECK: call {{.*}}takes_variadic_and_kw_only_param{{.*}}"<x, x, :variadic<index> [], x, x>()
    takes_variadic_and_kw_only_params[x, x, d=x, c=x]()

    # CHECK: call {{.*}}takes_variadic_and_kw_only_param{{.*}}"<x, x, :variadic<index> [x, x], x, 0>()
    takes_variadic_and_kw_only_params[x, x, x, x, c=x]()


# CHECK-LABEL: lit.fn @"test_variadic_and_kw_only_params_indirect
fn test_variadic_and_kw_only_params_indirect[x: Index,
    callee: fn [a: Index, b: Index, *args: Index, c: Index, d: Index = `0`]()->None]():

    # CHECK: lit.call{{.*}}bind_params(:!lit.generator<{{.*}}> callee, x, x, [], x, 0)]()
    callee[x, x, c=x]()

    # CHECK: call{{.*}}bind_params(:!lit.generator<{{.*}}> callee, x, x, [x, x], x, 0)]()
    callee[x, x, x, x, c=x]()


## Complex address space support

# Passing non-default address space through Self in an initializer.


# CHECK-LABEL: lit.fn @"initialize_in_addrspace
fn initialize_in_addrspace(
    ptr: UnsafePointer[ExampleRegPassable, address_space=AddressSpace(1)]
):

    # Get !lit.ref in addr space #1
    # CHECK-NEXT: [[PTRREF:%.*]] = lit.call{{.*}}@UnsafePointer::@"__getitem__{{.*}}(%ptr)

    # CHECK-NEXT: [[REGVAL:%.*]] = lit.call {{.*}}@ExampleRegPassable::@"__init__{{.*}}()

    # Use lit.ref.store to move into addrspace 1
    # CHECK-NEXT: lit.ref.store [[REGVAL]], [[PTRREF]] : <!ExampleRegPassable, mut #lit.any.origin, 1>
    ptr[] = ExampleRegPassable()


struct SomeRefItemStruct:
    fn __getitem__(self) -> ref [self] Int:
        pass


# CHECK-LABEL: lit.fn @"test_param_refitem
fn test_param_refitem[a: SomeRefItemStruct]():
    # CHECK-NEXT: !Int = <load_from_mem(:!lit.ref<!Int, imm {}> apply(:{{.*}}SomeRefItemStruct::@"__getitem__
    alias x = a[]


# Passing non-default address space through mut arg, must use temporary.
# CHECK-LABEL: lit.fn @"mutate_in_addrspace
fn mutate_in_addrspace(
    a: ExampleRegPassable,
    ptr: UnsafePointer[ExampleRegPassable, address_space=AddressSpace(1)],
):
    # Get !lit.ref in addr space #1
    # CHECK-NEXT: [[PTRREF:%.*]] = lit.call {{.*}}@UnsafePointer::@"__getitem__{{.*}}(%ptr)

    # Use a temporary to get an MLValue in the default address space.
    # CHECK-NEXT: %anonymous2A = lit.var.decl "anonymous
    # CHECK-NEXT: [[REGVAL:%.*]] = lit.ref.load [[PTRREF]] : <!ExampleRegPassable, mut #lit.any.origin, 1>
    # CHECK-NEXT: lit.ref.store [[REGVAL]], %anonymous2A
    # CHECK-NEXT: lit.call {{.*}}@ExampleRegPassable::@"mutateArg{{.*}}(%a, %anonymous2A)

    # Use lit.load/store to move back into addrspace 1
    # CHECK-NEXT: [[REGVAL:%.*]] = lit.load.consume %anonymous2A
    # CHECK-NEXT: lit.ref.store [[REGVAL]], [[PTRREF]] : <!ExampleRegPassable, mut #lit.any.origin, 1>
    a.mutateArg(ptr[])


@register_passable("trivial")
struct ExampleRegPassable:
    fn __init__(out self):
        pass

    fn mutateArg(self, mut other: Self):
        pass


## Partial Binding of Function Symbols With Implicit Parameters


struct Matrix[rows: Index, cols: Index]:
    pass


fn matmul_unrolled[I: Index](mut C: Matrix):
    pass


@always_inline
fn test_matrix_equal[
    func: fn (mut: Matrix) -> None
](mut C: Matrix) raises -> Bool:
    func(C)
    return True


# CHECK-LABEL: lit.fn @"partialBind
fn partialBind(mut C: Matrix[`1`, `2`]) raises:
    # CHECK-NEXT: %exp = lit.var.decl "exp
    # CHECK-NEXT: lit.call @{{.*}}::@"test_matrix_equal{{.*}}"[mut *"C`{{.*}}", mut *"__error__`{{.*}}", mut *"exp`{{.*}}"]
    # CHECK-SAME: <:!lit.generator<<?, index, index>[1](!lit.ref<@{{.*}}::@Matrix<*(0,0), *(0,1)>, mut *[0,0]> mut, |) -> !kgen.none>
    # CHECK-SAME: rebind(:!lit.generator<<?, index, index>[1]("C": !lit.ref<@{{.*}}::@Matrix<*(0,0), *(0,1)>, mut *[0,0]> mut) -> !kgen.none>
    # CHECK-SAME: @{{.*}}::@"matmul_unrolled{{.*}}"<0, ?, ?>), 1, 2>(%C, %__error__, %exp)
    var exp = test_matrix_equal[matmul_unrolled[`0`]](C)


# MOCO-692: [mojo-lang][ownership] Implicit conversion failure
# CHECK-LABEL: lit.fn @"test_implicit_conversion_bvalue
fn test_implicit_conversion_bvalue():
    # CHECK-NEXT: %foo = lit.var.decl
    # CHECK-NEXT: Struct1::@"__init__
    var foo = Struct1()
    # CHECK-NEXT: lit.ownership.use %foo
    # CHECK-NEXT: %anonymous2A = lit.var.decl
    # CHECK-NEXT: lit.call {{.*}}Struct2::@"__init__
    # CHECK-NEXT: lit.ref.immut
    # CHECK-NEXT: lit.call {{.*}}take_struct2
    take_struct2(foo^)


struct Struct1:
    fn __init__(out self):
        pass

    fn __moveinit__(out self, owned existing: Self):
        pass


struct Struct2:
    @implicit
    fn __init__(out self, owned foo: Struct1):
        pass


fn take_struct2(bar: Struct2):
    pass


fn pack_it[*Ts: AnyType](*args: *Ts) -> String:
    return String()


fn also_broken(r: Pointer[String]) -> String:
    return r[]


# MOCO-858: isSafeToUseValueDestForDirectResult doesn't handle aliasing through references
# CHECK-LABEL: lit.fn @"test_byref_slot_with_references
fn test_byref_slot_with_references():
    var f = String()

    # CHECK: [[RESULTTMP:%.*]] = lit.var.decl "__call_result_tmp__"
    # CHECK-NEXT: lit.call {{.*}}pack_it{{.*}}({{.*}},  [[RESULTTMP]])
    f = pack_it(f)
    # CHECK-NEXT: lit.call {{.*}}String::@"__moveinit__{{.*}}([[RESULTTMP]], %f)

    # CHECK: [[RESULTTMP:%.*]] = lit.var.decl "__call_result_tmp__"
    # CHECK-NEXT: lit.call {{.*}}also_broken{{.*}}({{.*}},  [[RESULTTMP]])
    f = also_broken(Pointer(to=f))
    # CHECK-NEXT: lit.call {{.*}}String::@"__moveinit__{{.*}}([[RESULTTMP]], %f)

    # CHECK: [[RESULTTMP:%.*]] = lit.var.decl "__call_result_tmp__"
    # CHECK-NEXT: lit.call {{.*}}also_broken{{.*}}({{.*}},  [[RESULTTMP]])
    f = also_broken(Pointer(to=f))
    # CHECK-NEXT: lit.call {{.*}}String::@"__moveinit__{{.*}}([[RESULTTMP]], %f)


# CHECK-LABEL: lit.fn @"test_byref_slot_closure_capture
fn test_byref_slot_closure_capture(owned x: String):
    # CHECK: lit.fn *"capture
    @parameter
    fn capture() -> String:
        return x

    # CHECK: %__call_result_tmp__
    # CHECK-NEXT: lit.call[{{.*}}: *"capture{{.*}}(%__call_result_tmp__)
    x = capture()
    # CHECK-NEXT: lit.call {{.*}}@String::@"__moveinit__{{.*}}(%__call_result_tmp__, %x)


fn test_int_ref(ref x: Int) -> ref [x] Int:
    return x


# CHECK-LABEL: lit.fn @"complex_ref_box_emission
fn complex_ref_box_emission[p: Int](a: Int):
    # Parameter ref just needs a box.
    _ = test_int_ref(p)
    # CHECK: [[VAR:%.*]] = lit.var.decl {{.*}}!lit.ref<!Int,
    # CHECK-NEXT: kgen.param.constant: !Int = <p>
    # CHECK-NEXT: lit.ref.store {{.*}}, [[VAR]]
    # CHECK-NEXT: [[TMP:%.*]] = lit.ref.immut [[VAR]]
    # CHECK-NEXT: lit.call {{.*}}test_int_ref{{.*}}([[TMP]])

    # Needs a conversion from IntegerLiteral to Int
    _ = test_int_ref(4)
    # CHECK: [[VAR:%.*]] = lit.var.decl {{.*}}!lit.ref<!Int,
    # CHECK-NEXT: kgen.param.constant: !Int = <{4}>
    # CHECK-NEXT: lit.ref.store {{.*}}, [[VAR]]
    # CHECK-NEXT: [[TMP:%.*]] = lit.ref.immut [[VAR]]
    # CHECK-NEXT: lit.call {{.*}}test_int_ref{{.*}}([[TMP]])

    # RValues infer as immutable, just like you can't pass them to mut.
    _ = test_int_ref(Int())
    # CHECK: [[VAR:%.*]] = lit.var.decl {{.*}}!lit.ref<!Int,
    # CHECK-NEXT: [[REGVAL:%.*]] = kgen.param.constant: !Int = <{0}>
    # CHECK-NEXT: lit.ref.store [[REGVAL]], [[VAR]]
    # CHECK-NEXT: [[TMP:%.*]] = lit.ref.immut [[VAR]]
    # CHECK-NEXT: lit.call {{.*}}test_int_ref{{.*}}([[TMP]])

    # TODO: Should work fine; needs generalized writeback.
    # _ = test_int_ref(a)
    # _ = test_int_ref(a+a)

# MOCO-1440 - Weird conditional conformance mismatch
struct ThingWithParam[X: Int]:
  @implicit
  fn __init__(out self: ThingWithParam[42], other: Bool): pass

fn test_cond_conformance(exclude: Bool):
    alias local_alias = 42
    var ptr : UnsafePointer[ThingWithParam[local_alias]]
    ptr[] = exclude


# MOCO-1442: Unnecessary copies being generated from owned values in constructors
@fieldwise_init  # This is copyable, but we don't want to.
struct Heavy(Copyable, Movable):
  pass

# This is intended to be a lightweight view of Heavy.
struct ViewOfHeavy:
  @implicit
  fn __init__(out self, h: Heavy): pass

fn takeOwnedValue(owned view: ViewOfHeavy): pass

# CHECK-LABEL: lit.fn @"testUnneededCopy
fn testUnneededCopy(heavy: Heavy):
  # CHECK-NEXT: [[TMP:%.*]] = lit.var.decl
  # CHECK-NEXT: lit.call {{.*}}ViewOfHeavy::@"__init__{{.*}}(%heavy, [[TMP]])
  # CHECK-NEXT: lit.call {{.*}}takeOwnedValue
  takeOwnedValue(heavy)
  # CHECK-NEXT: kgen.param.constant: none
