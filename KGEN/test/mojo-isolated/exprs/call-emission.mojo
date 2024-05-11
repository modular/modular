# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s


fn has_default_args(a: int, b: int = `1`, c: int = `2`):
    pass


# CHECK-LABEL: lit.func @"test_kw_arg_passing
fn test_kw_arg_passing(x: int, y: int, z: int):
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


# CHECK-LABEL: lit.func @"test_kw_arg_passing_indirect
fn test_kw_arg_passing_indirect(x: int, y: int, z: int):
    alias callee = has_default_args

    # CHECK-DAG: %[[C1:.*]] = kgen.param.constant = <1>
    # CHECK-NEXT: lit.call[{{.*}}](%x, %[[C1]], %z)
    callee(x, c=z)

    # CHECK-NEXT: lit.call[{{.*}}](%x, %y, %z)
    callee(c=z, b=y, a=x)


fn has_default_params[a: int, b: int = `1`, c: int = `2`]():
    pass


# CHECK-LABEL: lit.func @"test_kw_param_passing
fn test_kw_param_passing[x: int, y: int, z: int]():
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


# CHECK-LABEL: lit.func @"test_kw_param_passing_indirect
fn test_kw_param_passing_indirect[x: int, y: int, z: int]():
    # CHECK: lit.alias.decl [[CALLEE:.*]]: !lit.signature
    alias callee = has_default_params

    # CHECK: call{{.*}}bind_signature(:!lit.signature<{{.*}}> [[CALLEE]], x, 1, z)]()
    callee[x, c=z]()

    # CHECK: call{{.*}}bind_signature(:!lit.signature<{{.*}}> [[CALLEE]], x, y, z)]()
    callee[c=z, b=y, a=x]()


@value
struct MyCallable:
    fn __call__(self, m: int, n: int = `2`):
        pass


# CHECK-LABEL: lit.func @"test_callable_object
fn test_callable_object(x: int, y: int):
    # CHECK: %[[CALLABLE:.*]] = lit.var.decl {{.*}}: !lit.ref<!MyCallable
    var callable = MyCallable()

    # CHECK-DAG: %[[IMMREF:.*]] = lit.ref.immut %[[CALLABLE]]
    # CHECK-DAG: %[[C2:.*]] = kgen.param.constant = <2>
    # CHECK-NEXT: call {{.*}}@MyCallable::@"__call__{{.*}}(%[[IMMREF]], %x, %[[C2]])
    callable(x)

    # CHECK-DAG: %[[IMMREF:.*]] = lit.ref.immut %[[CALLABLE]]
    # CHECK-NEXT: call {{.*}}@MyCallable::@"__call__{{.*}}(%[[IMMREF]], %y, %x)
    callable(n=x, m=y)


fn takes_kw_only_args(a: int, b: int = `1`, *, c: int, d: int = `2`):
    pass


# CHECK-LABEL: lit.func @"test_kw_only_args
fn test_kw_only_args(x: int):
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


# CHECK-LABEL: lit.func @"test_kw_only_indirect
fn test_kw_only_indirect(x: int):
    alias callee = takes_kw_only_args

    # CHECK-DAG: %[[C1:.*]] = kgen.param.constant = <1>
    # CHECK-DAG: %[[C2:.*]] = kgen.param.constant = <2>
    # CHECK-NEXT: lit.call[{{.*}}](%x, %[[C1]], %x, %[[C2]])
    callee(x, c=x)

    # CHECK-DAG: %[[C1:.*]] = kgen.param.constant = <1>
    # CHECK-NEXT: lit.call[{{.*}}](%x, %[[C1]], %x, %x)
    callee(x, d=x, c=x)


fn takes_kw_only_params[a: int, b: int = `1`, *, c: int, d: int = `2`]():
    pass


# CHECK-LABEL: lit.func @"test_kw_only_params
fn test_kw_only_params[x: int]():
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


# CHECK-LABEL: lit.func @"test_kw_only_params_indirect
fn test_kw_only_params_indirect[x: int]():
    # CHECK: lit.alias.decl [[CALLEE:.*]]: !lit.signature
    alias callee = takes_kw_only_params

    # CHECK: call{{.*}}bind_signature(:!lit.signature<{{.*}}> [[CALLEE]], x, 1, x, 2)]()
    callee[x, c=x]()

    # CHECK: call{{.*}}bind_signature(:!lit.signature<{{.*}}> [[CALLEE]], x, 1, x, x)]()
    callee[x, d=x, c=x]()


fn takes_variadic_and_kw_only_args(
    a: int, b: int, *args: int, c: int, d: int = `0`
):
    pass


# CHECK-LABEL: lit.func @"test_variadic_and_kw_only_args
fn test_variadic_and_kw_only_args(x: int):
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
    a: int, b: int, *args: int, c: int, d: int = `0`
]():
    pass


# CHECK-LABEL: lit.func @"test_variadic_and_kw_only_params
fn test_variadic_and_kw_only_params[x: int]():
    # CHECK: call {{.*}}takes_variadic_and_kw_only_param{{.*}}"<x, x, :variadic<index> [], x, 0>()
    takes_variadic_and_kw_only_params[x, x, c=x]()

    # CHECK: call {{.*}}takes_variadic_and_kw_only_param{{.*}}"<x, x, :variadic<index> [], x, x>()
    takes_variadic_and_kw_only_params[x, x, d=x, c=x]()

    # CHECK: call {{.*}}takes_variadic_and_kw_only_param{{.*}}"<x, x, :variadic<index> [x, x], x, 0>()
    takes_variadic_and_kw_only_params[x, x, x, x, c=x]()


# CHECK-LABEL: lit.func @"test_variadic_and_kw_only_params_indirect
fn test_variadic_and_kw_only_params_indirect[x: int]():
    # CHECK: lit.alias.decl [[CALLEE:.*]]: !lit.signature
    alias callee = takes_variadic_and_kw_only_params

    # CHECK: call{{.*}}bind_signature(:!lit.signature<{{.*}}> [[CALLEE]], x, x, [], x, 0)]()
    callee[x, x, c=x]()

    # CHECK: call{{.*}}bind_signature(:!lit.signature<{{.*}}> [[CALLEE]], x, x, [x, x], x, 0)]()
    callee[x, x, x, x, c=x]()


## Complex address space support

# Passing non-default address space through initself.

# CHECK-LABEL: lit.func @"initialize_in_addrspace
fn initialize_in_addrspace(ptr: UnsafePointer[ExampleRegPassable, AddressSpace(1)]):
    # Get !lit.ref in addr space #1
    # CHECK-NEXT: [[PTRREF:%.*]] = lit.call {{.*}}@UnsafePointer::@"__refitem__{{.*}}(%ptr)
    # CHECK-NEXT: [[REFPTR1:%.*]] = lit.call {{.*}}Reference::@"__mlir_ref__{{.*}}([[PTRREF]]){{.*}}mut #lit.lifetime, 1>>

    # CHECK-NEXT: %anonymous2A = lit.var.decl "anonymous
    # CHECK-NEXT: lit.call {{.*}}@ExampleRegPassable::@"__init__{{.*}}(%anonymous2A)

    # Use lit.load/store to move into addrspace 1
    # CHECK-NEXT: [[REGVAL:%.*]] = lit.load.consume %anonymous2A
    # CHECK-NEXT: lit.ref.store [[REGVAL]], [[REFPTR1]] : <!ExampleRegPassable, mut #lit.lifetime, 1> 
    ptr[] = ExampleRegPassable()

# Passing non-default address space through inout arg, must use temporary.
# CHECK-LABEL: lit.func @"mutate_in_addrspace
fn mutate_in_addrspace(a: ExampleRegPassable, ptr: UnsafePointer[ExampleRegPassable, AddressSpace(1)]):
    # Get !lit.ref in addr space #1
    # CHECK-NEXT: [[PTRREF:%.*]] = lit.call {{.*}}@UnsafePointer::@"__refitem__{{.*}}(%ptr)
    # CHECK-NEXT: [[REFPTR1:%.*]] = lit.call {{.*}}Reference::@"__mlir_ref__{{.*}}([[PTRREF]]){{.*}}mut #lit.lifetime, 1>>

    # Use a temporary to get an MLValue in the default address space.
    # CHECK-NEXT: %anonymous2A = lit.var.decl "anonymous
    # CHECK-NEXT: [[REGVAL:%.*]] = lit.ref.load [[REFPTR1]] : <!ExampleRegPassable, mut #lit.lifetime, 1>
    # CHECK-NEXT: lit.ref.store [[REGVAL]], %anonymous2A
    # CHECK-NEXT: lit.call {{.*}}@ExampleRegPassable::@"mutateArg{{.*}}(%a, %anonymous2A)

    # Use lit.load/store to move back into addrspace 1
    # CHECK-NEXT: [[REGVAL:%.*]] = lit.load.consume %anonymous2A
    # CHECK-NEXT: lit.ref.store [[REGVAL]], [[REFPTR1]] : <!ExampleRegPassable, mut #lit.lifetime, 1> 
    a.mutateArg(ptr[])

@register_passable("trivial")
struct ExampleRegPassable:
  fn __init__(inout self): pass
  fn mutateArg(self, inout other: Self): pass



