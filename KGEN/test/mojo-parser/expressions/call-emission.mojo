# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s

# ===----------------------------------------------------------------------=== #
# Stubs to allow testing without builtins
# ===----------------------------------------------------------------------=== #

alias Int = __mlir_type.index

alias `1` = __mlir_attr.`1 : index`
alias `2` = __mlir_attr.`2 : index`

# ===----------------------------------------------------------------------=== #
# Actual tests
# ===----------------------------------------------------------------------=== #


fn has_default_args(a: Int, b: Int = `1`, c: Int = `2`):
    pass


# CHECK-LABEL: lit.func @"test_kw_arg_passing
fn test_kw_arg_passing(x: Int, y: Int, z: Int):
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
fn test_kw_arg_passing_indirect(x: Int, y: Int, z: Int):
    alias callee = has_default_args

    # CHECK-DAG: %[[C1:.*]] = kgen.param.constant = <1>
    # CHECK-NEXT: lit.call_param[{{.*}}](%x, %[[C1]], %z)
    callee(x, c=z)

    # CHECK-NEXT: lit.call_param[{{.*}}](%x, %y, %z)
    callee(c=z, b=y, a=x)


fn has_default_params[a: Int, b: Int = `1`, c: Int = `2`]():
    pass


# CHECK-LABEL: lit.func @"test_kw_param_passing
fn test_kw_param_passing[x: Int, y: Int, z: Int]():
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
fn test_kw_param_passing_indirect[x: Int, y: Int, z: Int]():
    # CHECK: lit.alias.decl [[CALLEE:.*]]: !lit.signature
    alias callee = has_default_params

    # CHECK: call_param{{.*}}bind_signature(:!lit.signature<{{.*}}> [[CALLEE]], x, 1, z)]()
    callee[x, c=z]()

    # CHECK: call_param{{.*}}bind_signature(:!lit.signature<{{.*}}> [[CALLEE]], x, y, z)]()
    callee[c=z, b=y, a=x]()


@value
struct MyCallable:
    fn __call__(self, m: Int, n: Int = `2`):
        pass


# CHECK-LABEL: lit.func @"test_callable_object
fn test_callable_object(x: Int, y: Int):
    # CHECK: %[[CALLABLE:.*]] = lit.varlet.decl {{.*}}: !lit.ref<!MyCallable
    var callable = MyCallable()

    # CHECK-DAG: %[[IMMREF:.*]] = lit.ref.immut %[[CALLABLE]]
    # CHECK-DAG: %[[C2:.*]] = kgen.param.constant = <2>
    # CHECK-NEXT: call {{.*}}@MyCallable::@"__call__{{.*}}(%[[IMMREF]], %x, %[[C2]])
    callable(x)

    # CHECK-DAG: %[[IMMREF:.*]] = lit.ref.immut %[[CALLABLE]]
    # CHECK-NEXT: call {{.*}}@MyCallable::@"__call__{{.*}}(%[[IMMREF]], %y, %x)
    callable(n=x, m=y)


fn takes_kw_only_args(a: Int, b: Int = `1`, *, c: Int, d: Int = `2`):
    pass


# CHECK-LABEL: lit.func @"test_kw_only_args
fn test_kw_only_args(x: Int):
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
fn test_kw_only_indirect(x: Int):
    alias callee = takes_kw_only_args

    # CHECK-DAG: %[[C1:.*]] = kgen.param.constant = <1>
    # CHECK-DAG: %[[C2:.*]] = kgen.param.constant = <2>
    # CHECK-NEXT: lit.call_param[{{.*}}](%x, %[[C1]], %x, %[[C2]])
    callee(x, c=x)

    # CHECK-DAG: %[[C1:.*]] = kgen.param.constant = <1>
    # CHECK-NEXT: lit.call_param[{{.*}}](%x, %[[C1]], %x, %x)
    callee(x, d=x, c=x)


fn takes_kw_only_params[a: Int, b: Int = `1`, *, c: Int, d: Int = `2`]():
    pass


# CHECK-LABEL: lit.func @"test_kw_only_params
fn test_kw_only_params[x: Int]():
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
fn test_kw_only_params_indirect[x: Int]():
    # CHECK: lit.alias.decl [[CALLEE:.*]]: !lit.signature
    alias callee = takes_kw_only_params

    # CHECK: call_param{{.*}}bind_signature(:!lit.signature<{{.*}}> [[CALLEE]], x, 1, x, 2)]()
    callee[x, c=x]()

    # CHECK: call_param{{.*}}bind_signature(:!lit.signature<{{.*}}> [[CALLEE]], x, 1, x, x)]()
    callee[x, d=x, c=x]()
