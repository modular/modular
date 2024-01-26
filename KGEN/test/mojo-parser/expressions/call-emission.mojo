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


fn takes_kw_only_args(a: Int, b: Int = `1`, *, c: Int, d: Int = `2`):
    pass


# CHECK-LABEL: lit.func @"test_kw_only_args{{.*}}"(%x: index borrow)
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


# CHECK-LABEL: lit.func @"test_kw_only_indirect{{.*}}"(%x: index borrow)
fn test_kw_only_indirect(x: Int):
    alias also_takes_kw_only_args = takes_kw_only_args

    # CHECK-DAG: %[[C1:.*]] = kgen.param.constant = <1>
    # CHECK-DAG: %[[C2:.*]] = kgen.param.constant = <2>
    # CHECK-NEXT: lit.call_param[{{.*}}](%x, %[[C1]], %x, %[[C2]])
    also_takes_kw_only_args(x, c=x)

    # CHECK-DAG: %[[C1:.*]] = kgen.param.constant = <1>
    # CHECK-NEXT: lit.call_param[{{.*}}](%x, %[[C1]], %x, %x)
    also_takes_kw_only_args(x, d=x, c=x)


fn takes_kw_only_params[a: Int, b: Int = `1`, *, c: Int, d: Int = `2`]():
    pass


# CHECK-LABEL: lit.func @"test_kw_only_params{{.*}}"<x>
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


# CHECK-LABEL: lit.func @"test_kw_only_params_indirect{{.*}}"<x>
fn test_kw_only_params_indirect[x: Int]():
    # CHECK: lit.alias.decl [[CALLEE:.*]]: !lit.signature
    alias also_takes_kw_only_params = takes_kw_only_params

    # CHECK: call_param{{.*}}bind_signature(:!lit.signature<{{.*}}> [[CALLEE]], x, 1, x, 2)]()
    also_takes_kw_only_params[x, c=x]()

    # CHECK: call_param{{.*}}bind_signature(:!lit.signature<{{.*}}> [[CALLEE]], x, 1, x, x)]()
    also_takes_kw_only_params[x, d=x, c=x]()
