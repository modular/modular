# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate -import-mojo %s -verify-diagnostics | kgen-opt -verify-parameters | FileCheck %s

from test_package.module import ParameterizedType

# CHECK-LABEL: lit.func @"reference_params_through_imported_struct
fn reference_params_through_imported_struct():
    # CHECK: kgen.param.constant: !Int = <#lit.struct<{value = 10}>>
    let cached_type: ParameterizedType[10]
    let value = cached_type.value

# CHECK-LABEL: lit.func @"ref_param_in_arg
# CHECK-SAME: <?, [[X:.*]]: !Int>
# CHECK-SAME: pointer<{{.*}}ParameterizedType<:!Int [[X]]>{{.*}}> byref_result
fn ref_param_in_arg(x: ParameterizedType) -> ParameterizedType[x.value]:
    # CHECK: lit.alias.fwd_decl "{{.*}}fn_type"
    # CHECK-SAME: signature<<?, !Int>("x":
    # CHECK-SAME: "y": !kgen.pointer<{{.*}}ParameterizedType<:!Int *(0,0)>
    alias fn_type: fn(x: ParameterizedType, y: ParameterizedType[x.value]) -> None
    return x
