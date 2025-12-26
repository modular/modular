# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s

from test_package.module import ParameterizedType


# CHECK-LABEL: lit.fn @"reference_params_through_imported_struct
fn reference_params_through_imported_struct():
    # CHECK: kgen.param.constant: !Int = <{10}>
    var cached_type: ParameterizedType[10]
    var value = cached_type.value


# CHECK-LABEL: lit.fn @"ref_param_in_arg
# CHECK-SAME: <?, [[X:.*]]: !Int>[
# CHECK-SAME: lit.ref<!lit.struct<#ParameterizedType <:!Int [[X]]>>{{.*}}> byref_result
fn ref_param_in_arg(x: ParameterizedType) -> ParameterizedType[x.value]:
    fn nested(x: ParameterizedType, y: ParameterizedType[x.value]):
        pass

    # CHECK: lit.alias.decl *"fn_type`3":
    # CHECK-SAME: generator<<?, "x.value`2x": !Int>[2]("x":
    # CHECK-SAME: "y": !lit.ref<{{.*}}#ParameterizedType <:!Int *(0,0)>
    comptime fn_type: fn (
        x: ParameterizedType, y: ParameterizedType[x.value]
    ) -> None = nested
    return x
