# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s -I=%S/inputs | FileCheck %s

from test_package.module import ParameterizedType


# CHECK-LABEL: lit.fn @"reference_params_through_imported_struct
def reference_params_through_imported_struct():
    # CHECK: kgen.param.constant: !Int = <{10}>
    var cached_type: ParameterizedType[10]
    var value = cached_type.value


# CHECK-LABEL: lit.fn @"ref_param_in_arg
# CHECK-SAME: <?, [[X:.*]]: !Int>[
# CHECK-SAME: lit.ref<!lit.struct<#ParameterizedType <:!Int [[X]]>>{{.*}}> byref_result
def ref_param_in_arg(x: ParameterizedType) -> ParameterizedType[x.value]:
    def nested(x: ParameterizedType, y: ParameterizedType[x.value]):
        pass

    # CHECK: lit.alias.decl *"def_type`3":
    # CHECK-SAME: generator<<?, "x.value`2x": !Int>[2]("x":
    # CHECK-SAME: "y": !lit.ref<{{.*}}#ParameterizedType <:!Int *(0,0)>
    comptime def_type: def(
        x: ParameterizedType, y: ParameterizedType[x.value]
    ) thin -> None = nested
    return x
