# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | kgen-opt -verify-parameters | FileCheck %s

from test_package.module import ParameterizedType


# CHECK-LABEL: lit.func @"reference_params_through_imported_struct
fn reference_params_through_imported_struct():
    # CHECK: kgen.param.constant = <10>
    var cached_type: ParameterizedType[__mlir_attr.`10 : index`]
    var value = cached_type.value


# CHECK-LABEL: lit.func @"ref_param_in_arg
# CHECK-SAME: <?, [[X:.*]]>[
# CHECK-SAME: lit.ref<{{.*}}ParameterizedType<[[X]]>{{.*}}> byref_result
fn ref_param_in_arg(x: ParameterizedType) -> ParameterizedType[x.value]:
    fn nested(x: ParameterizedType, y: ParameterizedType[x.value]):
        pass

    # CHECK: lit.alias.decl *"fn_type`3":
    # CHECK-SAME: signature<[2]<?, index>("x":
    # CHECK-SAME: "y": !lit.ref<{{.*}}ParameterizedType<*(0,0)>
    alias fn_type: fn (
        x: ParameterizedType, y: ParameterizedType[x.value]
    ) -> None = nested
    return x
