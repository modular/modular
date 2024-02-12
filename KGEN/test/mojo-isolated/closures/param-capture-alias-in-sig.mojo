# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %translate-with-packages %s | kgen-opt -verify-parameters | FileCheck %s


# CHECK: lit.struct.decl @"`_CI_{{.*}}"<[[X:.*]], |>
# CHECK: lit.struct.decl @"fn{{.*}}"<p0, |>


@register_passable
struct Param[y: __mlir_type.index]:
    pass


# CHECK-LABEL: lit.func @"param()"
fn param():
    # CHECK: lit.alias.decl [[X:.*]] = <2>
    alias X = __mlir_attr.`2 : index`

    # CHECK: call {{.*}}_CI_{{.*}}__init__{{.*}}<[[X]]>
    # CHECK: call {{.*}}fn{{.*}}__init__{{.*}}<[[X]]>
    fn in_sig(y: Param[X]) escaping:
        pass
