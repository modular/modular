# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen-translate %s -import-mojo --mojo-disable-builtins | kgen-opt -verify-parameters | FileCheck %s


# CHECK: lit.struct.decl @"`_CI_{{.*}}"<[[X:.*_X]], |>
# CHECK: lit.struct.decl @"fn{{.*}}"<p0, |>


@register_passable
struct Param[y: __mlir_type.index]:
    pass


# CHECK-LABEL: lit.func @"param()"
fn param():
    # CHECK: lit.alias.decl [[X:.*_X]] = <2>
    alias X = __mlir_attr.`2 : index`

    # CHECK: call {{.*}}_CI_{{.*}}__init__{{.*}}<[[X]]>
    # CHECK: call {{.*}}fn{{.*}}__init__{{.*}}<[[X]]>
    fn in_sig(y: Param[X]) escaping:
        pass
