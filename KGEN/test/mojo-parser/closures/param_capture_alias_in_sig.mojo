# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s


# CHECK: lit.struct.decl @"`_CI_{{.*}}"<[[X:.*]], |>
# CHECK: lit.struct.decl @"fn{{.*}}"<p0, |>


@register_passable
struct Param[y: __mlir_type.index]:
    pass


# CHECK-LABEL: lit.fn @"param
fn param[X: __mlir_type.index]():
    # CHECK: call {{.*}}_CI_{{.*}}__init__{{.*}}<X>
    # CHECK: call {{.*}}fn{{.*}}__init__{{.*}}<X>
    fn in_sig(y: Param[X]) escaping:
        pass
