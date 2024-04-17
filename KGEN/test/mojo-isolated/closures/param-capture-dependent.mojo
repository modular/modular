# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | kgen-opt -verify-parameters | FileCheck %s


@value
@register_passable
struct Foo[a: Int]:
    var b: Int


# CHECK: lit.struct.decl @"`_CI_{{.*}}"<[[a:.*a]]: !Int, [[X:\*".*"]]: [[FOO:.*]]<:!Int [[a]]>, |>
# CHECK: lit.func @"__call__{{.*}}({{.*}}<:!Int [[a]], :[[FOO]]<:!Int [[a]]>
# CHECK-NEXT: [[VAR1:%.*]] = lit.ref.struct.ger %0[field0]
# CHECK-NEXT: [[VAR2:%.*]] = lit.ref.load [[VAR1]]
# CHECK-NEXT: kgen.param.constant: !Int = <#lit.struct.extract<:[[FOO]]<:!Int [[a]]> [[X]], "b">>
fn parameter_capture[a: Int](c: Int) -> fn (x: Int) escaping -> Int:
    alias X = Foo[a](1)

    fn p_capture(x: Int) -> Int:
        return X.b + c

    return p_capture
