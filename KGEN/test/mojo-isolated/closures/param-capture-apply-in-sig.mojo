# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %translate-with-packages %s | kgen-opt -verify-parameters | FileCheck %s


@value
@register_passable
struct Foo[x: int]:
    var b: int

    fn get(self) -> int:
        return self.b


# CHECK: lit.struct.decl @"`_CI_{{.*}}"<[[a:.*]], [[Y:\*".*"]]: {{.*}}Foo<[[a]]>
# CHECK: lit.func @"__call__
# CHECK-SAME: @Foo<apply(:{{.*}}@Foo::@"get{{.*}}"<[[a]]>, [[Y]])>


fn alias_ref_apply_in_sig[a: int]():
    alias Y = Foo[a](__mlir_attr.`2 : index`)

    fn p_capture(x: int, y: Foo[Y.get()]) escaping -> int:
        return Foo[a](x).get()
