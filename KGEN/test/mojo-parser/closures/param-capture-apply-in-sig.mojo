# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen-translate -import-mojo --mojo-disable-builtins %s | kgen-opt -verify-parameters | FileCheck %s

alias Int = __mlir_type.index


@value
@register_passable
struct Foo[x: Int]:
    var b: Int

    fn get(self) -> Int:
        return self.b


# CHECK: lit.struct.decl @"`_CI_{{.*}}"<[[a:.*_a]], [[Y:\*".*"]]: {{.*}}Foo<[[a]]>
# CHECK: lit.func @"__call__
# CHECK-SAME: @Foo<apply(:{{.*}}@Foo::@"get{{.*}}"<[[a]]>, [[Y]])>


fn alias_ref_apply_in_sig[a: Int]():
    alias Y = Foo[a](__mlir_attr.`2 : index`)

    fn p_capture(x: Int, y: Foo[Y.get()]) escaping -> Int:
        return Foo[a](x).get()
