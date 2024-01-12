# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen-translate %s -import-mojo | FileCheck %s


@value
struct MemType:
    fn __add__(self, rhs: MemType) -> MemType:
        return MemType()


# CHECK-LABEL: lit.func @"makes_escaping_closure
fn makes_escaping_closure(m: MemType, w: Int):
    # CHECK: [[IMPL:%.*]] = lit.varlet.decl "anonymous*" synth : !lit.ref<mut !escaping
    # CHECK-NEXT: lit.call @{{.*}}::@"__init__{{.*}}([[IMPL]], %m)
    fn myclosure_with_mem_types(n: MemType) escaping -> MemType:
        return n + m

    # CHECK: [[IMPL:%.*]] = lit.varlet.decl "anonymous*" synth : !lit.ref<mut !escaping
    # CHECK-NEXT: [[A:%.*]] = lit.ref.load %a
    # CHECK-NEXT: lit.call @{{.*}}::@"__init__{{.*}}([[IMPL]], [[A]], %w)
    # CHECK-NEXT: [[WRAPPER:%.*]] = lit.varlet.decl "anonymous*" synth
    # CHECK-NEXT: lit.call @{{.*}}::@"__init__{{.*}}([[WRAPPER]], [[IMPL]])
    var a = w

    fn myclosure_with_reg_types(x: Int) escaping -> Int:
        a = a + 1
        return x + w
