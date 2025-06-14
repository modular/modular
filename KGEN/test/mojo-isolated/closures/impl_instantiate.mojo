# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s


@fieldwise_init
struct MemType(Copyable, Movable):
    fn __add__(self, rhs: MemType) -> MemType:
        return MemType()


# CHECK-LABEL: lit.fn @"makes_escaping_closure
fn makes_escaping_closure(m: MemType, w: Int):
    # CHECK: [[IMPL:%.*]] = lit.var.decl "anonymous*" synth
    # CHECK-NEXT: lit.call @{{.*}}::@"__init__{{.*}}(%m, [[IMPL]])
    fn myclosure_with_mem_types(n: MemType) escaping -> MemType:
        return n + m

    # CHECK: [[A:%.*]] = lit.ref.load %a
    # CHECK-NEXT: [[IMPL:%.*]] = lit.var.decl "anonymous*" synth
    # CHECK-NEXT: lit.call @{{.*}}::@"__init__{{.*}}([[A]], %w, [[IMPL]])
    # CHECK-NEXT: [[WRAPPER:%.*]] = lit.var.decl "myclosure_with_reg_types" var
    # CHECK-NEXT: lit.call @{{.*}}::@"__init__{{.*}}([[IMPL]], [[WRAPPER]])
    var a = w

    fn myclosure_with_reg_types(x: Int) escaping -> Int:
        a = a + 1
        return x + w
