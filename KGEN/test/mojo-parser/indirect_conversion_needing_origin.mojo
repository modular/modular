# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s

# Inferring the origin parameter of a callee's argument can require an implicit
# conversion whose constructor binds its operand by `ref`.  When the operand is
# an rvalue, it has to be materialized into memory first so that the inferred
# origin has something to point at.


struct RefWrapper[origin: ImmOrigin](Copyable, Movable):
    var n: Int

    @implicit
    def __init__[
        value_origin: ImmOrigin, //
    ](ref[value_origin] value: Int, out self: RefWrapper[value_origin]):
        self.n = value


def take_wrapper(w: RefWrapper) -> Int:
    return w.n


# CHECK-LABEL: lit.fn @"infer_origin_of_materialized_rvalue()"
def infer_origin_of_materialized_rvalue():
    var a = 1
    var b = 2

    # The `a + b` rvalue is spilled to a temporary, and both the implicit
    # `RefWrapper` conversion and `take_wrapper` itself are parameterized on
    # that temporary's origin.
    # CHECK: %[[SUM:.*]] = lit.call tail @std::@builtin::@stubs::@SIMD::@"__add__
    # CHECK: %[[TMP:.*]] = lit.var.decl "anonymous*" synth : !lit.ref<!Int, mut *"[[ORIGIN:[^"]*]]">
    # CHECK: lit.ref.store %{{.*}}, %[[TMP]]
    # CHECK: lit.call @{{.*}}@RefWrapper::@"__init__{{.*}}"[mut {{.*}}]<:origin<false> (mutcast mut *"[[ORIGIN]]")
    # CHECK: lit.call @{{.*}}@"take_wrapper{{.*}}"[muttoimm {{.*}}]<:origin<false> (mutcast mut *"[[ORIGIN]]")
    _ = take_wrapper(a + b)


def take_span(s: Span[...]):
    pass


# CHECK-LABEL: lit.fn @"infer_origin_of_materialized_literal()"
def infer_origin_of_materialized_literal():
    # Same thing one level deeper: the list literal first materializes an
    # `Array` temporary, which the implicit `Span` conversion then borrows.
    # CHECK: %[[ARRAY:.*]] = lit.var.decl "__call_result_tmp__" synth : !lit.ref<!lit.struct<#Array {{.*}}, mut *"[[ORIGIN:[^"]*]]">
    # CHECK: lit.call @std::@builtin::@stubs::@Span::@"__init__{{.*}}"<:!Bool {:scalar<bool> false}, :origin<false> (mutcast mut *"[[ORIGIN]]")
    # CHECK: lit.call @{{.*}}@"take_span{{.*}}"<:!Bool {:scalar<bool> false}, :origin<false> (mutcast mut *"[[ORIGIN]]")
    take_span([1, 2, 3])
