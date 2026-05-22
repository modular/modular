# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s

# This regression ensures associated alias witnesses are canonicalized the same
# way when a constraint is written directly on a trait parameter and when the
# same type value is rebound to an ancestor trait parameter.

from std.builtin.stubs import _type_is_eq_parse_time


trait WithA:
    comptime A: AnyType


@fieldwise_init
struct Concat[a: WithA, b: WithA] where _type_is_eq_parse_time[a.A, b.A]():
    pass


# CHECK-LABEL: lit.trait.decl @WithASum
trait WithASum(WithA):
    pass


# CHECK-LABEL: lit.fn @"concat_add
# CHECK-SAME: #kgen.get_witness<:!WithASum p, "constraint_trait_alias_upcast::WithA", "A">
def concat_add[
    p: WithASum, o: WithA
](lhs: p, rhs: o) -> Concat[p, o] where _type_is_eq_parse_time[p.A, o.A]():
    return {}
