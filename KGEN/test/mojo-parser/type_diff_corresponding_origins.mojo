# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated -verify-diagnostics %s

# Regression test for the diagnostic type-differ (`~MojoInflightDiag`).
#
# When a struct fails to satisfy a trait method, the differ must look past the
# implicit `self` origin (which positionally corresponds between the two
# signatures) and report the real argument-type mismatch: here `.argument` is
# `Float64` in the trait requirement but `Int` in the struct's method.
#
# The names are intentionally long (>30 chars) to trigger the differ's
# long-type drill-down heuristic. This exercises the differ directly and does
# not depend on the closure-trait machinery.

# expected-note @below {{trait 'LongishTraitNameForTheDiff' declared here}}
trait LongishTraitNameForTheDiff:
    # expected-note @below {{no 'method_with_a_name' candidates have type 'def(self: StructWithLongishNameForDiff, argument: Float64) thin -> None'}}
    def method_with_a_name(self, argument: Float64):
        ...


# expected-error @below {{'StructWithLongishNameForDiff' does not implement all requirements for 'LongishTraitNameForTheDiff'}}
struct StructWithLongishNameForDiff(LongishTraitNameForTheDiff):
    # expected-note @below {{candidate declared here with type 'def(self: StructWithLongishNameForDiff, argument: Int) thin -> None'}}
    # expected-note @below {{.argument of the first type is 'Float64' but the second type is 'Int'}}
    def method_with_a_name(self, argument: Int):
        pass
