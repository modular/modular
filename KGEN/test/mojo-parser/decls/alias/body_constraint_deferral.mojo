# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -verify-diagnostics %s

##===----------------------------------------------------------------------===##
# Auto-Predication During Parametric Alias Signature Emission
#
# Parametric aliases use the same parameter-list emission path as functions and
# structs. These cases verify that body constraints from types in an alias
# signature can be deferred until the alias's trailing `where` clauses are in
# scope.
##===----------------------------------------------------------------------===##


@fieldwise_init
struct PositiveOnly[N: Int] where N > 0:
    pass


comptime ConstrainedAlias[N: Int]: AnyType where N > 0 = PositiveOnly[N]


# Parameter declaration type position. Binding `PositiveOnly[K]` is unprovable
# while the parameter list is emitted, then discharged by the alias's trailing
# `where K > 0`.
comptime alias_parameter_type[K: Int, X: PositiveOnly[K]] where K > 0 = X


# Generator result type position. The declared alias result type binds
# `PositiveOnly[K]`, and the trailing `where` makes the binding valid.
comptime alias_result_type[
    K: Int
]: PositiveOnly[K] where K > 0 = PositiveOnly[K]()


# Parameter declaration type position through a constrained alias.
comptime alias_parameter_type_via_alias[
    K: Int, X: ConstrainedAlias[K]
] where K > 0 = X


# Generator result type position through a constrained alias.
comptime alias_result_type_via_alias[
    K: Int
]: ConstrainedAlias[K] where K > 0 = PositiveOnly[K]()
