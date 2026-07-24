# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s -verify-diagnostics -mojo-diagnose-missing-movable-conformance

# Struct with no conformance list at all -- the fix-it must synthesize a new
# `(...)` list from scratch.
# expected-warning @below {{struct does not explicitly conform to 'Movable'}}
struct NoConformanceList:
    pass


# Struct with an existing conformance list (constrained entry) that doesn't
# mention `Movable` (and, unlike `Copyable`, `ImplicitlyDeletable` doesn't
# transitively refine `Movable` either) -- the fix-it must append to the
# existing list.
# expected-warning @below {{struct does not explicitly conform to 'Movable'}}
struct NonEmptyConformanceListWithWhere(ImplicitlyDeletable where False):
    pass


# Same as above, but the existing entry has no `where` clause.
# expected-warning @below {{struct does not explicitly conform to 'Movable'}}
struct NonEmptyConformanceListNoWhere(ImplicitlyDeletable):
    pass


# An explicit but empty conformance list -- the fix-it must insert inside
# the existing parens, not create a second `(...)` list.
# expected-warning @below {{struct does not explicitly conform to 'Movable'}}
struct EmptyConformanceList():
    pass


# Already explicitly conforms to `Movable` -- no warning expected.
struct ExplicitlyMovable(Movable):
    pass


# Already explicitly opts out via the `where False` idiom -- no warning
# expected, since the conformance is already stated explicitly.
struct ExplicitlyOptedOut(Movable where False):
    pass


# Conforms to a trait that itself transitively refines `Movable`
# (RegisterPassable refines Movable) -- this already gives the struct an
# explicit, unambiguous `Movable` status, so no warning is expected even
# though `Movable` is never spelled out directly.
struct RefinesMovableTransitively(RegisterPassable):
    pass


# No conformance list, but a trailing `where` clause immediately follows the
# param list -- regression test for a fix-it that used to glue its synthesized
# `(...)` list onto `where` with no space in between.
# expected-warning @below {{struct does not explicitly conform to 'Movable'}}
struct PredicateOnStructOuter[value: Int] where value >= 0:
    pass
