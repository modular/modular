# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Test that the message on a conditional-conformance `where` clause
# (`struct S(Trait where (cond, "message"))`) is surfaced when a value of the
# struct type fails to conform to the trait. The message appears as a note at
# the `where` clause, across the conformance-failure diagnostics a user can
# hit: a generic trait-bounded parameter, a concrete trait-typed parameter,
# and a `var` initializer binding to a trait.
#
# Each scenario uses its own struct so per-section notes don't bleed across
# sites (mirrors the convention in `struct_body_constraints.mojo`).

# RUN: %parse-mojo-isolated -verify-diagnostics %s


trait Marker:
    pass


struct Yes(Marker):
    pass


struct No:
    pass


##===----------------------------------------------------------------------===##
# Satisfied conditional conformance - positive case (no error, no note).
##===----------------------------------------------------------------------===##


struct OkBox[T: Deinitable](
    Marker where (conforms_to(T, Marker), "OkBox[T] is a Marker only when T is")
):
    pass


def wants_marker_ok[U: Marker & Deinitable](x: U):
    pass


def use_ok(b: OkBox[Yes]):
    wants_marker_ok(b)


##===----------------------------------------------------------------------===##
# Violated via a generic trait-bounded parameter.
##===----------------------------------------------------------------------===##


struct GenBox[T: Deinitable](
    # expected-note @below {{unsatisfied conditional conformance: GenBox[T] requires T to be a Marker}}
    Marker where (conforms_to(T, Marker), "GenBox[T] requires T to be a Marker")
):
    pass


# expected-note @below {{function declared here}}
def wants_marker_gen[U: Marker & Deinitable](x: U):
    pass


def use_gen(b: GenBox[No]):
    # expected-error @below {{does not conform to trait 'Deinitable & Marker'}}
    wants_marker_gen(b)


##===----------------------------------------------------------------------===##
# Violated via a concrete trait-typed parameter.
##===----------------------------------------------------------------------===##


struct ConcreteBox[T: Deinitable](
    # expected-note @below {{unsatisfied conditional conformance: ConcreteBox[T] requires T to be a Marker}}
    Marker where (
        conforms_to(T, Marker), "ConcreteBox[T] requires T to be a Marker"
    )
):
    pass


# expected-note @below {{function declared here}}
def wants_marker_concrete(x: Marker & Deinitable):
    pass


def use_concrete(b: ConcreteBox[No]):
    # expected-error @below {{cannot be converted from 'ConcreteBox[No]' to 'Deinitable & Marker'}}
    wants_marker_concrete(b)


##===----------------------------------------------------------------------===##
# Violated via a `var` initializer binding to a trait.
##===----------------------------------------------------------------------===##


struct VarBox[T: Deinitable](
    # expected-note @below {{unsatisfied conditional conformance: VarBox[T] requires T to be a Marker}}
    Marker where (conforms_to(T, Marker), "VarBox[T] requires T to be a Marker")
):
    pass


def use_var(b: VarBox[No]):
    # expected-error @below {{cannot implicitly convert 'VarBox[No]' value to 'Deinitable & Marker' in 'var' initializer}}
    var m: Marker & Deinitable = b


##===----------------------------------------------------------------------===##
# Requiring a propagated ancestor trait: the message is written on the derived
# conformance but is carried down to the propagated ancestor constraint, so
# requiring the bare ancestor still surfaces it (see the design doc).
##===----------------------------------------------------------------------===##


trait Base:
    pass


trait Refined(Base):
    pass


struct RefinedBox[T: Deinitable](
    # expected-note @below {{unsatisfied conditional conformance: RefinedBox[T] requires T to be a Marker}}
    Refined where (conforms_to(T, Marker), "RefinedBox[T] requires T to be a Marker")
):
    pass


# expected-note @below {{function declared here}}
def wants_base[U: Base & Deinitable](x: U):
    pass


def use_ancestor(b: RefinedBox[No]):
    # expected-error @below {{does not conform to trait 'Deinitable & Base'}}
    wants_base(b)


##===----------------------------------------------------------------------===##
# A conformance satisfied by a caller `where` assumption is NOT reported: only
# the genuinely-unsatisfied conformance's message is surfaced. (The absence of
# a note for the CommonA message is enforced by -verify-diagnostics: an
# unexpected note fails the test.)
##===----------------------------------------------------------------------===##


trait MarkerA:
    pass


trait MarkerB:
    pass


trait CommonA:
    pass


trait CommonB:
    pass


struct TwoBox[T: Deinitable](
    CommonA where (conforms_to(T, MarkerA), "TwoBox needs MarkerA for CommonA"),
    # expected-note @below {{unsatisfied conditional conformance: TwoBox needs MarkerB for CommonB}}
    CommonB where (conforms_to(T, MarkerB), "TwoBox needs MarkerB for CommonB"),
):
    pass


# expected-note @below {{function declared here}}
def wants_both[U: CommonA & CommonB & Deinitable](x: U):
    pass


# V is assumed to be MarkerA (so the CommonA conformance holds) but nothing is
# assumed about MarkerB, so only CommonB genuinely fails.
def use_two[V: Deinitable](b: TwoBox[V]) where (
    conforms_to(V, MarkerA), "V is a MarkerA"
):
    # expected-error @below {{does not conform to trait}}
    wants_both(b)


##===----------------------------------------------------------------------===##
# When neither marker is available, BOTH conditional conformances fail and both
# messages surface as separate notes at the one call site. This exercises
# multiple messaged conformance errors reported together.
##===----------------------------------------------------------------------===##


struct BothBox[T: Deinitable](
    # expected-note @below {{unsatisfied conditional conformance: BothBox needs MarkerA for CommonA}}
    CommonA where (conforms_to(T, MarkerA), "BothBox needs MarkerA for CommonA"),
    # expected-note @below {{unsatisfied conditional conformance: BothBox needs MarkerB for CommonB}}
    CommonB where (conforms_to(T, MarkerB), "BothBox needs MarkerB for CommonB"),
):
    pass


# expected-note @below {{function declared here}}
def wants_both_nm[U: CommonA & CommonB & Deinitable](x: U):
    pass


def use_both_none[V: Deinitable](b: BothBox[V]):
    # expected-error @below {{does not conform to trait}}
    wants_both_nm(b)


##===----------------------------------------------------------------------===##
# Violated via a variadic trait-bounded pack element.
##===----------------------------------------------------------------------===##


struct PackBox[T: Deinitable](
    # expected-note @below {{unsatisfied conditional conformance: PackBox[T] requires T to be a Marker}}
    Marker where (conforms_to(T, Marker), "PackBox[T] requires T to be a Marker")
):
    pass


# expected-note @below {{function declared here}}
def wants_markers[*Ts: Marker & Deinitable](*args: *Ts):
    pass


def use_pack(b: PackBox[No]):
    # expected-error @below {{does not conform to trait}}
    wants_markers(b)
