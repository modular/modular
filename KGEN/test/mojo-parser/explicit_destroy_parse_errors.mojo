# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s -verify-diagnostics


# Test: @explicit_destroy cannot be used on a trait that conforms to
# ImplicitlyDeletable - this is an error at parse time.
# expected-error @+2 {{@explicit_destroy cannot be used on a trait that conforms to ImplicitlyDeletable}}
@explicit_destroy("Hmm, what am I?")
trait ExplicitDestroyOnImplicitlyDeletable(ImplicitlyDeletable):
    def __deinit__(deinit self):
        ...


# Test: Empty string message still counts as @explicit_destroy being present,
# so this should also error.
# expected-error @+2 {{@explicit_destroy cannot be used on a trait that conforms to ImplicitlyDeletable}}
@explicit_destroy("")
trait ExplicitDestroyEmptyMsgOnImplicitlyDeletable(ImplicitlyDeletable):
    def __deinit__(deinit self):
        ...


# ===----------------------------------------------------------------------=== #
# `@explicit_destroy` on a `trait` requires a string argument
# ===----------------------------------------------------------------------=== #


# expected-error @+2 {{@explicit_destroy requires an argument: `@explicit_destroy("...")`}}
# expected-note @+2 {{Use `ImplicitlyDeletable where False` conformance to opt out of implicit deletion. `@explicit_destroy` is no longer required.}}
@explicit_destroy
trait BareExplicitDestroyTrait:
    def consume(deinit self):
        ...


# expected-error @below {{expected exactly one argument: `@explicit_destroy("...")`}}
@explicit_destroy()
trait EmptyArgsExplicitDestroyTrait:
    def consume(deinit self):
        ...


# expected-error @below {{expected a string literal argument: `@explicit_destroy("...")`}}
@explicit_destroy(42)
trait NonStringArgExplicitDestroyTrait:
    def consume(deinit self):
        ...


# ===----------------------------------------------------------------------=== #
# Old `@explicit_destroy` usage
# ===----------------------------------------------------------------------=== #


# expected-error @+2 {{@explicit_destroy requires an argument: `@explicit_destroy("...")`}}
# expected-note @+2 {{Use `ImplicitlyDeletable where False` conformance to opt out of implicit deletion. `@explicit_destroy` is no longer required.}}
@explicit_destroy
struct UnconditionalDefault(Movable where False):
    pass


# expected-error @below {{expected exactly one argument: `@explicit_destroy("...")`}}
@explicit_destroy()
struct UnconditionalDefaultEmptyArgs(Movable where False):
    pass


# expected-error @below {{expected a string literal argument: `@explicit_destroy("...")`}}
@explicit_destroy(42)
struct NonStringArgExplicitDestroy:
    pass


# expected-error @+2 {{@explicit_destroy requires an argument: `@explicit_destroy("...")`}}
# expected-note @+2 {{Use `ImplicitlyDeletable where False` conformance to opt out of implicit deletion. `@explicit_destroy` is no longer required.}}
@explicit_destroy
struct ConditionalDefault[cond: Bool](ImplicitlyDeletable where cond, Movable where False):
    pass


# expected-error @+2 {{@explicit_destroy is not valid on `struct` with unconditional conformance to `ImplicitlyDeletable`}}
# expected-note @+2 {{Add `ImplicitlyDeletable where False` conformance or remove `@explicit_destroy`}}
@explicit_destroy("some error")
struct UnconditionalCustom(Movable where False):
    pass


# ===----------------------------------------------------------------------=== #
# Synthesized __deinit__ field checks
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct Linear(ImplicitlyDeletable where False, Movable where False):
    pass


struct CantSynthDtor(Movable where False):
    # expected-error @+1 {{field 'foo' has non-implicitly deletable type 'Linear'}}
    var foo: Linear
