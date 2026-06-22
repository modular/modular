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
    def __del__(deinit self):
        ...


# Test: Empty string message still counts as @explicit_destroy being present,
# so this should also error.
# expected-error @+2 {{@explicit_destroy cannot be used on a trait that conforms to ImplicitlyDeletable}}
@explicit_destroy("")
trait ExplicitDestroyEmptyMsgOnImplicitlyDeletable(ImplicitlyDeletable):
    def __del__(deinit self):
        ...

@explicit_destroy
@fieldwise_init
struct Linear:
     pass

struct CantSynthDtor:
    # expected-error @+1 {{field 'foo' has non-implicitly deletable type 'Linear'}}
    var foo: Linear
