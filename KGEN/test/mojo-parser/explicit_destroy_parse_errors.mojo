# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s -verify-diagnostics


# Test: @explicit_destroy cannot be used on a trait that conforms to
# ImplicitlyDestructible - this is an error at parse time.
# expected-error @+2 {{@explicit_destroy cannot be used on a trait that conforms to ImplicitlyDestructible}}
@explicit_destroy("Hmm, what am I?")
trait ExplicitDestroyOnImplicitlyDestructible(ImplicitlyDestructible):
    def __del__(deinit self):
        ...


# Test: Empty string message still counts as @explicit_destroy being present,
# so this should also error.
# expected-error @+2 {{@explicit_destroy cannot be used on a trait that conforms to ImplicitlyDestructible}}
@explicit_destroy("")
trait ExplicitDestroyEmptyMsgOnImplicitlyDestructible(ImplicitlyDestructible):
    def __del__(deinit self):
        ...
