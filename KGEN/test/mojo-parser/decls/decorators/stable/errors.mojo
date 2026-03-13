# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -mojo-search-paths=%S -verify-diagnostics %s

# Test error cases for @stable decorator.
#
# Note: Tests for "@stable member in unstable struct/trait" are located in
# test_std_mock/__init__.mojo since those errors only apply in opted-in packages.

# Error: @stable with a positional (non-keyword) argument is not supported.
# expected-error @+1 {{@stable only accepts the keyword argument 'since'}}
@stable("since 1.0")
struct StableWithArg:
    pass


# Error: @stable on local variable is not supported.
def test_local_var():
    # expected-error @+2 {{'var' statement in function body does not allow decorators}}
    @stable
    var x = 1


# @stable on alias is now supported (as of escape hatches feature).
# Verify it doesn't error - this alias will be verified in a separate test.
@stable
# expected-warning @+1 {{'alias' is deprecated, use 'comptime' instead}}
alias MY_STABLE_ALIAS = 42


# Verify that @stable members in stable structs are allowed (no error).
@stable
struct StableStruct:
    @stable
    def stable_method_in_stable(self):
        pass


# Verify that @stable members in stable traits are allowed (no error).
@stable
trait StableTrait:
    @stable
    def stable_method_in_stable_trait(self): ...


# Verify that @stable members in non-opted-in package types are allowed.
# This file is not in an opted-in package, so structs/traits here are stable
# by default, and @stable members should be allowed.
struct StructInNonOptedInPackage:
    @stable
    def stable_method(self):
        pass


trait TraitInNonOptedInPackage:
    @stable
    def stable_method(self): ...


# Error: Decorators are not allowed on import statements.
# expected-error @+2 {{'from' statement does not allow decorators}}
@stable
from test_std_mock import *
