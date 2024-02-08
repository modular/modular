# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %translate-with-packages %s -verify-diagnostics


struct Foo:
    pass


# COM: Issue #27654: Parser crash: Assertion failed: Types should match
# COM: https://github.com/modularml/mojo/issues/1607 Improved error message for this common error
fn test_return_type_instead_of_instance() -> Foo:
    # expected-error @+1 {{cannot implicitly convert 'Foo' type value to an instance of 'Foo' in return value (hint: did you mean to instantiate 'Foo'?)}}
    return Foo
