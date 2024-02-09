# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %translate-with-packages %s -verify-diagnostics


struct Foo:
    pass


fn take_instance_param[a: Foo]():
    pass


# expected-note @+1 {{function declared here}}
fn takes_instance_arg(a: Foo):
    pass


# COM: Issue #27654: Parser crash: Assertion failed: Types should match
# COM: https://github.com/modularml/mojo/issues/1607 Improved error message for this common error
fn test_type_instead_of_instance() -> Foo:
    # expected-error @+1 {{cannot pass 'Foo' type value, parameter expected an instance of 'Foo' (hint: did you mean to instantiate 'Foo'?)}}
    take_instance_param[Foo]
    # expected-error @+1 {{invalid call to 'takes_instance_arg': argument #0 cannot be converted from type value 'Foo' to an instance of 'Foo' (hint: did you mean to instantiate 'Foo'?)}}
    takes_instance_arg(Foo)
    # expected-error @+1 {{cannot implicitly convert 'Foo' type value to an instance of 'Foo' in return value (hint: did you mean to instantiate 'Foo'?)}}
    return Foo
