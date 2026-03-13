# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -verify-diagnostics %s

# Test error cases for @stable(since=...) argument.


# Error: positional string argument — only keyword 'since' is accepted.
# expected-error @+1 {{@stable only accepts the keyword argument 'since'}}
@stable("1.0")
def fn_positional_string():
    pass


# Error: unknown keyword argument.
# expected-error @+1 {{@stable only accepts the keyword argument 'since'}}
@stable(unknown="1.0")
def fn_unknown_kwarg():
    pass


# Error: since= value must be a string literal, not a number.
# expected-error @+1 {{'since' argument must be a string literal}}
@stable(since=1)
def fn_since_not_string():
    pass


# Error: version string must start with a digit.
# expected-error @+1 {{'since' argument must be a valid version string}}
@stable(since="v1.0")
def fn_since_starts_with_letter():
    pass


# Error: empty version string.
# expected-error @+1 {{'since' argument must be a valid version string}}
@stable(since="")
def fn_since_empty():
    pass


# Error: version string with invalid characters (hyphen is not allowed).
# expected-error @+1 {{'since' argument must be a valid version string}}
@stable(since="1.0-alpha")
def fn_since_hyphen():
    pass


# Error: too many arguments.
# expected-error @+1 {{@stable accepts only the 'since' argument}}
@stable(since="1.0", extra="x")
def fn_too_many_args():
    pass
