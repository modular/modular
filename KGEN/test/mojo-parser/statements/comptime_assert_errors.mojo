# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -verify-diagnostics %s

##===----------------------------------------------------------------------===##
# comptime assert errors
##===----------------------------------------------------------------------===##


struct NotBool:
    pass


def test_non_bool_type_error[x: NotBool]():
    # expected-error @below {{'NotBool' does not implement the '__bool__' method}}
    comptime assert x


def test_runtime_expr_error(x: Int, y: Int):
    # expected-error @below {{cannot use a dynamic value in 'comptime assert' expression}}
    comptime assert x == y


def test_non_string_literal_message_error():
    # expected-error @below {{cannot implicitly convert}}
    comptime assert True, 42


struct Notdef[x: Bool]:
    # expected-error @below {{'comptime assert' statements must be inside a function}}
    comptime assert x


# expected-note @below {{cannot prove constraint}}
# expected-note @below {{constraint declared here}}
def requires_natural[x: Int]() where x >= 0:
    pass


def test_assert_injects_assumption_correctly[x: Int]():
    comptime if x > 10:
        comptime assert x >= 0

        # This is OK.
        requires_natural[x]()
    else:
        # expected-error @below {{invalid call to 'requires_natural': lacking evidence to prove correctness}}
        # expected-note @below {{provide evidence for the constraint here to aid in candidate selection}}
        requires_natural[x]()

    # expected-error @below {{invalid call to 'requires_natural': lacking evidence to prove correctness}}
    # expected-note @below {{provide evidence for the constraint here to aid in candidate selection}}
    requires_natural[x]()


def test_newly_created_scope[x: Int]():
    # expected-error @below {{invalid call to 'requires_natural': lacking evidence to prove correctness}}
    # expected-note @below {{provide evidence for the constraint here to aid in candidate selection}}
    comptime y = requires_natural[x]()

    # COM: This assert should not validate the above statement.
    comptime assert x >= 0


def test_always_false_no_warning():
    comptime assert 2 < 1


def test_always_true_no_warning():
    comptime assert 2 > 1
