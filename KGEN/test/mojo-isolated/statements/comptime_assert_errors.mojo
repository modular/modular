# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -verify-diagnostics %s

##===----------------------------------------------------------------------===##
# __comptime_assert errors
##===----------------------------------------------------------------------===##

struct NotBool:
    pass

fn test_non_bool_type_error[x: NotBool]():
    # expected-error @below {{'NotBool' does not implement the '__bool__' method}}
    __comptime_assert x


fn test_runtime_expr_error(x: Int, y: Int):
    # expected-error @below {{cannot use a dynamic value in '__comptime_assert' expression}}
    __comptime_assert x == y


fn test_non_string_literal_message_error():
    # expected-error @below {{cannot implicitly convert}}
    __comptime_assert True, 42


struct NotFn[x: Bool]:
    # expected-error @below {{'__comptime_assert' statements must be inside a function}}
    __comptime_assert x


# expected-note @below {{cannot prove constraint}}
# expected-note @below {{constraint declared here}}
fn requires_natural[x: Int]() where x >= 0:
    pass


fn test_assert_injects_assumption_correctly[x: Int]():
    @parameter
    if x > 10:
        __comptime_assert x >= 0

        # This is OK.
        requires_natural[x]()
    else:
        # expected-error @below {{invalid call to 'requires_natural': lacking evidence to prove correctness}}
        # expected-note @below {{provide evidence for the constraint here to aid in candidate selection}}
        requires_natural[x]()

    # expected-error @below {{invalid call to 'requires_natural': lacking evidence to prove correctness}}
    # expected-note @below {{provide evidence for the constraint here to aid in candidate selection}}
    requires_natural[x]()


fn test_newly_created_scope[x: Int]():
    # expected-error @below {{invalid call to 'requires_natural': lacking evidence to prove correctness}}
    # expected-note @below {{provide evidence for the constraint here to aid in candidate selection}}
    comptime y = requires_natural[x]()

    # COM: This assert should not validate the above statement.
    __comptime_assert x >= 0


fn test_always_false_error():
    # expected-error @below {{failed __comptime_assert: condition is always False}}
    __comptime_assert 2 < 1


fn test_always_false_error_with_message():
    # expected-error @below {{failed __comptime_assert: condition is always False}}
    __comptime_assert False, "this message is ignored"


fn test_always_true_warning():
    # expected-warning @below {{redundant __comptime_assert: condition is always True}}
    __comptime_assert 2 > 1


fn test_always_true_warning_with_message():
    # expected-warning @below {{redundant __comptime_assert: condition is always True}}
    __comptime_assert True, "this message is also ignored"
