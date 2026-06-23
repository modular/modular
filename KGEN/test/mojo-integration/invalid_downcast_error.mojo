# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Test that Layout is not implicitly copyable.
# This is a common error that users were hitting,
# so it's good to make sure we don't regress.


# RUN: not kgen %s -elaborate 2>&1 | FileCheck %s


# CHECK: error: function instantiation failed
# CHECK: note: constraint failed: Invalid downcast


@fieldwise_init
struct Foo(ImplicitlyCopyable):
    pass


def print_foo_invalid[T: Writable & ImplicitlyDeletable](var x: T):
    print(x)


def invalid[T: ImplicitlyCopyable & ImplicitlyDeletable](x: T):
    print_foo_invalid(trait_downcast[Writable](x))


def main():
    invalid(Foo())
