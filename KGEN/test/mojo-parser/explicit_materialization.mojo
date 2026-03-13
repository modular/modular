# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s -verify-diagnostics


struct NonEM:
    def __init__(out self):
        pass

    def method(self):
        pass


struct Foo[v: NonEM]:
    def __init__(out self):
        pass

    def method(self):
        pass


def func(x: NonEM = NonEM()):
    pass


def main():
    # This should not raise a warning.
    func()

    comptime x = NonEM()
    # This should.
    # expected-error@+2 {{cannot materialize comptime value of type 'NonEM' to runtime because it is not 'ImplicitlyCopyable'}}
    # expected-note@+1 {{use 'materialize' to explicitly materialize the value}}
    func(x)
