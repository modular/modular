# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s -verify-diagnostics


struct NonEM:
    fn __init__(out self):
        pass

    fn method(self):
        pass


struct Foo[v: NonEM]:
    fn __init__(out self):
        pass

    fn method(self):
        pass


fn func(x: NonEM = NonEM()):
    pass


fn main():
    # This should not raise a warning.
    func()

    # This should.
    comptime x = NonEM()
    # expected-error@+2 {{cannot materialize comptime value of type 'NonEM' to runtime because it is not 'ImplicitlyCopyable'}}
    # expected-note@+1 {{use 'materialize' to explicitly materialize the value}}
    func(x)
