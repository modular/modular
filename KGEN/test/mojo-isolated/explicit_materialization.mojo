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


fn func(x: NonEM = NonEM()):
    pass


fn main():
    # This should raise a warning.
    alias x = NonEM()
    # expected-warning @below {{Try to implicitly materialize non-'ImplicitlyCopyable' type 'NonEM'. Try explicit materialization with 'materialize[value: T]()'}}
    func(x)

    # This should not.
    func()
