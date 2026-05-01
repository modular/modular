# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated -verify-diagnostics %s

# Regression test for MOCO-3883.


def call_it[F: def(Int)](func: F):
    func(5)


@fieldwise_init
struct MyCallableWithCapture:
    var value: Int

    # The `capturing` keyword is required to match the function trait signature.
    def __call__(self, arg: Int) capturing:
        _ = arg + self.value


def test_implicit_fn_trait_conformance():
    var c = MyCallableWithCapture(10)
    call_it(c)
