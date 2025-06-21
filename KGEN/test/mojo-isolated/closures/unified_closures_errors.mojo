# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated -verify-diagnostics %s


trait MyInterface:
    fn thing(self):
        ...


fn make_closure(x: Int) -> Int:
    fn parametric[T: MyInterface](a: T) unified:
        # expected-error @below {{use of unknown declaration 'A'}}
        alias X = A
        pass

    return x
