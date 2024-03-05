# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s -verify-diagnostics


trait SomeTrait:
    # expected-error @+1 {{'self' argument must have type 'Self' in trait method declaration, but actually has type 'index'}}
    fn add(self: int):
        ...


trait FooTrait:
    fn foo(self):
        ...


struct ParamType[x: int](FooTrait):
    fn foo(self):
        pass


fn invalid_trait_bind():
    # expected-error @below {{parametric type 'ParamType[?]' cannot bind to trait with missing parameters}}
    alias Bound: FooTrait = ParamType


fn different_trait_types[T: Copyable, U: Copyable](x: T) -> U:
    # expected-error @below {{cannot implicitly convert 'T' value to 'U' in return value}}
    return x
