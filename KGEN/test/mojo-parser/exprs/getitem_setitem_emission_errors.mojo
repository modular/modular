# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s -verify-diagnostics


struct WeirdArray:
    # expected-note @+1 {{function declared here}}
    def __getitem__(self, x: Int) -> Int:
        return x


def test_getitem(var a: WeirdArray, f: float, x: Int):
    # expected-error @+2 {{invalid call to '__getitem__': value passed to 'x' cannot be converted from 'float' to 'Int'}}
    # expected-note @+1 {{'float' is aka '__mlir_type.`!kgen.scalar<f64>`'}}
    _ = a[f]

    # expected-error @+1 {{invalid call to '__getitem__': expected at most 2 positional arguments, got 3}}
    _ = a[x, x]

    # expected-error @+1 {{expression must be mutable in assignment}}
    a[x] = x


struct NotSettable:
    def __getitem__(self) -> Int:
        pass

    # expected-error @+1 {{__setitem__ must take at least one argument for the value to set}}
    def __setitem__(self):
        pass


def test_setitem_kwargs(ns: NotSettable, x: Int):
    # expected-note @+1 {{used in an expression here}}
    ns[] = x

@fieldwise_init
struct VariadicIndexList:
    def __getitem__(mut self, *indices: Int) -> Int:
        pass

    def __setitem__(mut self, *indices: Int, val: Int):
        pass


# CHECK-LABEL: lit.fn @"testVariadicIndexList
# MOCO-696: Support variadic length keys in __setitem__
def testVariadicIndexList(mut foo: VariadicIndexList, i: Int, the_value: Int):
    # Getter is straight-forward.
    _ = foo[i, i]

    # Setter needs to pass the new value as 'val', not in the variadics.
    foo[i, i, i, i] = the_value
