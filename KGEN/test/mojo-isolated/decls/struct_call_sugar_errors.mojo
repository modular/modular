# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -verify-diagnostics %s


struct PCall:
    fn __init__(out self):
        pass

    fn __call__[x: Int](ref self, y: Int) -> Int:
        return x + y


struct PCallWithGetItem:
    fn __init__(out self):
        pass

    # expected-warning @below {{parametric '__call__' method cannot be called directly because 'PCallWithGetItem' defines '__getitem__', '__setitem__', or '__getattr__'; consider using a different name for this method}}
    fn __call__[x: Int](ref self, y: Int) -> Int:
        return x + y

    # expected-note @below {{__getitem__ defined here}}
    fn __getitem__(ref self, y: Int) -> Int:
        return y

    # expected-note @below {{__getitem__ defined here}}
    fn __getitem__(ref self, y: StringLiteral) -> Int:
        return 2


struct PCallWithSetItem:
    fn __init__(out self):
        pass

    # expected-warning @below {{parametric '__call__' method cannot be called directly because 'PCallWithSetItem' defines '__getitem__', '__setitem__', or '__getattr__'; consider using a different name for this method}}
    fn __call__[x: Int](ref self, y: Int) -> Int:
        return x + y

    # expected-note @below {{__setitem__ defined here}}
    fn __setitem__(mut self, x: Int, y: Int):
        return


struct PCallWithGetAttr:
    fn __init__(out self):
        pass

    # expected-warning @below {{parametric '__call__' method cannot be called directly because 'PCallWithGetAttr' defines '__getitem__', '__setitem__', or '__getattr__'; consider using a different name for this method}}
    fn __call__[_x: StringLiteral](ref self):
        pass

    # expected-note @below {{__getattr__ defined here}}
    fn __getattr__[x: StringLiteral](self) -> Int:
        return 2


fn main():
    var pc = PCall()
    _ = pc[1](2)

    # verifying that we don't break structs with __getitem__ defined
    var pcgi = PCallWithGetItem()
    _ = pcgi[1](2)  # expected-error {{'Int' does not implement the '__call__' method}}

    var pcsi = PCallWithSetItem()
    _ = pcsi[1](2)  # expected-error {{'Int' does not implement the '__call__' method}}

    var pcga = PCallWithGetAttr()
    _ = pcsi["test"]()  # expected-error {{'Int' does not implement the '__call__' method}}
