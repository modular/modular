# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s -verify-diagnostics


struct WeirdArray:
    # expected-note @+1 {{function declared here}}
    fn __getitem__(self, x: int) -> int:
        return x


fn test_getitem(owned a: WeirdArray, f: float, x: int):
    # expected-error @+1 {{invalid call to '__getitem__': index cannot be converted from 'scalar<f64>' to 'index'}}
    _ = a[f]

    # expected-error @+1 {{invalid call to '__getitem__': expected at most 2 positional arguments, got 3}}
    _ = a[x, x]

    # expected-error @+1 {{expression must be mutable in assignment}}
    a[x] = x


struct Settable:
    # expected-note @+1 {{function declared here}}
    fn __setitem__(self, x: int, y: int):
        pass


fn test_setitem_kwargs(c: Settable, x: int):
    # Issue #22580: Allow keyword arguments in __setitem__ calls
    # expected-error @+1 {{invalid call to '__setitem__': argument passed both as positional and keyword operand: 'x'}}
    c[x=x] = x
    # weird, but ok.
    c[y=x] = x


struct MultiSetItem:
    # expected-note @+1 {{candidate declared here}}
    fn __setitem__(self, x: int, y: int):
        pass

    # expected-note @+1 {{candidate declared here}}
    fn __setitem__(self, x: int, y: float):
        pass


fn test_setitem_overload(b: MultiSetItem, x: int):
    # expected-error @+1 {{'MultiSetItem' has overloaded __setitem__ implementations, which isn't supported}}
    b[x] = x
