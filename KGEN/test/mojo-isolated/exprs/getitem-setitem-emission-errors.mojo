# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %translate-with-packages %s -verify-diagnostics


struct WeirdArray:
    # expected-note @+1 {{function declared here}}
    fn __getitem__(self, x: Int) -> Int:
        return x


fn test_getitem(owned a: WeirdArray, f: Float, x: Int):
    # expected-error @+1 {{invalid call to '__getitem__': index cannot be converted from 'scalar<f64>' to 'index'}}
    _ = a[f]

    # expected-error @+1 {{invalid call to '__getitem__': expected at most 2 positional arguments, got 3}}
    _ = a[x, x]

    # expected-error @+1 {{expression must be mutable in assignment}}
    a[x] = x


struct Settable:
    fn __setitem__(self, x: Int, y: Int):
        pass


fn test_setitem_kwargs(c: Settable, x: Int):
    # expected-error @+1 {{keyword operands for __setitem__ not supported yet}}
    c[x=x] = x


struct MultiSetItem:
    # expected-note @+1 {{candidate declared here}}
    fn __setitem__(self, x: Int, y: Int):
        pass

    # expected-note @+1 {{candidate declared here}}
    fn __setitem__(self, x: Int, y: Float):
        pass


fn test_setitem_overload(b: MultiSetItem, x: Int):
    # expected-error @+1 {{'MultiSetItem' has overloaded __setitem__ implementations, which isn't supported}}
    b[x] = x
