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
    fn __setitem__(self, x: int, y: int): pass

struct NotSettable:
    fn __getitem__(self) -> Int: pass
    # expected-error @+1 {{__setitem__ must take at least one argument for the value to set}}
    fn __setitem__(self): pass


fn test_setitem_kwargs(c: Settable, ns: NotSettable, x: int):
    # Issue #22580: Allow keyword arguments in __setitem__ calls
    # weird but ok, value is passed as 'y'.
    c[x=x] = x
    # expected-error @+1 {{keyword argument 'y' may not be specified in the index list, it is needed for the new value}}
    c[y=x] = x
    # expected-note @+1 {{used in an expression here}}
    ns[] = x


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


@value
struct VariadicIndexList:
    fn __getitem__(inout self, *indices: Int) -> Int:
        pass

    fn __setitem__(inout self, *indices: Int, val: Int):
        pass

# CHECK-LABEL: lit.func @"testVariadicIndexList
# MOCO-696: Support variadic length keys in __setitem__
fn testVariadicIndexList(inout foo: VariadicIndexList, i: Int, the_value: Int):
    # Getter is straight-forward.
    # CHECK: [[VARIADIC:%.*]] = pop.variadic.splat 2, %i
    # CHECK: lit.call {{.*}}VariadicIndexList::@"__getitem__{{.*}}(%foo, [[VARIADIC]])
    _ = foo[i, i]

    # Setter needs to pass the new value as 'val', not in the variadics.
    # CHECK-NEXT: [[VARIADIC:%.*]] = pop.variadic.splat 4, %i
    # CHECK: lit.call {{.*}}VariadicIndexList::@"__setitem__{{.*}}(%foo, [[VARIADIC]], %the_value)
    foo[i, i, i, i] = the_value

struct Issue3142IntList:
    fn __getitem__[idx: Int](self) -> Int: pass
    # expected-note @+1 {{function declared here}}
    fn __setitem__[idx: Int](inout self, value: Int): pass

fn test(lst: Issue3142IntList):
    # expected-error @+1 {{invalid call to '__setitem__': could not deduce parameter 'idx' of callee '__setitem__'}}
    lst[0] = 0

