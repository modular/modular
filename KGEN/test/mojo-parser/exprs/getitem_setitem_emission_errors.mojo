# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s -verify-diagnostics


struct WeirdArray:
    # expected-note @+1 {{function declared here}}
    fn __getitem__(self, x: Int) -> Int:
        return x


fn test_getitem(var a: WeirdArray, f: float, x: Int):
    # expected-error @+2 {{invalid call to '__getitem__': value passed to 'x' cannot be converted from 'float' to 'Int'}}
    # expected-note @+1 {{'float' is aka '__mlir_type.`!pop.scalar<f64>`'}}
    _ = a[f]

    # expected-error @+1 {{invalid call to '__getitem__': expected at most 2 positional arguments, got 3}}
    _ = a[x, x]

    # expected-error @+1 {{expression must be mutable in assignment}}
    a[x] = x


struct Settable:
    fn __setitem__(self, x: Int, y: Int): pass

struct NotSettable:
    fn __getitem__(self) -> Int: pass
    # expected-error @+1 {{__setitem__ must take at least one argument for the value to set}}
    fn __setitem__(self): pass


fn test_setitem_kwargs(c: Settable, ns: NotSettable, x: Int):
    # Issue #22580: Allow keyword arguments in __setitem__ calls
    # weird but ok, value is passed as 'y'.
    c[x=x] = x
    # expected-error @+1 {{keyword argument 'y' may not be specified in the index list, it is needed for the new value}}
    c[y=x] = x
    # expected-note @+1 {{used in an expression here}}
    ns[] = x


struct MultiSetItem:
    # expected-note @+1 {{candidate declared here}}
    fn __setitem__(self, x: Int, y: Int):
        pass

    # expected-note @+1 {{candidate declared here}}
    fn __setitem__(self, x: Int, y: float):
        pass


fn test_setitem_overload(b: MultiSetItem, x: Int):
    # expected-error @+1 {{'MultiSetItem' has overloaded __setitem__ implementations, which isn't supported}}
    b[x] = x


@fieldwise_init
struct VariadicIndexList:
    fn __getitem__(mut self, *indices: Int) -> Int:
        pass

    fn __setitem__(mut self, *indices: Int, val: Int):
        pass

# CHECK-LABEL: lit.fn @"testVariadicIndexList
# MOCO-696: Support variadic length keys in __setitem__
fn testVariadicIndexList(mut foo: VariadicIndexList, i: Int, the_value: Int):
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
    fn __setitem__[idx: Int](mut self, value: Int): pass

fn test(lst: Issue3142IntList):
    # expected-error @below {{invalid call to '__setitem__': failed to infer parameter 'idx'}}
    lst[0] = 0
