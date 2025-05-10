# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s -mlir-print-debuginfo | kgen-opt -lower-semantic-cf -check-lifetimes -verify-diagnostics | FileCheck %s


struct MyAffine:
    fn __init__(out self):
        pass
    fn __del__(owned self):
        pass

# CHECK-LABEL: @"testAffineThing
fn testAffineThing():
    _ = MyAffine()
    # CHECK: lit.call {{.*}}MyAffine::@"__del__
    # CHECK: kgen.return

@explicit_destroy
struct EmptyExplicit:
    fn __init__(out self):
        pass

    # CHECK-LABEL: @"consume
    fn consume(owned self):
        # CHECK: lit.ownership.mark_destroyed %self
        __disable_del self
        # CHECK-NOT: lit.call {{.*}}__del__

fn correctUseExample():
    var l = EmptyExplicit()
    l^.consume()


struct ImplicitlyDestructibleContainerOfExplicit:
    var m: EmptyExplicit
    fn __init__(out self):
        self.m = EmptyExplicit()
    fn __del__(owned self):
        self.m^.consume()

fn foo1[T: Movable](owned x: T) -> T:
    # Is fine, we move it away instead of calling x.__del__()
    return x^

fn foo2[T: UnknownDestructibility](x: T):
    # Is fine, since x is a borrow
    pass

fn foo3[T: ImplicitlyDestructible](owned x: T):
    # Is fine, there's a x.__del__() available
    pass
