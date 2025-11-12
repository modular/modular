# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s -mlir-print-debuginfo | kgen-opt -lower-semantic-cf -check-lifetimes -verify-diagnostics | FileCheck %s


struct MyAffine:
    fn __init__(out self):
        pass
    fn __del__(deinit self):
        pass

# CHECK-LABEL: @"testAffineThing
fn testAffineThing():
    _ = MyAffine()
    # CHECK: lit.call {{.*}}MyAffine::@"__del__
    # CHECK: kgen.return

# CHECK-LABEL: lit.struct.decl @EmptyExplicit
@explicit_destroy
struct EmptyExplicit:
    fn __init__(out self):
        pass

    # CHECK-LABEL: @"consume
    fn consume(deinit self):
        # CHECK: lit.ownership.mark_destroyed %self
        pass
        # CHECK-NOT: lit.call {{.*}}__del__

    # Deinit method should be able to transfer all of self to another
    # deinit method.
    fn consume2(deinit self):
        self^.consume()


fn correctUseExample():
    var l = EmptyExplicit()
    l^.consume()

# CHECK-LABEL: lit.struct.decl @ExplicitDestroyThrowing
@explicit_destroy("end lifetime with foo()")
struct ExplicitDestroyThrowing:
    var field: MyAffine

    # CHECK-LABEL: lit.fn @"foo
    # CHECK: lit.call {{.*}}MyAffine::@"__del__
    fn foo(deinit self):
        pass

    # CHECK-LABEL: lit.fn @"method_that_raises
    fn method_that_raises(self) raises:
        pass

    # CHECK-LABEL: lit.fn @"raising_callee
    fn raising_callee(deinit self) raises:
        # This should be ok even though method() is throwing. The error path
        # should assume 't' is deinited.
        # CHECK: %__call_result_tmp__ = lit.var.decl
        # CHECK: lit.call {{.*}}@"method_that_raises{{.*}}(%0, %__error__, %__call_result_tmp__)
        # CHECK-NEXT: lit.ref.struct.ger %self[field]
        # CHECK-NEXT: lit.call {{.*}}MyAffine::@"__del__
        # CHECK-NEXT: hlcf.if
        # CHECK-NEXT: lit.ownership.mark_consumed %__call_result_tmp__
        # CHECK-NEXT: lit.var.lifetime.end %__call_result_tmp__
        # CHECK-NEXT: lit.ownership.mark_destroyed %self
        self.method_that_raises()



struct ImplicitlyDestructibleContainerOfExplicit:
    var m: EmptyExplicit
    fn __init__(out self):
        self.m = EmptyExplicit()
    fn __del__(deinit self):
        self.m^.consume()

fn foo1[T: Movable](var x: T) -> T:
    # Is fine, we move it away instead of calling x.__del__()
    return x^

fn foo2[T: UnknownDestructibility](x: T):
    # Is fine, since x is a borrow
    pass

fn foo3[T: ImplicitlyDestructible](var x: T):
    # Is fine, there's a x.__del__() available
    pass
