# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s -mlir-print-debuginfo | kgen-opt -lower-semantic-cf -check-lifetimes -verify-diagnostics | FileCheck %s


struct MyAffine:
    def __init__(out self):
        pass

    def __del__(deinit self):
        pass


# CHECK-LABEL: @"testAffineThing
def testAffineThing():
    _ = MyAffine()
    # CHECK: lit.call {{.*}}MyAffine::@"__del__
    # CHECK: kgen.return


# CHECK-LABEL: lit.struct.decl @EmptyExplicit
@explicit_destroy
struct EmptyExplicit:
    def __init__(out self):
        pass

    # CHECK-LABEL: @"consume
    def consume(deinit self):
        # CHECK: lit.ownership.mark_destroyed %self
        pass
        # CHECK-NOT: lit.call {{.*}}__del__

    # Deinit method should be able to transfer all of self to another
    # deinit method.
    def consume2(deinit self):
        self^.consume()


def correctUseExample():
    var l = EmptyExplicit()
    l^.consume()


# CHECK-LABEL: lit.struct.decl @ExplicitDestroyThrowing
@explicit_destroy("end lifetime with foo()")
struct ExplicitDestroyThrowing:
    var field: MyAffine

    # CHECK-LABEL: lit.fn @"foo
    # CHECK: lit.call {{.*}}MyAffine::@"__del__
    def foo(deinit self):
        pass

    # CHECK-LABEL: lit.fn @"method_that_raises
    def method_that_raises(self) raises:
        pass

    # CHECK-LABEL: lit.fn @"raising_callee
    def raising_callee(deinit self) raises:
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

    def __init__(out self):
        self.m = EmptyExplicit()

    def __del__(deinit self):
        self.m^.consume()


def foo1[T: Movable](var x: T) -> T:
    # Is fine, we move it away instead of calling x.__del__()
    return x^


def foo2[T: AnyType](x: T):
    # Is fine, since x is a borrow
    pass


def foo3[T: ImplicitlyDestructible](var x: T):
    # Is fine, there's a x.__del__() available
    pass


trait Iterator(ImplicitlyDestructible):
    comptime Element: AnyType


struct I(Iterator):
    comptime Element = Int


struct _MapIterator[
    InnerIteratorType: Iterator,
    //,
    Function: def (InnerIteratorType.Element) thin -> Int,
]():
    var _inner: Self.InnerIteratorType

    def __init__(out self):
        while True:
            pass


def f(x: Int) -> Int:
    return 1


def map[
    func: def (Int) thin -> Int,
](ref iterable: I) -> _MapIterator[InnerIteratorType=I, Function=func]:
    return {}


# CHECK-LABEL: lit.fn @"moco2373(
def moco2373(l: I):
    var l2 = map[f](l)
    # This shouldn't cause a crash, it should successfully destruct it.
    # CHECK: lit.call {{.*}}@_MapIterator::@"__del__
    _ = l2^
