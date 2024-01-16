# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s

# ===----------------------------------------------------------------------=== #
# Stubs to allow testing without builtins
# ===----------------------------------------------------------------------=== #

alias Int = __mlir_type.index


trait AnyType:
    fn __del__(owned self, /):
        ...


trait Copyable:
    fn __copyinit__(inout self, existing: Self, /):
        ...


# ===----------------------------------------------------------------------=== #
# Actual tests
# ===----------------------------------------------------------------------=== #


# COM: Test overloading precedence in the presence of static methods.
struct StaticOverloadStruct:
    fn __init__(inout self):
        pass

    fn foo(inout self):
        pass

    @staticmethod
    fn foo():
        pass


# CHECK-LABEL: lit.func @"test_static_overload()"
fn test_static_overload():
    var a = StaticOverloadStruct()
    # CHECK-NEXT: %a = lit.varlet.decl
    # CHECK-NEXT: lit.call{{.*}}__init__{{.*}}(%a)
    # CHECK-NEXT: lit.call @{{.*}}foo{{.*}}(%a)
    a.foo()


# COM: Issue https://github.com/modularml/mojo/issues/1408
# COM: Test that the number of implicit conversions is more important than
# COM: convention mismatches.
@register_passable("trivial")
struct MyElement(Copyable):
    pass


struct ConvertibleFromInt:
    fn __init__(inout self, a: Int):
        pass


struct MyContainer[T: Copyable]:
    var v: T

    fn foo(self, limits: ConvertibleFromInt):
        pass

    fn foo(self, index: Int) -> T:
        return self.v


# CHECK-LABEL: lit.func @"test_impl
fn test_impl(a: MyContainer[MyElement], b: Int):
    # CHECK: lit.call @{{.*}}@MyContainer::@"foo{{.*}}, "index": index borrow) -> !kgen.none
    _ = a.foo(b)
