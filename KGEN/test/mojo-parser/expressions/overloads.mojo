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

trait Destructable:
    fn __del__(owned self, /): ...

trait Copyable:
    fn __copyinit__(inout self, existing: Self, /): ...

# ===----------------------------------------------------------------------=== #
# Actual tests
# ===----------------------------------------------------------------------=== #

# COM: Issue https://github.com/modularml/mojo/issues/1408
# COM: Test that the number of implicit conversions is more important than
# COM: convention mismatches.

@register_passable("trivial")
struct MyElement(Copyable): pass

struct ConvertibleFromInt:
    fn __init__(inout self, a: Int): pass

struct MyContainer[T: Copyable]:
    var v: T

    fn foo(self, limits: ConvertibleFromInt): pass

    fn foo(self, index: Int) -> T:
        return self.v

# CHECK-LABEL: lit.func @"test_impl
fn test_impl(a: MyContainer[MyElement], b: Int):
    # CHECK: lit.call @{{.*}}@MyContainer::@"foo{{.*}}, "index": index borrow) -> !kgen.none
    _ = a.foo(b)
