# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen -elaborate -O0 %s -S | FileCheck %s

alias int = __mlir_type.index


trait SimpleTrait:
    @staticmethod
    fn bar() -> int:
        ...


struct MemType(SimpleTrait):
    @staticmethod
    @always_inline
    fn bar() -> int:
        return __mlir_attr.`1:index`


@register_passable
struct RegType(SimpleTrait):
    @staticmethod
    @always_inline
    fn bar() -> int:
        return __mlir_attr.`2:index`


@register_passable("trivial")
struct RegTypeTrivial(SimpleTrait):
    var x: int

    @staticmethod
    @always_inline
    fn bar() -> int:
        return __mlir_attr.`3:index`


fn generic_arg[T: SimpleTrait](x: T) -> int:
    return T.bar()


# CHECK: kgen.func @"{{.*}}generic_arg
# CHECK-SAME: %arg0: !kgen.pointer<struct<() memoryOnly>>
# CHECK-NEXT: <1>

# CHECK: kgen.func @"{{.*}}generic_arg
# CHECK-SAME: %arg0: !kgen.struct<()>
# CHECK-NEXT: <2>

# CHECK: kgen.func @"{{.*}}generic_arg
# CHECK-SAME: %arg0: index
# CHECK-NEXT: <3>


@export
fn top(a: MemType, b: RegType, c: RegTypeTrivial):
    _ = generic_arg(a)
    _ = generic_arg(b)
    _ = generic_arg(c)
