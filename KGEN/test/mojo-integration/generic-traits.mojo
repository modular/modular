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
# CHECK-SAME: %arg0: index
# CHECK-NEXT: <3>

# CHECK: kgen.func @"{{.*}}generic_arg
# CHECK-SAME: %arg0: !kgen.pointer<struct<() memoryOnly>>
# CHECK-NEXT: <1>

# CHECK: kgen.func @"{{.*}}generic_arg
# CHECK-SAME: %arg0: !kgen.struct<()>
# CHECK-NEXT: <2>


@export
fn top(a: MemType, b: RegType, c: RegTypeTrivial):
    _ = generic_arg(a)
    _ = generic_arg(b)
    _ = generic_arg(c)


trait GrandFather:
    fn bar(self):
        ...


trait Father(GrandFather):
    fn baz(self):
        ...


struct Son(Father):
    fn bar(self):
        pass

    fn baz(self):
        pass


# CHECK: kgen.func [[TAKE_GRAND_FATHER:@.*take_grand_father.*]](%arg0
fn take_grand_father[T: GrandFather](value: T):
    # CHECK: call {{.*}}Son::bar
    value.bar()


# CHECK: kgen.func [[TAKE_FATHER:@.*take_father.*]](%arg0
fn take_father[T: Father](value: T):
    # CHECK: call {{.*}}Son::baz
    value.baz()
    # CHECK: call [[TAKE_GRAND_FATHER]]
    take_grand_father(value)


# CHECK: kgen.func export @like_father_like
@export
fn like_father_like(value: Son):
    # CHECK: call [[TAKE_FATHER]]
    take_father(value)


@value
@register_passable
struct SomeType:
    fn __del__(owned self):
        pass


# CHECK-LABEL: kgen.func {{.*}}drop_copy
fn drop_copy[T: Copyable](value: T):
    # CHECK: [[V0:%.*]] = kgen.undef : !kgen.struct<()>
    # CHECK: call {{.*}}SomeType::__del__{{.*}}([[V0]])
    var unused = value


@export
fn copy_destroy(x: SomeType):
    drop_copy(x)
