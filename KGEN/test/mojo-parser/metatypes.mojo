# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen-translate -import-mojo --mojo-disable-builtins %s | FileCheck %s

# COM: Alias `!kgen.mlirtype` to isolate the test.
alias AnyType = __mlir_type.`!kgen.mlirtype`


@value
@register_passable("trivial")
struct Thing:
    fn __init__(x: NMType) -> Self:
        return Self {}

    fn foo(self):
        pass

    @staticmethod
    fn bar():
        pass


# COM: Test various things related to interfacing with generic types.
fn anytype[T: AnyType]():
    pass


fn anytype_arg[T: AnyType](x: T):
    pass


fn anytype_result[T: AnyType]() -> T:
    pass


@value
@nonmaterializable(Thing)
@register_passable
struct NMType:
    pass


# CHECK-LABEL: lit.func @"metatypes()"
fn metatypes():
    # COM: Test that a local alias can retain type properties.
    # CHECK: lit.alias.decl [[T:.*_T]]: metatype<[[THING:@.*]]> = <!Thing>
    alias T = Thing
    # CHECK: [[VAL:%.*]] = kgen.param.constant: !Thing =
    # CHECK: call {{.*}}@Thing::@"foo({{.*}})"([[VAL]])
    T().foo()
    # CHECK: call {{.*}}@Thing::@"bar()"
    T.bar()

    # COM: Test that binding to a generic type works.
    # CHECK: bound: !lit.signature<() -> !kgen.none> = <{{.*}}@"anytype[AnyType]()"<:type !Thing>>
    alias bound = anytype[Thing]

    # COM: Test that result types are bound correctly.
    # CHECK: call {{.*}}@"anytype_result[AnyType]()"<:type !Thing>
    let v: Thing = anytype_result[Thing]()

    # COM: Test that argument type inference works correctly.
    # CHECK: call {{.*}}@"anytype_arg[AnyType]($0)"<:type !Thing>
    anytype_arg(v)

    # COM: Test that metatypes are accepted as MLIR op result types.
    # CHECK: kgen.rebind %index2 : index to !Thing
    _ = __mlir_op.`kgen.rebind`[_type=Thing](__mlir_attr.`2 : index`)

    # COM: Test inferring from a non-materializable type.
    alias nm_alias = NMType()
    # CHECK: [[MVAL:%.*]] = kgen.param.constant: !Thing = <apply({{.*}}@Thing::@"__init__
    # CHECK: call {{.*}}@"anytype_arg[AnyType]($0)"<:type !Thing>([[MVAL]])
    anytype_arg(nm_alias)
