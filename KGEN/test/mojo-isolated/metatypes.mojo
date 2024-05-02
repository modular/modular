# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | kgen-opt -verify-parameters | FileCheck %s

# ===----------------------------------------------------------------------=== #
# Actual tests
# ===----------------------------------------------------------------------=== #


@value
@register_passable("trivial")
struct Thing:
    fn __init__(inout self, x: NMType):
        return

    fn foo(self):
        pass

    @staticmethod
    fn bar():
        pass


# COM: Test various things related to interfacing with generic types.
fn anytype[T: AnyRegType]():
    pass


fn anytype_arg[T: AnyRegType](x: T):
    pass


fn anytype_result[T: AnyRegType]() -> T:
    pass


@value
@nonmaterializable(Thing)
@register_passable
struct NMType:
    pass


# CHECK-LABEL: lit.func @"metatypes()"
fn metatypes():
    # COM: Test that a local alias can retain type properties.
    # CHECK: lit.alias.decl [[T:\*"T.*]]: !mt_Thing = <!Thing>
    alias T = Thing
    # CHECK: [[TMP:%.*]] = lit.var.decl "anonymous*"
    # CHECK: lit.call {{.*}}__init__{{.*}}([[TMP]])
    # CHECK: [[VAL:%.*]] = lit.ref.load [[TMP]]
    # CHECK: call {{.*}}@Thing::@"foo({{.*}})"([[VAL]])
    T().foo()
    # CHECK: call {{.*}}@Thing::@"bar()"
    T.bar()

    # COM: Test that binding to a generic type works.
    # CHECK: bound{{.*}}: !lit.signature<() -> !kgen.none> = <{{.*}}@"anytype[AnyRegType]()"<:type !Thing>>
    alias bound = anytype[Thing]

    # COM: Test that result types are bound correctly.
    # CHECK: call {{.*}}@"anytype_result[AnyRegType]()"<:type !Thing>
    var v: Thing = anytype_result[Thing]()

    # COM: Test that argument type inference works correctly.
    # CHECK: call {{.*}}@"anytype_arg[AnyRegType]($0)"<:type !Thing>
    anytype_arg(v)

    # COM: Test inferring from a non-materializable type.
    alias nm_alias = NMType()
    # CHECK: [[MVAL:%.*]] = kgen.param.constant: !Thing = <apply_result_slot({{.*}}@Thing::@"__init__
    # CHECK: call {{.*}}@"anytype_arg[AnyRegType]($0)"<:type !Thing>([[MVAL]])
    anytype_arg(nm_alias)
