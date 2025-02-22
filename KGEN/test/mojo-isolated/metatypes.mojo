# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s

# ===----------------------------------------------------------------------=== #
# Actual tests
# ===----------------------------------------------------------------------=== #


@value
@register_passable("trivial")
struct Thing:
    @implicit
    fn __init__(out self, x: NMType):
        return

    fn foo(self):
        pass

    @staticmethod
    fn bar():
        pass


# COM: Test various things related to interfacing with generic types.
fn anytype[T: AnyTrivialRegType]():
    pass


fn anytype_arg[T: AnyTrivialRegType](x: T):
    pass


fn anytype_result[T: AnyTrivialRegType]() -> T:
    pass


@value
@nonmaterializable(Thing)
@register_passable
struct NMType:
    pass


# CHECK-LABEL: lit.fn @"metatypes()"
fn metatypes():
    # COM: Test that a local alias can retain type properties.
    # CHECK: lit.alias.decl [[T:\*"T.*]]: !mt_Thing = <!Thing>
    alias T = Thing
    # CHECK: %anonymous2A = lit.var.decl 
    # CHECK-NEXT: lit.call {{.*}}Thing::@"__init__(){{.*}}(%anonymous2A)
    # CHECK-NEXT: [[TVAL:%.*]] = lit.ref.load %anonymous2A 
    # CHECK: call {{.*}}@Thing::@"foo({{.*}})"([[TVAL]])
    T().foo()
    # CHECK: call {{.*}}@Thing::@"bar()"
    T.bar()

    # COM: Test that binding to a generic type works.
    # CHECK: bound{{.*}}: !lit.generator<() -> !kgen.none> = <{{.*}}@"anytype[AnyTrivialRegType]()"<:type !Thing>>
    alias bound = anytype[Thing]

    # COM: Test that result types are bound correctly.
    # CHECK: call {{.*}}@"anytype_result[AnyTrivialRegType]()"<:type !Thing>
    var v: Thing = anytype_result[Thing]()

    # COM: Test that argument type inference works correctly.
    # CHECK: call {{.*}}@"anytype_arg[AnyTrivialRegType]($0)"<:type !Thing>
    anytype_arg(v)

    # COM: Test inferring from a non-materializable type.
    alias nm_alias = NMType()
    # CHECK: [[MVAL:%.*]] = kgen.param.constant: !Thing = <apply({{.*}}@Thing::@"__init__
    # CHECK: call {{.*}}@"anytype_arg[AnyTrivialRegType]($0)"<:type !Thing>([[MVAL]])
    anytype_arg(nm_alias)
