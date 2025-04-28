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
    # CHECK-NEXT: [[TVAL:%.*]] = lit.call {{.*}}Thing::@"__init__(){{.*}}()
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
    # CHECK: [[DNMVAL:%.*]] = lit.var.decl "anonymous*"
    # CHECK-NEXT: [[NMVAL:%.*]] = kgen.param.materialize: !NMType = <apply(:!lit.generator<() -> !NMType> @metatypes::@NMType::@"__init__()")>
    # CHECK-NEXT: lit.ref.store [[NMVAL]], [[DNMVAL]]
    # CHECK-NEXT: [[IMMUT:%.*]] = lit.ref.immut [[DNMVAL]]
    # CHECK-NEXT: [[MVAL:%.*]] = lit.call {{.*}}@Thing::@"__init__{{.*}}([[IMMUT]])
    # CHECK-NEXT: call {{.*}}@"anytype_arg[AnyTrivialRegType]($0)"<:type !Thing>([[MVAL]])
    anytype_arg(nm_alias)


# Stef's crazy metatype stress test.
@value
struct StefStressTest[x: Int]:
    @staticmethod
    fn increment() -> __type_of(StefStressTest[x+1]):
      while True: pass
      #return StefStressTest[x+1]  # Doesn't work yet.

fn use_int(a: Int): pass

# CHECK-LABEL: lit.fn @"access_param_from_metatype()"
fn access_param_from_metatype():
    # CHECK-NEXT: lit.alias.decl *"f1`": meta<!lit.struct<#StefStressTest <:!Int {1}>>>
    alias f1 = StefStressTest[0].increment()
    # CHECK: [[TMP:%.*]] = kgen.param.constant: !Int = <{1}>
    # CHECK: lit.call {{.*}}@"use_int{{.*}}([[TMP]]) 
    use_int(f1.x)

    # CHECK: lit.call {{.*}}@"increment()"<:!Int {0}>()
    # CHECK: lit.call {{.*}}@"increment()"<:!Int {1}>()
    # CHECK: [[TMP:%.*]] = kgen.param.constant: !Int = <{2}>
    # CHECK-NEXT: lit.call {{.*}}@"use_int{{.*}}([[TMP]]) 
    use_int(StefStressTest[0].increment().increment().x)


