# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s

# ===----------------------------------------------------------------------=== #
# Actual tests
# ===----------------------------------------------------------------------=== #


@fieldwise_init
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


@fieldwise_init
@nonmaterializable(Thing)
@register_passable
struct NMType(ImplicitlyCopyable):
    pass


# CHECK-LABEL: lit.fn @"metatypes()"
fn metatypes():
    # COM: Test that a local alias can retain type properties.
    # CHECK: lit.alias.decl [[T:\*"T.*]]: !mt_Thing = <!Thing>
    comptime T = Thing
    # CHECK-NEXT: [[TVAL:%.*]] = lit.call {{.*}}Thing::@"__init__(){{.*}}()
    # CHECK: call {{.*}}@Thing::@"foo({{.*}})"([[TVAL]])
    T().foo()
    # CHECK: call {{.*}}@Thing::@"bar()"
    T.bar()

    # COM: Test that binding to a generic type works.
    # CHECK: bound{{.*}}: !lit.generator<() -> !kgen.none> = <{{.*}}@"anytype[AnyTrivialRegType]()"<:type !Thing>>
    comptime bound = anytype[Thing]

    # COM: Test that result types are bound correctly.
    # CHECK: call {{.*}}@"anytype_result[AnyTrivialRegType]()"<:type !Thing>
    var v: Thing = anytype_result[Thing]()

    # COM: Test that argument type inference works correctly.
    # CHECK: call {{.*}}@"anytype_arg[AnyTrivialRegType]($0)"<:type !Thing>
    anytype_arg(v)

    # COM: Test inferring from a non-materializable type.
    comptime nm_alias = NMType()
    # CHECK: [[DNMVAL:%.*]] = lit.var.decl "anonymous*"
    # CHECK-NEXT: [[NMVAL:%.*]] = kgen.param.materialize: !NMType = <sugar_alias{{.*}}apply(:!lit.generator<() -> !NMType> @metatypes::@NMType::@"__init__()"){{.*}}>
    # CHECK-NEXT: lit.ref.store [[NMVAL]], [[DNMVAL]]
    # CHECK-NEXT: [[IMMUT:%.*]] = lit.ref.immut [[DNMVAL]]
    # CHECK-NEXT: [[MVAL:%.*]] = lit.call {{.*}}@Thing::@"__init__{{.*}}([[IMMUT]])
    # CHECK-NEXT: call {{.*}}@"anytype_arg[AnyTrivialRegType]($0)"<:type !Thing>([[MVAL]])
    anytype_arg(nm_alias)


# Stef's crazy metatype stress test.
@fieldwise_init
struct StefStressTest[x: Int]:
    @staticmethod
    fn increment() -> type_of(StefStressTest[x + 1]):
        while True:
            pass
        # return StefStressTest[x+1]  # Doesn't work yet.


fn use_int(a: Int):
    pass


# CHECK-LABEL: lit.fn @"access_param_from_metatype()"
fn access_param_from_metatype():
    # CHECK-NEXT: lit.alias.decl *"f1`": meta<!lit.struct<#StefStressTest <:!Int {1}>>>
    comptime f1 = StefStressTest[0].increment()
    # CHECK: [[TMP:%.*]] = kgen.param.constant: !Int = <{1}>
    # CHECK: lit.call {{.*}}@"use_int{{.*}}([[TMP]])
    use_int(f1.x)

    # CHECK: lit.call {{.*}}@"increment()"<:!Int {0}>()
    # CHECK: lit.call {{.*}}@"increment()"<:!Int {1}>()
    # CHECK: [[TMP:%.*]] = kgen.param.constant: !Int = <{2}>
    # CHECK-NEXT: lit.call {{.*}}@"use_int{{.*}}([[TMP]])
    use_int(StefStressTest[0].increment().increment().x)


# COM: we should handle mt_Int : mt_mt_Int -> copyable : any_trait<Copyable>
# correctly, since Int is a Copyable
fn meta_type_to_trait[T: type_of(Copyable), //, W: T](t: W):
    pass


fn meta_type_to_trait_driver():
    # CHECK: lit.call @metatypes::@"meta_type_to_trait[{{.*}}]<:!lit.anytrait<!Copyable> !mt_Int, :!mt_Int !Int>(%1)
    meta_type_to_trait[Int](1)
