# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s


@fieldwise_init
struct ParamType[a: Int]:
    pass


struct SomeStruct[a: Int, b: Int, c: Int = 2]:
    @staticmethod
    def foo(arg: ParamType[Self.b]) -> Int:
        return Self.b


# This install default c = 2 and create `SomeStruct[1, ?, 2]`
# CHECK: lit.alias.decl *"c0`{{.*}}": meta<!lit.struct<#SomeStruct <:!Int {1}, :!Int ?, :!Int {2}>
comptime c0 = SomeStruct[1, _]

# This is SomeStruct[1, ?, ?]

# CHECK: lit.alias.decl *"c1`{{.*}}": meta<!lit.struct<#SomeStruct <:!Int {1}, :!Int ?, :!Int ?>
comptime c1 = SomeStruct[1, _, _]


# This is the same as SomeStruct[1, 3, 2], this is because
# SomeStruct[1, _] binds a and c (since [] always produces the MOST concrete type), the second [] binds b to 3.
# CHECK: lit.alias.decl *"c2`{{.*}}": meta<!lit.struct<#SomeStruct <:!Int {1}, :!Int {3}, :!Int {2}>
comptime c2 = SomeStruct[1, _][3]


def foo[a: Int, b: Int, c: Int, d: Int](x: SomeStruct[a, b, c]):
    pass


def test(x: SomeStruct[1, 2, 3]):
    # Make sure we handle call binding correctly without requiring:
    # foo[d=1, ...] or foo[_, _, _, d=1]

    # CHECK: lit.call tail @parameter_binding::@"foo[::Int,::Int,::Int,::Int]
    foo[d=1](x)

    # Although they are all valid syntax ofc.

    # CHECK: lit.call tail @parameter_binding::@"foo[::Int,::Int,::Int,::Int]
    foo[d=1, ...](x)

    # CHECK: lit.call tail @parameter_binding::@"foo[::Int,::Int,::Int,::Int]
    foo[_, _, _, d=1](x)

    # CHECK: lit.call @parameter_binding::@SomeStruct::@"foo({{.*}})"{{.*}}<:!Int {1}, :!Int {4}, :!Int {2}>
    var _ = SomeStruct[1].foo(ParamType[4]())


def foo[T: def[a: Int, b: Int](ParamType[b]) thin -> Int](param: ParamType[1]):
    # CHECK: lit.call tail{{.*}}bind_params(:{{.*}} T, :!Int {2}, :!Int {1})]
    T[2](param)


# We allow default values on inferred parameters.
# CHECK: lit.struct.decl @MySpan<mut: !Bool = {:scalar<bool> false}
struct MySpan[
    mut: Bool = False,
    //,
    origin: Origin[mut=mut],
]():
    pass
