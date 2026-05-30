# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated --verify-diagnostics %s | FileCheck %s


struct DT[a: Int]:
    pass


# CHECK-LABEL: lit.trait.decl @B
trait B:
    comptime a: Int

    # This is a default value
    # CHECK:      lit.alias.decl *"c`2": !Int = <sugar_member_alias(!kgen.param<:!B *"_Self`">, "a", #kgen.get_witness<:!B *"_Self`", "{{.*}}::B", "a">)>
    # CHECK-SAME:   {defaultedAssociatedAlias}
    comptime c = Self.a

    # This is a dependent default type alias
    # CHECK:      lit.alias.decl *"T`3": meta<!lit.struct<#DT <:!Int sugar_member_alias(!kgen.param<:!B *"_Self`">, "a", #kgen.get_witness<:!B *"_Self`", "{{.*}}::B", "a">)>>> = <@{{.*}}::@DT<:!Int sugar_member_alias(!kgen.param<:!B *"_Self`">, "a", #kgen.get_witness<:!B *"_Self`", "{{.*}}::B", "a">)>>
    # CHECK-SAME:   {defaultedAssociatedAlias}
    comptime T = DT[Self.a]


# CHECK-LABEL: lit.struct.decl @Foo
struct Foo(B):
    comptime a: Int = 1

    # CHECK: lit.alias.decl *"c`2": !Int = <sugar_member_alias(!Foo, "a", {1})>
    #
    # Make sure that trait._Self is replaced properly.
    # CHECK: lit.alias.decl *"T`3": meta<!lit.struct<#DT <:!Int sugar_member_alias(!Foo, "a", {1})>>> = <@default_associated_alias::@DT<:!Int sugar_member_alias(!Foo, "a", {1})>>


# COM: A defaulted alias provided by a refining trait must be honored by a struct
# COM: that conforms only to the refinement.

# CHECK-LABEL: lit.trait.decl @C
trait C:
    # CHECK:      lit.alias.decl *"T`{{[0-9]+}}": !AnyType
    # CHECK-NOT:    {defaultedAssociatedAlias}
    comptime T: AnyType

# CHECK-LABEL: lit.trait.decl @D
trait D(C):
    # CHECK:      lit.alias.decl *"T`{{[0-9]+}}": !AnyType = <!Int>
    # CHECK-SAME:   {defaultedAssociatedAlias}
    comptime T: AnyType = Int

# CHECK-LABEL: lit.struct.decl @Bar
struct Bar(D):
    # COM: The defaulted `T = Int` from D must materialize on the struct exactly
    # COM: once with `{defaultedAssociatedAlias}` preserved, and both conformances
    # COM: (to C, the abstract parent, and to D, the refining provider) must emit
    # COM: a witness binding `T = Int`.
    # CHECK:      lit.alias.decl *"T`{{[0-9]+}}": !AnyType = <!Int>
    # CHECK-SAME:   {defaultedAssociatedAlias}
    # CHECK:      kgen.conformance @"{{.*}}::C"
    # CHECK-NEXT:   kgen.witness "T" : !AnyType = !Int
    # CHECK:      kgen.conformance @"{{.*}}::D"
    # CHECK-NEXT:   kgen.witness "T" : !AnyType = !Int
    pass
