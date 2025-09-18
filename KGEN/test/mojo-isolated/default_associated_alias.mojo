# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated --verify-diagnostics %s


struct DT[a: Int]:
    pass


# CHECK-LABEL: lit.trait.decl @B
trait B:
    alias a: Int

    # This is a default value
    # CHECK:      lit.alias.decl *"c`2": !Int = <#kgen.get_witness<:!B *"_Self`", "{{.*}}::B", "a">>
    # CHECK-SAME:   {defaultedAssociatedAlias}
    alias c = Self.a

    # This is a dependent default type alias
    # CHECK:      lit.alias.decl *"T`3": meta<!lit.struct<#DT <:!Int #kgen.get_witness<:!B *"_Self`", "{{.*}}::B", "a">>>> = <@{{.*}}::@DT<:!Int #kgen.get_witness<:!B *"_Self`", "{{.*}}::B", "a">>>
    # CHECK-SAME:   {defaultedAssociatedAlias}
    alias T = DT[Self.a]


# CHECK-LABEL: lit.struct.decl @Foo
struct Foo(B):
    alias a: Int = 1

    # TODO: can parser fold it?
    #
    # CHECK: lit.alias.decl *"c`2": !Int = <#kgen.get_witness<:!mt_Foo !Foo, "{{.*}}::B", "a">>
    #
    # Make sure that trait._Self is replaced properly.
    # CHECK: lit.alias.decl *"T`3": meta<!lit.struct<#DT <:!Int #kgen.get_witness<:!mt_Foo !Foo, "{{.*}}::B", "a">>>> = <@{}::@DT<:!Int #kgen.get_witness<:!mt_Foo !Foo, "{}::B", "a">>>
