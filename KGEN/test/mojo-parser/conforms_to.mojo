# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s


struct Foo:
    pass


trait Bar:
    pass


comptime Alias = Bar
# CHECK:      lit.alias.decl *"CONFORMS_TO_CHECK
# CHECK-SAME: conforms_to(:!mt_Foo !Foo, [@conforms_to::@Bar, @std::@builtin::@stubs::@AnyType])
comptime CONFORMS_TO_CHECK = conforms_to(Foo, Alias)


# CHECK:      lit.alias.decl *"meta_meta_type_conforms_to
# CHECK-SAME: conforms_to(:!kgen.param<:meta<!mt_Int> *(0,0)> *(0,1), [@conforms_to::@Bar, @std::@builtin::@stubs::@AnyType])})>>
comptime meta_meta_type_conforms_to[
    MM: type_of(type_of(Int)), type: MM
] = conforms_to(type, Bar)

# CHECK:      lit.alias.decl *"any_trait_type_conforms_to
# CHECK-SAME: conforms_to(:!kgen.param<:!lit.anytrait<!AnyType> *(0,0)> *(0,1), [@conforms_to::@Bar, @std::@builtin::@stubs::@AnyType])})
comptime any_trait_type_conforms_to[
    MM: type_of(AnyType), type: MM
] = conforms_to(type, Bar)
