# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s


struct Foo:
    pass


struct Baz:
    pass


trait Bar:
    pass


comptime Alias = Bar
# CHECK:      lit.alias.decl *"CONFORMS_TO_CHECK
# CHECK-SAME: conforms_to(:!mt_Foo !Foo, :meta<!Bar_AnyType> !Bar_AnyType)
comptime CONFORMS_TO_CHECK = conforms_to(Foo, Alias)


# CHECK:      lit.alias.decl *"meta_meta_type_conforms_to
# CHECK-SAME: conforms_to(:!kgen.param<:meta<!lit.meta<!Int>> *(0,0)> *(0,1), :meta<!Bar_AnyType> !Bar_AnyType)
comptime meta_meta_type_conforms_to[
    MM: type_of(type_of(Int)), type: MM
] = conforms_to(type, Bar)

# CHECK:      lit.alias.decl *"any_trait_type_conforms_to
# CHECK-SAME: conforms_to(:!kgen.param<:meta<!AnyType> *(0,0)> *(0,1), :meta<!Bar_AnyType> !Bar_AnyType)
comptime any_trait_type_conforms_to[
    MM: type_of(AnyType), type: MM
] = conforms_to(type, Bar)


# Opaque param_list operand: not a 1-element `ParamListAttr` literal, so the
# CHECK:      lit.alias.decl *"param_list_conforms_to
# CHECK-SAME: conforms_to(:param_list<!AnyType> *(0,0), :meta<!Bar_AnyType> !Bar_AnyType)
comptime param_list_conforms_to[*Ts: AnyType] = conforms_to(Ts.values, Bar)
