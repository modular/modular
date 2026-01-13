# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s

# ===----------------------------------------------------------------------=== #
# Helpers
# ===----------------------------------------------------------------------=== #

comptime struct_field_index_by_name[
    T: AnyType,
    name: StringLiteral,
]: Int = Int(
        mlir_value=__mlir_attr[
            `#kgen.struct_field_index_by_name<`,
            T,
            `, `,
            name.value,
            `> : index`,
        ]
    )

comptime struct_field_type_by_name[
    StructT: AnyType,
    name: StringLiteral,
] = __mlir_attr[
    `#kgen.struct_field_type_by_name<`,
    StructT,
    `, `,
    name.value,
    `> : !kgen.type`,
]

# ===----------------------------------------------------------------------=== #
# Parse-Time Folding
# ===----------------------------------------------------------------------=== #

trait HasInt:
    fn __init__(out self):
        ...

    fn get_int(self) -> Int:
        ...


@fieldwise_init
struct MyStruct(HasInt, ImplicitlyCopyable):
    fn get_int(self) -> Int:
        return 42


@fieldwise_init
struct MyParam[x: Int](HasInt, ImplicitlyCopyable):
    fn get_int(self) -> Int:
        return Self.x


# CHECK-LABEL: lit.struct.decl @MyWrapper
@fieldwise_init
struct MyWrapper[x: Int]:
    # CHECK: lit.struct.field nested : !MyStruct
    var nested: MyStruct
    # CHECK: lit.struct.field nested_param : !lit.struct<#MyParam <:!Int x>>
    var nested_param: MyParam[Self.x]


# CHECK-LABEL: lit.fn @"main()"
fn main():
    # Test struct_field_types folding
    # CHECK: lit.alias.decl *"fieldType0`{{[0-9]*}}": type = <!MyStruct>
    comptime fieldType0 = __struct_field_types(MyWrapper[37])[0]
    # CHECK: lit.alias.decl *"fieldType1`{{[0-9]*}}": type = <!lit.struct<#MyParam <:!Int {37}>>>
    comptime fieldType1 = __struct_field_types(MyWrapper[37])[1]
    # CHECK: lit.alias.decl *"nestedParamFields`{{[0-9]*}}": variadic<type> = <[]>
    comptime nestedParamFields = __struct_field_types(fieldType1)

    # Test struct_field_names folding
    # CHECK: lit.alias.decl *"fieldName0`{{[0-9]*}}": string = <"nested">
    comptime fieldName0 = __struct_field_names(MyWrapper[37])[0]
    # CHECK: lit.alias.decl *"fieldName1`{{[0-9]*}}": string = <"nested_param">
    comptime fieldName1 = __struct_field_names(MyWrapper[37])[1]

    # Test struct_field_index_by_name folding
    # CHECK: lit.alias.decl *"fieldIdx0`{{[0-9]*}}": !Int = <{0}>
    comptime fieldIdx0 = struct_field_index_by_name[MyWrapper[37], "nested"]
    # CHECK: lit.alias.decl *"fieldIdx1`{{[0-9]*}}": !Int = <{1}>
    comptime fieldIdx1 = struct_field_index_by_name[MyWrapper[37], "nested_param"]

    # Test struct_field_type_by_name folding
    # CHECK: lit.alias.decl *"fieldTypeByName0`{{[0-9]*}}": type = <!MyStruct>
    comptime fieldTypeByName0 = struct_field_type_by_name[MyWrapper[37], "nested"]
    # CHECK: lit.alias.decl *"fieldTypeByName1`{{[0-9]*}}": type = <!lit.struct<#MyParam <:!Int {37}>>>
    comptime fieldTypeByName1 = struct_field_type_by_name[MyWrapper[37], "nested_param"]
