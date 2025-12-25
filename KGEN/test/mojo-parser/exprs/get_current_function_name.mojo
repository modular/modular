# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s

# Test __get_current_function_name() magic function returns the correct
# function name in various contexts.


# CHECK-LABEL: lit.fn @"test_simple_function
fn test_simple_function():
    # CHECK: #StringLiteral <:string "test_simple_function">
    var name = __get_current_function_name()
    _ = name


# CHECK-LABEL: lit.struct.decl @MyStruct
struct MyStruct:
    # should return an empty string
    # CHECK: lit.alias.decl *"comptimeField`": !lit.struct<#StringLiteral <:string "">>
    comptime comptimeField = __get_current_function_name()

    fn __init__(out self):
        pass

    # CHECK-LABEL: lit.fn @"my_method
    fn my_method(self):
        # CHECK: #StringLiteral <:string "my_method">
        var name = __get_current_function_name()
        _ = name

    # CHECK-LABEL: lit.fn @"my_static_method
    @staticmethod
    fn my_static_method():
        # CHECK: #StringLiteral <:string "my_static_method">
        var name = __get_current_function_name()
        _ = name


# CHECK-LABEL: lit.trait.decl @MyTrait
trait MyTrait:
    # CHECK-LABEL: lit.fn @"trait_method
    fn trait_method(self):
        # CHECK: #StringLiteral <:string "trait_method">
        var name = __get_current_function_name()
        _ = name


struct AnotherStruct(MyTrait):
    fn __init__(out self):
        pass


# CHECK-LABEL: lit.extension.decl @"extension:AnotherStruct"
__extension AnotherStruct:
    # CHECK-LABEL: lit.fn @"extension_method
    fn extension_method(self: AnotherStruct):
        # CHECK: #StringLiteral <:string "extension_method">
        var name = __get_current_function_name()
        _ = name


# CHECK-LABEL: lit.fn @"test_nested_function
fn test_nested_function():
    # CHECK-LABEL: lit.fn *"inner
    fn inner():
        # CHECK: #StringLiteral <:string "inner">
        var name = __get_current_function_name()
        _ = name

    inner()


# CHECK-LABEL: lit.fn @"test_unified_clsoures
fn test_unified_clsoures():
    var capture = 1

    fn closure[param: Int](arg: Int) unified {var capture}:
        fn nested_closure[param: Int](arg: Int) unified {var capture}:
            _ = param + arg + capture
            # CHECK: #StringLiteral <:string "nested_closure">
            var name = __get_current_function_name()
            _ = name

        nested_closure[param](arg)

    closure[1](1)


fn main():
    test_simple_function()
    test_nested_function()
    var s = MyStruct()
    s.my_method()
    MyStruct.my_static_method()
    var a = AnotherStruct()
    a.extension_method()
    a.trait_method()
    test_unified_clsoures()
