# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# COM: This file is used to test imported from cached bytecode modules.

alias StringLiteralAlias = __mlir_attr.`"foobar" : !kgen.string`

var __global_variable = __mlir_attr.`1234 : index`


# COM: AnyType stub to allow testing without builtins.
trait AnyType:
    pass


trait Trait:
    pass


struct FuncRef[fn_type: __mlir_type.`!kgen.type`, f: fn_type]:
    pass


struct FuncRefField:
    var func_ref: FuncRef[fn () -> None, FuncRefField.foo]

    @staticmethod
    fn foo():
        pass
