# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# COM: This file is used to test imported from cached bytecode modules.

comptime StringLiteralAlias = __mlir_attr.`"foobar" : !kgen.string`


trait Trait:
    pass


struct FuncRef[fn_type: __mlir_type.`!kgen.non_struct_type`, f: fn_type]:
    pass


struct FuncRefField:
    var func_ref: FuncRef[fn() -> None, FuncRefField.foo]

    @staticmethod
    fn foo():
        pass
