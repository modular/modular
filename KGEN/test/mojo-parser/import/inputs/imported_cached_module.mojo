# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# COM: This file is used to test imported from cached bytecode modules.

comptime StringLiteralAlias = __mlir_attr.`"foobar" : !kgen.string`


trait Trait:
    pass


struct FuncRef[def_type: __mlir_type.`!kgen.non_struct_type`, f: def_type]:
    pass


struct FuncRefField:
    var func_ref: FuncRef[def() thin -> None, FuncRefField.foo]

    @staticmethod
    def foo():
        pass
