# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# COM: This file is used to test imported from cached bytecode modules.

alias StringLiteralAlias = __mlir_attr.`"foobar" : !kgen.string`
var global_variable = __mlir_attr.`1234 : index`


# COM: AnyType stub to allow testing without builtins.
trait AnyType:
    pass


trait Trait:
    pass
