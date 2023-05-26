##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_mlir_dialect(HLCF hlcf -dialect=hlcf)
add_modular_doc(HLCF -gen-dialect-doc HLCFDialect KGEN/docs/ -dialect=hlcf)
add_mlir_interface(HLCFInterfaces)
