##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

# Generating LIT filename prefix into the "kgen" C++ namespace.
add_mlir_dialect(LIT lit)
add_modular_doc(LIT -gen-dialect-doc LITDialect KGEN/docs/ -dialect=lit)

set(LLVM_TARGET_DEFINITIONS LIT.td)

mlir_tablegen(LITAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=lit)
mlir_tablegen(LITAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=lit)
add_public_tablegen_target(MLIRLITAttrsIncGen)
add_dependencies(MLIRLITIncGen MLIRLITAttrsIncGen)

add_mlir_interface(LITInterfaces)
