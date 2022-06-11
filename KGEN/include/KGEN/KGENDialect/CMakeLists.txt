#===- CMakeLists.txt ------------------------------------------*- cmake -*-===##
#
# This file is Modular Inc proprietary.
#
#===-----------------------------------------------------------------------===##

# Generating KGEN filename prefix into the "kgen" C++ namespace.
add_mlir_dialect(KGEN kgen)
add_modular_doc(KGEN -gen-dialect-doc KGENDialect KGEN/docs/)

set(LLVM_TARGET_DEFINITIONS KGEN.td)
mlir_tablegen(KGENAttrs.h.inc -gen-attrdef-decls -dialect=kgen)
mlir_tablegen(KGENAttrs.cpp.inc -gen-attrdef-defs -dialect=kgen)
add_public_tablegen_target(MLIRKGENAttrsIncGen)

add_dependencies(MLIRKGENIncGen MLIRKGENAttrsIncGen)
