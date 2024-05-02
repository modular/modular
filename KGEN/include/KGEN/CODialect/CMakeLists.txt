##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_mlir_dialect(CO co -dialect=co)
add_modular_doc(CO -gen-dialect-doc CODialect KGEN/docs/ -dialect=co)

set(LLVM_TARGET_DEFINITIONS CODialectBytecode.td)
mlir_tablegen(CODialectBytecode.cpp.inc -gen-bytecode -bytecode-dialect="co")
add_public_tablegen_target(MLIRCODialectBytecodeIncGen)
add_dependencies(MLIRPOPIncGen MLIRCODialectBytecodeIncGen)
