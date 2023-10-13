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

set(LLVM_TARGET_DEFINITIONS LITEnums.td)
mlir_tablegen(LITEnums.h.inc -gen-enum-decls)
mlir_tablegen(LITEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRLITEnumsIncGen)
add_dependencies(MLIRLITIncGen MLIRLITEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS LITDialectBytecode.td)
mlir_tablegen(LITDialectBytecode.cpp.inc -gen-bytecode -bytecode-dialect="lit")
add_public_tablegen_target(MLIRLITDialectBytecodeIncGen)
add_definitions(MLIRLITIncGen MLIRLITDialectBytecodeIncGen)

add_mlir_interface(LITInterfaces)
