##===- CMakeLists.txt -----------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_mlir_dialect(Index index -dialect=index)
add_modular_doc(Index
  -gen-dialect-doc IndexDialect Support/docs/ -dialect=index)

set(LLVM_TARGET_DEFINITIONS Index.td)

mlir_tablegen(IndexAttrs.h.inc -gen-attrdef-decls -dialect=index)
mlir_tablegen(IndexAttrs.cpp.inc -gen-attrdef-defs -dialect=index)
add_public_tablegen_target(MLIRIndexAttrsIncGen)
add_dependencies(MLIRIndexIncGen MLIRIndexAttrsIncGen)

mlir_tablegen(IndexEnums.h.inc -gen-enum-decls)
mlir_tablegen(IndexEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRIndexEnumsIncGen)
add_dependencies(MLIRIndexIncGen MLIRIndexEnumsIncGen)
