##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_mlir_dialect(M M -dialect=M)
add_modular_doc(M -gen-dialect-doc MDialect Support/docs -dialect=M)

set(LLVM_TARGET_DEFINITIONS M.td)
mlir_tablegen(MAttrs.h.inc -gen-attrdef-decls -dialect=M)
mlir_tablegen(MAttrs.cpp.inc -gen-attrdef-defs -dialect=M)
add_public_tablegen_target(MLIRMAttrsIncGen)
add_dependencies(MLIRMIncGen MLIRMAttrsIncGen)

set(LLVM_TARGET_DEFINITIONS MTypeInterfaces.td)
mlir_tablegen(MTypeInterfaces.h.inc -gen-type-interface-decls)
mlir_tablegen(MTypeInterfaces.cpp.inc -gen-type-interface-defs)
add_public_tablegen_target(MLIRMTypeInterfacesIncGen)
add_dependencies(MLIRMIncGen MLIRMTypeInterfacesIncGen)
