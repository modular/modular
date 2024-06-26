##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_mlir_dialect(M M -dialect=M)
add_modular_doc(M -gen-dialect-doc MDialect Support/docs/ -dialect=M)

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

set(LLVM_TARGET_DEFINITIONS MAttrInterfaces.td)
mlir_tablegen(MAttrInterfaces.h.inc -gen-attr-interface-decls)
mlir_tablegen(MAttrInterfaces.cpp.inc -gen-attr-interface-defs)
add_public_tablegen_target(MLIRMAttrInterfacesIncGen)
add_dependencies(MLIRMIncGen MLIRMAttrInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS MDialectBytecode.td)
mlir_tablegen(MDialectBytecode.cpp.inc -gen-bytecode -bytecode-dialect="M")
add_public_tablegen_target(MLIRMDialectBytecodeIncGen)
add_dependencies(MLIRMIncGen MLIRMDialectBytecodeIncGen)
