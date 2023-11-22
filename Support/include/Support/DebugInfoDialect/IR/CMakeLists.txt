##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_mlir_dialect(DebugInfo debuginfo -dialect=debuginfo)
add_modular_doc(DebugInfo
  -gen-dialect-doc DebugInfoDialect Support/docs/ -dialect=debuginfo)

set(LLVM_TARGET_DEFINITIONS DebugInfoExprAttrInterfaces.td)
mlir_tablegen(DebugInfoExprAttrInterfaces.h.inc -gen-attr-interface-decls)
mlir_tablegen(DebugInfoExprAttrInterfaces.cpp.inc -gen-attr-interface-defs)
add_public_tablegen_target(MLIRDebugInfoExprAttrInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS DebugInfo.td)
mlir_tablegen(DebugInfoEnums.h.inc -gen-enum-decls -dialect=debuginfo)
mlir_tablegen(DebugInfoEnums.cpp.inc -gen-enum-defs -dialect=debuginfo)
mlir_tablegen(DebugInfoAttrs.h.inc -gen-attrdef-decls -dialect=debuginfo)
mlir_tablegen(DebugInfoAttrs.cpp.inc -gen-attrdef-defs -dialect=debuginfo)
add_public_tablegen_target(MLIRDebugInfoAttrsIncGen)
add_dependencies(MLIRDebugInfoAttrsIncGen MLIRDebugInfoExprAttrInterfacesIncGen)
add_dependencies(MLIRDebugInfoIncGen MLIRDebugInfoAttrsIncGen)

set(LLVM_TARGET_DEFINITIONS DebugInfoDialectBytecode.td)
mlir_tablegen(DebugInfoDialectBytecode.cpp.inc -gen-bytecode -bytecode-dialect="debuginfo")
add_public_tablegen_target(MLIRDebugInfoDialectBytecodeIncGen)
add_definitions(MLIRDebugInfoIncGen MLIRDebugInfoDialectBytecodeIncGen)

add_mlir_interface(DebugInfoInterfaces)
