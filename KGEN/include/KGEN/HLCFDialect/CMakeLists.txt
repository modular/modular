##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_mlir_dialect(HLCF hlcf -dialect=hlcf)
add_modular_doc(HLCF -gen-dialect-doc HLCFDialect KGEN/docs/ -dialect=hlcf)
add_mlir_interface(HLCFInterfaces)

set(LLVM_TARGET_DEFINITIONS HLCF.td)
mlir_tablegen(HLCFAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=hlcf)
mlir_tablegen(HLCFAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=hlcf)
add_public_tablegen_target(MLIRHLCFAttrsIncGen)
add_dependencies(MLIRHLCFIncGen MLIRHLCFAttrsIncGen)

mlir_tablegen(HLCFEnums.h.inc -gen-enum-decls)
mlir_tablegen(HLCFEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRHLCFEnumsIncGen)
add_dependencies(MLIRHLCFIncGen MLIRHLCFEnumsIncGen)
