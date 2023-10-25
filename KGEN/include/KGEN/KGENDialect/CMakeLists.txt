##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

# Generating KGEN filename prefix into the "kgen" C++ namespace.
add_mlir_dialect(KGEN kgen)
add_modular_doc(KGEN -gen-dialect-doc KGENDialect KGEN/docs/ -dialect=kgen)

add_mlir_interface(KGENInterfaces)

set(LLVM_TARGET_DEFINITIONS KGEN.td)
mlir_tablegen(KGENAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=kgen)
mlir_tablegen(KGENAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=kgen)
add_public_tablegen_target(MLIRKGENAttrsIncGen)
add_dependencies(MLIRKGENIncGen MLIRKGENAttrsIncGen)

set(LLVM_TARGET_DEFINITIONS KGENDialectBytecode.td)
mlir_tablegen(KGENDialectBytecode.cpp.inc -gen-bytecode -bytecode-dialect="kgen")
add_public_tablegen_target(MLIRKGENDialectBytecodeIncGen)
add_definitions(MLIRKGENIncGen MLIRKGENDialectBytecodeIncGen)

set(LLVM_TARGET_DEFINITIONS KGENEnums.td)
mlir_tablegen(KGENEnums.h.inc -gen-enum-decls)
mlir_tablegen(KGENEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(KGENEnumAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=kgen)
mlir_tablegen(KGENEnumAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=kgen)
add_public_tablegen_target(MLIRKGENEnumsIncGen)
add_dependencies(MLIRKGENIncGen MLIRKGENEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS KGENAttrInterfaces.td)
mlir_tablegen(KGENAttrInterfaces.h.inc -gen-attr-interface-decls)
mlir_tablegen(KGENAttrInterfaces.cpp.inc -gen-attr-interface-defs)
add_public_tablegen_target(MLIRKGENAttrInterfacesIncGen)
add_dependencies(MLIRKGENIncGen MLIRKGENAttrInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS KGENTypeInterfaces.td)
mlir_tablegen(KGENTypeInterfaces.h.inc -gen-type-interface-decls)
mlir_tablegen(KGENTypeInterfaces.cpp.inc -gen-type-interface-defs)
add_public_tablegen_target(MLIRKGENTypeInterfacesIncGen)
add_dependencies(MLIRKGENIncGen MLIRKGENTypeInterfacesIncGen)
