# ===- CMakeLists.txt ------------------------------------------*- cmake -*-===##
#
# This file is Modular Inc proprietary.
#
# ===-----------------------------------------------------------------------===##

# Generating Meta filename prefix into the "kgen" C++ namespace.
add_mlir_dialect(Meta meta)
add_modular_doc(Meta -gen-dialect-doc MetaDialect KGEN/docs/)

# set(LLVM_TARGET_DEFINITIONS KGEN.td)
# mlir_tablegen(KGENAttrs.h.inc -gen-attrdef-decls -dialect=kgen)
# mlir_tablegen(KGENAttrs.cpp.inc -gen-attrdef-defs -dialect=kgen)
# add_public_tablegen_target(MLIRKGENAttrsIncGen)
# add_dependencies(MLIRKGENIncGen MLIRKGENAttrsIncGen)

# mlir_tablegen(KGENEnums.h.inc -gen-enum-decls)
# mlir_tablegen(KGENEnums.cpp.inc -gen-enum-defs)
# add_public_tablegen_target(MLIRKGENEnumsIncGen)
# add_dependencies(MLIRKGENIncGen MLIRKGENEnumsIncGen)
