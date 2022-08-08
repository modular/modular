#===- CMakeLists.txt ------------------------------------------*- cmake -*-===##
#
# This file is Modular Inc proprietary.
#
#===-----------------------------------------------------------------------===##

add_subdirectory(KGENDialect)
add_subdirectory(HLKGENDialect)
add_subdirectory(MetaDialect)
add_subdirectory(POPDialect)

set(LLVM_TARGET_DEFINITIONS KGENPasses.td)
mlir_tablegen(KGENPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(MLIRKGENPasses)

