##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_subdirectory(KGENDialect)
add_subdirectory(LITDialect)
add_subdirectory(POPDialect)
add_subdirectory(ZAPDialect)

set(LLVM_TARGET_DEFINITIONS KGENPasses.td)
mlir_tablegen(KGENPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(MLIRKGENPasses)
