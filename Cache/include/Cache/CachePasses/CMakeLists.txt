##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

set(LLVM_TARGET_DEFINITIONS CachePasses.td)
mlir_tablegen(CachePasses.h.inc -gen-pass-decls)
add_public_tablegen_target(MLIRCachePassesIncGen)
