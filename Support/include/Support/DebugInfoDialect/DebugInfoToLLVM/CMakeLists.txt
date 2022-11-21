##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

set(LLVM_TARGET_DEFINITIONS DebugInfoToLLVM.td)
mlir_tablegen(DebugInfoToLLVM.h.inc -gen-pass-decls -name DebugInfoToLLVM)
add_public_tablegen_target(DebugInfoToLLVMIncGen)
