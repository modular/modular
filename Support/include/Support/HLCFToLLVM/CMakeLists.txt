##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

set(LLVM_TARGET_DEFINITIONS HLCFToLLVM.td)
mlir_tablegen(HLCFToLLVM.h.inc -gen-pass-decls)
add_public_tablegen_target(HLCFToLLVMIncGen)
