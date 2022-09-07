##===- CMakeLists.txt -----------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

set(LLVM_TARGET_DEFINITIONS IndexToLLVM.td)
mlir_tablegen(IndexToLLVM.h.inc -gen-pass-decls)
add_public_tablegen_target(IndexToLLVMIncGen)
