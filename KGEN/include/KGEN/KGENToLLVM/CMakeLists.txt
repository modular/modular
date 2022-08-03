#===- CMakeLists.txt ------------------------------------------*- cmake -*-===##
#
# This file is Modular Inc proprietary.
#
#===-----------------------------------------------------------------------===##

set(LLVM_TARGET_DEFINITIONS ConvertKGENToLLVM.td)
mlir_tablegen(ConvertKGENToLLVM.h.inc -gen-pass-decls -name ConvertKGENToLLVM)
add_public_tablegen_target(MLIRConvertKGENToLLVM)
