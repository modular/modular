##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_tool(llvm-module-split
  llvm-module-split.cpp

  LINK_LIBS PRIVATE
  KGENToolCommon
  KGENObjectCompiler
  LLVMSupport
  MSupportGlobals
)
