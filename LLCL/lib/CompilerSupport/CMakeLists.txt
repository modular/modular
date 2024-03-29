##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_library(LLCLCompilerSupport
  AsyncSideEffectMap.cpp
  LLVMThreadPool.cpp
  MLIRLocationDecoder.cpp

  LINK_LIBS PUBLIC
  LLCLSupport
  MLIRIR
)
target_link_libraries(MDialect PRIVATE LLCLCompilerSupport)
