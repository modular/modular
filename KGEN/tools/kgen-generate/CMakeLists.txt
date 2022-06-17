#===- CMakeLists.txt ------------------------------------------*- cmake -*-===##
#
# This file is Modular Inc proprietary.
#
#===-----------------------------------------------------------------------===##

set(LLVM_LINK_COMPONENTS
  Support
)

add_llvm_tool(kgen-generate
  GenerateKernels.cpp
  kgen-generate.cpp
)
llvm_update_compile_flags(kgen-generate)

target_link_libraries(kgen-generate
  PRIVATE
  KGENDialect
  MetaDialect
  MSupport

  MLIRIR
  MLIRArithmetic
  MLIROptLib
  MLIRParser
  MLIRSupport
  MLIRTransforms
)
