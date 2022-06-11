#===- CMakeLists.txt ------------------------------------------*- cmake -*-===##
#
# This file is Modular Inc proprietary.
#
#===-----------------------------------------------------------------------===##

set(LLVM_LINK_COMPONENTS
  Support
  )

add_llvm_tool(kgen-opt
  kgen-opt.cpp
)
llvm_update_compile_flags(kgen-opt)

target_link_libraries(kgen-opt
  PRIVATE
  KGENDialect
  GRTDialect
  
  MLIRIR
  MLIROptLib
  MLIRParser
  MLIRSupport
  MLIRTransforms
  )
