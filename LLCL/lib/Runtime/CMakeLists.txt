#===- CMakeLists.txt - LLCL Runtime Build ----------------------*- cmake -*-===#
#
# This file is Modular Inc proprietary.
#
#===------------------------------------------------------------------------===#

add_mlir_library(LLCLRuntime
  MallocAllocator.cpp
  Runtime.cpp
  
  ADDITIONAL_HEADER_DIRS

  DEPENDS

  LINK_COMPONENTS

  LINK_LIBS PUBLIC
  LLVMSupport
  )
