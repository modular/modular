#===- CMakeLists.txt - LLCL Support Build ----------------------*- cmake -*-===#
#
# This file is Modular Inc proprietary.
#
#===------------------------------------------------------------------------===#

add_mlir_library(LLCLSupport
  ReferenceCounted.cpp
  
  ADDITIONAL_HEADER_DIRS

  DEPENDS

  LINK_COMPONENTS

  LINK_LIBS PUBLIC
  )
