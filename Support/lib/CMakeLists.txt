#===- CMakeLists.txt - Modular Support Build -------------------*- cmake -*-===#
#
# This file is Modular Inc proprietary.
#
#===------------------------------------------------------------------------===#

add_mlir_library(MSupport
  AlignedAlloc.cpp
  Error.cpp
  
  ADDITIONAL_HEADER_DIRS

  DEPENDS

  LINK_COMPONENTS

  LINK_LIBS PUBLIC
  )
