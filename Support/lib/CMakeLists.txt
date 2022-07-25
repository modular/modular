#===- CMakeLists.txt ------------------------------------------*- cmake -*-===##
#
# This file is Modular Inc proprietary.
#
#===-----------------------------------------------------------------------===##

add_subdirectory(Benchmark)

add_modular_library(MSupport
  AlignedAlloc.cpp
  ConcatenationTree.cpp
  DType.cpp
  Error.cpp

  ADDITIONAL_HEADER_DIRS

  DEPENDS

  LINK_COMPONENTS

  LINK_LIBS PUBLIC
  )
