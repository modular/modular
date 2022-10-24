##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

set(CMAKE_CXX_VISIBILITY_PRESET default)

add_modular_mlir_library(KGENCompilerRT SHARED
  DebugAssert.cpp
  Initialize.cpp
  Print.cpp
  Random.cpp
)
