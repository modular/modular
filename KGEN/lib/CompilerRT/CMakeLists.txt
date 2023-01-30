##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

set(CMAKE_CXX_VISIBILITY_PRESET default)

add_modular_mlir_library(KGENCompilerRT SHARED
  Benchmark.cpp
  DebugAssert.cpp
  Initialize.cpp
  LLCL.cpp
  Print.cpp
  Tracing.cpp
  Random.cpp
  System.cpp
  InitIntelAMX.cpp

  LINK_LIBS PUBLIC
  MSupport
  LLCLRuntime
)
