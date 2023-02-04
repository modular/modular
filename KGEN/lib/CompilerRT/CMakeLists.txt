##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_mlir_library(KGENCompilerRT
  Benchmark.cpp
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
