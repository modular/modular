##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_mlir_library(KGENCompilerRT
  Initialize.cpp
  LLCL.cpp
  Memory.cpp
  Print.cpp
  Random.cpp
  Tracing.cpp
  System.cpp
  InitIntelAMX.cpp

  LINK_LIBS PUBLIC
  MSupport
  LLCLRuntime
)
