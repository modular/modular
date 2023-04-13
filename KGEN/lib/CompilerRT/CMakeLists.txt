##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_mlir_library(KGENCompilerRT
  InitIntelAMX.cpp
  Initialize.cpp
  LLCL.cpp
  Memory.cpp
  OutputChain.cpp
  Print.cpp
  Python.cpp
  Random.cpp
  System.cpp
  Tracing.cpp

  LINK_LIBS PUBLIC
  MSupport
  LLCLRuntime
)
