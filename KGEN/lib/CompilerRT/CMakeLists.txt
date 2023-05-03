##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_library(KGENCompilerRT
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

target_compile_definitions(KGENCompilerRT PRIVATE MODULAR_BUILDING_COMPILERRT)
