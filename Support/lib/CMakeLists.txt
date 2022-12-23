##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_subdirectory(Benchmark)
add_subdirectory(DebugInfoDialect)
add_subdirectory(HLCFDialect)
add_subdirectory(HLCFToLLVM)
add_subdirectory(Interpreter)
add_subdirectory(MDialect)
add_subdirectory(ML)
add_subdirectory(Compiler)
add_subdirectory(Telemetry)

add_modular_library(MSupport
  AlignedAlloc.cpp
  CommonCLOptions.cpp
  ConcatenationTree.cpp
  Error.cpp
  HMAC.cpp
  Host.cpp
  MicroBenchmark.cpp
  TempFile.cpp
  TimeProfiler.cpp

  LINK_LIBS PUBLIC
  LLVMSupport

  DEPENDS
  MConfig
  )
