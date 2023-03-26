##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_library(MSupport
  AlignedAlloc.cpp
  CommonCLOptions.cpp
  Error.cpp
  HMAC.cpp
  Host.cpp
  MicroBenchmark.cpp
  Target.cpp
  TempFile.cpp
  TimeProfiler.cpp

  LINK_LIBS PUBLIC
  LLVMSupport
  LLVMTarget
  LLVMTargetParser
  MLIRSupport

  DEPENDS
  )

add_subdirectory(ADT)
add_subdirectory(Benchmark)
add_subdirectory(MDialect)
add_subdirectory(ML)
add_subdirectory(Compiler)
add_subdirectory(Interpreter)
add_subdirectory(DebugInfoDialect)
