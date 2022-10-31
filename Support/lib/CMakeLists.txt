##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_subdirectory(Benchmark)
add_subdirectory(MDialect)
add_subdirectory(ML)
add_subdirectory(Compiler)

add_modular_library(MSupport
  AlignedAlloc.cpp
  BlobCache.cpp
  CommonCLOptions.cpp
  ConcatenationTree.cpp
  Error.cpp
  HMAC.cpp
  MicroBenchmark.cpp
  TempFile.cpp
  TimeProfiler.cpp

  LINK_LIBS PUBLIC
  LLVMSupport

  DEPENDS
  MConfig
  )
