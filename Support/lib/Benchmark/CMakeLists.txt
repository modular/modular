#===- CMakeLists.txt -------------------------------------------*- cmake -*-===#
#
# This file is Modular Inc proprietary.
#
#===------------------------------------------------------------------------===#

if (NOT BUILD_BENCHMARKS)
  # The target is disabled if not built with benchmarking support
  return()
endif (NOT BUILD_BENCHMARKS)

add_mlir_library(MSupportBenchmark
  QuerySampleLibrary.cpp
  SystemUnderTest.cpp

  ADDITIONAL_HEADER_DIRS

  DEPENDS

  LINK_COMPONENTS

  LINK_LIBS PUBLIC
  mlperf_loadgen
)
