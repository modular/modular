##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_library(MSupportBenchmark
  QuerySampleLibrary.cpp
  SystemUnderTest.cpp

  LINK_LIBS PUBLIC
  mlperf-loadgen
)
