#===- CMakeLists.txt ------------------------------------------*- cmake -*-===##
#
# This file is Modular Inc proprietary.
#
#===-----------------------------------------------------------------------===##

add_modular_library(MSupportBenchmark
  QuerySampleLibrary.cpp
  SystemUnderTest.cpp

  ADDITIONAL_HEADER_DIRS

  DEPENDS

  LINK_COMPONENTS

  LINK_LIBS PUBLIC
  mlperf-loadgen
)
