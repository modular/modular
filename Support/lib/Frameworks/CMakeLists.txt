##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_library(FrameworkSupport
  StatsReport.cpp

  LINK_LIBS PUBLIC
  MTelemetry
)
