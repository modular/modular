##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##
add_modular_library(MProfiling
  STATIC OBJECT # Always generate an obj.${name} target.

  TimeProfiler.cpp

  LINK_LIBS PUBLIC
  MArchTargetMinimal
  )
