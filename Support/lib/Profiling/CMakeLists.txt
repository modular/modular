##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##
add_modular_library(MProfiling
  STATIC OBJECT # Always generate an obj.${name} target.

  MallocProfiler.cpp
  TimeProfiler.cpp

  LINK_LIBS PUBLIC
  MArchTargetMinimal
  )

# Only if MODULAR_MALLOC_PROFILER is set: link to malloc override lib.
modular_malloc_profile(MProfiling)
