##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

# This target is intended to appear exactly once in all top-level
# executables/dynamic libraries.
add_modular_library(MSupportGlobals
  SHARED
  GlobalProfilerContext.cpp
)


target_include_directories(MSupportGlobals
  PUBLIC
  ${CMAKE_SOURCE_DIR}/Support/include
)

target_compile_definitions(MSupportGlobals PRIVATE MODULAR_BUILDING_LIBRARY)
