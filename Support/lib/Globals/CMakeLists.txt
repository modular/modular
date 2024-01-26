##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

set(SOURCES
  Globals.cpp)

# This target is intended to appear exactly once in all top-level
# executables/dynamic libraries.
add_modular_library(MSupportGlobals
  SHARED
  ${SOURCES}
)


target_include_directories(MSupportGlobals
  PUBLIC
  ${CMAKE_SOURCE_DIR}/Support/include
)

target_compile_definitions(MSupportGlobals PRIVATE MODULAR_BUILDING_LIBRARY)

# CAUTION: Only use this if you know that you won't be dlopen'ing another modular binary!
add_modular_library(MSupportGlobals-static
  OBJECT # Always generate an obj.${name} target.
  ${SOURCES}
)
target_include_directories(MSupportGlobals-static
  PUBLIC
  ${CMAKE_SOURCE_DIR}/Support/include
)
target_compile_definitions(MSupportGlobals-static PRIVATE MODULAR_NO_EXPORT)
target_compile_definitions(obj.MSupportGlobals-static PRIVATE MODULAR_NO_EXPORT)
