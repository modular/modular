##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

set(SOURCES
    CompactRuntimeTable.cpp
    Globals.cpp
    TypeInfoTable.cpp)

# This target is intended to appear exactly once in all top-level
# executables/dynamic libraries.
add_modular_library(LLCLRuntimeGlobals
  SHARED
  ${SOURCES}

  LINK_LIBS
  PUBLIC
  LLVMSupport
)

target_include_directories(LLCLRuntimeGlobals
  PUBLIC
  ${CMAKE_SOURCE_DIR}/LLCL/include
)

target_compile_definitions(LLCLRuntimeGlobals PRIVATE MODULAR_BUILDING_LIBRARY)

# CAUTION: Only use this if you know that you won't be dlopen'ing another modular binary!
add_modular_library(LLCLRuntimeGlobals-static
  OBJECT # Always generate an obj.${name} target.
  ${SOURCES}

  LINK_LIBS
  PUBLIC
  LLVMSupport
)

target_include_directories(LLCLRuntimeGlobals-static
  PUBLIC
  ${CMAKE_SOURCE_DIR}/LLCL/include
)
target_compile_definitions(LLCLRuntimeGlobals-static PRIVATE MODULAR_NO_EXPORT)
target_compile_definitions(obj.LLCLRuntimeGlobals-static PRIVATE MODULAR_NO_EXPORT)

