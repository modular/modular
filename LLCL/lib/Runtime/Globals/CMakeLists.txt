##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

# This target is intended to appear exactly once in all top-level
# executables/dynamic libraries.
add_modular_library(LLCLRuntimeGlobals
  SHARED
  CompactRuntimeTable.cpp
  Globals.cpp
  TypeInfoTable.cpp

  LINK_LIBS
  PUBLIC
  LLVMSupport
)

target_include_directories(LLCLRuntimeGlobals
  PUBLIC
  ${CMAKE_SOURCE_DIR}/LLCL/include
)

target_compile_definitions(LLCLRuntimeGlobals PRIVATE MODULAR_BUILDING_LIBRARY)
