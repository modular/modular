##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

# This target is intended to appear exactly once in all top-level
# executables/dynamic libraries.
add_modular_library(AsyncRTRuntimeGlobals
  SHARED
  Globals.cpp

  LINK_LIBS
  PUBLIC
  LLVMSupport
)

target_include_directories(AsyncRTRuntimeGlobals
  PUBLIC
  ${CMAKE_SOURCE_DIR}/AsyncRT/include
)

target_compile_definitions(AsyncRTRuntimeGlobals PRIVATE MODULAR_BUILDING_LIBRARY)

if (TCMALLOC_LIBRARY)
  target_link_libraries(AsyncRTRuntimeGlobals PRIVATE libtcmalloc)
  target_include_directories(AsyncRTRuntimeGlobals PUBLIC ${TCMALLOC_PATH}/include)
endif()


# CAUTION: Only use this if you know that you won't be dlopen'ing another modular binary!
add_modular_library(AsyncRTRuntimeGlobals-static
  OBJECT # Always generate an obj.${name} target.
  Globals.cpp

  LINK_LIBS
  PUBLIC
  LLVMSupport
)

target_include_directories(AsyncRTRuntimeGlobals-static
  PUBLIC
  ${CMAKE_SOURCE_DIR}/AsyncRT/include
)
target_compile_definitions(AsyncRTRuntimeGlobals-static PRIVATE MODULAR_NO_EXPORT)
target_compile_definitions(obj.AsyncRTRuntimeGlobals-static PRIVATE MODULAR_NO_EXPORT)

if (TCMALLOC_LIBRARY)
  target_link_libraries(AsyncRTRuntimeGlobals-static PUBLIC libtcmalloc)
  target_link_libraries(obj.AsyncRTRuntimeGlobals-static PUBLIC libtcmalloc)
  target_include_directories(AsyncRTRuntimeGlobals-static PUBLIC ${TCMALLOC_PATH}/include)
endif()
