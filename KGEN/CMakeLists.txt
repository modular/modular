#===- CMakeLists.txt ------------------------------------------*- cmake -*-===##
#
# This file is Modular Inc proprietary.
#
#===-----------------------------------------------------------------------===##

set(KGEN_SOURCE_DIR ${MODULAR_SOURCE_DIR}/KGEN)
set(KGEN_BINARY_DIR ${MODULAR_BINARY_DIR}/KGEN)

# Add KGEN files to the include path
set(KGEN_INCLUDE_DIR ${KGEN_SOURCE_DIR}/include)
include_directories(${KGEN_INCLUDE_DIR})
set(KGEN_INCLUDE_DIR ${KGEN_BINARY_DIR}/include)
include_directories(${KGEN_INCLUDE_DIR})

# add_kgen_translation_library is a wrapper around add_modular_library that
# automatically links the library into kgen-translate.
function(add_kgen_translation_library name)
  set_property(GLOBAL APPEND PROPERTY KGEN_TRANSLATION_LIBS ${name})
  add_modular_library(${ARGV} DEPENDS modular-headers)
endfunction()

# add_kgen_library is a wrapper around add_modular_library
function(add_kgen_library name)
  set_property(GLOBAL APPEND PROPERTY KGEN_LIBS ${name})
  add_modular_library(${ARGV} DEPENDS modular-headers)
endfunction()

add_subdirectory(include/KGEN)
add_subdirectory(lib)
add_subdirectory(tools)

# Testing infrastructure
#add_subdirectory(unittests)
add_subdirectory(test)
#add_subdirectory(integration_test)


