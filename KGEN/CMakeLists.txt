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
add_subdirectory(lib/KGENDialect)
add_subdirectory(lib/HLKGENDialect)
add_subdirectory(lib/MetaDialect)
add_subdirectory(lib/POPDialect)
add_subdirectory(lib/LowerHLKGEN)
add_subdirectory(lib/KGENToLLVM)
add_subdirectory(lib/KernelElaborator)
add_subdirectory(lib/ExecutionEngine)
add_subdirectory(lib/CLOptions)
add_subdirectory(tools/kgen)
add_subdirectory(tools/kgen-opt)
add_subdirectory(tools/kgen-execute)

# This function takes a target name, a single source file, and a list of
# kernels to be processed. It cannot execute kernels, but it will generate
# the CMake targets needed to depend on them from C/C++ code. Usage:
#
#  add_kgen_kernel(exp
#    SOURCE exp.mlir
#    KERNELS run_exp
#    INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIRECTORY}/test/kernels")
#
# This function also respects `include_directories` directives whenever possible.
function(add_kgen_kernel target)
  cmake_parse_arguments(KGEN "" "SOURCE" "KERNELS;INCLUDE_DIRS" ${ARGN})

  # Get the include_directories directives that are scoped to this file and below.
  get_property(KGEN_INCLUDES DIRECTORY PROPERTY INCLUDE_DIRECTORIES)

  set(KGEN_INCLUDE_FLAGS "")
  foreach (dir ${KGEN_INCLUDES})
    list(APPEND KGEN_INCLUDE_FLAGS "-I=${dir}")
  endforeach ()

  foreach (dir ${KGEN_INCLUDE_DIRS})
    list(APPEND KGEN_INCLUDE_FLAGS "-I=${dir}")
  endforeach ()

  # Add a target that controls emission of all of the kernels.
  set(OBJS "")
  foreach (kernel ${KGEN_KERNELS})
    set(object_file_name "${kernel}${CMAKE_CXX_OUTPUT_EXTENSION}")
    add_custom_command(
      OUTPUT ${object_file_name}
      OUTPUT "${kernel}.h"
      COMMAND $<TARGET_FILE:kgen> -emit -kernel=${kernel}:${object_file_name} ${KGEN_INCLUDE_FLAGS} ${KGEN_SOURCE}
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      DEPENDS kgen ${KGEN_SOURCE}
    )

    add_library(${target}_${kernel} ${object_file_name})
    set_target_properties(${target}_${kernel} PROPERTIES LINKER_LANGUAGE C)

    list(APPEND OBJS ${object_file_name})
  endforeach ()

  add_library(${target} ${OBJS})
  set_target_properties(${target} PROPERTIES LINKER_LANGUAGE C)
endfunction()

# Testing infrastructure
#add_subdirectory(unittests)
add_subdirectory(test)
#add_subdirectory(integration_test)
