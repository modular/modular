##===- CMakeLists.txt -----------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

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

# This function reads ${KGEN_LINK_LIBS} and puts the include flags into
# ${KGEN_INCLUDE_FLAGS}.
macro(_setup_libs)
  set(KGEN_LIB_DEPENDS "")
  foreach (lib ${KGEN_LINK_LIBS})
    if (NOT TARGET ${lib})
      message(FATAL_ERROR "${lib} is not a target, it can't be linked")
    endif ()
    get_target_property(lib_include_dir ${lib} INCLUDE_DIRECTORIES)
    list(APPEND KGEN_INCLUDE_FLAGS "-I=${lib_include_dir}")
    list(APPEND KGEN_LIB_DEPENDS "${lib}")
  endforeach ()
endmacro()

# This function takes a target name, a single source file, and a list of
# targets to link. It generates the pre-elaboration library files to be used
# by other kgen files.
#
#  add_kgen_library(lib
#    SOURCE library.mlir
#    LINK_LIBS someother-lib)
#
# This attaches the directory for the output to the target, so you can simply
# specify this target in the LINK_LIBS of another kgen_library or kgen_kernel
# and the include flags will be setup correctly.
function(add_kgen_library target)
  cmake_parse_arguments(KGEN "" "SOURCE" "LINK_LIBS" ${ARGN})

  _setup_libs()

  get_filename_component(outfile ${KGEN_SOURCE} NAME_WLE)

  add_custom_command(
    OUTPUT "${outfile}.mlir"
    COMMAND $<TARGET_FILE:kgen> -gen-lib ${KGEN_SOURCE} ${KGEN_INCLUDE_FLAGS} -o "${outfile}.mlir"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS kgen ${KGEN_SOURCE} ${KGEN_LIB_DEPENDS}
  )

  add_custom_target(${target} DEPENDS "${outfile}.mlir")
  set_target_properties(${target} PROPERTIES INCLUDE_DIRECTORIES ${CMAKE_CURRENT_BINARY_DIR})
endfunction()

# This function takes a target name, a single source file, and a list of
# kernels to be processed. It cannot execute kernels, but it will generate
# the CMake targets needed to depend on them from C/C++ code. Usage:
#
#  add_kgen_kernel(exp
#    SOURCE exp.mlir
#    KERNELS run_exp
#    LINK_LIBS lib)
#
# This function will pull its include flags from the linked libs to avoid
# having too many search paths.
function(add_kgen_kernel target)
  cmake_parse_arguments(KGEN "" "SOURCE" "KERNELS;LINK_LIBS" ${ARGN})

  _setup_libs()

  # Add a target that controls emission of all of the kernels.
  set(OBJS "")
  foreach (kernel ${KGEN_KERNELS})
    set(object_file_name "${kernel}${CMAKE_CXX_OUTPUT_EXTENSION}")
    add_custom_command(
      OUTPUT ${object_file_name}
      OUTPUT "${kernel}.h"
      COMMAND $<TARGET_FILE:kgen> -emit -kernel=${kernel}:${object_file_name} ${KGEN_INCLUDE_FLAGS} ${KGEN_SOURCE}
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      DEPENDS kgen ${KGEN_SOURCE} ${KGEN_LIB_DEPENDS}
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
