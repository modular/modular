##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

set(KGEN_SOURCE_DIR ${MODULAR_SOURCE_DIR}/KGEN)
set(KGEN_BINARY_DIR ${MODULAR_BINARY_DIR}/KGEN)

# add_kgen_translation_library is a wrapper around add_modular_mlir_library that
# automatically links the library into kgen-translate.
function(add_kgen_translation_library name)
  set_property(GLOBAL APPEND PROPERTY KGEN_TRANSLATION_LIBS ${name})
  add_modular_mlir_library(${ARGV} DEPENDS modular-headers)
endfunction()

add_subdirectory(include/KGEN)
add_subdirectory(lib/KGENDialect)
add_subdirectory(lib/LITDialect)
add_subdirectory(lib/POPDialect)
add_subdirectory(lib/ZAPDialect)
add_subdirectory(lib/LowerLIT)
add_subdirectory(lib/LowerZAPToPOP)
add_subdirectory(lib/KGENToLLVM)
add_subdirectory(lib/LowerToObject)
add_subdirectory(lib/CompilerRT)
add_subdirectory(lib/ExecutionEngine)
add_subdirectory(lib/Elaborator)
add_subdirectory(lib/CLOptions)
add_subdirectory(lib/ParseLit)
add_subdirectory(lib/Transforms)
add_subdirectory(tools/kgen)
add_subdirectory(tools/kgen-opt)
add_subdirectory(tools/kgen-execute)
add_subdirectory(tools/kgen-translate)

# This function reads ${KGEN_LINK_LIBS} and puts the include flags into
# ${KGEN_INCLUDE_FLAGS}.
macro(_setup_libs)
  set(KGEN_LIB_DEPENDS "")
  set(KGEN_INCLUDE_FLAGS "")
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
#    OUTPUT library.mlirbc
#    source_file_1.lib.mlir
#    source_file_2.lib.mlir
#    LINK_LIBS someother-lib)
#
# This attaches the directory for the output to the target, so you can simply
# specify this target in the LINK_LIBS of another kgen_library or kgen_kernel
# and the include flags will be setup correctly.
function(add_kgen_library target)
  cmake_parse_arguments(KGEN "SAVE_TEMPS" "OUTPUT" "LINK_LIBS" ${ARGN})

  _setup_libs()

  if (NOT KGEN_OUTPUT)
    set(KGEN_OUTPUT ${target}.lib.mlirbc)
  endif ()

  add_custom_command(
    OUTPUT ${KGEN_OUTPUT}
    COMMAND $<TARGET_FILE:kgen>
      $<$<BOOL:${KGEN_SAVE_TEMPS}>:-save-temps>
      -gen-lib ${KGEN_UNPARSED_ARGUMENTS} -o ${KGEN_OUTPUT}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS kgen ${KGEN_LIB_DEPENDS} ${KGEN_UNPARSED_ARGUMENTS} ${KGEN_LINK_LIBS}
  )

  add_custom_target(${target} DEPENDS ${KGEN_OUTPUT} ${KGEN_UNPARSED_ARGUMENTS} kgen ${KGEN_LIB_DEPENDS})
  set_target_properties(${target} PROPERTIES INCLUDE_DIRECTORIES ${CMAKE_CURRENT_BINARY_DIR})
endfunction()

# This function takes a target name, a single source file, and a list of
# kernels to be processed. It cannot execute kernels, but it will generate
# the CMake targets needed to depend on them from C/C++ code. Usage:
#
#  add_kgen_kernel(exp
#    SFF_DEBUG #< This prints information about the SelectFastestFunction
#              #< search process.
#
#    SOURCE exp.mlir
#    KERNELS run_exp
#    LINK_LIBS lib)
#
# This function will pull its include flags from the linked libs to avoid
# having too many search paths.
function(add_kgen_kernel target)
  cmake_parse_arguments(KGEN "SFF_DEBUG;ENABLE_SEARCH" "SOURCE;OUTPUT_NAME" "KERNELS;LINK_LIBS" ${ARGN})

  _setup_libs()

  # If we built with MODULAR_KGEN_ENABLE_SEARCH=ON, then turn on
  # KGEN_ENABLE_SEARCH unconditionally.
  if (MODULAR_KGEN_ENABLE_SEARCH)
    set(KGEN_ENABLE_SEARCH ON)
  endif ()

  if (NOT KGEN_OUTPUT_NAME)
    get_filename_component(KGEN_OUTPUT_NAME ${KGEN_SOURCE} NAME_WE)
  endif ()

  # CMake emits build targets as relative paths but Ninja doesn't identify
  # absolute paths (in *.d) as relative paths (in build.ninja).
  set(object_file_name "${KGEN_OUTPUT_NAME}${CMAKE_CXX_OUTPUT_EXTENSION}")
  file(RELATIVE_PATH object_file_name_rel ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR}/${object_file_name})

  add_custom_command(
    OUTPUT ${object_file_name}
    OUTPUT "${KGEN_OUTPUT_NAME}.h"

    COMMAND $<TARGET_FILE:kgen> -emit
    -func=${kernel}
    $<$<BOOL:${KGEN_SFF_DEBUG}>:-debug-only="select-fastest-function">
    $<$<BOOL:${KGEN_ENABLE_SEARCH}>:-enable-search>
    -o ${object_file_name_rel} ${KGEN_INCLUDE_FLAGS} ${KGEN_SOURCE}
    -d ${object_file_name_rel}.d

    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    DEPFILE ${CMAKE_CURRENT_BINARY_DIR}/${object_file_name}.d
    DEPENDS kgen ${KGEN_SOURCE} ${KGEN_LIB_DEPENDS}
  )

  add_library(${target} ${object_file_name})
  add_dependencies(${target} ${KGEN_LIB_DEPENDS} kgen)
  set_target_properties(${target}
    PROPERTIES LINKER_LANGUAGE C
    INCLUDE_DIRECTORIES ${CMAKE_CURRENT_BINARY_DIR})
endfunction()

# Testing infrastructure
add_subdirectory(test)
add_subdirectory(unittests)
#add_subdirectory(integration_test)
