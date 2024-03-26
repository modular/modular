##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_subdirectory(include/Cache/CacheDialect)
add_subdirectory(lib)
add_subdirectory(tools/cache-opt)
add_subdirectory(tools/cache-mgr)
add_subdirectory(test)
add_subdirectory(unittests)

# This function allows us to cache build artifacts from within cmake, and
# generate header files that contain those CAS IDs. Use like this (example
# stolen from KGEN/lib/ExecutionEngine):
#
# cache_files(cache_a_file
#             BASE_DIR ${MODULAR_DERIVED_PATH}/.mojo_cache/orc
#             HEADER_NAME ${MODULAR_BINARY_DIR}/KGEN/include/KGEN/ExecutionEngine/ORCCASID.h
#             HEADER_RELATIVE_NAME KGEN/ExecutionEngine/ORCCASID.h
#             INPUT_NAMES "kOrcRT" # Note that these must be globally unique - they're turned into C++ identifiers.
#             RELATIVE_INSTALL_DIR ".mojo_cache/orc")
function(cache_files target)
  set(options)
  set(one_value_keywords BASE_DIR HEADER_NAME HEADER_RELATIVE_NAME VERSION)
  set(multi_value_keywords INPUTS INPUT_NAMES RELATIVE_INSTALL_DIR)
  cmake_parse_arguments(
    PARSE_ARGV 1
    CACHE_MGR
    "${options}"
    "${one_value_keywords}"
    "${multi_value_keywords}"
  )

  # For debugging, but default to always for now.
  # This is the name of the directory used in `.derived/.mojo_cache`
  # for where to store the relevant CAS artifacts.  Before filing a bug
  # report, please check the contents there.
  include(CMakePrintHelpers)
  cmake_print_variables(CACHE_MGR_VERSION)

  set(script ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/cmake/gen-casid-header.py)
  set(args
    ${Python3_EXECUTABLE} ${script}
    --output-relative-path="${CACHE_MGR_HEADER_RELATIVE_NAME}"
    --output-header-file="${CACHE_MGR_HEADER_NAME}"
    --cache-mgr=$<TARGET_FILE:cache-mgr>
    --cas-base-dir="${CACHE_MGR_BASE_DIR}"
    --backend-version="${CACHE_MGR_VERSION}")

  foreach(input_name ${CACHE_MGR_INPUT_NAMES})
    list(APPEND args "--input-name=${input_name}")
  endforeach()
  list(APPEND args ${CACHE_MGR_INPUTS})

  add_custom_command(
    OUTPUT ${CACHE_MGR_HEADER_NAME}
    COMMAND ${args}
    COMMENT "Caching files for ${CACHE_MGR_HEADER_NAME}"
    DEPENDS KGENVersion ${script} ${CACHE_MGR_INPUTS})
  add_custom_target(
    ${target}
    DEPENDS ${CACHE_MGR_HEADER_NAME})

  install(DIRECTORY "${CACHE_MGR_BASE_DIR}/"
    DESTINATION ${CACHE_MGR_RELATIVE_INSTALL_DIR})
endfunction()
