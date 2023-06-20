##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_subdirectory(include/Cache/CacheDialect)
add_subdirectory(include/Cache/CachePasses)
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
#             BASE_DIR ${MODULAR_DERIVED_PATH}/.kgen_cache/orc
#             HEADER_NAME ${MODULAR_BINARY_DIR}/KGEN/include/KGEN/ExecutionEngine/ORCCASID.h
#             INPUT_NAMES "kOrcRT" # Note that these must be globally unique - they're turned into C++ identifiers.
#             RELATIVE_INSTALL_DIR ".kgen_cache/orc")
function(cache_files target)
  cmake_parse_arguments(CACHE_MGR "" "BASE_DIR;HEADER_NAME;VERSION" "INPUTS;INPUT_NAMES;RELATIVE_INSTALL_DIR" ${ARGN})
  # Input names must all be unique, so we can use the first one.
  list(GET CACHE_MGR_INPUT_NAMES 0 UNIQUE_PATH)
  set(UNIQUE_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/${UNIQUE_PATH})
  # Set the variables the script is gonna need.
  set(FILENAME ${UNIQUE_BINARY_DIR}/hashes.out)
  set(OUTFILE ${CACHE_MGR_HEADER_NAME})
  set(NAMES ${CACHE_MGR_INPUT_NAMES})
  configure_file(${CMAKE_SOURCE_DIR}/Cache/cmake/ReadHashes.cmake.in ${UNIQUE_BINARY_DIR}/ReadHashes.cmake @ONLY)
  # Add a target to run that script to generate the header file.
  add_custom_target(${target} ALL
    COMMAND $<TARGET_FILE:cache-mgr> -base-dir=${CACHE_MGR_BASE_DIR} -input=${CACHE_MGR_INPUTS} -backend-version=${CACHE_MGR_VERSION} -o=${FILENAME}
    COMMAND ${CMAKE_COMMAND} -P ${UNIQUE_BINARY_DIR}/ReadHashes.cmake)

  install(DIRECTORY "${CACHE_MGR_BASE_DIR}/"
    DESTINATION ${CACHE_MGR_RELATIVE_INSTALL_DIR})
endfunction()
